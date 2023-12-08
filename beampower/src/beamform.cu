#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define BLOCKSIZE 512
#define KILOBYTE 1024
#define MEGABYTE 1024*1024 

extern "C"{
#include "beamform_gpu.h"

inline void gpuAssert(
        cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void __global__ _find_minmax_time_delays_ker(
        int* time_delays,
        float* weights,
        size_t n_sources,
        size_t n_stations,
        size_t n_phases,
        int* time_delays_minmax
        ){

    /* Find the minimum and maximum time_delays for each point of the grid.
     * Even indexes correspond to minimum time_delays,
     * odd indexes correspond to maximum time_delays. */

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_sources) return; // skip threads that are out-of-bound

    int min_time_delay = INT_MAX;
    int max_time_delay = INT_MIN;
    int time_delay;
    size_t weight_offset;
    size_t mv_offset;

    for (size_t s=0; s<n_stations; s++){
        weight_offset = i*n_stations + s;
        if (weights[weight_offset] == 0.) continue; // the station is not used

        for (size_t p=0; p<n_phases; p++){
            mv_offset = i*n_stations*n_phases + s*n_phases + p;
            time_delay = time_delays[mv_offset];
            if (time_delay > max_time_delay) max_time_delay = time_delay;
            if (time_delay < min_time_delay) min_time_delay = time_delay;
            }
    }
    time_delays_minmax[i * 2 + 0] = min_time_delay;
    time_delays_minmax[i * 2 + 1] = max_time_delay;

}

void __global__ _beam(
        float *waveform_features,
        int *time_delays,
        int *time_delays_minmax,
        float *weights,
        int global_time_index,
        size_t n_samples,
        size_t n_stations,
        size_t n_phases, 
        size_t dim0_beam,
        float *beam){

    size_t i = blockIdx.x; // source index
    size_t t_idx = threadIdx.x; // thread-private time counter
    int signed_t_idx = (int)threadIdx.x; // thread-private index in int format
    float beam_i = 0.; // sum
    size_t feature_offset; // position on input pointer
    // number of elements to store in shared memory
    size_t size_time_delays = n_stations*n_phases;
    // declare shared arrays
    extern __shared__ int shared[];
    int *time_delays_s = &shared[0];
    float *weights_s = (float *)&shared[size_time_delays];

    // read data into shared memory
    while (t_idx < size_time_delays){
        time_delays_s[t_idx] = time_delays[i*n_stations*n_phases + t_idx];
        if (t_idx < n_stations){
            weights_s[t_idx] = weights[i*n_stations + t_idx];
        }
        t_idx += blockDim.x;
    }
    // wait for all threads to be done with reading
    __syncthreads();

    // compute this beam only if it stays within time bounds
    if (
            ((global_time_index + signed_t_idx + time_delays_minmax[2*i + 0]) >= 0)
            & 
            ((global_time_index + signed_t_idx + time_delays_minmax[2*i + 1]) < n_samples)
            ){
        // start shift and stack
        for (size_t s=0; s<n_stations; s++){
            for (size_t p=0; p<n_phases; p++){
                feature_offset = s * n_samples * n_phases + p\
                                + n_phases * time_delays_s[s * n_phases + p]\
                                + n_phases * threadIdx.x;
                beam_i += weights_s[s] * waveform_features[feature_offset];
            }
        }
        // update beam
        beam[i*dim0_beam + threadIdx.x] = beam_i;
    }
}

void __global__ _beam_check_out_of_bounds(
        float *waveform_features,
        int *time_delays,
        int *time_delays_minmax,
        float *weights,
        int global_time_index,
        size_t n_samples,
        size_t n_stations,
        size_t n_phases, 
        size_t dim0_beam,
        float *beam
        ){

    size_t i = blockIdx.x; // source index
    size_t t_idx = threadIdx.x; // thread-private time counter
    int signed_t_idx = (int)threadIdx.x; // thread-private index in int format
    float beam_i = 0.; // sum
    size_t feature_offset; // position on input pointer
    // number of elements to store in shared memory
    size_t size_time_delays = n_stations*n_phases;
    // declare shared arrays
    extern __shared__ int shared[];
    int *time_delays_s = &shared[0];
    float *weights_s = (float *)&shared[size_time_delays];

    // read data into shared memory
    while (t_idx < size_time_delays){
        time_delays_s[t_idx] = time_delays[i*n_stations*n_phases + t_idx];
        if (t_idx < n_stations){
            weights_s[t_idx] = weights[i*n_stations + t_idx];
        }
        t_idx += blockDim.x;
    }
    // wait for all threads to be done with reading
    __syncthreads();

    // start shift and stack
    for (size_t s=0; s<n_stations; s++){
        for (size_t p=0; p<n_phases; p++){
            // check temporal bounds (flexible mode)
            if (
                    ((global_time_index + signed_t_idx + time_delays_s[s * n_phases + p]) >= 0)
                    & 
                    ((global_time_index + signed_t_idx + time_delays_s[s * n_phases + p]) < n_samples)
                    ){
                feature_offset = s * n_samples * n_phases + p\
                                + n_phases * time_delays_s[s * n_phases + p]\
                                + n_phases * threadIdx.x;
                beam_i += weights_s[s] * waveform_features[feature_offset];
            }
        }
    }
    // update beam
    beam[i*dim0_beam + threadIdx.x] = beam_i;
}


void __global__ _beam_max(float *beam, size_t n_sources, float *beam_max,
        int *source_index_beam_max){

    float max_beam = FLT_MIN;
    int max_beam_index;

    // loop over all sources and search for the maximum
    for (int i=0; i<n_sources; i++){
        if (beam[i*blockDim.x + threadIdx.x] > max_beam){
            max_beam = beam[i*blockDim.x + threadIdx.x];
            max_beam_index = i;
        }
    }
    
    // update output arrays
    beam_max[threadIdx.x] = max_beam;
    source_index_beam_max[threadIdx.x] = max_beam_index;

}

void beamform_max(
        float* waveform_features,
        int* time_delays,
        float* weights,
        size_t n_samples,
        size_t n_sources,
        size_t n_stations,
        size_t n_phases,
        int out_of_bounds,
        float* beam_max, 
        int* source_index_beam_max
        ){

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their time_delays. The output is a vector with length
     * (n_samples x n_sources), which therefore can potentially be very large
     * for long time series. This function is fit for monitoring the network
     * response in 4D (time and space) with applications for event detection
     * but also rupture progation imaging (back-projection). */

    int nGPUs = 0;
    size_t Mb = MEGABYTE;
    //cudaError_t cuda_result;

    // count the number of available GPUs
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(nGPUs);

    // compute the number of sources processed by each GPU
    size_t n_sources_per_GPU = n_sources/nGPUs + 1;

    // compute the amount of shared memory requested by _beam
    size_t shared_mem = n_stations*n_phases*sizeof(int)\
                        + n_stations*sizeof(float);

    // start a parallel section to distribute tasks across GPUs
#pragma omp parallel firstprivate(n_sources_per_GPU, nGPUs)\
    shared(waveform_features, time_delays, weights, beam_max)
    {
        // associate thread to a single GPU and get
        // GPU characteristics such as memory capacity
        int id = omp_get_thread_num();
        cudaSetDevice(id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, id);

        // Card-dependent settings: prefer L1 cache or shared memory
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        // Check that the amount of shared memory is achievable
        size_t max_shared_mem = props.sharedMemPerBlock;
        if (shared_mem > max_shared_mem){
            printf("Too much shared memory requested on GPU %d"\
                   " (%zu kb requested vs %zu kb available)."\
                   "Consider reducing the number of stations processed "\
                   "at once.\n", id, shared_mem/KILOBYTE,
                   max_shared_mem/KILOBYTE);
            exit(0);
        }


        // compute the start and end indexes of the grid sources
        // processed by the GPU
        size_t src_idx_start = id*n_sources_per_GPU;
        size_t src_idx_end = (id+1)*n_sources_per_GPU;
        if (src_idx_end > n_sources){
            src_idx_end = n_sources;
            n_sources_per_GPU = src_idx_end - src_idx_start;
        }

        // declare device pointers
        float *waveform_features_d;
        int *time_delays_d;
        int *time_delays_minmax_d;
        float *weights_d;
        float *beam_d;
        float *beam_max_d;
        int *source_index_beam_max_d;
        // declare host pointers
        float *beam_max_thread;
        int *source_index_beam_max_thread;

        // size of arrays on device
        size_t sizeofdata = n_stations*n_samples*n_phases*sizeof(float);
        size_t sizeoftime_delays = n_sources_per_GPU*n_stations*n_phases*sizeof(int);
        size_t sizeoftime_delays_minmax = 2*n_sources_per_GPU*sizeof(int);
        size_t sizeofweights = n_sources_per_GPU*n_stations*sizeof(float);
        size_t sizeofbeam = BLOCKSIZE*n_sources_per_GPU*sizeof(float);
        size_t sizeofbeam_max = n_samples*sizeof(float);
        size_t sizeoftotal = sizeofdata + sizeoftime_delays + sizeoftime_delays_minmax\
                             + sizeofweights + sizeofbeam + 2*sizeofbeam_max;

        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (sizeoftotal > freeMem) {
            printf("%zu Mb are requested on GPU #%i whereas it has only %zu free Mb.\n",
                   sizeoftotal/Mb, id, freeMem/Mb);
            printf("Consider reducing the duration of the seismograms processed"\
                   " at once, or downsample the source grid.\n");
            exit(0);
        }


        // allocate GPU memory
        cudaMalloc((void**)&waveform_features_d, sizeofdata);
        cudaMalloc((void**)&time_delays_d, sizeoftime_delays);
        cudaMalloc((void**)&time_delays_minmax_d, sizeoftime_delays_minmax);
        cudaMalloc((void**)&weights_d, sizeofweights);
        cudaMalloc((void**)&beam_d, sizeofbeam);
        cudaMalloc((void**)&beam_max_d, sizeofbeam_max);
        cudaMalloc((void**)&source_index_beam_max_d, sizeofbeam_max);
        // declare host pointers and allocate CPU memory
        beam_max_thread = (float *)malloc(sizeofbeam_max);
        source_index_beam_max_thread = (int *)malloc(sizeofbeam_max);

        // transfer data from host (CPU) to device (GPU)
        cudaMemcpy(
                waveform_features_d, waveform_features, sizeofdata, cudaMemcpyHostToDevice
                );
        cudaMemcpy(
                time_delays_d,
                time_delays + src_idx_start*n_stations*n_phases,
                sizeoftime_delays, 
                cudaMemcpyHostToDevice
               );
        cudaMemcpy(
                weights_d,
                weights + src_idx_start*n_stations, 
                sizeofweights,
                cudaMemcpyHostToDevice
                );

        // compute time_delays min and max
        _find_minmax_time_delays_ker<<<n_sources_per_GPU/BLOCKSIZE+1, BLOCKSIZE>>>(
                time_delays_d,
                weights_d, 
                n_sources_per_GPU,
                n_stations,
                n_phases,
                time_delays_minmax_d
                );

        // initialize beam_max and source_index_beam_max
        cudaMemset(beam_max_d, 0., sizeofbeam_max);
        cudaMemset(source_index_beam_max_d, 0., sizeofbeam_max);

        // initialize GPU time index
        int time_GPU = 0;

        // compute network response
        while (time_GPU < n_samples){
            // initialize beam_d to zeros
            cudaMemset(beam_d, 0., sizeofbeam);

            // backproject the wavefield onto n_sources_per_GPU
            // grid locations and at BLOCKSIZE time locations
            if (out_of_bounds == 0){
                // check out-of-bound operations for the whole network
                _beam<<<n_sources_per_GPU, BLOCKSIZE, shared_mem>>>(
                        waveform_features_d + n_phases*time_GPU,
                        time_delays_d,
                        time_delays_minmax_d,
                        weights_d,
                        time_GPU,
                        n_samples,
                        n_stations,
                        n_phases,
                        BLOCKSIZE,
                        beam_d
                        );
            }
            else if (out_of_bounds == 1){
                // check out-of-bound operations separately for each station
                _beam_check_out_of_bounds<<<n_sources_per_GPU, BLOCKSIZE, shared_mem>>>(
                        waveform_features_d + n_phases*time_GPU,
                        time_delays_d,
                        time_delays_minmax_d,
                        weights_d,
                        time_GPU,
                        n_samples,
                        n_stations,
                        n_phases,
                        BLOCKSIZE,
                        beam_d
                        );
            }

            // find the maximum beam_max and beam_max source index across the
            // n_sources_per_GPU grid locations and at the BLOCKSIZE time
            // locations
            _beam_max<<<1, BLOCKSIZE>>>(
                    beam_d,
                    n_sources_per_GPU,
                    beam_max_d + time_GPU,
                    source_index_beam_max_d + time_GPU
                    );

            // increment time
            time_GPU += BLOCKSIZE;
        }

        // get results back to the host
        cudaMemcpy(beam_max_thread, beam_max_d, sizeofbeam_max, cudaMemcpyDeviceToHost);
        cudaMemcpy(
                source_index_beam_max_thread,
                source_index_beam_max_d,
                sizeofbeam_max,
                cudaMemcpyDeviceToHost
                );

        // wait for all GPUs to finish processing their part of the grid
        cudaDeviceSynchronize();

        // return an error if something happened in the kernel
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // critical section to merge all the single-GPU beam_max into one
#pragma omp critical
        for (size_t t=0; t<n_samples; t++){
            {
                if (beam_max_thread[t] > beam_max[t]){
                    beam_max[t] = beam_max_thread[t];
                    source_index_beam_max[t] = src_idx_start+source_index_beam_max_thread[t];
                }
            }
        }

        // free memory
        cudaFree(waveform_features_d);
        cudaFree(time_delays_d);
        cudaFree(time_delays_minmax_d);
        cudaFree(weights_d);
        cudaFree(beam_d);
        cudaFree(beam_max_d);
        cudaFree(source_index_beam_max_d);

        // done!

    } // omp parallel

}

void beamform(
        float* waveform_features,
        int* time_delays,
        float* weights,
        size_t n_samples,
        size_t n_sources,
        size_t n_stations,
        size_t n_phases,
        int out_of_bounds,
        float* beam
        ){

    int nGPUs = 0;
    size_t Mb = MEGABYTE;
    //cudaError_t cuda_result;

    // count the number of available GPUs
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(nGPUs);

    // compute the number of sources processed by each GPU
    size_t n_sources_per_GPU = n_sources/nGPUs + 1;

    // compute the amount of shared memory requested by _beam
    size_t shared_mem = n_stations*n_phases*sizeof(int)\
                        + n_stations*sizeof(float);

    // start a parallel section to distribute tasks across GPUs
#pragma omp parallel firstprivate(n_sources_per_GPU, nGPUs)\
    shared(waveform_features, time_delays, weights, beam)
    {
        // associate thread to a single GPU and get
        // GPU characteristics such as memory capacity
        int id = omp_get_thread_num();
        cudaSetDevice(id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, id);

        // Card-dependent settings: prefer L1 cache or shared memory
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        // Check that the amount of shared memory is achievable
        size_t max_shared_mem = props.sharedMemPerBlock;
        if (shared_mem > max_shared_mem){
            printf("Too much shared memory requested on GPU %d"\
                   " (%zu kb requested vs %zu kb available)."\
                   "Consider reducing the number of stations processed "\
                   "at once.\n", id, shared_mem/KILOBYTE,
                   max_shared_mem/KILOBYTE);
            exit(0);
        }


        // compute the start and end indexes of the grid sources
        // processed by the GPU
        size_t src_idx_start = id*n_sources_per_GPU;
        size_t src_idx_end = (id+1)*n_sources_per_GPU;
        if (src_idx_end > n_sources){
            src_idx_end = n_sources;
            n_sources_per_GPU = src_idx_end - src_idx_start;
        }

        // declare device pointers
        float *waveform_features_d;
        int *time_delays_d;
        int *time_delays_minmax_d;
        float *weights_d;
        float *beam_d;

        // size of arrays on device
        size_t sizeofdata = n_stations*n_samples*n_phases*sizeof(float);
        size_t sizeoftime_delays = n_sources_per_GPU*n_stations*n_phases*sizeof(int);
        size_t sizeoftime_delays_minmax = 2*n_sources_per_GPU*sizeof(int);
        size_t sizeofweights = n_sources_per_GPU*n_stations*sizeof(float);
        size_t sizeofbeam = n_sources_per_GPU*n_samples*sizeof(float);
        size_t sizeoftotal = sizeofdata + sizeoftime_delays + sizeoftime_delays_minmax\
                             + sizeofweights + sizeofbeam;

        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (sizeoftotal > freeMem) {
            printf("%zu Mb are requested on GPU #%i whereas it has only %zu free Mb.\n",
                   sizeoftotal/Mb, id, freeMem/Mb);
            printf("Consider reducing the duration of the seismograms processed"\
                   " at once, or downsample the source grid.\n");
            exit(0);
        }


        // allocate GPU memory
        cudaMalloc((void**)&waveform_features_d, sizeofdata);
        cudaMalloc((void**)&time_delays_d, sizeoftime_delays);
        cudaMalloc((void**)&time_delays_minmax_d, sizeoftime_delays_minmax);
        cudaMalloc((void**)&weights_d, sizeofweights);
        cudaMalloc((void**)&beam_d, sizeofbeam);

        // transfer data from host (CPU) to device (GPU)
        cudaMemcpy(
                waveform_features_d, waveform_features, sizeofdata, cudaMemcpyHostToDevice
                );
        cudaMemcpy(
                time_delays_d,
                time_delays + src_idx_start*n_stations*n_phases,
                sizeoftime_delays,
                cudaMemcpyHostToDevice
                );
        cudaMemcpy(
                weights_d,
                weights + src_idx_start*n_stations,
                sizeofweights,
                cudaMemcpyHostToDevice
                );

        // compute time_delays min and max
        _find_minmax_time_delays_ker<<<n_sources_per_GPU/BLOCKSIZE+1, BLOCKSIZE>>>(
                time_delays_d,
                weights_d,
                n_sources_per_GPU,
                n_stations,
                n_phases,
                time_delays_minmax_d
                );

        // initialize beam_d to zeros
        cudaMemset(beam_d, 0., sizeofbeam);


        // initialize GPU time index
        int time_GPU = 0;

        //printf("GPU %d done with allocating and copying data.\n", id);

        // compute network response
        while (time_GPU < n_samples){

            // backproject the wavefield onto n_sources_per_GPU
            // grid locations and at BLOCKSIZE time locations

            if (out_of_bounds == 0){
                // check out-of-bound operations for the whole network
                _beam<<<n_sources_per_GPU, BLOCKSIZE, shared_mem>>>(
                        waveform_features_d + n_phases*time_GPU,
                        time_delays_d,
                        time_delays_minmax_d,
                        weights_d,
                        time_GPU,
                        n_samples,
                        n_stations,
                        n_phases,
                        BLOCKSIZE,
                        beam_d + time_GPU
                        );
            }
            else if (out_of_bounds == 1){
                // check out-of-bound operations separately for each station
                _beam_check_out_of_bounds<<<n_sources_per_GPU, BLOCKSIZE, shared_mem>>>(
                        waveform_features_d + n_phases*time_GPU,
                        time_delays_d,
                        time_delays_minmax_d,
                        weights_d,
                        time_GPU,
                        n_samples,
                        n_stations,
                        n_phases,
                        BLOCKSIZE,
                        beam_d + time_GPU
                        );
            }

            // increment time
            time_GPU += BLOCKSIZE;
        }

        // get results back to the host
        cudaMemcpy(
                beam + src_idx_start*n_samples, beam_d, sizeofbeam, cudaMemcpyDeviceToHost
                );

        // free memory
        cudaFree(waveform_features_d);
        cudaFree(time_delays_d);
        cudaFree(time_delays_minmax_d);
        cudaFree(weights_d);
        cudaFree(beam_d);

        // return an error if something happened in the kernel
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        // done!

    } // omp parallel


}

} // extern C
