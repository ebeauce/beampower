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
#include "beamformed_nr_GPU.h"

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

void __global__ _find_minmax_moveouts_ker(int* moveouts, float* weights,
        size_t n_sources, size_t n_stations, size_t n_phases,
        int* moveouts_minmax){

    /* Find the minimum and maximum moveouts for each point of the grid.
     * Even indexes correspond to minimum moveouts,
     * odd indexes correspond to maximum moveouts. */

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_sources) return; // skip threads that are out-of-bound

    int min_moveout = INT_MAX;
    int max_moveout = INT_MIN;
    int moveout;
    size_t weight_offset;
    size_t mv_offset;

    for (size_t s=0; s<n_stations; s++){
        weight_offset = i*n_stations + s;
        if (weights[weight_offset] == 0.) continue; // the station is not used

        for (size_t p=0; p<n_phases; p++){
            mv_offset = i*n_stations*n_phases + s*n_phases + p;
            moveout = moveouts[mv_offset];
            if (moveout > max_moveout) max_moveout = moveout;
            if (moveout < min_moveout) min_moveout = moveout;
            }
    }
    moveouts_minmax[i * 2 + 0] = min_moveout;
    moveouts_minmax[i * 2 + 1] = max_moveout;

}

void __global__ _beam(float *detection_traces, int *moveouts,
        int *moveouts_minmax, float *weights, size_t global_time_index,
        size_t n_samples, size_t n_stations, size_t n_phases, 
        size_t dim0_nr, float *nr){

    size_t i = blockIdx.x; // source index
    size_t t_idx = threadIdx.x; // thread-private index
    float beam = 0.; // sum
    size_t det_tr_offset; // position on input pointer
    // number of elements to store in shared memory
    size_t size_moveouts = n_stations*n_phases;
    // declare shared arrays
    extern __shared__ int shared[];
    int *moveouts_s = &shared[0];
    float *weights_s = (float *)&shared[size_moveouts];

    // read data into shared memory
    while (t_idx < size_moveouts){
        moveouts_s[t_idx] = moveouts[i*n_stations*n_phases + t_idx];
        if (t_idx < n_stations){
            weights_s[t_idx] = weights[i*n_stations + t_idx];
        }
        t_idx += blockDim.x;
    }
    // wait for all threads to be done with reading
    __syncthreads();

    // compute this beam only if it stays within time bounds
    if ((global_time_index >= -moveouts_minmax[2*i + 0])
            & ((global_time_index + threadIdx.x + moveouts_minmax[2*i + 1]) < n_samples)){
        // start shift and stack
        for (size_t s=0; s<n_stations; s++){
            for (size_t p=0; p<n_phases; p++){
                det_tr_offset = s*n_samples*n_phases + p\
                                + n_phases*moveouts_s[s*n_phases + p]\
                                + n_phases*threadIdx.x;
                beam += weights_s[s]*detection_traces[det_tr_offset];
            }
        }
        // update nr
        nr[i*dim0_nr + threadIdx.x] = beam;
    }
}

void __global__ _cnr(float *nr, size_t n_sources, float *cnr,
        int *source_index_cnr){

    float max_nr = FLT_MIN;
    int max_nr_index;

    // loop over all sources and search for the maximum
    for (int i=0; i<n_sources; i++){
        if (nr[i*blockDim.x + threadIdx.x] > max_nr){
            max_nr = nr[i*blockDim.x + threadIdx.x];
            max_nr_index = i;
        }
    }
    
    // update output arrays
    cnr[threadIdx.x] = max_nr;
    source_index_cnr[threadIdx.x] = max_nr_index;

}

void composite_network_response(float* detection_traces, int* moveouts, float* weights,
        size_t n_samples, size_t n_sources, size_t n_stations, size_t n_phases,
        float* cnr, int* source_index_cnr){

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their moveouts. The output is a vector with length
     * (n_samples x n_sources), which therefore can potentially be very large
     * for long time series. This function is fit for monitoring the network
     * response in 4D (time and space) with applications for event detection
     * but also rupture progation imaging (back-projection). */

    int nGPUs=0;
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
    shared(detection_traces, moveouts, weights, cnr)
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
        float *detection_traces_d;
        int *moveouts_d;
        int *moveouts_minmax_d;
        float *weights_d;
        float *nr_d;
        float *cnr_d;
        int *source_index_cnr_d;
        // declare host pointers
        float *cnr_thread;
        int *source_index_cnr_thread;

        // size of arrays on device
        size_t sizeofdata = n_stations*n_samples*n_phases*sizeof(float);
        size_t sizeofmoveouts = n_sources_per_GPU*n_stations*n_phases*sizeof(int);
        size_t sizeofmoveouts_minmax = 2*n_sources_per_GPU*sizeof(int);
        size_t sizeofweights = n_sources_per_GPU*n_stations*sizeof(float);
        size_t sizeofnr = BLOCKSIZE*n_sources_per_GPU*sizeof(float);
        size_t sizeofcnr = n_samples*sizeof(float);
        size_t sizeoftotal = sizeofdata + sizeofmoveouts + sizeofmoveouts_minmax\
                             + sizeofweights + sizeofnr + 2*sizeofcnr;

        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (sizeoftotal > freeMem) {
            printf("%zu Mb are requested on GPU #%i whereas it has only %zu free Mb.\n",
                   sizeoftotal/MEGABYTE, id, freeMem/MEGABYTE);
            printf("Consider reducing the duration of the seismograms processed"\
                   " at once, or downsample the source grid.\n");
            exit(0);
        }


        // allocate GPU memory
        cudaMalloc((void**)&detection_traces_d, sizeofdata);
        cudaMalloc((void**)&moveouts_d, sizeofmoveouts);
        cudaMalloc((void**)&moveouts_minmax_d, sizeofmoveouts_minmax);
        cudaMalloc((void**)&weights_d, sizeofweights);
        cudaMalloc((void**)&nr_d, sizeofnr);
        cudaMalloc((void**)&cnr_d, sizeofcnr);
        cudaMalloc((void**)&source_index_cnr_d, sizeofcnr);
        // declare host pointers and allocate CPU memory
        cnr_thread = (float *)malloc(sizeofcnr);
        source_index_cnr_thread = (int *)malloc(sizeofcnr);

        // transfer data from host (CPU) to device (GPU)
        cudaMemcpy(detection_traces_d, detection_traces, sizeofdata,
                cudaMemcpyHostToDevice);
        cudaMemcpy(moveouts_d, moveouts + src_idx_start*n_stations*n_phases,
                sizeofmoveouts, cudaMemcpyHostToDevice);
        cudaMemcpy(weights_d, weights + src_idx_start*n_stations, sizeofweights,
                cudaMemcpyHostToDevice);

        // compute moveouts min and max
        _find_minmax_moveouts_ker<<<n_sources_per_GPU/BLOCKSIZE+1, BLOCKSIZE>>>(
                moveouts_d, weights_d, n_sources_per_GPU, n_stations,
                n_phases, moveouts_minmax_d);

        // initialize cnr and source_index_cnr
        cudaMemset(cnr_d, 0., sizeofcnr);
        cudaMemset(source_index_cnr_d, 0., sizeofcnr);

        // initialize GPU time index
        size_t time_GPU = 0;

        // compute network response
        while (time_GPU < n_samples){
            // initialize nr_d to zeros
            cudaMemset(nr_d, 0., sizeofnr);

            // backproject the wavefield onto n_sources_per_GPU
            // grid locations and at BLOCKSIZE time locations
            _beam<<<n_sources_per_GPU,
                    BLOCKSIZE, shared_mem>>>(
                            detection_traces_d + n_phases*time_GPU, moveouts_d,
                            moveouts_minmax_d, weights_d, time_GPU, n_samples,
                            n_stations, n_phases, BLOCKSIZE, nr_d);

            // find the maximum cnr and cnr source index across the
            // n_sources_per_GPU grid locations and at the BLOCKSIZE time
            // locations
            _cnr<<<1, BLOCKSIZE>>>(
                    nr_d, n_sources_per_GPU, cnr_d + time_GPU,
                    source_index_cnr_d + time_GPU);

            // increment time
            time_GPU += BLOCKSIZE;
        }

        // get results back to the host
        cudaMemcpy(cnr_thread, cnr_d, sizeofcnr, cudaMemcpyDeviceToHost);
        cudaMemcpy(source_index_cnr_thread, source_index_cnr_d,
                sizeofcnr, cudaMemcpyDeviceToHost);

        // wait for all GPUs to finish processing their part of the grid
        cudaDeviceSynchronize();

        // return an error if something happened in the kernel
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // critical section to merge all the single-GPU cnr into one
        for (size_t t=0; t<n_samples; t++){
#pragma omp critical
            {
                if (cnr_thread[t] > cnr[t]){
                    cnr[t] = cnr_thread[t];
                    source_index_cnr[t] = src_idx_start+source_index_cnr_thread[t];
                }
            }
        }

        // free memory
        cudaFree(detection_traces_d);
        cudaFree(moveouts_d);
        cudaFree(moveouts_minmax_d);
        cudaFree(weights_d);
        cudaFree(cnr_d);
        cudaFree(source_index_cnr_d);

        // done!

    } // omp parallel

}

void network_response(float* detection_traces, int* moveouts, float* weights,
        size_t n_samples, size_t n_sources, size_t n_stations, size_t n_phases,
        float* nr){

    int nGPUs=0;
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
    shared(detection_traces, moveouts, weights, nr)
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
        float *detection_traces_d;
        int *moveouts_d;
        int *moveouts_minmax_d;
        float *weights_d;
        float *nr_d;

        // size of arrays on device
        size_t sizeofdata = n_stations*n_samples*n_phases*sizeof(float);
        size_t sizeofmoveouts = n_sources_per_GPU*n_stations*n_phases*sizeof(int);
        size_t sizeofmoveouts_minmax = 2*n_sources_per_GPU*sizeof(int);
        size_t sizeofweights = n_sources_per_GPU*n_stations*sizeof(float);
        size_t sizeofnr = n_sources_per_GPU*n_samples*sizeof(float);
        size_t sizeoftotal = sizeofdata + sizeofmoveouts + sizeofmoveouts_minmax\
                             + sizeofweights + sizeofnr;

        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (sizeoftotal > freeMem) {
            printf("%zu Mb are requested on GPU #%i whereas it has only %zu free Mb.\n",
                   sizeoftotal/MEGABYTE, id, freeMem/MEGABYTE);
            printf("Consider reducing the duration of the seismograms processed"\
                   " at once, or downsample the source grid.\n");
            exit(0);
        }


        // allocate GPU memory
        cudaMalloc((void**)&detection_traces_d, sizeofdata);
        cudaMalloc((void**)&moveouts_d, sizeofmoveouts);
        cudaMalloc((void**)&moveouts_minmax_d, sizeofmoveouts_minmax);
        cudaMalloc((void**)&weights_d, sizeofweights);
        cudaMalloc((void**)&nr_d, sizeofnr);

        // transfer data from host (CPU) to device (GPU)
        cudaMemcpy(detection_traces_d, detection_traces, sizeofdata,
                cudaMemcpyHostToDevice);
        cudaMemcpy(moveouts_d, moveouts + src_idx_start*n_stations*n_phases,
                sizeofmoveouts, cudaMemcpyHostToDevice);
        cudaMemcpy(weights_d, weights + src_idx_start*n_stations, sizeofweights,
                cudaMemcpyHostToDevice);

        // compute moveouts min and max
        _find_minmax_moveouts_ker<<<n_sources_per_GPU/BLOCKSIZE+1, BLOCKSIZE>>>(
                moveouts_d, weights_d, n_sources_per_GPU, n_stations,
                n_phases, moveouts_minmax_d);

        // initialize nr_d to zeros
        cudaMemset(nr_d, 0., sizeofnr);


        // initialize GPU time index
        size_t time_GPU = 0;

        //printf("GPU %d done with allocating and copying data.\n", id);

        // compute network response
        while (time_GPU < n_samples){

            // backproject the wavefield onto n_sources_per_GPU
            // grid locations and at BLOCKSIZE time locations
            _beam<<<n_sources_per_GPU,
                    BLOCKSIZE, shared_mem>>>(
                            detection_traces_d + n_phases*time_GPU, moveouts_d,
                            moveouts_minmax_d, weights_d, time_GPU, n_samples,
                            n_stations, n_phases, n_samples, nr_d + time_GPU);

            // increment time
            time_GPU += BLOCKSIZE;
        }

        // get results back to the host
        cudaMemcpy(nr + src_idx_start*n_samples, nr_d, sizeofnr,
                cudaMemcpyDeviceToHost);

        // free memory
        cudaFree(detection_traces_d);
        cudaFree(moveouts_d);
        cudaFree(moveouts_minmax_d);
        cudaFree(weights_d);
        cudaFree(nr_d);

        // return an error if something happened in the kernel
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());


        // done!

    } // omp parallel


}

} // extern C
