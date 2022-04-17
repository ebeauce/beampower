#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include "beamformed_nr_CPU.h"

void _find_minmax_moveouts(int* moveouts, float* weights, size_t n_sources, 
                           size_t n_stations, size_t n_phases,
                           int *moveouts_minmax) {

    /* Find the minimum and maximum moveouts for each point of the grid.
     * Even indexes correspond to minimum moveouts,
     * odd indexes correspond to maximum moveouts. */

#pragma omp parallel for\
    shared(moveouts, weights, moveouts_minmax)
    for (size_t i = 0; i < n_sources; i++){
        int max_moveout = INT_MIN;
        int min_moveout = INT_MAX;
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
}

float _beam(float* detection_traces, int* moveouts, float* weights,
            size_t n_samples, size_t n_stations, size_t n_phases) {

    /* Build a beam of the detection traces using the input moveouts.
     * This routine uses the detection traces prestacked for each phase. */

    float beam = 0.; // shifted and stacked traces
    size_t det_tr_offset; // position on input pointer

    for (size_t s=0; s<n_stations; s++){
        if (weights[s] == 0) continue;
        // station loop
        for (size_t p=0; p<n_phases; p++){
            // phase loop
            det_tr_offset = s*n_samples*n_phases + p\
                            + n_phases*moveouts[s*n_phases + p];
            beam += weights[s]*detection_traces[det_tr_offset];
        }
    }
    return beam;
}


void prestack_detection_traces(
        float* detection_traces, float* weights_phases,
        size_t n_samples, size_t n_stations, size_t n_channels,
        size_t n_phases, float* prestack_traces){

    /* The channel dimension can be reduced ahead of the beamforming
     * for each phase since, for a given phase, all channels of a same
     * station are always stacked with no relative time-shift. */

    int prestack_offset;
    int det_tr_offset;
    int weight_offset;

#pragma omp parallel for\
    private(prestack_offset, det_tr_offset, weight_offset)\
    shared(detection_traces, weights_phases, prestack_traces)
    for (size_t s=0; s<n_stations; s++){
        // station loop
        for (size_t t=0; t<n_samples; t++){
            // time loop
            for (size_t p=0; p<n_phases; p++){
                // phase loop
                prestack_offset = s*n_samples*n_phases + t*n_phases + p;
                // initialize stack
                prestack_traces[prestack_offset] = 0.;
                for (size_t c=0; c<n_channels; c++){
                    // channel loop
                    // stack detection traces along the channel axis
                    det_tr_offset = s*n_channels*n_samples + c*n_samples + t;
                    weight_offset = s*n_channels*n_phases + c*n_phases + p;
                    prestack_traces[prestack_offset] += \
                        weights_phases[weight_offset]*detection_traces[det_tr_offset];
                }
            }
        }
    }
}


void network_response(float* detection_traces, int* moveouts, float* weights,
                      size_t n_samples, size_t n_sources, size_t n_stations,
                      size_t n_phases, float* nr) {

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their moveouts. The output is a vector with length
     * (n_samples x n_sources), which therefore can potentially be very large
     * for long time series. This function is fit for monitoring the network
     * response in 4D (time and space) with applications for event detection
     * but also rupture progation imaging (back-projection). */

    size_t mv_offset; // location on moveouts (use size_t to handle large numbers)
    size_t weights_offset; // location on weights pointer
    size_t nr_offset; // location on nr
    int *moveouts_minmax; // vector with min and max mv of each source
    int mv_min, mv_max;

    // search for min and max moveout of each source
    moveouts_minmax = (int *)malloc(2*n_sources*sizeof(int));
   
    _find_minmax_moveouts(moveouts,
                          weights,
                          n_sources,
                          n_stations,
                          n_phases,
                          moveouts_minmax);
#pragma omp parallel for\
    private(mv_offset, weights_offset, nr_offset)\
    shared(detection_traces, moveouts, nr)
    for (size_t i=0; i<n_sources; i++){
        mv_offset = i*n_stations*n_phases;
        weights_offset = i*n_stations;
        nr_offset = i*n_samples;
        mv_min = moveouts_minmax[2*i+0];
        mv_max = moveouts_minmax[2*i+1];
        for (size_t t=0; t<n_samples; t++){
            // check out-of-bound operations
            if ((t + mv_max) > n_samples) continue;
            if ((t + mv_min) < 0) continue;

            // compute the beamformed network responses for all sources
            nr[nr_offset + t] = _beam(detection_traces + n_phases*t,
                                      moveouts + mv_offset,
                                      weights + weights_offset,
                                      n_samples,
                                      n_stations,
                                      n_phases);
        }
    }
}

void composite_network_response(
        float *detection_traces, int *moveouts, float *weights,
        size_t n_samples, size_t n_sources, size_t n_stations,
        size_t n_phases, float *nr, int *source_index_nr) {

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their moveouts. This routine keeps only the largest
     * network response value and associated source index at each time step.
     * The output is a pair of vectors with length (n_samples). This output
     * is much more memory friendly than network_response's. Therefore, this
     * routine should be preferred whenever the goal of the task does not
     * require the full picture of the network response at each time. */

    size_t mv_offset; // location on moveouts pointer
    size_t weights_offset; // location on weights pointer
    int *moveouts_minmax; // vector with min and max mv of each source
    float current_nr = 0.; // value of currently computed nr
    float largest_nr = 0.; // current largest visited nr
    int largest_nr_index = 0; // source index of current largest visited nr

    // search for min and max moveout of each source
    moveouts_minmax = (int *)malloc(2*n_sources*sizeof(int));
    
    _find_minmax_moveouts(moveouts,
                          weights,
                          n_sources,
                          n_stations,
                          n_phases,
                          moveouts_minmax);
#pragma omp parallel for\
    private(mv_offset, weights_offset, largest_nr, largest_nr_index, current_nr)\
    shared(detection_traces, moveouts, weights, nr, source_index_nr)
    for (size_t t=0; t<n_samples; t++){
        largest_nr = 0.;
        largest_nr_index = 0;

        for (size_t i=0; i<n_sources; i++){

            // check out-of-bound operations
            if (t + moveouts_minmax[2*i+1] > n_samples) continue;
            if (t + moveouts_minmax[2*i+0] < 0) continue;

            mv_offset = i*n_stations*n_phases;
            weights_offset = i*n_stations;

            // compute the beamformed network responses for all sources
            current_nr = _beam(detection_traces + n_phases*t,
                               moveouts + mv_offset,
                               weights + weights_offset,
                               n_samples,
                               n_stations,
                               n_phases);
   
            if (current_nr > largest_nr){
                largest_nr = current_nr;
                largest_nr_index = i;
            }
        }
        nr[t] = largest_nr;
        source_index_nr[t] = largest_nr_index;
    }
}

