#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <limits.h>
#include "beamform_cpu.h"

void _find_minmax_time_delays(int *time_delays, float *weights, size_t n_sources,
                              size_t n_stations, size_t n_phases,
                              int *time_delays_minmax)
{

    /* Find the minimum and maximum time_delays for each point of the grid.
     * Even indexes correspond to minimum time_delays,
     * odd indexes correspond to maximum time_delays. */

#pragma omp parallel for shared(time_delays, weights, time_delays_minmax)
    for (size_t i = 0; i < n_sources; i++)
    {
        int max_time_delay = INT_MIN;
        int min_time_delay = INT_MAX;
        int time_delay;
        size_t weight_offset;
        size_t time_delay_offset;

        for (size_t s = 0; s < n_stations; s++)
        {
            weight_offset = i * n_stations + s;
            if (weights[weight_offset] == 0.)
                continue; // the station is not used

            for (size_t p = 0; p < n_phases; p++)
            {
                time_delay_offset = i * n_stations * n_phases + s * n_phases + p;
                time_delay = time_delays[time_delay_offset];
                if (time_delay > max_time_delay)
                    max_time_delay = time_delay;
                if (time_delay < min_time_delay)
                    min_time_delay = time_delay;
            }
        }
        time_delays_minmax[i * 2 + 0] = min_time_delay;
        time_delays_minmax[i * 2 + 1] = max_time_delay;
    }
}

float _beam(float *waveform_features, int *time_delays, float *weights,
            size_t n_samples, size_t n_stations, size_t n_phases)
{

    /* Build a beam of the detection traces using the input time_delays.
     * This routine uses the detection traces prestacked for each phase. */

    float beam = 0.;      // shifted and stacked traces
    size_t feature_offset; // position on input pointer

    for (size_t s = 0; s < n_stations; s++)
    {
        if (weights[s] == 0)
            continue;
        // station loop
        for (size_t p = 0; p < n_phases; p++)
        {
            // phase loop
            feature_offset = s * n_samples * n_phases + p + n_phases * time_delays[s * n_phases + p];
            beam += weights[s] * waveform_features[feature_offset];
        }
    }
    return beam;
}

float _beam_check_out_of_bounds(
        float *waveform_features, int *time_delays, float *weights,
        size_t time_index, size_t n_samples, size_t n_stations, size_t n_phases
        )
{

    /* Build a beam of the detection traces using the input time_delays.
     * It checks whether t + time_delay[s, p] is out of bound for each
     * station and phase.
     * This routine uses the detection traces prestacked for each phase. */

    float beam = 0.;      // shifted and stacked traces
    size_t feature_offset; // position on input pointer
    size_t max_offet;

    for (size_t s = 0; s < n_stations; s++)
    {
        if (weights[s] == 0)
            continue;
        // station loop
        for (size_t p = 0; p < n_phases; p++)
        {
            // check out-of-bounds
            if (time_index + time_delays[s * n_phases + p] >= n_samples) continue;
            if (time_index + time_delays[s * n_phases + p] < 0) continue;
            // phase loop
            feature_offset = s * n_samples * n_phases + p + n_phases * time_delays[s * n_phases + p];
            beam += weights[s] * waveform_features[feature_offset];
        }
    }
    return beam;
}


void prestack_waveform_features(
    float *waveform_features, float *weights_phases,
    size_t n_samples, size_t n_stations, size_t n_channels,
    size_t n_phases, float *prestack_traces)
{

    /* The channel dimension can be reduced ahead of the beamforming
     * for each phase since, for a given phase, all channels of a same
     * station are always stacked with no relative time-shift. */

    size_t prestack_offset;
    size_t feature_offset;
    size_t weight_offset;

#pragma omp parallel for private(prestack_offset, feature_offset, weight_offset) \
    shared(waveform_features, weights_phases, prestack_traces)
    for (size_t s = 0; s < n_stations; s++)
    {
        // station loop
        for (size_t t = 0; t < n_samples; t++)
        {
            // time loop
            for (size_t p = 0; p < n_phases; p++)
            {
                // phase loop
                prestack_offset = s * n_samples * n_phases + t * n_phases + p;
                // initialize stack
                prestack_traces[prestack_offset] = 0.;
                for (size_t c = 0; c < n_channels; c++)
                {
                    // channel loop
                    // stack detection traces along the channel axis
                    feature_offset = s * n_channels * n_samples + c * n_samples + t;
                    weight_offset = s * n_channels * n_phases + c * n_phases + p;
                    prestack_traces[prestack_offset] +=
                        weights_phases[weight_offset] * waveform_features[feature_offset];
                }
            }
        }
    }
}

void beamform(float *waveform_features, int *time_delays, float *weights,
              size_t n_samples, size_t n_sources, size_t n_stations,
              size_t n_phases, int out_of_bounds, float *beam)
{

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their time_delays. The output is a vector with length
     * (n_samples x n_sources), which therefore can potentially be very large
     * for long time series. This function is fit for monitoring the network
     * response in 4D (time and space) with applications for event detection
     * but also rupture progation imaging (back-projection). */

    size_t time_delay_offset; // location on time_delays (use size_t to handle large numbers)
    size_t weights_offset;    // location on weights pointer
    size_t beam_offset;       // location on beam
    int *time_delays_minmax;  // vector with min and max mv of each source
    int time_delay_min, time_delay_max;

    // search for min and max time_delay of each source
    time_delays_minmax = (int *)malloc(2 * n_sources * sizeof(int));

    _find_minmax_time_delays(time_delays,
                             weights,
                             n_sources,
                             n_stations,
                             n_phases,
                             time_delays_minmax);
#pragma omp parallel for private(time_delay_offset, weights_offset, beam_offset, time_delay_min, time_delay_max) \
    shared(waveform_features, time_delays, beam, time_delays_minmax)
    for (size_t i = 0; i < n_sources; i++)
    {
        time_delay_offset = i * n_stations * n_phases;
        weights_offset = i * n_stations;
        beam_offset = i * n_samples;
        time_delay_min = time_delays_minmax[2 * i + 0];
        time_delay_max = time_delays_minmax[2 * i + 1];
        for (int t = 0; t < n_samples; t++)
        {
            if (out_of_bounds == 0){
                // check out-of-bound operations for the whole network
                if ((t + time_delay_max) >= n_samples)
                    continue;
                if ((t + time_delay_min) < 0)
                    continue;

                // compute the beamformed network responses for all sources
                beam[beam_offset + t] = _beam(waveform_features + n_phases * t,
                                              time_delays + time_delay_offset,
                                              weights + weights_offset,
                                              n_samples,
                                              n_stations,
                                              n_phases);

                }
            else if (out_of_bounds == 1){
                // check out-of-bound operations separately for each station
                //
                // compute the beamformed network responses for all sources
                beam[beam_offset + t] = _beam_check_out_of_bounds(
                        waveform_features + n_phases * t,
                        time_delays + time_delay_offset,
                        weights + weights_offset,
                        t,
                        n_samples,
                        n_stations,
                        n_phases
                        );
            }
        }
    }
}

void beamform_max(
    float *waveform_features, int *time_delays, float *weights,
    size_t n_samples, size_t n_sources, size_t n_stations,
    size_t n_phases, int out_of_bounds, float *beam, int *source_index_beam)
{

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their time_delays. This routine keeps only the largest
     * network response value and associated source index at each time step.
     * The output is a pair of vectors with length (n_samples). This output
     * is much more memory friendly than network_response's. Therefore, this
     * routine should be preferred whenever the goal of the task does not
     * require the full picture of the network response at each time. */

    size_t time_delay_offset; // location on time_delays pointer
    size_t weights_offset;    // location on weights pointer
    int *time_delays_minmax;  // vector with min and max mv of each source
    float current_beam;       // value of currently computed beam
    float largest_beam;       // current largest visited beam
    int largest_beam_index;   // source index of current largest visited beam

    // search for min and max time_delay of each source
    time_delays_minmax = (int *)malloc(2 * n_sources * sizeof(int));

    _find_minmax_time_delays(time_delays,
                             weights,
                             n_sources,
                             n_stations,
                             n_phases,
                             time_delays_minmax);
#pragma omp parallel for private(time_delay_offset, weights_offset, largest_beam, largest_beam_index, current_beam) \
    shared(waveform_features, time_delays, weights, beam, source_index_beam)
    for (size_t t = 0; t < n_samples; t++)
    {
        largest_beam = -FLT_MAX;
        largest_beam_index = 0;

        for (size_t i = 0; i < n_sources; i++)
        {
            time_delay_offset = i * n_stations * n_phases;
            weights_offset = i * n_stations;

            if (out_of_bounds == 0){
                // check out-of-bound operations for the whole network
                if ((t + time_delays_minmax[2 * i + 1]) >= n_samples)
                    continue;
                if ((t + time_delays_minmax[2 * i + 0]) < 0)
                    continue;

                // compute the beamformed network responses for all sources
                current_beam = _beam(
                        waveform_features + n_phases * t,
                        time_delays + time_delay_offset,
                        weights + weights_offset,
                        n_samples,
                        n_stations,
                        n_phases);

                }
            else if (out_of_bounds == 1){
                // check out-of-bound operations separately for each station
                //
                // compute the beamformed network responses for all sources
                current_beam = _beam_check_out_of_bounds(
                        waveform_features + n_phases * t,
                        time_delays + time_delay_offset,
                        weights + weights_offset,
                        t,
                        n_samples,
                        n_stations,
                        n_phases
                        );
            }

            //// check out-of-bound operations
            //if (t + time_delays_minmax[2 * i + 1] >= n_samples)
            //    continue;
            //if (t + time_delays_minmax[2 * i + 0] < 0)
            //    continue;

            //time_delay_offset = i * n_stations * n_phases;
            //weights_offset = i * n_stations;

            //// compute the beamformed network responses for all sources
            //current_beam = _beam(waveform_features + n_phases * t,
            //                     time_delays + time_delay_offset,
            //                     weights + weights_offset,
            //                     n_samples,
            //                     n_stations,
            //                     n_phases);

            if (current_beam > largest_beam)
            {
                largest_beam = current_beam;
                largest_beam_index = i;
            }
        }
        if (largest_beam > -FLT_MAX)
            beam[t] = largest_beam;
        source_index_beam[t] = largest_beam_index;
    }
}

void beamform_differential(float *waveform_features, int *time_delays, float *weights,
                           size_t n_samples, size_t n_sources, size_t n_stations,
                           size_t n_phases, float *beam)
{

    /* Compute the beamformed network response at each input theoretical source
     * characterized by their time_delays. The output is a vector with length
     * (n_samples x n_sources), which therefore can potentially be very large
     * for long time series. This function is fit for monitoring the network
     * response in 4D (time and space) with applications for event detection
     * but also rupture progation imaging (back-projection). */

    size_t time_delay_offset;    // location on time_delays (use size_t to handle large numbers)
    size_t weights_offset;       // location on weights pointer
    size_t beam_offset;          // location on beam
    int t = (n_samples - 1) / 2; // location or zero-lag
    int *time_delays_minmax;     // vector with min and max mv of each source
    int time_delay_min, time_delay_max;

    // search for min and max time_delay of each source
    time_delays_minmax = (int *)malloc(2 * n_sources * sizeof(int));

    _find_minmax_time_delays(time_delays,
                             weights,
                             n_sources,
                             n_stations,
                             n_phases,
                             time_delays_minmax);
#pragma omp parallel for private(time_delay_offset, weights_offset, beam_offset) \
    shared(waveform_features, time_delays, beam)
    for (size_t i = 0; i < n_sources; i++)
    {
        time_delay_offset = i * n_stations * n_phases;
        weights_offset = i * n_stations;
        beam_offset = i;
        time_delay_min = time_delays_minmax[2 * i + 0];
        time_delay_max = time_delays_minmax[2 * i + 1];

        // check out-of-bound operations
        if ((t + time_delay_max) >= n_samples)
            continue;
        if ((t + time_delay_min) < 0)
            continue;

        // compute the beamformed network responses for all sources
        beam[beam_offset] = _beam(waveform_features + n_phases * t,
                                  time_delays + time_delay_offset,
                                  weights + weights_offset,
                                  n_samples,
                                  n_stations,
                                  n_phases);
    }
}
