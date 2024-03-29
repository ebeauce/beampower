void _find_minmax_time_delays(int *, float *, size_t, size_t, size_t, int *);
float _beam(float *, int *, float *, size_t, size_t, size_t);
float _beam_check_out_of_bounds(float *, int *, float *, size_t, size_t, size_t, size_t);
void prestack_waveform_features(float *, float *, size_t, size_t, size_t, size_t, int, float *);
void beamform(float *, int *, float *, size_t, size_t, size_t, size_t, int, int, float *);
void beamform_differential(float *, int *, float *, size_t, size_t, size_t, size_t, int, float *);
void beamform_max(float *, int *, float *, size_t, size_t, size_t, size_t, int, int, float *, int *);
