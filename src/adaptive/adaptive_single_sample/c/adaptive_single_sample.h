#ifndef ADAPTIVE_SINGLE_SAMPLE_H
#define ADAPTIVE_SINGLE_SAMPLE_H

typedef struct {
    int num_taps;
    float mu;
    int num_channels;
    float **weights;
    float **buffer;
    int iteration;
} AdaptiveFilter;

AdaptiveFilter* filter_create(int num_taps, float mu, int num_channels);
void filter_adapt(AdaptiveFilter* filter, float* x, float desired_signal, float* outputs);
void filter_free(AdaptiveFilter* filter);

#endif