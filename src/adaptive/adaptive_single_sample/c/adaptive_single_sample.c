#include "adaptive_single_sample.h"
#include <stdlib.h>
#include <stdio.h>

AdaptiveFilter *filter_create(int num_taps, float mu, int num_channels) {
    AdaptiveFilter *filter = malloc(sizeof(AdaptiveFilter));
    if (!filter)
        return NULL;

    filter->num_taps = num_taps;
    filter->mu = mu;
    filter->num_channels = num_channels;
    filter->iteration = 0;

    filter->weights = malloc(num_channels * sizeof(float *));
    filter->buffer = malloc(num_channels * sizeof(float *));

    for (int i = 0; i < num_channels; i++) {
        filter->weights[i] = calloc(num_taps, sizeof(float));
        filter->buffer[i] = calloc(num_taps, sizeof(float));
    }

    return filter;
}

void filter_adapt(AdaptiveFilter *filter, float *x, float desired_signal, float *outputs) {
    printf("Iteration %d\n", filter->iteration);
    filter->iteration++;

    for (int channel = 0; channel < filter->num_channels; channel++) {
        // Shift buffer
        for (int i = filter->num_taps - 1; i > 0; i--) {
            filter->buffer[channel][i] = filter->buffer[channel][i - 1];
        }
        filter->buffer[channel][0] = x[channel];
        
        // Print the buffer
        for (int i = 0; i < filter->num_taps; i++) {
            printf("  Buffer[%d][%d]: %.10f\n", channel, i, filter->buffer[channel][i]);
        }
        // print the desired signal
        printf("  Desired: %.10f\n", desired_signal);

        // Print the weights
        for (int i = 0; i < filter->num_taps; i++) {
            printf("  Weights[%d][%d]: %.10f\n", channel, i, filter->weights[channel][i]);
        }

        // Compute output
        outputs[channel] = 0;
        for (int i = 0; i < filter->num_taps; i++) {
            outputs[channel] += filter->weights[channel][i] * filter->buffer[channel][i];
        }

        // Print the output
        printf("  Output: %.10f\n", outputs[channel]);

        // Update weights
        float error = desired_signal - outputs[channel];

        // print the error
        printf("  Error: %.10f\n", error);

        for (int i = 0; i < filter->num_taps; i++) {
            filter->weights[channel][i] += filter->mu * error * filter->buffer[channel][i];
        }
        // print the updated weights
        for (int i = 0; i < filter->num_taps; i++) {
            printf("  Weights Updated[%d][%d]: %.10f\n", channel, i, filter->weights[channel][i]);
        }
        break;
    }
}

void filter_free(AdaptiveFilter *filter) {
    for (int i = 0; i < filter->num_channels; i++) {
        free(filter->weights[i]);
        free(filter->buffer[i]);
    }
    free(filter->weights);
    free(filter->buffer);
    free(filter);
}