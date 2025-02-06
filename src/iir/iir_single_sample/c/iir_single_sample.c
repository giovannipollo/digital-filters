#include "iir_single_sample.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int iir_filter_init(IIRFilter* filter, const double* b, const double* a, int filter_order) {
    filter->num_taps = filter_order + 1;
    
    // Allocate memory for coefficients and buffers
    filter->b = (double*)malloc(filter->num_taps * sizeof(double));
    filter->a = (double*)malloc(filter->num_taps * sizeof(double));
    filter->input_buffer = (double*)malloc(filter_order * sizeof(double));
    filter->output_buffer = (double*)malloc(filter_order * sizeof(double));
    
    if (!filter->b || !filter->a || !filter->input_buffer || !filter->output_buffer) {
        iir_filter_destroy(filter);
        return -1;
    }
    
    // Copy coefficients
    memcpy(filter->b, b, filter->num_taps * sizeof(double));
    memcpy(filter->a, a, filter->num_taps * sizeof(double));
    
    // Initialize buffers to zero
    memset(filter->input_buffer, 0, filter_order * sizeof(double));
    memset(filter->output_buffer, 0, filter_order * sizeof(double));
    
    return 0;
}

double iir_filter_apply(IIRFilter* filter, double x) {
    double y = filter->b[0] * x;
    
    // Apply filter
    for (int i = 1; i < filter->num_taps; i++) {
        y += filter->b[i] * filter->input_buffer[i - 1];
    }
    for (int i = 1; i < filter->num_taps; i++) {
        y -= filter->a[i] * filter->output_buffer[i - 1];
    }
    
    // Rotate input buffer
    for (int i = filter->num_taps - 2; i > 0; i--) {
        filter->input_buffer[i] = filter->input_buffer[i - 1];
    }
    filter->input_buffer[0] = x;
    
    // Print the input buffer
    // printf("Input buffer: ");
    // for (int i = 0; i < filter->num_taps - 1; i++) {
    //     printf("%f ", filter->input_buffer[i]);
    // }
    // printf("\n");

    // Rotate output buffer
    for (int i = filter->num_taps - 2; i > 0; i--) {
        filter->output_buffer[i] = filter->output_buffer[i - 1];
    }
    filter->output_buffer[0] = y;
    
    // Print the output buffer
    // printf("Output buffer: ");
    // for (int i = 0; i < filter->num_taps - 1; i++) {
    //     printf("%f ", filter->output_buffer[i]);
    // }
    // printf("\n");
    
    return y;
}

void iir_filter_destroy(IIRFilter* filter) {
    free(filter->b);
    free(filter->a);
    free(filter->input_buffer);
    free(filter->output_buffer);
}