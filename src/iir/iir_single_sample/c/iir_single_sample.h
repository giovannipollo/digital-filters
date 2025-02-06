#ifndef IIR_SINGLE_SAMPLE_H
#define IIR_SINGLE_SAMPLE_H

typedef struct {
    double* b;           // Numerator coefficients
    double* a;           // Denominator coefficients
    double* input_buffer;  // Input history
    double* output_buffer; // Output history
    int num_taps;         // Filter order + 1
} IIRFilter;

int iir_filter_init(IIRFilter* filter, const double* b, const double* a, int filter_order);
double iir_filter_apply(IIRFilter* filter, double x);
void iir_filter_destroy(IIRFilter* filter);

#endif