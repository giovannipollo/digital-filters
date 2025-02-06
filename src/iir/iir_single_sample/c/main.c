#include "iir_single_sample.h"
#include <stdio.h>

int main() {
    double b[] = {0.00718404, 0.0, -0.02873616, 0.0, 0.04310425, 0.0, -0.02873616, 0.0, 0.00718404}; // Numerator coefficients
    double a[] = {1.0, -5.97101379, 15.76907227, -24.15665094, 23.54903262,
                  -14.9792961, 6.06950005, -1.43110205, 0.15046446}; // Denominator coefficients

    int filter_order = 8;

    IIRFilter filter;
    if (iir_filter_init(&filter, b, a, filter_order) != 0) {
        printf("Failed to initialize filter\n");
        return -1;
    }

    // Open the file called "considered_ppg_patient_1.txt" for reading
    FILE* file = fopen("../considered_ppg_patient_1.txt", "r");
    if (!file) {
        printf("Failed to open file\n");
        return -1;
    }

    // Read the first 10000 samples from the file
    for (int i = 0; i < 10; i++) {
        double x;
        if (fscanf(file, "%lf", &x) != 1) {
            printf("Failed to read sample\n");
            return -1;
        }

        double y = iir_filter_apply(&filter, x);
        printf("%f\n", y);
    }

    iir_filter_destroy(&filter);
    return 0;
}