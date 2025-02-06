#include "adaptive_single_sample.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_TAPS 5
#define MU 0.01f
#define NUM_CHANNELS 3

int main() {
    AdaptiveFilter *filter = filter_create(NUM_TAPS, MU, NUM_CHANNELS);
    if (!filter) {
        printf("Failed to create filter\n");
        return 1;
    }

    FILE *input_file = fopen("../input_signal.txt", "r");
    FILE *desired_file = fopen("../desired_signal.txt", "r");
    if (!input_file || !desired_file) {
        printf("Failed to open input files\n");
        return 1;
    }

    float x[NUM_CHANNELS];
    float d;
    float outputs[NUM_CHANNELS];
    int sample = 0;
    char line[256];

    // Process signals line by line
    while (!feof(input_file) && !feof(desired_file)) {
        // Read input signal line
        if (fgets(line, sizeof(line), input_file)) {
            if (sscanf(line, "[%f %f %f]", &x[0], &x[1], &x[2]) != 3) {
                printf("Failed to parse input line: %s\n", line);
                continue;
            }
        }

        // Read desired signal
        if (fscanf(desired_file, "%f", &d) != 1) {
            printf("Failed to read desired signal\n");
            break;
        }

        filter_adapt(filter, x, d, outputs);

        /*
        printf("Sample %d:\n", sample);
        printf("  Inputs: %.6f, %.6f, %.6f\n", x[0], x[1], x[2]);
        printf("  Desired: %.6f\n", d);
        printf("  Outputs: %.6f, %.6f, %.6f\n",
               outputs[0], outputs[1], outputs[2]);
        printf("\n");
        */
        if (sample == 100) {
            break;
        }
        sample++;
    }

    printf("Total samples processed: %d\n", sample);

    fclose(input_file);
    fclose(desired_file);
    filter_free(filter);
    return 0;
}