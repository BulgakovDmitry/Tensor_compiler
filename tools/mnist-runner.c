#include "ModelAPI/ModelAPI.h"

#include <stdio.h>
#include <stdlib.h>

enum {
    MNIST_INPUT_SIZE = 1 * 1 * 28 * 28,
    MNIST_OUTPUT_SIZE = 10,
};

static int readInput(const char *path, float *input) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        return 1;
    }

    size_t readCount =
        fread(input, sizeof(float), MNIST_INPUT_SIZE, file);
    fclose(file);

    if (readCount != MNIST_INPUT_SIZE) {
        fprintf(stderr, "expected %d float32 values, got %zu\n",
                MNIST_INPUT_SIZE, readCount);
        return 1;
    }

    return 0;
}

int main(int argc, char **argv) {
    float input[MNIST_INPUT_SIZE] = {0.0f};
    float output[MNIST_OUTPUT_SIZE] = {0.0f};

    if (argc > 2) {
        fprintf(stderr, "usage: %s [input-f32.bin]\n", argv[0]);
        return 1;
    }

    if (argc == 2 && readInput(argv[1], input) != 0) {
        return 1;
    }

    int status = tensorCompForward(input, output);
    if (status != 0) {
        fprintf(stderr, "tensorCompForward failed: %d\n", status);
        return status;
    }

    int best = 0;
    for (int i = 1; i < MNIST_OUTPUT_SIZE; ++i) {
        if (output[i] > output[best]) {
            best = i;
        }
    }

    printf("logits:");
    for (int i = 0; i < MNIST_OUTPUT_SIZE; ++i) {
        printf(" %.6f", output[i]);
    }
    printf("\nprediction: %d\n", best);
    return 0;
}
