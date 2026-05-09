#include "ModelAPI/ModelAPI.h"

extern int tensorCompForwardImpl(const float* input, float* output);

int tensorCompForward(const float* input, float* output) {
    if (!input || !output) return -1;

    return tensorCompForwardImpl(input, output);
}
