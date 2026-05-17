#include "ModelAPI/ModelAPI.h"
#include <string.h>
#include <stdint.h>

extern int tensorCompForwardImpl(float* input, float* output);

int tensorCompForward(const float* input, float* output) {
    return tensorCompForwardImpl((float*)input, output);
}
