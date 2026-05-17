#include "ModelAPI/ModelAPI.h"
#include <string.h>
#include <stdint.h>

extern int tensorCompForwardImpl(float* input, float* output);

int tensorCompForward(const float* input, float* output) {
    return tensorCompForwardImpl((float*)input, (float*)input);
}

/// Stub for MLIR's memrefCopy runtime function.
/// MLIR emits calls to this when lowering memref.copy ops.
void memrefCopy(int64_t elemSize, void *src, void *dst) {
    if (elemSize == 4 && src && dst && src != dst) {
        // Hardcode for input [1,3,224,224]: 1*3*224*224*4 = 602112 bytes
        memcpy(dst, src, 602112);
    }
}
