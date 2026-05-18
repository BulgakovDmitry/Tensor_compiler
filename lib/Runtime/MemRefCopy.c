#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t rank;
    void *descriptor;
} UnrankedMemRefType;

typedef struct {
    char *basePtr;
    char *data;
    int64_t offset;
    int64_t sizesAndStrides[];
} StridedMemRefDescriptor;

void memrefCopy(int64_t elemSize, UnrankedMemRefType *srcArg,
                UnrankedMemRefType *dstArg) {
    int64_t rank = srcArg->rank;
    StridedMemRefDescriptor *src =
        (StridedMemRefDescriptor *)srcArg->descriptor;
    StridedMemRefDescriptor *dst =
        (StridedMemRefDescriptor *)dstArg->descriptor;

    int64_t *srcSizes = src->sizesAndStrides;
    int64_t *srcStrides = src->sizesAndStrides + rank;
    int64_t *dstSizes = dst->sizesAndStrides;
    int64_t *dstStrides = dst->sizesAndStrides + rank;

    char *srcPtr = src->data + src->offset * elemSize;
    char *dstPtr = dst->data + dst->offset * elemSize;

    if (rank == 0) {
        memcpy(dstPtr, srcPtr, (size_t)elemSize);
        return;
    }

    for (int64_t i = 0; i < rank; ++i) {
        if (srcSizes[i] == 0) {
            return;
        }
    }

    int64_t indices[rank];
    memset(indices, 0, sizeof(indices));
    int64_t readIndex = 0;
    int64_t writeIndex = 0;

    for (;;) {
        memcpy(dstPtr + writeIndex, srcPtr + readIndex, (size_t)elemSize);

        for (int64_t axis = rank - 1; axis >= 0; --axis) {
            int64_t newIndex = ++indices[axis];
            readIndex += srcStrides[axis] * elemSize;
            writeIndex += dstStrides[axis] * elemSize;

            if (srcSizes[axis] != newIndex) {
                break;
            }
            if (axis == 0) {
                return;
            }

            indices[axis] = 0;
            readIndex -= srcSizes[axis] * srcStrides[axis] * elemSize;
            writeIndex -= dstSizes[axis] * dstStrides[axis] * elemSize;
        }
    }
}
