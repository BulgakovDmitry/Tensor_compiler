#ifndef TENSORDRIVER_TENSORDRVAPI_H
#define TENSORDRIVER_TENSORDRVAPI_H

#include <dlpack/dlpack.h>

void tensorDrvRun(const DLTensor* inputs, int numInputs,
                  DLTensor* outputs, int numOutputs);


#endif // TENSORDRIVER_TENSORDRVAPI_H
