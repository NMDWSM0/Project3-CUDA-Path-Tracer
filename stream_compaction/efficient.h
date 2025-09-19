#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientSharedMem {

        void initializeBuffers(int N);

        void freeBuffers();

        int partitionStable(int n, int elemSize, void* dev_odata, const void* dev_idata, int* dev_bdata);

        int radixSortMatType(int n, int* dev_odata, const int* dev_idata);

        int radixSortMatTypeCUB(int n, int* dev_odata, const int* dev_idata);
    }
}
