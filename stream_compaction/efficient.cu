#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/binary_search.h>

#define blockSize 512

namespace StreamCompaction {
    namespace EfficientSharedMem {

        /*struct PathSegMirror { int d[11]; };*/

        static std::vector<int*> dev_datas;
        static int* dev_ndataA, * dev_ndataB, *dev_totalFalse;

        __global__ void kernExclusiveScanEachBlock(int N, int* odata, int* blocksum) {
            static_assert(blockSize % 32 == 0, "blockSize must be multiple of warpSize");
            constexpr int WARPS = blockSize / 32;

            __shared__ int warpSums[WARPS];
            const int tid = threadIdx.x;
            const int gid = threadIdx.x + (blockIdx.x * blockDim.x);
            const int lid = tid & 31;
            const int wid = tid >> 5;

            int x = odata[gid];

            int inclusiveInWarp = x;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int inclusiveInWarp_by_offset = __shfl_up_sync(0xFFFFFFFF, inclusiveInWarp, offset);
                if (lid >= offset) {
                    inclusiveInWarp += inclusiveInWarp_by_offset;
                }
            }
            int exclusiveInWarp = inclusiveInWarp - x;

            if (lid == 31) {
                warpSums[wid] = inclusiveInWarp;
            }
            __syncthreads();

            if (wid == 0) {
                int inclusiveWarpSum = (lid < WARPS) ? warpSums[lid] : 0;
#pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    int inclusiveWarpSum_by_offset = __shfl_up_sync(0xFFFFFFFF, inclusiveWarpSum, offset);
                    if (lid >= offset) {
                        inclusiveWarpSum += inclusiveWarpSum_by_offset;
                    }
                }
                if (lid < WARPS) {
                    // warpSums should be exlusive warpsum scan
                    warpSums[lid] = inclusiveWarpSum - warpSums[lid];
                }
            }
            __syncthreads();
            const int warpPrefix = warpSums[wid];

            if (gid < N) {
                odata[gid] = warpPrefix + exclusiveInWarp;
            }

            if (tid == blockSize - 1) {
                blocksum[blockIdx.x] = warpPrefix + inclusiveInWarp;
            }
        }

        __global__ void kernExclusiveScanOneBlock(int N, int* odata) {
            extern __shared__ int shared_odata[];

            const int tid = threadIdx.x;
            int maxdepth = ilog2ceil(N);

            // load into sharedmemory
            shared_odata[tid] = odata[tid];
            __syncthreads();

            // upsweep
            for (int d = 0; d < maxdepth; ++d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < N / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    shared_odata[tid_t] += shared_odata[tid_t - offset];
                }
                __syncthreads();
            }

            // save the last data(total sum of a block) 
            if (tid == N - 1) {
                shared_odata[tid] = 0; // and set the last data to 0
            }
            __syncthreads();

            // downsweep
            for (int d = maxdepth - 1; d >= 0; --d) {
                const int offset = 1 << d;
                const int twooffset = offset << 1;
                if (tid < N / twooffset) {
                    const int tid_t = (tid + 1) * twooffset - 1;
                    int temp = shared_odata[tid_t - offset];
                    shared_odata[tid_t - offset] = shared_odata[tid_t];
                    shared_odata[tid_t] += temp;
                }
                __syncthreads();
            }

            // write back to global memory
            odata[tid] = shared_odata[tid];
        }

        __global__ void kernAddBlockSum(int N, int* odata, int* scanned_blocksumdata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            odata[index] += scanned_blocksumdata[blockIdx.x];
        }

        __global__ void kernMapToInverseBooleanDigit(int n, int* bools, const int* idata, int digit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            bools[index] = 1 - GET_BIT(idata[index], digit);
        }

        __global__ void kernWritePartition(int n, int words4, void* odata, const void* idata, const int* fdata, const int* bdata, int totalKeep) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int shouldKeep = bdata[index];
            int fIndex = fdata[index];
            int tIndex = index - fIndex + totalKeep;
            int sortedIndex = shouldKeep ? fIndex : tIndex;

            //((PathSegMirror*)(odata))[sortedIndex] = ((const PathSegMirror*)(idata))[index];
            size_t src = (size_t)index * (size_t)words4;
            size_t dst = (size_t)sortedIndex * (size_t)words4;   
            const uint32_t* srcw = reinterpret_cast<const uint32_t*>(idata);
            uint32_t* dstw = reinterpret_cast<uint32_t*>(odata);

#pragma unroll
            for (int w = 0; w < words4; ++w)
                dstw[dst + w] = srcw[src + w];
        }

        __global__ void kernWriteRadixSort(int n, int* odata, const int* idata, const int* fdata, const int* totalFalse, int digit) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int in = idata[index];
            int fIndex = fdata[index];
            int sortedIndex = GET_BIT(in, digit) ? (index - fIndex + (*totalFalse)) : fdata[index];
            odata[sortedIndex] = in;
        }

       
        __global__ void kernSetSpecificValue(int index, int* idata, int* odata) {
            if (threadIdx.x == 0) *odata = idata[index];
        }

        __global__ void kernAddSpecificValue(int index, int* idata, int* odata) {
            if (threadIdx.x == 0) *odata += idata[index];
        }

        void recursiveExclusiveScan(int N, const std::vector<int*>& odatas, int bufferlvl) {
            if (N <= blockSize)
            {
                // I'm sure that N is power of 2
                kernExclusiveScanOneBlock <<<1, N, N * sizeof(int) >> > (N, odatas[bufferlvl]);
                return;
            }
            else
            {
                int fullBlocksPerGrid = ((N + blockSize - 1) / blockSize);
                kernExclusiveScanEachBlock << < fullBlocksPerGrid, blockSize >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
                recursiveExclusiveScan(fullBlocksPerGrid, odatas, bufferlvl + 1);
                kernAddBlockSum << <fullBlocksPerGrid, blockSize >> > (N, odatas[bufferlvl], odatas[bufferlvl + 1]);
            }
        }

        /**
         * Initialize buffers used for scan
         *
         * @param N      The size of next power-of-2 
         */
        void initializeBuffers(int N) {
            // memory allocation
            int bufferSize = N;
            do {
                int* dev_data;
                cudaMalloc(&dev_data, bufferSize * sizeof(int));
                cudaMemset(dev_data, 0, bufferSize * sizeof(int));
                dev_datas.push_back(dev_data);
                bufferSize /= blockSize;
            } while (bufferSize > 1);
            cudaMalloc(&dev_ndataA, N * sizeof(int));
            cudaMemset(dev_ndataA, 0, N * sizeof(int));
            cudaMalloc(&dev_ndataB, N * sizeof(int));
            cudaMemset(dev_ndataB, 0, N * sizeof(int));
            cudaMalloc(&dev_totalFalse, sizeof(int));
            cudaMemset(dev_totalFalse, 0, sizeof(int));
        }

        void freeBuffers() {
            for (auto dev_data : dev_datas) {
                cudaFree(dev_data);
            }
            cudaFree(dev_ndataA);
            cudaFree(dev_ndataB);
        }

        /**
         * Performs partition on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param dev_odata  The array into which to store elements.
         * @param dev_idata  The array of elements to partition.
         * @param dev_bdata  The array of elements showing that it should keep in the front
         * @return       The number of remaining elements
         */
        int partitionStable(int n, int elemSize, void* dev_odata, const void* dev_idata, int* dev_bdata) {
            int N = 1 << ilog2ceil(n);
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;
            // set dev_data to 0
            cudaMemset(dev_datas[0], 0, N * sizeof(int));
            cudaMemcpy(dev_datas[0], dev_bdata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            int fLast, eLast;
            if (n < N) {
                eLast = 0;
            }
            else {
                cudaMemcpy(&eLast, dev_datas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is e[n ¨C 1]
            }
            // do exclusive scan
            recursiveExclusiveScan(N, dev_datas, 0);
            // compute totalKeep
            cudaMemcpy(&fLast, dev_datas[0] + N - 1, sizeof(int), cudaMemcpyDeviceToHost); // this is f[n ¨C 1]
            int totalKeep = fLast + eLast;
            // write
            if (totalKeep > 0) {
                kernWritePartition << <fullBlocksPerGrid, blockSize >> > (n, elemSize / 4, dev_odata, dev_idata, dev_datas[0], dev_bdata, totalKeep);
            }
            return totalKeep;
        }

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param dev_odata  The array into which to store the index of elements. (can be the same as dev_idata)
         * @param dev_idata  The array of elements to sort (last 4 bit is mat_type, first 27 bit is index(exclude sign)).
         */
        int radixSortMatType(int n, int* dev_odata, const int* dev_idata) {
            constexpr int sort_bit = 4;
            int N = 1 << ilog2ceil(n);
            int fullBlocksPerGrid = (N + blockSize - 1) / blockSize;

            // copy data
            cudaMemset(dev_ndataA, 0xF, N * sizeof(int)); // put mant 7 into dev_datas[0], those 7s will be at the back of the array after sort
            cudaMemcpy(dev_ndataA, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // start kernel
            int totalRemain = 0;
            int* in_dev_ndata, * out_dev_ndata;
            for (int digit = 0; digit < sort_bit; ++digit) {
                // radix sort for one digit
                in_dev_ndata = (digit & 0x1) ? dev_ndataB : dev_ndataA;
                if (digit == sort_bit - 1) {
                    out_dev_ndata = dev_odata;
                }
                else {
                    out_dev_ndata = (digit & 0x1) ? dev_ndataA : dev_ndataB;
                }
                // compute e into fdatas[0]
                kernMapToInverseBooleanDigit<<<fullBlocksPerGrid, blockSize>>>(N, dev_datas[0], in_dev_ndata, digit);
                kernSetSpecificValue<<<1, 1>>>(N - 1, dev_datas[0], dev_totalFalse);
                // do exclusive scan
                recursiveExclusiveScan(N, dev_datas, 0);
                // compute totalFalse
                kernAddSpecificValue<<<1, 1>>>(N - 1, dev_datas[0], dev_totalFalse);
                // write this sort
                kernWriteRadixSort<<<fullBlocksPerGrid, blockSize>>>(n, out_dev_ndata, in_dev_ndata, dev_datas[0], dev_totalFalse, digit);
            }
            cudaMemcpy(&totalRemain, dev_totalFalse, sizeof(int), cudaMemcpyDeviceToHost);
            return totalRemain;
        }

        struct LowBits {
            __host__ __device__ int operator()(int x) const { return x & 0xF; }
        };

        /**
         * Performs radix sort on idata, storing the result into odata.
         *
         * @param n      The number of elements in idata.
         * @param dev_odata  The array into which to store the index of elements. (can be the same as dev_idata)
         * @param dev_idata  The array of elements to sort (last 4 bit is mat_type, first 27 bit is index(exclude sign)).
         */
        int radixSortMatTypeCUB(int n, int* dev_odata, const int* dev_idata) {
            constexpr int sort_bit = 4;
            constexpr int terminate_flag = 1 << (sort_bit - 1);
            cudaMemcpy(dev_ndataA, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            static void* temp = nullptr; static size_t temp_bytes = 0;
            if (!temp) {
                cub::DeviceRadixSort::SortKeys(temp, temp_bytes, dev_ndataA, dev_odata, n,
                    0, sort_bit, 0);
                cudaMalloc(&temp, temp_bytes);
            }
            cub::DeviceRadixSort::SortKeys(temp, temp_bytes, dev_ndataA, dev_odata, n,
                0, sort_bit, 0);

            thrust::device_ptr<int> p(dev_odata);
            auto first = thrust::make_transform_iterator(p, LowBits{});
            auto last = first + n;
            auto it = thrust::lower_bound(thrust::device, first, last, terminate_flag);
            int totalRemain = static_cast<int>(it - first);
            return totalRemain;
        }
    }
}
