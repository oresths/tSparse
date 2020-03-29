/*
 * mm.h
 *
 *  Created on: Mar 26, 2019
 *      Author: ore
 */

#ifndef SRC_MM_H_
#define SRC_MM_H_


#include <stddef.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <cusp/csr_matrix.h>

#define HALF_MAX 65520.f
#define HALF_MIN 0.00000006f

#define BMP_DIM 8U //The dimension of the bitmap block

// Reading or writing to TCUs, although the pattern is theoretically unknown, leads to bank conflicts. To avoid
// them we pad. But reading/writing to fragments needs 128-bit alignment, so the minimum padding is 8 halfs.
#define PAD_HALF 8
#define PAD_FLOAT 4

#define WARPS_BLOCK 2 //warps per block
#define TILES_WARP 2 //tiles per warp
#define TILES_BLOCK (WARPS_BLOCK * TILES_WARP) //tiles per block

#define DEBUG 0 // Print additional info from the algo
#define DEBUG_API 0 // Collect CUDA API errors
#define TIMING_VERBOSE 0 // Time specific algorithm steps

#define GPU_WARMUP 1 // Make GPU come out of low power mode
// How many times the implementation will be repeated in order to get average of results.
// 1 repetition is gives faster time, something with memory allocation/deallocation?
// Also CUSP has memory problems with more than 1 repetitions
#define REPETITIONS 1

//Set to 1 to try bringing out of fp16 range matrices in the fp16 range. Scaling keeps positive definiteness and
//we can scale the output to its original range losslessly. Not sure if useful.
#define SCALE 0

//#define USE_NVTX // Enable, disable profiler markers


#define gpuErrchk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char const *const source, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s: %s %s %d: %s\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line,
                source);
        if (abort) {
            cudaDeviceReset();
            exit(code);
        }
    }
}

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


using IndexTypeH = int;  //indices
using ElemIndexTypeH = uint32_t;
using ValueTypeBMPH = thrust::tuple<uint32_t, uint64_t>; //index in values array, bitmap
using ValueTypeBMPH_NT = uint64_t; //bitmap
using ValueTypeH = float; // the actual values of values array
using MatrixTypeH = cusp::coo_matrix<IndexTypeH,ValueTypeBMPH,cusp::device_memory>;
using MatrixTypeH_NT = cusp::coo_matrix<IndexTypeH,ValueTypeBMPH_NT,cusp::device_memory>;

using MatrixTypeCOO = cusp::coo_matrix<IndexTypeH,ValueTypeH,cusp::host_memory>;

void multiplyBmp(const MatrixTypeH& A, const thrust::device_vector<ValueTypeH>& A_elems, const MatrixTypeH& B,
    const thrust::device_vector<ValueTypeH>& B_elems, MatrixTypeH& C, thrust::device_vector<ValueTypeH>& C_elems);
void multiplyBmp_noTuple(const MatrixTypeH_NT& A, const thrust::device_vector<ValueTypeH>& A_elems,
        const thrust::device_vector<ElemIndexTypeH>& A_idx, const MatrixTypeH_NT& B,
        const thrust::device_vector<ValueTypeH>& B_elems, const thrust::device_vector<ElemIndexTypeH>& B_idx,
        MatrixTypeH_NT& C, thrust::device_vector<ValueTypeH>& C_elems, thrust::device_vector<ElemIndexTypeH>& C_idx);
void get_characteristics(const MatrixTypeH& A, const thrust::device_vector<ValueTypeH>& A_elems, const MatrixTypeH& B,
    const thrust::device_vector<ValueTypeH>& B_elems, MatrixTypeH& C, thrust::device_vector<ValueTypeH>& C_elems,
    const MatrixTypeCOO& A_coo, const MatrixTypeCOO& B_coo, MatrixTypeCOO& C_coo);


#endif /* SRC_MM_H_ */
