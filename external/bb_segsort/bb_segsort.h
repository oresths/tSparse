/*
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)   
*                                                                              
*   This program is free software: you can redistribute it and/or modify       
*   it under the terms of the GNU General Public License as published by       
*   the Free Software Foundation, version 2.1                                  
*                                                                              
*   This program is distributed in the hope that it will be useful,            
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              
*   GNU General Public License, version 2.1, for more details.                 
*                                                                              
*   You should have received a copy of the GNU General Public License          
*                                                                              
*/

#ifndef _H_BB_SEGSORT
#define _H_BB_SEGSORT

#include <iostream>
#include <vector>
#include <algorithm>
template<class T>
void show_d(T *arr_d, int n, std::string prompt);

#include "bb_bin.h"
#include "bb_comput_s.h"
#include "bb_comput_l.h"

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }


template<class K, class T>
int bb_segsort(K *keys_d, T *vals_d, int n,  int *d_segs, int length)
{
    cudaError_t cuda_err;
    int *h_bin_counter = new int[SEGBIN_NUM];

    int *d_bin_counter;
    int *d_bin_segs_id;
    cuda_err = cudaMalloc((void **)&d_bin_counter, SEGBIN_NUM * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, length * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

    cuda_err = cudaMemset(d_bin_counter, 0, SEGBIN_NUM * sizeof(int));
    CUDA_CHECK(cuda_err, "memset d_bin_counter");

    K *keysB_d;
    T *valsB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, n * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc keysB_d");
    cuda_err = cudaMalloc((void **)&valsB_d, n * sizeof(T));
    CUDA_CHECK(cuda_err, "alloc valsB_d");

    bb_bin(d_bin_segs_id, d_bin_counter, d_segs, length, n, h_bin_counter);

    cudaStream_t streams[SEGBIN_NUM-1];
    for(int i = 0; i < SEGBIN_NUM-1; i++) cudaStreamCreate(&streams[i]);

    int subwarp_size, subwarp_num, factor;
    dim3 blocks(256, 1, 1);
    dim3 grids(1, 1, 1);

    blocks.x = 256;
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];
    grids.x = (subwarp_num+blocks.x-1)/blocks.x;
    if(subwarp_num > 0)
    gen_copy<<<grids, blocks, 0, streams[0]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[0], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp2_tc1_r2_r2_orig<<<grids, blocks, 0, streams[1]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[1], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc2_r3_r4_orig<<<grids, blocks, 0, streams[2]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[2], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc4_r5_r8_orig<<<grids, blocks, 0, streams[3]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[3], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 4;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp4_tc4_r9_r16_strd<<<grids, blocks, 0, streams[4]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[4], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp8_tc4_r17_r32_strd<<<grids, blocks, 0, streams[5]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[5], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp16_tc4_r33_r64_strd<<<grids, blocks, 0, streams[6]>>>(keys_d, vals_d, keysB_d, valsB_d, 
        n, d_segs, d_bin_segs_id+h_bin_counter[6], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp8_tc16_r65_r128_strd<<<grids, blocks, 0, streams[7]>>>(keys_d, vals_d, keysB_d, valsB_d,  
        n, d_segs, d_bin_segs_id+h_bin_counter[7], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 32;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp32_tc8_r129_r256_strd<<<grids, blocks, 0, streams[8]>>>(keys_d, vals_d, keysB_d, valsB_d,  
        n, d_segs, d_bin_segs_id+h_bin_counter[8], subwarp_num, length);

    blocks.x = 128;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk128_tc4_r257_r512_orig<<<grids, blocks, 0, streams[9]>>>(keys_d, vals_d, keysB_d, valsB_d,   
        n, d_segs, d_bin_segs_id+h_bin_counter[9], subwarp_num, length);

    blocks.x = 256;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk256_tc4_r513_r1024_orig<<<grids, blocks, 0, streams[10]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[10], subwarp_num, length);

    blocks.x = 512;
    subwarp_num = h_bin_counter[12]-h_bin_counter[11];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk512_tc4_r1025_r2048_orig<<<grids, blocks, 0, streams[11]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[11], subwarp_num, length);

    // sort long segments
    subwarp_num = length-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[12], subwarp_num, length);
    
    // std::swap(keys_d, keysB_d);
    // std::swap(vals_d, valsB_d);
    cuda_err = cudaMemcpy(keys_d, keysB_d, sizeof(K)*n, cudaMemcpyDeviceToDevice);
    CUDA_CHECK(cuda_err, "copy to keys_d from keysB_d");
    cuda_err = cudaMemcpy(vals_d, valsB_d, sizeof(T)*n, cudaMemcpyDeviceToDevice);
    CUDA_CHECK(cuda_err, "copy to vals_d from valsB_d");

    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    CUDA_CHECK(cuda_err, "free keysB");
    cuda_err = cudaFree(valsB_d);
    CUDA_CHECK(cuda_err, "free valsB");

    for (int i = 0; i < SEGBIN_NUM - 1; i++) cudaStreamDestroy(streams[i]);
    delete[] h_bin_counter;
    return 1;
}

template<class T>
void show_d(T *arr_d, int n, std::string prompt)
{
    std::vector<T> arr_h(n);
    cudaMemcpy(&arr_h[0], arr_d, sizeof(T)*n, cudaMemcpyDeviceToHost);
    std::cout << prompt;
    for(auto v: arr_h) std::cout << v << ", "; std::cout << std::endl;
}
#endif
