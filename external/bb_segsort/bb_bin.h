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

#ifndef _H_BB_BIN
#define _H_BB_BIN

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define SEGBIN_NUM 13

__global__
void bb_bin_histo(int *d_bin_counter, const int *d_segs, int length, int n);

__global__
void bb_bin_group(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, int length, int n);

void bb_bin(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, 
        const int length, const int n, int *h_bin_counter)
{
    const int num_threads = 256;
    const int num_blocks = ceil((double)length/(double)num_threads);

    bb_bin_histo<<< num_blocks, num_threads >>>(d_bin_counter, d_segs, length, n);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    thrust::device_ptr<int> d_arr = thrust::device_pointer_cast<int>(d_bin_counter);
    thrust::exclusive_scan(d_arr, d_arr + SEGBIN_NUM, d_arr);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    cudaMemcpyAsync(h_bin_counter, d_bin_counter, SEGBIN_NUM*sizeof(int), cudaMemcpyDeviceToHost);

    // group segment IDs (that belong to the same bin) together
    bb_bin_group<<< num_blocks, num_threads >>>(d_bin_segs_id, d_bin_counter, d_segs, length, n);

    // show_d(d_bin_segs_id, length, "d_bin_segs_id:\n");
}

__global__
void bb_bin_histo(int *d_bin_counter, const int *d_segs, int length, int n)
{
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x; 

    __shared__ int local_histo[SEGBIN_NUM];
    if (tid < SEGBIN_NUM)
        local_histo[tid] = 0;
    __syncthreads();

    if (gid < length)
    {
        const int size = ((gid==length-1)?n:d_segs[gid+1]) - d_segs[gid];

        if (size <= 1)
            atomicAdd((int *)&local_histo[0 ], 1);
        if (1  < size && size <= 2 )
            atomicAdd((int *)&local_histo[1 ], 1);
        if (2  < size && size <= 4 )
            atomicAdd((int *)&local_histo[2 ], 1);
        if (4  < size && size <= 8 )
            atomicAdd((int *)&local_histo[3 ], 1);
        if (8  < size && size <= 16)
            atomicAdd((int *)&local_histo[4 ], 1);
        if (16 < size && size <= 32)
            atomicAdd((int *)&local_histo[5 ], 1);
        if (32 < size && size <= 64)
            atomicAdd((int *)&local_histo[6 ], 1);
        if (64 < size && size <= 128)
            atomicAdd((int *)&local_histo[7 ], 1);
        if (128 < size && size <= 256)
            atomicAdd((int *)&local_histo[8 ], 1);
        if (256 < size && size <= 512)
            atomicAdd((int *)&local_histo[9 ], 1);
        if (512 < size && size <= 1024)
            atomicAdd((int *)&local_histo[10], 1);
        if (1024 < size && size <= 2048)
            atomicAdd((int *)&local_histo[11], 1);
        if (2048 < size)
            atomicAdd((int *)&local_histo[12], 1);
    }
    __syncthreads();

    if (tid < SEGBIN_NUM)
        atomicAdd((int *)&d_bin_counter[tid], local_histo[tid]);
}

__global__
void bb_bin_group(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, int length, int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length)
    {
        const int size = ((gid==length-1)?n:d_segs[gid+1]) - d_segs[gid];
        int position;
        if (size <= 1)
            position = atomicAdd((int *)&d_bin_counter[0 ], 1);
        else if (size <= 2)                              
            position = atomicAdd((int *)&d_bin_counter[1 ], 1);
        else if (size <= 4)                              
            position = atomicAdd((int *)&d_bin_counter[2 ], 1);
        else if (size <= 8)                              
            position = atomicAdd((int *)&d_bin_counter[3 ], 1);
        else if (size <= 16)                             
            position = atomicAdd((int *)&d_bin_counter[4 ], 1);
        else if (size <= 32)                             
            position = atomicAdd((int *)&d_bin_counter[5 ], 1);
        else if (size <= 64)                             
            position = atomicAdd((int *)&d_bin_counter[6 ], 1);
        else if (size <= 128)                            
            position = atomicAdd((int *)&d_bin_counter[7 ], 1);
        else if (size <= 256)                            
            position = atomicAdd((int *)&d_bin_counter[8 ], 1);
        else if (size <= 512)                            
            position = atomicAdd((int *)&d_bin_counter[9 ], 1);
        else if (size <= 1024)
            position = atomicAdd((int *)&d_bin_counter[10], 1);
        else if (size <= 2048)
            position = atomicAdd((int *)&d_bin_counter[11], 1);
        else
            position = atomicAdd((int *)&d_bin_counter[12], 1);
        d_bin_segs_id[position] = gid;
    }
}

#endif
