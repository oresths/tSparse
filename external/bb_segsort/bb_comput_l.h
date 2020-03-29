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

#ifndef _H_BB_COMPUT_L
#define _H_BB_COMPUT_L

#define ERR_INFO(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <limits>

__device__
int binary_search(int *blk_stat, int bin_size, int gid, int blk_num)
{
    int l = 0;
    int h = bin_size;
    int m;
    int lr, rr;
    while(l < h)
    {
        m = l + (h-l)/2;
        lr = blk_stat[m];
        rr = (m==bin_size)?blk_num:blk_stat[m+1];
        if(lr<=gid && gid<rr)
        {
            return m;
        } else if(gid < lr)
        {
            h = m;
        } else
        {
            l = m+1;
        }
    }
    return m;
}

__device__ inline
int upper_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

__device__ inline
int log2(int u)
{
    int s, t;
    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;
    return (t | (u >> 1));
}

__global__
void kern_get_num_blk_init(int *max_segsize, int *segs, int *bin, int *blk_stat, 
        int n, int bin_size, int length, int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        int seg_size = ((bin[gid]==length-1)?n:segs[bin[gid]+1])-segs[bin[gid]];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
        atomicMax(max_segsize, seg_size);
    }
}

__global__
void kern_get_init_pos(int *blk_stat, int *blk_innerid, int *blk_seg_start, 
        int blk_num, int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < blk_num)
    {
        int found = binary_search(blk_stat, bin_size, gid, blk_num);
        blk_innerid[gid] = gid - blk_stat[found];
        blk_seg_start[gid] = found;
    }
}

template<class K, class T>
__global__
void kern_block_sort(K *key, T *val, K *keyB, T *valB, int *segs, 
        int *bin, int *blk_innerid, int *blk_seg_start, int length, int n)
{
    /*** codegen ***/
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];
    /*** codegen ***/
    const int tid = threadIdx.x;
    // const int bin_it = blockIdx.x;
    __shared__ K smem[2048];
    __shared__ int tmem[2048];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    // int k;
    // int ext_seg_size;
    /*** codegen ***/
    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    k = k + (innerbid<<11);
    seg_size = min(seg_size-(innerbid<<11), 2048);
    /*** codegen ***/
    rg_k0  = (tid1+(warp_id<<7)+0   <seg_size)?key[k+tid1+(warp_id<<7)+0   ]:std::numeric_limits<K>::max();
    rg_k1  = (tid1+(warp_id<<7)+32  <seg_size)?key[k+tid1+(warp_id<<7)+32  ]:std::numeric_limits<K>::max();
    rg_k2  = (tid1+(warp_id<<7)+64  <seg_size)?key[k+tid1+(warp_id<<7)+64  ]:std::numeric_limits<K>::max();
    rg_k3  = (tid1+(warp_id<<7)+96  <seg_size)?key[k+tid1+(warp_id<<7)+96  ]:std::numeric_limits<K>::max();
    if(tid1+(warp_id<<7)+0   <seg_size) rg_v0  = tid1+(warp_id<<7)+0   ;
    if(tid1+(warp_id<<7)+32  <seg_size) rg_v1  = tid1+(warp_id<<7)+32  ;
    if(tid1+(warp_id<<7)+64  <seg_size) rg_v2  = tid1+(warp_id<<7)+64  ;
    if(tid1+(warp_id<<7)+96  <seg_size) rg_v3  = tid1+(warp_id<<7)+96  ;
    // exch_intxn: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
    CMP_SWP(K,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(K,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x3,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(K,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x7,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(K,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0xf,bit4);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x4,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(K,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1f,bit5);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x8,bit4);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x4,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(K,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(K,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(K,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    
    smem[(warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
    smem[(warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
    smem[(warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
    smem[(warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
    tmem[(warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
    tmem[(warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
    tmem[(warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
    tmem[(warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
    __syncthreads();
    // Merge in 4 steps
    int grp_start_wp_id;
    int grp_start_off;
    // int tmp_wp_id;
    int lhs_len;
    int rhs_len;
    int gran;
    int s_a;
    int s_b;
    bool p;
    K tmp_k0;
    K tmp_k1;
    int tmp_v0;
    int tmp_v1;
    K *start;
    // Step 0
    grp_start_wp_id = ((warp_id>>1)<<1);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (128 );
    rhs_len = (128 );
    gran = (tid1<<2);
    if((warp_id&1)==0){
        gran += 0;
    }
    if((warp_id&1)==1){
        gran += (128 );
    }
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 1
    grp_start_wp_id = ((warp_id>>2)<<2);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (256 );
    rhs_len = (256 );
    gran = (tid1<<2);
    gran += (warp_id&3)*128;
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 2
    grp_start_wp_id = ((warp_id>>3)<<3);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (512 );
    rhs_len = (512 );
    gran = (tid1<<2);
    gran += (warp_id&7)*128;
    
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 3
    grp_start_wp_id = ((warp_id>>4)<<4);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (1024);
    rhs_len = (1024);
    gran = (tid1<<2);
    gran += (warp_id&15)*128;
    
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    
    if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
    if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
    if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
    if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    T t_v0 ;
    T t_v1 ;
    T t_v2 ;
    T t_v3 ;
    if((tid<<2)+0 <seg_size) t_v0  = val[k+rg_v0 ];
    if((tid<<2)+1 <seg_size) t_v1  = val[k+rg_v1 ];
    if((tid<<2)+2 <seg_size) t_v2  = val[k+rg_v2 ];
    if((tid<<2)+3 <seg_size) t_v3  = val[k+rg_v3 ];
    if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = t_v0 ;
    if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = t_v1 ;
    if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = t_v2 ;
    if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = t_v3 ;
}

__global__
void kern_get_num_blk(int *segs, int *bin, int *blk_stat, 
        int n, int bin_size, int length, int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        int seg_size = ((bin[gid]==length-1)?n:segs[bin[gid]+1])-segs[bin[gid]];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
    }
}

template<class K, class T>
__global__
void kern_block_merge(K *keys, T *vals, K *keysB, T *valsB, int *segs, int *bin, 
        int *blk_innerid, int *blk_seg_start, int length, int n, int stride)
{
    __shared__ K smem[128*16];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];

    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    if(stride < seg_size)
    {
        int loc_a, loc_b;
        int cnt_a, cnt_b;
        int coop = (stride<<1)>>11; // how many blocks coop
        int coop_bid = innerbid%coop;
        int l_gran, r_gran;
        loc_a = (innerbid/coop)*(stride<<1);
        cnt_a = min(stride, seg_size-loc_a);
        loc_b = min(loc_a + stride, seg_size);
        cnt_b = min(stride, seg_size-loc_b);
        l_gran = coop_bid<<11;
        r_gran = min((coop_bid+1)<<11, seg_size-loc_a);
        int l_s_a, l_s_b;
        int r_s_a, r_s_b;
        l_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, l_gran);
        l_s_b = l_gran - l_s_a;
        r_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, r_gran);
        r_s_b = r_gran - r_s_a;
        int l_st = 0;
        int l_cnt = r_s_a - l_s_a;
        if(l_s_a+tid     <r_s_a) smem[l_st+tid     ] = keys[k+loc_a+l_s_a+tid     ];
        if(l_s_a+tid+128 <r_s_a) smem[l_st+tid+128 ] = keys[k+loc_a+l_s_a+tid+128 ];
        if(l_s_a+tid+256 <r_s_a) smem[l_st+tid+256 ] = keys[k+loc_a+l_s_a+tid+256 ];
        if(l_s_a+tid+384 <r_s_a) smem[l_st+tid+384 ] = keys[k+loc_a+l_s_a+tid+384 ];
        if(l_s_a+tid+512 <r_s_a) smem[l_st+tid+512 ] = keys[k+loc_a+l_s_a+tid+512 ];
        if(l_s_a+tid+640 <r_s_a) smem[l_st+tid+640 ] = keys[k+loc_a+l_s_a+tid+640 ];
        if(l_s_a+tid+768 <r_s_a) smem[l_st+tid+768 ] = keys[k+loc_a+l_s_a+tid+768 ];
        if(l_s_a+tid+896 <r_s_a) smem[l_st+tid+896 ] = keys[k+loc_a+l_s_a+tid+896 ];
        if(l_s_a+tid+1024<r_s_a) smem[l_st+tid+1024] = keys[k+loc_a+l_s_a+tid+1024];
        if(l_s_a+tid+1152<r_s_a) smem[l_st+tid+1152] = keys[k+loc_a+l_s_a+tid+1152];
        if(l_s_a+tid+1280<r_s_a) smem[l_st+tid+1280] = keys[k+loc_a+l_s_a+tid+1280];
        if(l_s_a+tid+1408<r_s_a) smem[l_st+tid+1408] = keys[k+loc_a+l_s_a+tid+1408];
        if(l_s_a+tid+1536<r_s_a) smem[l_st+tid+1536] = keys[k+loc_a+l_s_a+tid+1536];
        if(l_s_a+tid+1664<r_s_a) smem[l_st+tid+1664] = keys[k+loc_a+l_s_a+tid+1664];
        if(l_s_a+tid+1792<r_s_a) smem[l_st+tid+1792] = keys[k+loc_a+l_s_a+tid+1792];
        if(l_s_a+tid+1920<r_s_a) smem[l_st+tid+1920] = keys[k+loc_a+l_s_a+tid+1920];
        int r_st = r_s_a - l_s_a;
        int r_cnt = r_s_b - l_s_b;
        if(l_s_b+tid     <r_s_b) smem[r_st+tid     ] = keys[k+loc_b+l_s_b+tid     ];
        if(l_s_b+tid+128 <r_s_b) smem[r_st+tid+128 ] = keys[k+loc_b+l_s_b+tid+128 ];
        if(l_s_b+tid+256 <r_s_b) smem[r_st+tid+256 ] = keys[k+loc_b+l_s_b+tid+256 ];
        if(l_s_b+tid+384 <r_s_b) smem[r_st+tid+384 ] = keys[k+loc_b+l_s_b+tid+384 ];
        if(l_s_b+tid+512 <r_s_b) smem[r_st+tid+512 ] = keys[k+loc_b+l_s_b+tid+512 ];
        if(l_s_b+tid+640 <r_s_b) smem[r_st+tid+640 ] = keys[k+loc_b+l_s_b+tid+640 ];
        if(l_s_b+tid+768 <r_s_b) smem[r_st+tid+768 ] = keys[k+loc_b+l_s_b+tid+768 ];
        if(l_s_b+tid+896 <r_s_b) smem[r_st+tid+896 ] = keys[k+loc_b+l_s_b+tid+896 ];
        if(l_s_b+tid+1024<r_s_b) smem[r_st+tid+1024] = keys[k+loc_b+l_s_b+tid+1024];
        if(l_s_b+tid+1152<r_s_b) smem[r_st+tid+1152] = keys[k+loc_b+l_s_b+tid+1152];
        if(l_s_b+tid+1280<r_s_b) smem[r_st+tid+1280] = keys[k+loc_b+l_s_b+tid+1280];
        if(l_s_b+tid+1408<r_s_b) smem[r_st+tid+1408] = keys[k+loc_b+l_s_b+tid+1408];
        if(l_s_b+tid+1536<r_s_b) smem[r_st+tid+1536] = keys[k+loc_b+l_s_b+tid+1536];
        if(l_s_b+tid+1664<r_s_b) smem[r_st+tid+1664] = keys[k+loc_b+l_s_b+tid+1664];
        if(l_s_b+tid+1792<r_s_b) smem[r_st+tid+1792] = keys[k+loc_b+l_s_b+tid+1792];
        if(l_s_b+tid+1920<r_s_b) smem[r_st+tid+1920] = keys[k+loc_b+l_s_b+tid+1920];
        __syncthreads();

        int gran = tid<<4;
        int s_a, s_b;
        bool p;
        K rg_k0 ;
        K rg_k1 ;
        K rg_k2 ;
        K rg_k3 ;
        K rg_k4 ;
        K rg_k5 ;
        K rg_k6 ;
        K rg_k7 ;
        K rg_k8 ;
        K rg_k9 ;
        K rg_k10;
        K rg_k11;
        K rg_k12;
        K rg_k13;
        K rg_k14;
        K rg_k15;
        int rg_v0 ;
        int rg_v1 ;
        int rg_v2 ;
        int rg_v3 ;
        int rg_v4 ;
        int rg_v5 ;
        int rg_v6 ;
        int rg_v7 ;
        int rg_v8 ;
        int rg_v9 ;
        int rg_v10;
        int rg_v11;
        int rg_v12;
        int rg_v13;
        int rg_v14;
        int rg_v15;
        K tmp_k0,tmp_k1;
        int tmp_v0,tmp_v1;

        s_a = find_kth3(smem+l_st, l_cnt, smem+r_st, r_cnt, gran);
        s_b = gran - s_a;
        tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k0 = p ? tmp_k0 : tmp_k1;
        rg_v0 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k1 = p ? tmp_k0 : tmp_k1;
        rg_v1 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k2 = p ? tmp_k0 : tmp_k1;
        rg_v2 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k3 = p ? tmp_k0 : tmp_k1;
        rg_v3 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k4 = p ? tmp_k0 : tmp_k1;
        rg_v4 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k5 = p ? tmp_k0 : tmp_k1;
        rg_v5 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k6 = p ? tmp_k0 : tmp_k1;
        rg_v6 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k7 = p ? tmp_k0 : tmp_k1;
        rg_v7 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k8 = p ? tmp_k0 : tmp_k1;
        rg_v8 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k9 = p ? tmp_k0 : tmp_k1;
        rg_v9 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k10 = p ? tmp_k0 : tmp_k1;
        rg_v10 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k11 = p ? tmp_k0 : tmp_k1;
        rg_v11 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k12 = p ? tmp_k0 : tmp_k1;
        rg_v12 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k13 = p ? tmp_k0 : tmp_k1;
        rg_v13 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k14 = p ? tmp_k0 : tmp_k1;
        rg_v14 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k15 = p ? tmp_k0 : tmp_k1;
        rg_v15 = p ? tmp_v0 : tmp_v1;

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x1 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x1 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x1 );
        rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1 );
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1 );
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1 );
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x1 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x1 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x1 );
        if(lane_id&0x1) SWP(K, rg_k0 ,rg_k1 , int, rg_v0 ,rg_v1 );
        if(lane_id&0x1) SWP(K, rg_k2 ,rg_k3 , int, rg_v2 ,rg_v3 );
        if(lane_id&0x1) SWP(K, rg_k4 ,rg_k5 , int, rg_v4 ,rg_v5 );
        if(lane_id&0x1) SWP(K, rg_k6 ,rg_k7 , int, rg_v6 ,rg_v7 );
        if(lane_id&0x1) SWP(K, rg_k8 ,rg_k9 , int, rg_v8 ,rg_v9 );
        if(lane_id&0x1) SWP(K, rg_k10,rg_k11, int, rg_v10,rg_v11);
        if(lane_id&0x1) SWP(K, rg_k12,rg_k13, int, rg_v12,rg_v13);
        if(lane_id&0x1) SWP(K, rg_k14,rg_k15, int, rg_v14,rg_v15);
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x1 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x1 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x1 );
        rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x1 );
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x1 );
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x1 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x1 );
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x1 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x1 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x1 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x2 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x2 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x2 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x2 );
        rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2 );
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2 );
        rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2 );
        rg_v10 = __shfl_xor_sync(0xffffffff,rg_v10, 0x2 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x2 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x2 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x2 );
        if(lane_id&0x2 ) SWP(K, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(K, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(K, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(K, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        if(lane_id&0x2 ) SWP(K, rg_k8 , rg_k10, int, rg_v8 , rg_v10);
        if(lane_id&0x2 ) SWP(K, rg_k9 , rg_k11, int, rg_v9 , rg_v11);
        if(lane_id&0x2 ) SWP(K, rg_k12, rg_k14, int, rg_v12, rg_v14);
        if(lane_id&0x2 ) SWP(K, rg_k13, rg_k15, int, rg_v13, rg_v15);
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x2 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x2 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x2 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x2 );
        rg_v2  = __shfl_xor_sync(0xffffffff,rg_v2 , 0x2 );
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x2 );
        rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x2 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x2 );
        rg_v10 = __shfl_xor_sync(0xffffffff,rg_v10, 0x2 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x2 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x2 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x2 );
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x4 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x4 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x4 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x4 );
        rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4 );
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4 );
        rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4 );
        rg_v12 = __shfl_xor_sync(0xffffffff,rg_v12, 0x4 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x4 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x4 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x4 );
        if(lane_id&0x4 ) SWP(K, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(K, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(K, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(K, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        if(lane_id&0x4 ) SWP(K, rg_k8 , rg_k12, int, rg_v8 , rg_v12);
        if(lane_id&0x4 ) SWP(K, rg_k9 , rg_k13, int, rg_v9 , rg_v13);
        if(lane_id&0x4 ) SWP(K, rg_k10, rg_k14, int, rg_v10, rg_v14);
        if(lane_id&0x4 ) SWP(K, rg_k11, rg_k15, int, rg_v11, rg_v15);
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x4 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x4 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x4 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x4 );
        rg_v4  = __shfl_xor_sync(0xffffffff,rg_v4 , 0x4 );
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x4 );
        rg_v6  = __shfl_xor_sync(0xffffffff,rg_v6 , 0x4 );
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x4 );
        rg_v12 = __shfl_xor_sync(0xffffffff,rg_v12, 0x4 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x4 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x4 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x4 );
        rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x8 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x8 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x8 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x8 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x8 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x8 );
        rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8 );
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8 );
        rg_v10 = __shfl_xor_sync(0xffffffff,rg_v10, 0x8 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x8 );
        rg_v12 = __shfl_xor_sync(0xffffffff,rg_v12, 0x8 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x8 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x8 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x8 );
        if(lane_id&0x8 ) SWP(K, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
        if(lane_id&0x8 ) SWP(K, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
        if(lane_id&0x8 ) SWP(K, rg_k2 , rg_k10, int, rg_v2 , rg_v10);
        if(lane_id&0x8 ) SWP(K, rg_k3 , rg_k11, int, rg_v3 , rg_v11);
        if(lane_id&0x8 ) SWP(K, rg_k4 , rg_k12, int, rg_v4 , rg_v12);
        if(lane_id&0x8 ) SWP(K, rg_k5 , rg_k13, int, rg_v5 , rg_v13);
        if(lane_id&0x8 ) SWP(K, rg_k6 , rg_k14, int, rg_v6 , rg_v14);
        if(lane_id&0x8 ) SWP(K, rg_k7 , rg_k15, int, rg_v7 , rg_v15);
        rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x8 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x8 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x8 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x8 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x8 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x8 );
        rg_v8  = __shfl_xor_sync(0xffffffff,rg_v8 , 0x8 );
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x8 );
        rg_v10 = __shfl_xor_sync(0xffffffff,rg_v10, 0x8 );
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x8 );
        rg_v12 = __shfl_xor_sync(0xffffffff,rg_v12, 0x8 );
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x8 );
        rg_v14 = __shfl_xor_sync(0xffffffff,rg_v14, 0x8 );
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x8 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x10);
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x10);
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x10);
        rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x10);
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x10);
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x10);
        if(lane_id&0x10) SWP(K, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(K, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x10) SWP(K, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x10) SWP(K, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x10) SWP(K, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x10) SWP(K, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x10) SWP(K, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x10) SWP(K, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x10);
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x10);
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x10);
        rg_v1  = __shfl_xor_sync(0xffffffff,rg_v1 , 0x10);
        rg_v3  = __shfl_xor_sync(0xffffffff,rg_v3 , 0x10);
        rg_v5  = __shfl_xor_sync(0xffffffff,rg_v5 , 0x10);
        rg_v7  = __shfl_xor_sync(0xffffffff,rg_v7 , 0x10);
        rg_v9  = __shfl_xor_sync(0xffffffff,rg_v9 , 0x10);
        rg_v11 = __shfl_xor_sync(0xffffffff,rg_v11, 0x10);
        rg_v13 = __shfl_xor_sync(0xffffffff,rg_v13, 0x10);
        rg_v15 = __shfl_xor_sync(0xffffffff,rg_v15, 0x10);

        if((innerbid<<11)+(warp_id<<9)+0  +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+0  +lane_id] = rg_k0 ;
        if((innerbid<<11)+(warp_id<<9)+32 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+32 +lane_id] = rg_k2 ;
        if((innerbid<<11)+(warp_id<<9)+64 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+64 +lane_id] = rg_k4 ;
        if((innerbid<<11)+(warp_id<<9)+96 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+96 +lane_id] = rg_k6 ;
        if((innerbid<<11)+(warp_id<<9)+128+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+128+lane_id] = rg_k8 ;
        if((innerbid<<11)+(warp_id<<9)+160+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+160+lane_id] = rg_k10;
        if((innerbid<<11)+(warp_id<<9)+192+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+192+lane_id] = rg_k12;
        if((innerbid<<11)+(warp_id<<9)+224+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+224+lane_id] = rg_k14;
        if((innerbid<<11)+(warp_id<<9)+256+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+256+lane_id] = rg_k1 ;
        if((innerbid<<11)+(warp_id<<9)+288+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+288+lane_id] = rg_k3 ;
        if((innerbid<<11)+(warp_id<<9)+320+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+320+lane_id] = rg_k5 ;
        if((innerbid<<11)+(warp_id<<9)+352+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+352+lane_id] = rg_k7 ;
        if((innerbid<<11)+(warp_id<<9)+384+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+384+lane_id] = rg_k9 ;
        if((innerbid<<11)+(warp_id<<9)+416+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+416+lane_id] = rg_k11;
        if((innerbid<<11)+(warp_id<<9)+448+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+448+lane_id] = rg_k13;
        if((innerbid<<11)+(warp_id<<9)+480+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+480+lane_id] = rg_k15;

        if((innerbid<<11)+(warp_id<<9)+0  +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+0  +lane_id]=vals[k+rg_v0 ];
        if((innerbid<<11)+(warp_id<<9)+32 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+32 +lane_id]=vals[k+rg_v2 ];
        if((innerbid<<11)+(warp_id<<9)+64 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+64 +lane_id]=vals[k+rg_v4 ];
        if((innerbid<<11)+(warp_id<<9)+96 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+96 +lane_id]=vals[k+rg_v6 ];
        if((innerbid<<11)+(warp_id<<9)+128+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+128+lane_id]=vals[k+rg_v8 ];
        if((innerbid<<11)+(warp_id<<9)+160+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+160+lane_id]=vals[k+rg_v10];
        if((innerbid<<11)+(warp_id<<9)+192+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+192+lane_id]=vals[k+rg_v12];
        if((innerbid<<11)+(warp_id<<9)+224+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+224+lane_id]=vals[k+rg_v14];
        if((innerbid<<11)+(warp_id<<9)+256+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+256+lane_id]=vals[k+rg_v1 ];
        if((innerbid<<11)+(warp_id<<9)+288+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+288+lane_id]=vals[k+rg_v3 ];
        if((innerbid<<11)+(warp_id<<9)+320+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+320+lane_id]=vals[k+rg_v5 ];
        if((innerbid<<11)+(warp_id<<9)+352+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+352+lane_id]=vals[k+rg_v7 ];
        if((innerbid<<11)+(warp_id<<9)+384+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+384+lane_id]=vals[k+rg_v9 ];
        if((innerbid<<11)+(warp_id<<9)+416+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+416+lane_id]=vals[k+rg_v11];
        if((innerbid<<11)+(warp_id<<9)+448+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+448+lane_id]=vals[k+rg_v13];
        if((innerbid<<11)+(warp_id<<9)+480+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+480+lane_id]=vals[k+rg_v15];
    }
}

template<class K, class T>
__global__
void kern_copy(K *srck, T *srcv, K *dstk, T *dstv, int *segs, int *bin, 
        int *blk_innerid, int *blk_seg_start, int length, int n, int res)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];
    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    int stride = upper_power_of_two(seg_size);
    int steps = log2(stride/2048);

    if((steps&1))
    {
        if((innerbid<<11)+tid     <seg_size) dstk[k+(innerbid<<11)+tid     ] = srck[k+(innerbid<<11)+tid     ];
        if((innerbid<<11)+tid+128 <seg_size) dstk[k+(innerbid<<11)+tid+128 ] = srck[k+(innerbid<<11)+tid+128 ];
        if((innerbid<<11)+tid+256 <seg_size) dstk[k+(innerbid<<11)+tid+256 ] = srck[k+(innerbid<<11)+tid+256 ];
        if((innerbid<<11)+tid+384 <seg_size) dstk[k+(innerbid<<11)+tid+384 ] = srck[k+(innerbid<<11)+tid+384 ];
        if((innerbid<<11)+tid+512 <seg_size) dstk[k+(innerbid<<11)+tid+512 ] = srck[k+(innerbid<<11)+tid+512 ];
        if((innerbid<<11)+tid+640 <seg_size) dstk[k+(innerbid<<11)+tid+640 ] = srck[k+(innerbid<<11)+tid+640 ];
        if((innerbid<<11)+tid+768 <seg_size) dstk[k+(innerbid<<11)+tid+768 ] = srck[k+(innerbid<<11)+tid+768 ];
        if((innerbid<<11)+tid+896 <seg_size) dstk[k+(innerbid<<11)+tid+896 ] = srck[k+(innerbid<<11)+tid+896 ];
        if((innerbid<<11)+tid+1024<seg_size) dstk[k+(innerbid<<11)+tid+1024] = srck[k+(innerbid<<11)+tid+1024];
        if((innerbid<<11)+tid+1152<seg_size) dstk[k+(innerbid<<11)+tid+1152] = srck[k+(innerbid<<11)+tid+1152];
        if((innerbid<<11)+tid+1280<seg_size) dstk[k+(innerbid<<11)+tid+1280] = srck[k+(innerbid<<11)+tid+1280];
        if((innerbid<<11)+tid+1408<seg_size) dstk[k+(innerbid<<11)+tid+1408] = srck[k+(innerbid<<11)+tid+1408];
        if((innerbid<<11)+tid+1536<seg_size) dstk[k+(innerbid<<11)+tid+1536] = srck[k+(innerbid<<11)+tid+1536];
        if((innerbid<<11)+tid+1664<seg_size) dstk[k+(innerbid<<11)+tid+1664] = srck[k+(innerbid<<11)+tid+1664];
        if((innerbid<<11)+tid+1792<seg_size) dstk[k+(innerbid<<11)+tid+1792] = srck[k+(innerbid<<11)+tid+1792];
        if((innerbid<<11)+tid+1920<seg_size) dstk[k+(innerbid<<11)+tid+1920] = srck[k+(innerbid<<11)+tid+1920];

        if((innerbid<<11)+tid     <seg_size) dstv[k+(innerbid<<11)+tid     ] = srcv[k+(innerbid<<11)+tid     ];
        if((innerbid<<11)+tid+128 <seg_size) dstv[k+(innerbid<<11)+tid+128 ] = srcv[k+(innerbid<<11)+tid+128 ];
        if((innerbid<<11)+tid+256 <seg_size) dstv[k+(innerbid<<11)+tid+256 ] = srcv[k+(innerbid<<11)+tid+256 ];
        if((innerbid<<11)+tid+384 <seg_size) dstv[k+(innerbid<<11)+tid+384 ] = srcv[k+(innerbid<<11)+tid+384 ];
        if((innerbid<<11)+tid+512 <seg_size) dstv[k+(innerbid<<11)+tid+512 ] = srcv[k+(innerbid<<11)+tid+512 ];
        if((innerbid<<11)+tid+640 <seg_size) dstv[k+(innerbid<<11)+tid+640 ] = srcv[k+(innerbid<<11)+tid+640 ];
        if((innerbid<<11)+tid+768 <seg_size) dstv[k+(innerbid<<11)+tid+768 ] = srcv[k+(innerbid<<11)+tid+768 ];
        if((innerbid<<11)+tid+896 <seg_size) dstv[k+(innerbid<<11)+tid+896 ] = srcv[k+(innerbid<<11)+tid+896 ];
        if((innerbid<<11)+tid+1024<seg_size) dstv[k+(innerbid<<11)+tid+1024] = srcv[k+(innerbid<<11)+tid+1024];
        if((innerbid<<11)+tid+1152<seg_size) dstv[k+(innerbid<<11)+tid+1152] = srcv[k+(innerbid<<11)+tid+1152];
        if((innerbid<<11)+tid+1280<seg_size) dstv[k+(innerbid<<11)+tid+1280] = srcv[k+(innerbid<<11)+tid+1280];
        if((innerbid<<11)+tid+1408<seg_size) dstv[k+(innerbid<<11)+tid+1408] = srcv[k+(innerbid<<11)+tid+1408];
        if((innerbid<<11)+tid+1536<seg_size) dstv[k+(innerbid<<11)+tid+1536] = srcv[k+(innerbid<<11)+tid+1536];
        if((innerbid<<11)+tid+1664<seg_size) dstv[k+(innerbid<<11)+tid+1664] = srcv[k+(innerbid<<11)+tid+1664];
        if((innerbid<<11)+tid+1792<seg_size) dstv[k+(innerbid<<11)+tid+1792] = srcv[k+(innerbid<<11)+tid+1792];
        if((innerbid<<11)+tid+1920<seg_size) dstv[k+(innerbid<<11)+tid+1920] = srcv[k+(innerbid<<11)+tid+1920];
    }
}
template<class K, class T>
int gen_grid_kern_r2049(K *keys_d, T *vals_d, K *keysB_d, T *valsB_d, 
        int n, int *segs_d, int *bin_d, int bin_size, int length)
{
    cudaError_t err;
    int *blk_stat_d; // histogram of how many blocks for each seg
    err = cudaMalloc((void **)&blk_stat_d, bin_size*sizeof(int));
    ERR_INFO(err, "alloc blk_stat_d");

    int blk_num; // total number of blocks
    int *max_segsize_d;
    err = cudaMalloc((void **)&max_segsize_d, sizeof(int));
    ERR_INFO(err, "alloc max_segsize_d");

    err = cudaMemset(max_segsize_d, 0, sizeof(int));
    ERR_INFO(err, "memset max_segsize_d");

    dim3 blocks(256, 1, 1);
    dim3 grids((bin_size+blocks.x-1)/blocks.x, 1, 1);

    // this kernel gets how many blocks for each seg; get max seg length;
    // last parameter is how many pairs one block can handle
    kern_get_num_blk_init<<<grids, blocks>>>(max_segsize_d, segs_d, bin_d, blk_stat_d, 
            n, bin_size, length, 2048); // 512thread*4key /*** codegen ***/

    int max_segsize;
    err = cudaMemcpy(&max_segsize, max_segsize_d, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from max_segsize_d");
    // store the last number from blk_stat_d in case of the exclusive scan later on
    err = cudaMemcpy(&blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  

    thrust::device_ptr<int> d_arr0 = thrust::device_pointer_cast<int>(blk_stat_d);
    thrust::exclusive_scan(d_arr0, d_arr0+bin_size, d_arr0);

    int part_blk_num;
    err = cudaMemcpy(&part_blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  
    blk_num = blk_num + part_blk_num;

    int *blk_innerid; // record each block's inner id
    err = cudaMalloc((void **)&blk_innerid, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_innerid");  

    int *blk_seg_start; // record each block's segment's starting position
    err = cudaMalloc((void **)&blk_seg_start, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_seg_start");  

    grids.x = (blk_num+blocks.x-1)/blocks.x;
    kern_get_init_pos<<<grids, blocks>>>(blk_stat_d, blk_innerid, blk_seg_start, 
            blk_num, bin_size);

    /*** codegen ***/
    blocks.x = 512;
    grids.x = blk_num;
    kern_block_sort<<<grids, blocks>>>(keys_d, vals_d, keysB_d, valsB_d, segs_d, bin_d, 
            blk_innerid, blk_seg_start, length, n);

    blocks.x = 256;
    grids.x = (bin_size+blocks.x-1)/blocks.x;
    kern_get_num_blk<<<grids, blocks>>>(segs_d, bin_d, blk_stat_d, 
            n, bin_size, length, 2048); // 128t*16k /*** codegen ***/

    err = cudaMemcpy(&blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  

    thrust::device_ptr<int> d_arr1 = thrust::device_pointer_cast<int>(blk_stat_d);
    thrust::exclusive_scan(d_arr1, d_arr1+bin_size, d_arr1);

    err = cudaMemcpy(&part_blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  
    blk_num = blk_num + part_blk_num;

    err = cudaFree(blk_innerid);
    ERR_INFO(err, "free blk_innerid");  

    err = cudaMalloc((void **)&blk_innerid, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_innerid");  

    err = cudaFree(blk_seg_start);
    ERR_INFO(err, "free blk_seg_start");  

    err = cudaMalloc((void **)&blk_seg_start, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_seg_start");  

    grids.x = (blk_num+blocks.x-1)/blocks.x;
    kern_get_init_pos<<<grids, blocks>>>(blk_stat_d, blk_innerid, blk_seg_start, 
            blk_num, bin_size);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    int stride = 2048; // unit for already sorted
    int cnt = 0;
    blocks.x = 128;
    grids.x = blk_num;

    // cout << "max_segsize " << max_segsize << endl;
    while(stride < max_segsize)
    {
        kern_block_merge<<<grids, blocks>>>(keys_d, vals_d, keysB_d, valsB_d, segs_d, bin_d, 
                blk_innerid, blk_seg_start, length, n, stride);
        stride <<= 1;
        std::swap(keys_d, keysB_d);
        std::swap(vals_d, valsB_d);
        cnt++;
    }

    // cout << "cnt " << cnt << endl;

    blocks.x = 128;
    grids.x = blk_num;
    K *srck = (cnt&1)?keys_d:keysB_d;
    K *dstk = (cnt&1)?keysB_d:keys_d;
    T *srcv = (cnt&1)?vals_d:valsB_d;
    T *dstv = (cnt&1)?valsB_d:vals_d;
    kern_copy<<<grids, blocks>>>(srck, srcv, dstk, dstv, segs_d, bin_d, 
                blk_innerid, blk_seg_start, length, n, cnt);
    
    err = cudaFree(blk_stat_d);
    ERR_INFO(err, "free blk_stat_d");  
    err = cudaFree(blk_innerid);
    ERR_INFO(err, "free blk_innerid");  
    err = cudaFree(blk_seg_start);
    ERR_INFO(err, "free blk_seg_start");  
    err = cudaFree(max_segsize_d);
    ERR_INFO(err, "free max_segsize_d");  

    return cnt-1;
}
#endif
