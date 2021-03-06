/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// reserve 0 for undefined
#define CUSP_DEVICE_BLAS_CUBLAS    1
#define CUSP_DEVICE_BLAS_CBLAS     2
#define CUSP_DEVICE_BLAS_GENERIC   3

#ifndef CUSP_DEVICE_BLAS_SYSTEM
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA_
  #define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_CUBLAS
#else
  #define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_GENERIC
#endif
#endif // CUSP_DEVICE_BLAS_SYSTEM

#define CUSP_DEVICE_BACKEND_CUBLAS  CUSP_DEVICE_BLAS_CUBLAS
#define CUSP_DEVICE_BACKEND_CBLAS   CUSP_DEVICE_BLAS_CBLAS
#define CUSP_DEVICE_BACKEND_GENERIC CUSP_DEVICE_BLAS_GENERIC

#if CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_CUBLAS
#define __CUSP_DEVICE_BLAS_NAMESPACE cublas
#elif CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_CBLAS
#define __CUSP_DEVICE_BLAS_NAMESPACE cblas
#elif CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_GENERIC
#define __CUSP_DEVICE_BLAS_NAMESPACE
#endif

#define __CUSP_DEVICE_BLAS_ROOT __CUSP_DEVICE_SYSTEM_ROOT/detail/__CUSP_DEVICE_BLAS_NAMESPACE

