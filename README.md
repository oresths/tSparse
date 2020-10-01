# Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores
In this repository we provide the source code of our accelerated Sparse Matrix-Matrix multiplication (SpGEMM) implementation, which we desrcribe in our paper "Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores" (see references).

## How to build
Cmake, CUDA and CUSP are required. Modification to the CMakeLists.txt files may be necessary, e.g. to change GPU architecture.

Instructions:
1. Download the source code into a folder e.g. tSparse-src.
2. Create a folder for building the executable in the same directory as tSparse-src, e.g tSparse-release.
3. Inside tSparse-release and call cmake:

> cmake ../tSparse-src

4. Finally, call make:

> make -j{number of CPU cores}

## How to run
In order to test SPGEMM run "spmm" executable. spmm accepts 1 or 2 arguments. In case of 1 argument it performs matrix squaring (A\*A). In case of 2 arguments it performs the matrix multiplication (A\*B).

### example
> ./spmm A.mtx
or
> ./spmm A.mtx B.mtx

## Troubleshooting
- The first compilation after running cmake may give an error similar to : "error: undefined reference to '__cudaRegisterLinkedBinary_12_spmm_cpp1_ii_handle'". This error is related to the CUDA library that is used by CUDA dynamic parallelism.
Running make a second time solves this issue.
- The latest multiplication and counting kernels do not support Volta. The reason is that we use direct access to fragments (instead of through shared memory) for performance reasons.

## Contact data
Orestis Zachariadis (orestis.zachariadis@uco.es)

## References
O. Zachariadis, N. Satpute, J. Gómez-Luna, and J. Olivares, “Accelerating sparse matrix–matrix multiplication with GPU Tensor Cores,” Computers & Electrical Engineering, vol. 88, p. 106848, Dec. 2020, doi: 10.1016/j.compeleceng.2020.106848.
[arXiv:](http://arxiv.org/abs/2009.14600)


