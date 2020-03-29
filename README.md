# Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores
In this repository we provide the source code of our accelerated Sparse Matrix-Matrix multiplication (SpGEMM) implementations, which we desrcribe in our paper "Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores" (under revision).

## How to build
Cmake, CUDA and CUSP are required. Modification to the CMakeLists.txt files may be necessary, e.g. to change GPU architecture. Call cmake like this:

"cmake ../spmm-src"

Then call make:

"make -j{number of CPU cores}

## How to run
In order to test SPGEMM run "spmm" executable. spmm accepts 1 or 2 arguments. In case of 1 argument it performs matrix squaring (A*A). In case of 2 arguments it performs the matrix multiplicaiton (A*B).

### example
./spmm A.mtx
or
./spmm A.mtx B.mtx

## Troubleshooting
The first compilation after running cmake may give an error similar to : "error: undefined reference to '__cudaRegisterLinkedBinary_12_spmm_cpp1_ii_handle'". This error is related to the CUDA library that is used by CUDA dynamic parallelism.
Running make a second time solves this issue.

## Contact data
Orestis Zachariadis (orestis.zachariadis@uco.es)

## References
<Placeholder for citing the paper and the research data>

