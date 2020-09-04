project (SPMM)

cuda_add_executable(spmm spmm.cu mm.cu)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(spmm PRIVATE -march=native -fopenmp)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

target_link_libraries(spmm ${TARGET_LINK_LIBS} ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} ${CUPTI_LIB} gomp)
