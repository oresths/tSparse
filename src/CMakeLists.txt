project (SPMM)

cuda_add_executable(spmm spmm.cu mm.cu)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(spmm PRIVATE -march=native -fopenmp)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

target_include_directories(spmm
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        /usr/local/cuda-10.1/NsightSystems-2019.3/Target-x86_64/nvtx/include/
)

set(PARALLEL_STUDIO_INSTALL_DIR /opt/intel/parallel_studio_xe_2018/vtune_amplifier_2018)
target_include_directories(spmm PRIVATE ${PARALLEL_STUDIO_INSTALL_DIR}/include)
link_directories(${PARALLEL_STUDIO_INSTALL_DIR}/lib64)

find_library(ITTNOTIFY_LIBRARY ittnotify HINTS ${PARALLEL_STUDIO_INSTALL_DIR}/lib64)
if(ITTNOTIFY_LIBRARY)
    list(APPEND TARGET_LINK_LIBS ${ITTNOTIFY_LIBRARY})
else(ITTNOTIFY_LIBRARY)
    message(STATUS "Intel ITT profiling library not found")
endif(ITTNOTIFY_LIBRARY)

target_link_libraries(spmm ${TARGET_LINK_LIBS} ${CUDA_cusparse_LIBRARY} ${CUDA_cudadevrt_LIBRARY} gomp)

#target_link_libraries(spmm /home/ore/git/gunrock/gunrock-debug/lib/libgunrock.so)
