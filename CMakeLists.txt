cmake_minimum_required (VERSION 3.4)
project (tSparse)

set(tSparse_VERSION_MAJOR 0)
set(tSparse_VERSION_MINOR 4)

set (CMAKE_CXX_STANDARD 11)

# ccache, cmake version > 3.4
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_CXX_COMPILER_LAUNCHER "ccache") # cmake version > 3.4
    set(CMAKE_C_COMPILER_LAUNCHER "ccache") # cmake version > 3.4
else(CCACHE_FOUND)
    message(STATUS "ccache not found")
endif(CCACHE_FOUND)

# Faster linker (ccache doesn't support linking caching)
if (UNIX AND NOT APPLE)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE ld_version)
  if ("${ld_version}" MATCHES "GNU gold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
  endif()
endif()

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-D_USE_CUDA)
    message(STATUS "Found CUDA - the GPU code will be compiled.")
else(CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please turn the USE_CUDA flag off.")
endif(CUDA_FOUND)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11") # set (CMAKE_CXX_STANDARD 11) does not create flags to be propagated with CUDA_PROPAGATE_HOST_FLAGS
set (CUDA_PROPAGATE_HOST_FLAGS ON)

list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -G -g -O0 --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 --expt-relaxed-constexpr)
# -res-usage -maxrregcount 128
# -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -lineinfo -keep -src-in-ptx --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 --expt-relaxed-constexpr)

include_directories( ${CMAKE_SOURCE_DIR}/external )

add_subdirectory(src)

