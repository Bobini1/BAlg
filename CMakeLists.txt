cmake_minimum_required(VERSION 3.21)
project(BAlg CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CXX_STANDARD 20)

include_directories(NdArray)

add_executable(BAlg
        NdArray/Algorithms.cu
        NdArray/Algorithms.cuh
        NdArray/kernel.cu
        NdArray/NdArray.cuh NdArray/NdArray.cu NdArray/Shape.h NdArray/Shape.cpp)

set_target_properties(BAlg PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")