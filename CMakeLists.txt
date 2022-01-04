cmake_minimum_required(VERSION 3.21)
project(BAlg CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CXX_STANDARD 20)

include_directories(NdArray)

add_subdirectory(Boost_tests)

add_executable(BAlg
        NdArray/Algorithms/Algorithms.h
        NdArray/kernel.cu
        NdArray/NdArrayFlex.h NdArray/NdArrayFlex.cpp NdArray/Shape.h NdArray/Shape.cpp NdArray/NdArrayVariadic.h NdArray/NdArrayVariadic.cpp NdArray/Algorithms/Algorithms.cuh NdArray/Algorithms/Algorithms.cu)

set_target_properties(BAlg PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")