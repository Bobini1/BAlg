cmake_minimum_required(VERSION 3.21)
project(BAlg CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CXX_STANDARD 20)

include_directories(NdArray)

add_subdirectory(Boost_tests)

add_executable(BAlg
        NdArray/Algorithms/Algorithms.h
        NdArray/kernel.cu
        NdArray/NdArrayVariadic.h NdArray/NdArrayVariadic.cpp NdArray/Algorithms/Reduce.cuh NdArray/Algorithms/Reduce.cu NdArray/Algorithms/Map.cuh NdArray/Algorithms/commonFunctionality.cuh NdArray/Algorithms/commonFunctionality.cu NdArray/Algorithms/Map.cu NdArray/Algorithms/Combine.cuh NdArray/Algorithms/Combine.cu)

set_target_properties(BAlg PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(Doxygen)

if(DOXYGEN_FOUND)

    set(DOXYGEN_INPUT ${SOURCE_FILES})
    set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    add_custom_command(
            OUTPUT ${DOXYGEN_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${DOXYGEN_INPUT}
    )

    add_custom_target(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})

    add_custom_target(apidoc_forced
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

endif(DOXYGEN_FOUND)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")