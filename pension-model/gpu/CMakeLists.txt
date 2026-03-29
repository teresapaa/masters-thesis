cmake_minimum_required(VERSION 3.18)
project(pension_model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

option(USE_FLOAT "Use float instead of double" OFF)

add_executable(model
    model_gpu.cpp
    cuda_bellman.cu
)

if(USE_FLOAT)
    target_compile_definitions(model PRIVATE USE_REAL)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(model PRIVATE CUDA::cudart)
target_compile_options(model PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G -g>
)