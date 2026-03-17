cmake_minimum_required(VERSION 3.18)
project(pension_model LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

add_executable(model
    model_gpu.cpp
    cuda_bellman.cu
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(model PRIVATE CUDA::cudart)