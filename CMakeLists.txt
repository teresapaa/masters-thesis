cmake_minimum_required(VERSION 3.18)
project(masters_thesis LANGUAGES CXX)

# Option to enable CUDA; default OFF so CPU laptop can build without CUDA installed.
option(ENABLE_CUDA "Build with CUDA support" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If user asked for CUDA, try to find it; also allow automatic enable if found and not explicitly disabled.
if(ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Building with CUDA enabled")
else()
  # Auto-detect if CUDA exists; do NOT force enable if user explicitly set -DENABLE_CUDA=OFF.
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit detected. You can enable it with -DENABLE_CUDA=ON")
  else()
    message(STATUS "CUDA toolkit not found or disabled; building CPU-only")
  endif()
endif()

# Sources
set(SRC_MAIN src/main.cpp)
set(SRC_CPU src/cpu_impl.cpp)
set(SRC_GPU src/gpu_impl.cu)

add_executable(masters_thesis ${SRC_MAIN})

# Add CPU implementation always (safe on both machines)
add_library(impl_cpu STATIC ${SRC_CPU})
target_include_directories(impl_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add GPU implementation only if CUDA was requested
if(ENABLE_CUDA)
  enable_language(CUDA)
  add_library(impl_gpu STATIC ${SRC_GPU})
  target_include_directories(impl_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(impl_gpu PUBLIC CUDA::cudart)
  # Example: compile flags for CUDA device debugging/profiling
  target_compile_options(impl_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# Choose which implementation to link at configure time
if(ENABLE_CUDA)
  target_link_libraries(masters_thesis PRIVATE impl_gpu)
else()
  target_link_libraries(masters_thesis PRIVATE impl_cpu)
endif()

# Install / packaging targets optional
install(TARGETS masters_thesis
  RUNTIME DESTINATION bin)