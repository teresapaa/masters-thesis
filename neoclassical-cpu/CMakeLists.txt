cmake_minimum_required(VERSION 3.18)
project(neoclassical-cpu LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources
set(SRC_MAIN src/main.cpp)
set(SRC_CPU src/cpu_impl.cpp)

add_executable(neoclassical-cpu ${SRC_MAIN})

# Add CPU implementation
add_library(impl_cpu STATIC ${SRC_CPU})
target_include_directories(impl_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link CPU implementation
target_link_libraries(neoclassical-cpu PRIVATE impl_cpu)

# GCC <9 may need to link stdc++fs for std::filesystem
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(neoclassical-cpu PRIVATE stdc++fs)
  endif()
endif()

# Install target
install(TARGETS neoclassical-cpu
  RUNTIME DESTINATION bin)
