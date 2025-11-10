```markdown
# masters_thesis (CUDA-capable)

This repository contains a minimal layout to develop on both a laptop without a GPU and one with a GPU. The build uses CMake and CUDA is optional.

Key idea
- Default build is CPU-only (no CUDA required).
- Enable CUDA explicitly on your GPU laptop using -DENABLE_CUDA=ON or set the CMake GUI/preset in Visual Studio.

Build & run (CLI)

CPU-only laptop (no CUDA toolchain required):
1. mkdir build && cd build
2. cmake -DENABLE_CUDA=OFF ..
3. cmake --build . --config Release
4. ./masters_thesis (or Release\\masters_thesis.exe on Windows)

GPU laptop (CUDA required):
1. mkdir build && cd build
2. cmake -DENABLE_CUDA=ON ..
3. cmake --build . --config Release
4. ./masters_thesis

Visual Studio (full IDE, CMake workflow)
- Open the repository folder or the top-level CMakeLists.txt in Visual Studio.
- In CMake -> Cache or CMake Settings, set ENABLE_CUDA = ON on the GPU laptop; leave it OFF on the CPU laptop.
- Configure and press Build / Debug like a normal VS project.

Notes & strategies
- Compile-time selection (this repo): we pick CPU vs GPU implementation during the CMake configure step. This is simple and avoids requiring the CUDA toolkit on the CPU-only laptop.
- Runtime selection (advanced): you could compile both implementations and choose at runtime, or build a GPU plugin (.dll/.so) on the GPU laptop and dynamically load it. These approaches require the CUDA toolchain be present for at least the GPU build and complicate development/artifacts.
- CI: run CPU builds on GitHub Actions (ubuntu-latest / windows-latest). For GPU tests, use a self-hosted runner with an NVIDIA GPU or a cloud CI service that provides GPUs. Document this in README.
- Keep documentation of required CUDA version and GPU driver in README.
- Consider using Docker + nvidia-container for reproducible GPU builds (useful for CI/self-hosted runners).

License, data, and large files
- Do not commit large datasets or GPU binaries. Use Git LFS for large files, or store datasets outside the repo and document download steps.
```