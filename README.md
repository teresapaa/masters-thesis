```markdown
# masters_thesis 


Key idea
- Default build is CPU-only (no CUDA required).
- Enable CUDA explicitly using -DENABLE_CUDA=ON or set the CMake GUI/preset in Visual Studio.

Build & run (CLI)

CPU-only laptop:
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


```
