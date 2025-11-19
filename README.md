```markdown
# masters_thesis 


Key idea
- Default build is CPU-only (no CUDA required).
- Enable CUDA explicitly modifying CMakePresets.json

Then

cmake -S . -B out/build -G "Ninja" -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build out/build --config Debug

.\build\masters_thesis.exe  

```
