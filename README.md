```markdown
# masters_thesis 

Includes all code related to my master's thesis: 
GPU Accelerated Computing and Machine Learning for Economic Modeling

Currently contains following directories:

neoclassical-cpu:
- an easy cpu implementation for the neoclassical growth model
- files:
	- src/cpu_impl.cpp: the actual solution, no I/O operations for benchmarking
	- src/cpu-plots.cpp: the solution including writing the results for plotting
	- scripts/plot_vfi.py: python plotting to verify the results

neoclassical-gpu
- three different gpu implementations for the neoclassical growth model
	- GPU-v1: a naive cuda implementation distributing each state i for one thread to process
	- GPU-v2: an advandced implementation using parallel reduction and multiple kernels, distributing each state i to a block of threads to process

neoclassical-rl
- rl-solution: solving the problem using reinforcement learning

Compiling and running from x64 Native Tools Command Prompt for VS 2022:

CPU (cpu_impl.cpp, for benchmarking):
cmake -S . -B out/build && cmake --build out/build
out\build\Debug\neoclassical-cpu.exe

GPU-v1:
nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -Xcompiler "/std:c++17" GPU-v1.cu -o GPU-v1
GPU-v1.exe

GPU-v2:
nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -Xcompiler "/std:c++17" GPU-v2.cu -o GPU-v2
GPU-v2.exe

RL (could be compiled with something else as well):
nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -Xcompiler "/std:c++17" rl-solution.cpp -o rl-solution
rl-solution.exe

```
