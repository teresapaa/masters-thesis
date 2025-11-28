```markdown
# masters_thesis 

Includes all code related to my master's thesis: 
GPU Accelerated Computing and Machine Learning for Economic Modeling

Currently contains following directories:

neoclassical-cpu:
- an easy cpu implementation for the neoclassical growth model
- python plotting to verify the results

neoclassical-gpu
- three different gpu implementations for the neoclassical growth model
	- a naive cuda implementation (v1)
	- an implementation relying heavily on shared memory (not efficient) (v2)
	- an advandced implementation using parallel reduction and multiple kernels (v3)


TODO:
- add time tracking to the CPU implementation as well; also standardize the time tracking so that check ups won't affect
- speed up v3 reduction in kernel 
- add a reinforcement learning implementation for the neoclassical growth model
- clean the code of each implemetation
- consider adding profiling to cuda implementations
- consider making a cmake solution for the cuda implementations
- consider adding a shared test library

```
