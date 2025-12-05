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
- add a reinforcement learning implementation for the neoclassical growth model
- clean the code of each implemetation
- create simple cmake/cmakes and a shared file containing common functions
	- switch cpu implemetation doubles -> floats or make both implementations
- add value function plotting to the gpu implementations as well

CONSIDERATIONS FOR FURTHER WORK:
- consider adding profiling to cuda implementations
- consider making a cmake solution for the cuda implementations
- consider adding a shared file for testing the correctness of each implementation
- making CPU mroe comparable to GPU implementations in terms of performance measurement

```
