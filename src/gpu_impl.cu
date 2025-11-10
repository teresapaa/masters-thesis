#include <iostream>
#include <cuda_runtime.h>

__global__ void add_one(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] += 1.0f;
}

extern "C" void run_compute() {
    std::cout << "[GPU] running simple compute (CUDA)\n";
    const int N = 8;
    float host[N];
    for (int i = 0; i < N; ++i) host[i] = i;

    float* device = nullptr;
    if (cudaMalloc(&device, N * sizeof(float)) != cudaSuccess) {
        std::cerr << "[GPU] cudaMalloc failed\n";
        return;
    }
    cudaMemcpy(device, host, N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 8, grid = (N + block - 1) / block;
    add_one<<<grid, block>>>(device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host, device, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);

    std::cout << "[GPU] result: ";
    for (int i = 0; i < N; ++i) std::cout << host[i] << " ";
    std::cout << "\n";
}