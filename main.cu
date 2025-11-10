#include <iostream>
#include <cuda_runtime.h>

__global__ void add_one(float* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] += 1.0f;
}

int main() {
    const int N = 16;
    float host[N];
    for (int i = 0; i < N; ++i) host[i] = float(i);

    float* device = nullptr;
    cudaMalloc(&device, N * sizeof(float));
    cudaMemcpy(device, host, N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 8, grid = (N + block - 1) / block;
    add_one<<<grid, block>>>(device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host, device, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device);

    std::cout << "Result: ";
    for (int i = 0; i < N; ++i) std::cout << host[i] << " ";
    std::cout << std::endl;
    return 0;
}