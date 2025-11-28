// nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -Xcompiler "/std:c++17" GPU-solution.cu -o GPU-solution

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <chrono>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

using namespace std;


void find_crossing(vector<float> K, int n_k, thrust::host_vector<int> policy) {
    int crossing = -1;
    for (int i = 0; i < n_k; ++i) {
        double Kp = K[policy[i]];
        if (Kp > K[i]) crossing = i;
    }
    if (crossing >= 0) {
        cout << "Numerical steady-state approx at K ~ " << K[crossing]
            << ", K' at that state = " << K[policy[crossing]] << ", index = " << crossing << endl;
    }
    else {
        cout << "No crossing found (policy never suggests K' > K)." << endl;
    }
}


__global__ void value_function_iteration_kernel(
    const float* __restrict__ K,
    const float* __restrict__ K_pow,
    const float* __restrict__ V_old,
    float* __restrict__ V_new,
    int* __restrict__ policy,
    int n_k,
    float alpha,
    float z,
    float beta,
    float delta,
    float NEG_INF
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_k) return;

    float max_value = NEG_INF;
    int best_j = 0;

	float base = z * K_pow[i] + (1 - delta) * K[i];
    for (int j = 0; j < n_k; j++) {
        float consumption = base - K[j];
        float value = (consumption <= 0.0) ? NEG_INF : logf(consumption) + beta * V_old[j];
        if (value > max_value) {
            max_value = value;
            best_j = j;
        }
    }
    V_new[i] = max_value;
    policy[i] = best_j;


}


      


void run_compute() {
    cout << "Neoclassical Growth model [GPU]" << endl;

    //Setting up variables
    int n_k = 1000; // number of grid points
    float Kmin = 0.5f; // lower bound of the state space
    float Kmax = 100.0f; // upper bound of the state space
    float epsilon = 0.001f; //tolerance of error
    float alpha = 0.5f; //capital share
    float z = 1.0f; //productivity
    float beta = 0.96f; //annual discounting
    float delta = 0.025f; //annual depreciation


    //Set the grid points and calculate K^alpha for each grid point
    vector<float> K(n_k);
    float step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * step;

    thrust::device_vector<float> d_K = K;
    thrust::device_vector<float> d_K_pow(K.size());
    thrust::transform(d_K.begin(), d_K.end(), d_K_pow.begin(),
        [=] __device__(float k) { return powf(k, alpha); });

    //Initialize the value function and policy arrays
    thrust::device_vector<float> d_V_new(n_k, 0.0f);
    thrust::device_vector<float> d_V_old(n_k, 0.0f);
    thrust::device_vector<int> d_policy(n_k, 0);


    //Initialize values for the VF iteration loop

    float diff = std::numeric_limits<float>::max();
    int iteration = 0;
    const int max_iter = 20000;
    const float NEG_INF = -std::numeric_limits<float>::max();


    while (diff > epsilon && iteration < max_iter && ++iteration) {
        //Launch the kernel
        int threadsPerBlock = 256;
        int numBlocks = (n_k + threadsPerBlock -1) / threadsPerBlock;

        value_function_iteration_kernel << <numBlocks, threadsPerBlock >> > (
            thrust::raw_pointer_cast(d_K.data()),
            thrust::raw_pointer_cast(d_K_pow.data()),
            thrust::raw_pointer_cast(d_V_old.data()),
            thrust::raw_pointer_cast(d_V_new.data()),
            thrust::raw_pointer_cast(d_policy.data()),
            n_k,
            alpha,
            z,
            beta,
            delta,
            NEG_INF
            );


        diff = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(d_V_new.begin(), d_V_old.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_V_new.end(), d_V_old.end())),
            [] __host__ __device__(thrust::tuple<float, float> V) {
            float x = thrust::get<0>(V);
            float y = thrust::get<1>(V);
            return x > y ? x - y : y - x;
            },
            0.0,
            thrust::maximum<float>()
        );

        d_V_old.swap(d_V_new);

    }

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    thrust::host_vector<int> policy(d_policy);
    find_crossing(K, n_k, policy);

    /*
    for (int idx : policy) {
        std::cout << idx << " ";
    }
    */

}





int main()
{
    std::cout << "masters_thesis: starting compute\n";
    auto host_start = std::chrono::steady_clock::now();
    run_compute();
    auto host_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> host_ms = host_end - host_start;
    std::cout << "End-to-end host wall-clock time: " << host_ms.count() << " ms\n";
    std::cout << "masters_thesis: finished\n";
    return 0;
}
