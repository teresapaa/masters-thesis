// nvcc -G hello.cu -o hello

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

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

using namespace std;




__global__ void value_function_iteration_kernel(
    const double* K,
    const double* K_pow,
    const double* V_old,
    double* V_new,
    int* policy,
    int n_k,
    double alpha,
    double z,
    double beta,
    double delta,
    double NEG_INF
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_k) return;

    double max_value = NEG_INF;
    int best_j = 0;

    for (int j = 0; j < n_k; j++) {
        double consumption = z * K_pow[i] + (1 - delta) * K[i] - K[j];
        double value = (consumption <= 0.0) ? NEG_INF : log(consumption) + beta * V_old[j];
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
    double Kmin = 0.5; // lower bound of the state space
    double Kmax = 100.0; // upper bound of the state space
    double epsilon = 0.001; //tolerance of error
    double alpha = 0.5; //capital share
    double z = 1.0; //productivity
    double beta = 0.96; //annual discounting
    double delta = 0.025; //annual depreciation


    //Set the grid points and calculate K^alpha for each grid point
    vector<double> K(n_k);
    double step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * step;

    thrust::device_vector<double> d_K = K;
    thrust::device_vector<double> d_K_pow(K.size());
    thrust::transform(d_K.begin(), d_K.end(), d_K_pow.begin(),
        [=] __device__(double k) { return pow(k, alpha); });

    //Initialize the value function and policy arrays
    thrust::device_vector<double> d_V_new(n_k, 0.0);
    thrust::device_vector<double> d_V_old(n_k, 0.0);
    thrust::device_vector<int> d_policy(n_k, 0);

    vector<double> V_new(n_k, 0.0);
    vector<int> policy(n_k, 0);
    vector<double> V_old(n_k, 0.0);

    //Initialize values for the VF iteration loop

    double diff = DBL_MAX;
    int iteration = 0;
    const int max_iter = 20000;
    const double NEG_INF = -1.0e30;

    while (diff > epsilon && iteration < max_iter && ++iteration) {
        //Launch the kernel
        int threadsPerBlock = 256;
        int numBlocks = (n_k + threadsPerBlock) / threadsPerBlock;

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

        cudaDeviceSynchronize();
        //Copy V_new back to host
        thrust::copy(d_V_new.begin(), d_V_new.end(), V_new.begin());

        diff = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(d_V_new.begin(), d_V_old.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_V_new.end(), d_V_old.end())),
            [] __host__ __device__(thrust::tuple<double, double> V) {
            double x = thrust::get<0>(V);
            double y = thrust::get<1>(V);
            return x > y ? x - y : y - x;
            },
            0.0,
            thrust::maximum<double>()
        );

        V_old = V_new;
        thrust::copy(V_old.begin(), V_old.end(), d_V_old.begin());
        ++iteration;

        cudaDeviceSynchronize();

    }

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;
    //cout << "Alt diff: " << difference << endl;
}





int main()
{
    std::cout << "masters_thesis: starting compute\n";
    run_compute();
    std::cout << "masters_thesis: finished\n";
    return 0;
}
