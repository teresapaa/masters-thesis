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


using namespace std;

/*
* Helper function to find out if the boundaries of the state space are reached; a.k.a. if Kmax is too small
*/

void check_if_binding(const vector<int>& policy, int n_k) {    
    bool isBinding = false;
    for ( int idx : policy) {
        if (idx == 0 || idx == n_k - 1) {
            isBinding = true;
            break;
        }
    }
    if (isBinding) {
    cout << "State space is binding" << endl;
    }
    else {
        cout << "State space is not binding :) " << endl;
    }
}

/*
* Helper function to calculate the maximum pointwise distance of two vectors
*/

//This can be given to GPU
double max_abs_difference(const std::vector<double>& V0, const std::vector<double>& V1) {
    double d = 0.0;
    for (size_t i = 0; i < V0.size(); ++i) {
        double diff = std::abs(V0[i] - V1[i]);
        if (diff > d) d = diff;
    }
    return d;
}

/*
* A helper function to find index where K' - K changes sign: last i with K'[i] > K[i]
*/
void find_crossing(vector<double> K, int n_k, vector<int> policy) {
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





/*
* The main function calculating the Neoclassical Growth Model
*/
extern "C" void run_compute() {
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
    string output_dir = R"(C:\Users\Administrator\source\repos\masters-thesis)";
    
	//Set the grid points and calculate K^alpha for each grid point
    vector<double> K(n_k);
    double step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * step;

    thrust::device_vector<double> d_K = K;
    thrust::device_vector<double> d_K_pow(K.size());
    thrust::transform(d_K.begin(), d_K.end(), d_K_pow.begin(), 
        [=] __device__ (double k) { return pow(k, alpha); });

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
    const double NEG_INF = -1.0e300;

    while (diff > epsilon && iteration < max_iter && ++iteration) {
        //Launch the kernel
        int threadsPerBlock = 256;
		int numBlocks = (n_k + threadsPerBlock) / threadsPerBlock;

        value_function_iteration_kernel<<<numBlocks, threadsPerBlock>>>(
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

        diff = max_abs_difference(V_old, V_new);
        V_old = V_new;
        thrust::copy(V_old.begin(), V_old.end(), d_V_old.begin());
        ++iteration;

	} 



    //Checks to make sure iteration works as desired:

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    //Are the bounds of the state space binding?
    check_if_binding(policy, n_k);

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
    find_crossing(K, n_k, policy);

    /* Start without I/O
   
    //Print out the optimal policy:
    std::cout << "Optimal policy g:" << std::endl;
    for (int x : policy) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

       
    cout << "Corresponding K' values for each K:" << endl;

    for (int x : policy) {
        std::cout << K[x] << " ";
    }
    std::cout << std::endl;

    cout << "value function" << endl;

    for (int x : policy) {
        std::cout << K[x] << " ";
    }
    std::cout << std::endl;
    */
    
}


