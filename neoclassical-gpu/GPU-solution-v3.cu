// nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -Xcompiler "/std:c++17" GPU-solution-v3.cu -o GPU-solution-v3

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

//working on a test file:
//extern void find_crossing(vector<float> K, int n_k, thrust::host_vector<int> policy);

//check the crossing point where K' > K to ensure everything is working
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


//one block calculates one state i
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

	//find the current state
    int i = blockIdx.x;
    if (i >= n_k) return;

	//intitialize variables
	int thread = threadIdx.x;  //startingpoint for the iteration over j
	int bDim = blockDim.x; //256 with the curent setup
    float local_max_value = NEG_INF;
    int local_best_j = 0;
	float base = z * K_pow[i] + (1.0f - delta) * K[i]; //pre-compute the base part
    
    for (int j = thread ; j < n_k; j+= bDim) {
        float consumption = base - K[j];
        float value = (consumption <= 0.0) ? NEG_INF : logf(consumption) + beta * V_old[j];
        if (value > local_max_value) {
            local_max_value = value;
            local_best_j = j;
        }
    }


	//do the first round of reduction with warp shuffles 
	const unsigned FULL = 0xffffffffu; //mask for active threads (all in this case)
	const int W = 32; //warp size
	int numWarps = (bDim + W - 1) / W;

	//allocate memory for warp results
    extern __shared__ unsigned char s_mem_raw[];
    float* w_max_values = reinterpret_cast<float*>(s_mem_raw);
    int* w_best_js = reinterpret_cast<int*>(w_max_values + numWarps);

    //warp values
	int lane = thread & (W - 1); //thread index within the warp, faster way for t % W
	int warpId = thread >> 5; //warp index within the block, faster way for t / W

	//initialize the registers to suffle within the warp
	float v = local_max_value;
	int j = local_best_j;

	for (int offset = 16; offset > 0; offset >>= 1) { //divides offset by 2 each iteration
        float v_other = __shfl_down_sync(FULL, v, offset); //reads the value from the register offset lanes above
		float j_other = __shfl_down_sync(FULL, j, offset); //reads the value from the register offset lanes above
        if (v_other > v) {
            v = v_other;
            j = j_other;
		}
    }

	//store the warp results to shared memory
    if (lane == 0) {
        w_max_values[warpId] = v;
		w_best_js[warpId] = j;
    }

    /*
    //recuction using the shared memory 
    __syncthreads();
    if (thread == 0) {
        float max_value = NEG_INF;
        int best_j = 0;
        for (int w = 0; w < numWarps; ++w) {
            if (w_max_values[w] > max_value) {
                max_value = w_max_values[w];
                best_j = w_best_js[w];
            }
        }
        V_new[i] = max_value;
        policy[i] = best_j;
    }
    

    */

    //move onto doing the reduction across warps with shuffles    
	__syncthreads();

    if (warpId == 0) { //use only the warp 0 for this
        float vv = (thread < numWarps) ? w_max_values[thread] : NEG_INF; //load the warp results into registers
        float jj = (thread < numWarps) ? w_best_js[thread] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) { //divides offset by 2 each iteration
            float vv_other = __shfl_down_sync(FULL, vv, offset); //reads the value from the register offset lanes above
            float jj_other = __shfl_down_sync(FULL, jj, offset); //reads the value from the register offset lanes above

            if (vv_other > vv) {
                vv = vv_other;
                jj = jj_other;
            }
        }

        if (thread == 0) { //only one thread writes into global memory
            V_new[i] = vv;
            policy[i] = jj;
        }
    }

    

      

}

      


void run_compute() {

    cout << "Neoclassical Growth model [GPU -v3]" << endl;

    auto host_start = std::chrono::steady_clock::now();

    //Setting up variables
    int n_k = 1000; // number of grid points
    float Kmin = 0.5f; // lower bound of the state space
    float Kmax = 100.0f; // upper bound of the state space
    float epsilon = 0.001f; //tolerance of error
    float alpha = 0.5f; //capital share
    float z = 1.0f; //productivity
    float beta = 0.96f; //annual discounting
    float delta = 0.025f; //annual depreciation


    //Set the grid points 
    vector<float> K(n_k);
    float step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * step;

    //Calculate K^alpha for each grid point
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

		//Parameters for kernel launch
        int threadsPerBlock = 256;
        int numBlocks = n_k;
        int numWarps = (threadsPerBlock + 31) / 32;
		int sharedMemBytes = numWarps * (sizeof(float) + sizeof(int));

        //Launch the kernel
        value_function_iteration_kernel << <numBlocks, threadsPerBlock, sharedMemBytes >> > (
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

		//Calculate the maximum difference for convergence check on the device
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

	//cheks to ensure everything is working
    auto host_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> host_ms = host_end - host_start;
    std::cout << "End-to-end host wall-clock time: " << host_ms.count() << " ms\n";

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    thrust::host_vector<int> policy(d_policy);
    find_crossing(K, n_k, policy);

}




int main()
{
    std::cout << "masters_thesis: starting compute\n";
    run_compute();
    std::cout << "masters_thesis: finished\n";
    return 0;
}
