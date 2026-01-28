// nvcc --extended-lambda -G -arch=sm_86 -std=c++17 -DUSE_DOUBLE=1 -Xcompiler "/std:c++17" gpu.cu -o gpu-double-g
// nvcc --extended-lambda -G -arch=sm_86 -std=c++17  -Xcompiler "/std:c++17" gpu.cu -o gpu-float-g
// G pois, jos haluaa ajaa releasena, -DUSE_DOUBLE=1 pois jos haluu ajaa floatina
// parametrit komentoriviltä: n_k rounds warmups

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
#include <cuda_runtime.h>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

using namespace std;

//Determine whether to compile as doubles or floats
#if defined(USE_DOUBLE)
using Real = double;
constexpr const char* REAL_NAME = "double";
#else
using Real = float;
constexpr const char* REAL_NAME = "float";
#endif

/*
* Helper function to calculate medians of the running times
*/
template<typename T>
Real median_of_vector(const std::vector<T>& input) {
    if (input.empty()) throw std::domain_error("median of empty vector");
    std::vector<T> v = input;            // make a copy if we must preserve original
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (n % 2 == 1) {
        return static_cast<Real>(v[mid]);
    }
    else {
        // v[mid] is the upper middle. Find the maximum in the lower partition [0, mid).
        T upper = v[mid];
        T lower = *std::max_element(v.begin(), v.begin() + mid);
        return (static_cast<Real>(lower) + static_cast<Real>(upper)) * 0.5;
    }
}

/*
* A helper function to find index where K' - K changes sign: last i with K'[i] > K[i]
*/
std::tuple<int, int> find_crossing(vector<Real> K, int n_k, thrust::host_vector<int> policy) {
    int crossing_min = -1;
    int crossing_max = -1;
    for (int i = 0; i < n_k; ++i) {
        Real Kp = K[policy[i]];
        if (Kp > K[i]) crossing_min = i;
        if (Kp >= K[i]) crossing_max = i;
    }
    if (crossing_min < 0) {
        cout << "No crossing found (policy never suggests K' > K)." << endl;
    }

    return { crossing_min, crossing_max };
}


//one block calculates one state i
__global__ void value_function_iteration_kernel(
    const Real* __restrict__ K,
    const Real* __restrict__ K_pow,
    const Real* __restrict__ V_old,
    Real* __restrict__ V_new,
    int* __restrict__ policy,
    int n_k,
    Real alpha,
    Real z,
    Real beta,
    Real delta,
    Real NEG_INF
) {

	//find the current state
    int i = blockIdx.x;
    if (i >= n_k) return;

	//intitialize variables
	int thread = threadIdx.x;  //startingpoint for the iteration over j
	int bDim = blockDim.x; //256 with the curent setup
    Real local_max_value = NEG_INF;
    int local_best_j = 0;
	Real base = z * K_pow[i] + (1.0f - delta) * K[i]; //pre-compute the base part
    
    for (int j = thread ; j < n_k; j+= bDim) {
        Real consumption = base - K[j];
        Real value = (consumption <= 0.0) ? NEG_INF : logf(consumption) + beta * V_old[j];
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
    Real* w_max_values = reinterpret_cast<Real*>(s_mem_raw);
    int* w_best_js = reinterpret_cast<int*>(w_max_values + numWarps);

    //warp values
	int lane = thread & (W - 1); //thread index within the warp, faster way for t % W
	int warpId = thread >> 5; //warp index within the block, faster way for t / W

	//initialize the registers to suffle within the warp
	Real v = local_max_value;
	int j = local_best_j;

	for (int offset = 16; offset > 0; offset >>= 1) { //divides offset by 2 each iteration
        Real v_other = __shfl_down_sync(FULL, v, offset); //reads the value from the register offset lanes above
		Real j_other = __shfl_down_sync(FULL, j, offset); //reads the value from the register offset lanes above
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

    //do the reduction across warps with shuffles    
	__syncthreads();

    if (warpId == 0) { //use only the warp 0 for this
        Real vv = (thread < numWarps) ? w_max_values[thread] : NEG_INF; //load the warp results into registers
        Real jj = (thread < numWarps) ? w_best_js[thread] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) { //divides offset by 2 each iteration
            Real vv_other = __shfl_down_sync(FULL, vv, offset); //reads the value from the register offset lanes above
            Real jj_other = __shfl_down_sync(FULL, jj, offset); //reads the value from the register offset lanes above

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

      

/*
Where the actual calculation happens
*/
std::tuple<Real, Real, int, Real, int, int, Real, Real> run_compute(int n_k) {

    //set up cuda timing
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    

    //Setting up default variables
    Real Kmin = 0.5f; // lower bound of the state space
    Real Kmax = 100.0f; // upper bound of the state space
    Real epsilon = 0.001f; //tolerance of error
    Real alpha = 0.5f; //capital share
    Real z = 1.0f; //productivity
    Real beta = 0.96f; //annual discounting
    Real delta = 0.025f; //annual depreciation

    //start host timer
    auto host_start = std::chrono::steady_clock::now();

    //Set the grid points 
    vector<Real> K(n_k);
    Real step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * step;

    //Initialize the value function and policy arrays
    thrust::device_vector<Real> d_V_new(n_k, 0.0f);
    thrust::device_vector<Real> d_V_old(n_k, 0.0f);
    thrust::device_vector<int> d_policy(n_k, 0);

    //Initialize values for the VF iteration loop
    Real diff = std::numeric_limits<Real>::max();
    int iteration = 0;
    const int max_iter = 20000;
    const Real NEG_INF = -std::numeric_limits<Real>::max();

    //Start gpu timing
    cudaEventRecord(gpu_start);

    //Calculate K^alpha for each grid point
    thrust::device_vector<Real> d_K = K;
    thrust::device_vector<Real> d_K_pow(K.size());
    thrust::transform(d_K.begin(), d_K.end(), d_K_pow.begin(),
        [=] __device__(Real k) { return powf(k, alpha); });

    while (diff > epsilon && iteration < max_iter && ++iteration) {

		//Parameters for kernel launch
        int threadsPerBlock = 256;
        int numBlocks = n_k;
        int numWarps = (threadsPerBlock + 31) / 32;
		int sharedMemBytes = numWarps * (sizeof(Real) + sizeof(int));

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
            [] __host__ __device__(thrust::tuple<Real, Real> V) {
            Real x = thrust::get<0>(V);
            Real y = thrust::get<1>(V);
            return x > y ? x - y : y - x;
            },
            0.0,
            thrust::maximum<Real>()
        );

        d_V_old.swap(d_V_new);

    }
    
    //record run times
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    auto host_end = std::chrono::steady_clock::now();
    std::chrono::duration<Real, std::milli> host_ms = host_end - host_start;
    float gpu_total_ms_f = 0.0f;
    cudaEventElapsedTime(&gpu_total_ms_f, gpu_start, gpu_stop);
    Real gpu_total_ms = static_cast<Real>(gpu_total_ms_f);

    //find crossing
    thrust::host_vector<int> policy(d_policy);
    auto [crossing_min, crossing_max] = find_crossing(K, n_k, policy);

    
    //write the value function into a file in out/rl-test and name the file so that it includes the used grid size
	///is this necessary to do here? If so, make it only write it once per n_k
    /*
    {
        // ensure directory exists
        std::filesystem::path out_dir("out/rl-test");
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            std::cerr << "Failed to create directory: " << out_dir << " : " << ec.message() << std::endl;
        }

        // build filename: value_function_nk_<n_k>_<REAL_NAME>.csv
        std::ostringstream fname;
        fname << "vf_" << n_k << "_" << REAL_NAME << ".csv";
        std::filesystem::path filepath = out_dir / fname.str();

        // copy converged value function to host
        thrust::host_vector<Real> h_V(d_V_old);

        std::ofstream ofs(filepath, std::ios::out);
        if (!ofs) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
        }
        else {
            // header and precision
            ofs << "K,V,policy\n";
            ofs << std::setprecision(std::numeric_limits<Real>::digits10 + 2);

            for (int i = 0; i < n_k; ++i) {
                int pol = (i < static_cast<int>(policy.size())) ? policy[i] : 0;
                ofs << K[i] << "," << h_V[i] << "," << pol << "\n";
            }
        }
    }
    */



    return{ gpu_total_ms, host_ms.count(), iteration, diff, crossing_min, crossing_max , K[crossing_min], K[crossing_max] };
}


/*
Function to handle the running with parameters from command line
*/
void handle_running(int argc, char* argv[]) {
    cout << "Neoclassical Growth model [GPU]" << endl;

    //declare variables
    Real diff, crossing_min_K, crossing_max_K, GPU_median, wall_clock_median;
    int iteration, crossing_min, crossing_max;
    int rounds = 1;
    int warmups = 0;
    int n_k = 145;

    //if command line parameters set, run according to those
    if (argc > 3) {
        n_k = std::atoi(argv[1]);
        rounds = std::atoi(argv[2]);
        warmups = std::atoi(argv[3]);

        vector<Real> GPU_times(rounds, 0);
        vector<Real> wall_clocks(rounds, 0);

        for (int warmup = 0; warmup < warmups; warmup++) {
            Real GPU_time;
            Real wall_clock;
            std::tie(GPU_time, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
        }


        //actual recordings
        for (int i = 0; i < rounds; i++) {
            Real GPU_time;
            Real wall_clock;
            std::tie(GPU_time, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
            GPU_times[i] = GPU_time;
            wall_clocks[i] = wall_clock;
        }

        //calculate medians
        GPU_median = median_of_vector(GPU_times);
        wall_clock_median = median_of_vector(wall_clocks);
    }

    else {
        //default to one run with n_k = 100
        Real GPU_time;
        Real wall_clock;
        std::tie(GPU_time, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
        GPU_median = static_cast<Real>(GPU_time);
        wall_clock_median = wall_clock;
    }

    //write results to a file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string filename = "out\\gpu";
    std::filesystem::path p{ filename };

    std::ofstream ofs(filename, std::ios::out | std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open file: " << filename << '\n';
    }

    else {
        ofs << std::ctime(&now_time);
        ofs << "Using Real = " << REAL_NAME << std::endl;
        ofs << "Grid size(n_k): " << n_k << std::endl;
        ofs << "Found a solution after " << iteration << " iterations" << endl;
        ofs << "Final diff: " << diff << endl;
        ofs << "Numerical steady-state approx between K ~ " << crossing_min_K << " and K ~ " << crossing_max_K
            << ", indexes = " << crossing_min << ", " << crossing_max << endl;
        ofs << "GPU-time median of " << rounds << " rounds: " << GPU_median << std::endl;
        ofs << "Wall-clock median of " << rounds << " rounds: " << wall_clock_median << "\n";
        ofs << "\n";
    }
}


int main(int argc, char* argv[])
{
    std::cout << "masters_thesis: starting compute\n";
    handle_running(argc, argv);
    std::cout << "masters_thesis: finished\n";
    return 0;
}
