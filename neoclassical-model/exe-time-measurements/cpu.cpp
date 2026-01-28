// release double: cl /std:c++17 /O2 /EHsc /DUSE_DOUBLE=1 cpu.cpp /Fe:cpu-double.exe
// release float: cl /std:c++17 /O2 /EHsc cpu.cpp /Fe:cpu-float.exe
// parametrit komentoriviltä: n_k rounds warmups

#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <windows.h>

using namespace std;

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
        //cout << "State space is not binding :) " << endl;
    }
}

/*
* Helper function to calculate the maximum pointwise distance of two vectors
*/
Real max_abs_difference(const std::vector<Real>& V0, const std::vector<Real>& V1) {
    Real d = 0.0;
    for (size_t i = 0; i < V0.size(); ++i) {
        Real diff = std::abs(V0[i] - V1[i]);
        if (diff > d) d = diff;
    }
    return d;
}

/*
* A helper function to find index where K' - K changes sign: last i with K'[i] > K[i]
*/
std::tuple<int, int> find_crossing(vector<Real> K, int n_k, vector<int> policy) {
    
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


/*
* The main function calculating the Neoclassical Growth Model
*/
std::tuple<std::uint64_t, Real, int, Real, int, int, Real, Real> run_compute(int n_k) {

    //Setting up variables
    Real Kmin = 0.5; // lower bound of the state space
    Real Kmax = 100.0; // upper bound of the state space
    Real epsilon = 0.001; //tolerance of error
    Real alpha = 0.5; //capital share
    Real z = 1.0; //productivity
    Real beta = 0.96; //annual discounting
    Real delta = 0.025; //annual depreciation
   
    auto host_start = std::chrono::steady_clock::now();
    HANDLE hProcess = GetCurrentProcess();
    ULONG64 startCycles = 0, endCycles = 0;
    QueryProcessCycleTime(hProcess, &startCycles);

    //Set the grid points
    vector<Real> K(n_k);
    Real step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

    //Initialize the value function and policy arrays
    vector<Real> V_old(n_k, 0.0);
    vector<Real> V_new(n_k, 0.0);
    vector<int> policy(n_k, 0);

    //Initialize values for the VF iteration loop
    const Real NEG_INF = -std::numeric_limits<Real>::infinity();
    Real diff = 0.0;
    int iteration = 0;
    const int max_iter = 20000;

    //precompute K^alpha
    vector<Real> powK(n_k);
    std::transform(K.begin(), K.end(), powK.begin(),
        [alpha] (Real k) { return pow(k, alpha); });


    //Value function iteration loop
    do {
        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {
            
            int current_maxidx = 0;
            Real current_max = NEG_INF;

            //Go through all the possible transitions from i
            for (int j = 0; j < n_k; j++) {

                //Calculate consumption
                Real c = z * powK[i] + (1 - delta) * K[i] - K[j];

                //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K
                if (c <= 0) break;

                //Update the best value found so far
                else {
                    Real value = log(c) + beta * V_old[j];
                    if (value > current_max) {
                        current_max = value;
                        current_maxidx = j;
                    }
                }

            }
            //update policy and value functions 
            policy[i] = current_maxidx;
            V_new[i] = current_max;
        }
        
        diff = max_abs_difference(V_old, V_new);
        V_old = V_new;
        ++iteration;


    } while (diff > epsilon && iteration < max_iter);

    //stop measuring time and CPU cycles
    QueryProcessCycleTime(hProcess, &endCycles);
    auto host_end = std::chrono::steady_clock::now();
    std::uint64_t CPU_cycles = (endCycles - startCycles);
    std::chrono::duration<Real, std::milli> host_ms = host_end - host_start;

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
    auto [crossing_min, crossing_max] = find_crossing(K, n_k, policy);

    return{ CPU_cycles, host_ms.count(), iteration, diff, crossing_min, crossing_max , K[crossing_min], K[crossing_max]};
}

/*
Function to handle the running with parameters from command line
*/
void handle_running(int argc, char* argv[]) {
    cout << "Neoclassical Growth model [no GPU]" << endl;

    //results to be recorded
    Real diff, crossing_min_K, crossing_max_K, CPU_median, wall_clock_median;
    int iteration, crossing_min, crossing_max;

    int rounds = 1;
	int warmups = 0;
    int n_k = 145;

    //check if the user has given parameters
    if (argc > 3) {
        //assign parameters from the user
        n_k = std::atoi(argv[1]);
        rounds = std::atoi(argv[2]);
		warmups = std::atoi(argv[3]);

        //vectors to record results
        vector<std::uint64_t> CPU_clocks(rounds, 0);
        vector<Real> wall_clocks(rounds, 0);

        //warm-ups
        int warmups = 5;
        for (int warmup = 0; warmup < warmups; warmup++) {
            std::uint64_t CPU_cycles;
            Real wall_clock;
            std::tie(CPU_cycles, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
        }

        //actual recordings
        for (int i = 0; i < rounds; i++) {
            std::uint64_t CPU_cycles;
            Real wall_clock;
            std::tie(CPU_cycles, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
            CPU_clocks[i] = CPU_cycles;
            wall_clocks[i] = wall_clock;
        }

        //calculate medians
        CPU_median = median_of_vector(CPU_clocks);
        wall_clock_median = median_of_vector(wall_clocks);
    }

    else {
        //run once with default parameters
        std::uint64_t CPU_cycles;
        Real wall_clock;
        std::tie(CPU_cycles, wall_clock, iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);
        CPU_median = static_cast<Real>(CPU_cycles);
        wall_clock_median = wall_clock;
    }

    //record results to file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string filename = "out\\cpu";
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
        ofs << "CPU-time median of " << rounds << " rounds: " << CPU_median << std::endl;
        ofs << "Wall-clock median of " << rounds << " rounds: " << wall_clock_median << "\n";
        ofs << "\n";
    }

}


int main(int argc, char* argv[]) 
{
    std::cout << "masters_thesis: starting compute\n";
    handle_running(argc, argv);
    std::cout << "masters_thesis: finished\n\n";
    return 0;
}
