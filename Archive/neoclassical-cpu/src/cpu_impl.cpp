// cmake -S . -B out/build && cmake --build out/build
// out\build\Debug\neoclassical-cpu.exe

#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace std;
namespace fs = std::filesystem;
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
    
    int crossing_min = -1;
    int crossing_max = -1;
    for (int i = 0; i < n_k; ++i) {
        double Kp = K[policy[i]];
        if (Kp > K[i]) crossing_min = i;
        if (Kp >= K[i]) crossing_max = i;
    }
    if (crossing_min >= 0) {
        cout << "Numerical steady-state approx between K ~ " << K[crossing_min] << " and K ~ " << K[crossing_max]
            << ", K' at the max state = " << K[policy[crossing_max]] << ", indexes = " << crossing_min << ", " << crossing_max << endl;
    }
    else {
        cout << "No crossing found (policy never suggests K' > K)." << endl;
    }
}



/*
* The main function calculating the Neoclassical Growth Model
*/
extern "C" void run_compute(int argc, char* argv[]) {
    cout << "Neoclassical Growth model [no GPU]" << endl;
	auto host_start = std::chrono::steady_clock::now();

    //Setting up variables
    int n_k = 1000; // number of grid points
    double Kmin = 0.5; // lower bound of the state space
    double Kmax = 100.0; // upper bound of the state space
    double epsilon = 0.001; //tolerance of error
    double alpha = 0.5; //capital share
    double z = 1.0; //productivity
    double beta = 0.96; //annual discounting
    double delta = 0.025; //annual depreciation

    //Parsing command line arguments if set:
    if (argc > 1) {
        n_k = std::atoi(argv[1]);
    }
    std::cout << "n_k: " << n_k << std::endl;
   

    //Set the grid points
    vector<double> K(n_k);
    double step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

    //Initialize the value function and policy arrays
    vector<double> V_old(n_k, 0.0);
    vector<double> V_new(n_k, 0.0);
    vector<int> policy(n_k, 0);

    //Initialize values for the VF iteration loop
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    double diff = 0.0;
    int iteration = 0;
    const int max_iter = 20000;

    //precompute K^alpha
    vector<double> powK(n_k);
    std::transform(K.begin(), K.end(), powK.begin(),
        [alpha] (double k) { return pow(k, alpha); });


    //Value function iteration loop
    do {
        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {
            
            int current_maxidx = 0;
            double current_max = NEG_INF;

            //Go through all the possible transitions from i
            for (int j = 0; j < n_k; j++) {

                //Calculate consumption
                double c = z * powK[i] + (1 - delta) * K[i] - K[j];
                //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K
                if (c <= 0) break;

                //Update the best value found so far
                else {
                    double value = log(c) + beta * V_old[j];
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

    auto host_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> host_ms = host_end - host_start;
    std::cout << "End-to-end host wall-clock time: " << host_ms.count() << " ms\n";

    //Checks to make sure iteration works as desired:

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    //Are the bounds of the state space binding?
    check_if_binding(policy, n_k);

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
    find_crossing(K, n_k, policy);


}


