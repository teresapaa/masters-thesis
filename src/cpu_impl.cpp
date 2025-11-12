#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

/*
* Helper function to find out if the boundaries of the state space are reached; a.k.a. if Kmax is too small
*/
bool is_bindinpolicy(const vector<int>& policy, int n_k) {

    for ( int idx : policy) {
        if (idx == 0 || idx == n_k -1 ) return true; ;
    }
    return false;
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
* The main function calculating the Neoclassical Growth Model
*/
extern "C" void run_compute() {
    cout << "Neoclassical Growth model [no GPU]" << endl;

    //Setting up variables
    int n_k = 1000; // number of grid points
    double Kmin = 0.5; // lower bound of the state space
    double Kmax = 100.0; // upper bound of the state space
    double epsilon = 0.001; //tolerance of error
    double alpha = 0.5; //capital share
    double z = 1.0; //productivity
    double beta = 0.96; //annual discounting
    double delta = 0.025; //annual depreciation

    // production function
    auto F = [alpha](double k) { return powf(k, alpha);};

    // utility function 
    auto u = [](double c) {return log(c); };

    // consumption function
    auto C = [delta, z, F](double K_i, double K_j) {return z * F(K_i) + (1 - delta) * K_i - K_j;};

    //Value conditional on the choice of K
        //K_i = capital from previous period 
        //K_j = capital-option of the current period
        //V_j = the previous value function at the point J 
    auto V = [z, delta, beta, F, u, C](double K_i, double K_j, double V_j) { return u(C(K_i, K_j)) + beta * V_j;};
    

    
    //Set the grid points
    vector<double> K(n_k);
    double step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

    //Initialize the value function and policy arrays
    vector<double> V_old(n_k, 0.0);
    vector<double> V_new(n_k, 0.0);
    vector<double> V_conditionals(n_k, 0.0);
    vector<int> policy(n_k, 0);

    //Initialize values for the VF iteration loop
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    double diff =0.0;
    int iteration = 0;
    const int max_iter = 20000;

    //Value function iteration loop
    do {
        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {

            //Go through all the possible transitions from i
            for (int j = 0; j < n_k; j++) {

                //If consumption is nonpositive, assign a large negative number to make sure that state won't be chosen
                if (C(K[i], K[j]) <= 0) {
                    V_conditionals[j] = NEG_INF; 
                }
                //Update the value conditionals
                else { V_conditionals[j] = V(K[i], K[j], V_old[j]); }

            }
            //find the state j maximizing VF, update that to policy and value functions 
            auto max_elem = max_element(V_conditionals.begin(), V_conditionals.end());
            policy[i] = std::distance(V_conditionals.begin(), max_elem);
            V_new[i] = *max_elem;
        }
        
        diff = max_abs_difference(V_old, V_new);
        V_old = V_new;
        ++iteration;

    } while (diff > epsilon && iteration < max_iter);


    //Checks to make sure iteration works as desired:

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    //Are the bounds of the state space binding?
    if (is_bindinpolicy(policy, n_k)) {
        cout << "State space is binding" << endl;
    }
    else {
        cout << "State space is not binding :) " << endl;
    }

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
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

    /*
    Print out the optimal policy:
    std::cout << "Optimal policy g:" << std::endl;
    for (int x : g) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
       
    cout << "Corresponding K' values for each K:" << endl;

    for (int x : g) {
        std::cout << K[x] << " ";
    }
    std::cout << std::endl;
    */
}


