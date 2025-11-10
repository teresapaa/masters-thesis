#include <iostream>
#include <cmath>
#include <vector>
using namespace std;



extern "C" void run_compute() {
    cout << "Neoclassical Growth model [no GPU]\n";

    //Setting up variables
    int n_k = 100; // number of grid points
    float Kmin = 0.5f; // lower bound of the state space
    float Kmax = 4.5f; // upper bound of the state space
    float epsilon = 0.001f; //tolerance of error
    float alpha = 0.5f; //capital share
    int z = 1; //productivity
    float delta = 0.03f; //fix this to be something meaningful
    float beta = 0.96f; //annual discounting
    float delta = 0.025f; //annual depreciation

    // production function - now as a lambda function, could be implemented as a normal function as well
    auto F = [alpha](float k) { return powf(k, alpha);};

    // utility function: u(c) = log(C)
    auto u = [](float c) {return log(c); };

    //Value conditional on the choice of K
        //K_i = capital from previous period 
        //K_j = capital-option of the current period
        //V_j = the previous value function at the point J 
    auto V = [z, delta, beta, F, u](float K_i, float K_j, float V_j) { return u(z * F(K_i) + (1 - delta) * K_i - K_j) + beta * V_j;};
    auto C = [delta, z, F](float K_i, float K_j) {return z * F(K_i) + (1 - delta) * K_i - K_j;};
    
    //Set the grid points
    vector<float> K(n_k);
    float step = (Kmax - Kmin) / (n_k - 1);
    for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

    //Initialize the value function to 0 - what kind of data structure?
    vector<float> V_old(n_k, 0.0);
    vector<float> V_new(n_k, 0.0);
    vector<float> V_conditionals(n_k, 0.0);
    vector<int> g;

    double diff;
    int iteration = 0;

    //Value function iteration loop
    do {
        for (int i = 0; i < n_k; i++) {
            for (int j = 0; j < n_k; j++) {
                if (C(K[i], K[j]) < 0) { V_conditionals[j] = 100000000.0f; }
                else { V_conditionals[j] = V(K[i], K[j], V_old[j]); }

            }
            float max = *max_element(V_conditionals.begin(), V_conditionals.end());
            // int index = std::distance(V_conditionals.begin(), max);  <- what is the problem?
            // g.push_back(index);
            V_new[i] = max;
        }

        //diff = ...

    } while (diff < epsilon);
     
    //Check the results

        
}