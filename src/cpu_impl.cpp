#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

bool is_binding(const vector<int>& g, float n_k) {

    for ( int idx : g) {
        if (idx == 1 || idx == n_k -1 ) return true; ;
    }
    return false;
}

float max_abs_difference(const std::vector<float>& V0, const std::vector<float>& V1) {
    float d = 0.0;
    for (size_t i = 0; i < V0.size(); ++i) {
        float diff = std::abs(V0[i] - V1[i]);
        if (diff > d) d = diff;
    }
    return d;
}

extern "C" void run_compute() {
    cout << "Neoclassical Growth model [no GPU]" << endl;

    //Setting up variables
    int n_k = 1000; // number of grid points
    float Kmin = 0.5f; // lower bound of the state space
    float Kmax = 100.0f; // upper bound of the state space
    float epsilon = 0.001f; //tolerance of error
    float alpha = 0.5f; //capital share
    float z = 1.0f; //productivity
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
    vector<int> g(n_k, 0);

    const float NEG_INF = -1e300;
    float diff =0.0f;
    int iteration = 0;
    const int max_iter = 20000;

    //Value function iteration loop
    do {
        for (int i = 0; i < n_k; i++) {
            for (int j = 0; j < n_k; j++) {
                if (C(K[i], K[j]) < 0) { V_conditionals[j] = NEG_INF; }
                else { V_conditionals[j] = V(K[i], K[j], V_old[j]); }

            }
            auto max_elem = max_element(V_conditionals.begin(), V_conditionals.end());
            int index = std::distance(V_conditionals.begin(), max_elem);
            g[i] = index;
            V_new[i] = *max_elem;
        }
    
        diff = max_abs_difference(V_old, V_new);

        V_old = V_new;
        ++iteration;

    } while (diff > epsilon && iteration < max_iter);

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    //Are the bounds of the state space binding?
    if (is_binding(g, n_k)) {
        cout << "State space is binding" << endl;
    }

    //Print out the optimal policy:
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

}


