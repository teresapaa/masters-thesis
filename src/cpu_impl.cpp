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
    string output_dir = R"(C:\Users\Administrator\source\repos\masters-thesis)";

    

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

    // Ensure output directory exists so file writes succeed
    fs::path outdir = fs::path(output_dir) / "out" / "data";

    try {
        fs::create_directories(outdir);
    }
    catch (const std::exception& e) {
        cout << "Warning: failed to create output directory '" << outdir.string() << "': " << e.what() << endl;
    }

    // snapshot frequency: save V every save_every iterations (and always final)
    const int save_every = 20;
    auto save_snapshot = [&](int iter) {
        std::ostringstream fname;
        fname << outdir.string() << "/vfi_iter_" << setw(4) << setfill('0') << iter << ".csv";
        ofstream f(fname.str());
        if (!f.is_open()) {
            cout << "Warning: could not open snapshot file for writing: " << fname.str() << endl;
            return;
        }
        f << "i,K,V\n";
        for (int i = 0; i < n_k; ++i) {
            f << i << "," << K[i] << "," << V_old[i] << "\n";
        }
        f.close();
        };

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

        if (iteration % save_every == 0) save_snapshot(iteration);


    } while (diff > epsilon && iteration < max_iter);


    //Checks to make sure iteration works as desired:

    cout << "Found a solution after " << iteration << " iterations" << endl;
    cout << "Final diff: " << diff << endl;

    //Are the bounds of the state space binding?
    check_if_binding(policy, n_k);

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
    find_crossing(K, n_k, policy);

    // write final CSV of policy/value
    {
        fs::path final_path = outdir / "vfi_final.csv";
        ofstream fout(final_path.string());
        if (!fout.is_open()) {
            cout << "Warning: could not open final CSV for writing: " << final_path.string() << endl;
        }
        else {
            fout << "i,K,V,Kp_index,Kp,c\n";
            for (int i = 0; i < n_k; ++i) {
                int j = policy[i];
                double Ki = K[i];
                double Kj = K[j];
                double c = C(Ki, Kj);
                fout << i << "," << K[i] << "," << V_new[i] << "," << j << "," << Kj << "," << c << "\n";
            }
            fout.close();
        }
    }

    // save final V snapshot too
    {
        fs::path snap_path = outdir / "vfi_iter_final.csv";
        ofstream f(snap_path.string());
        if (!f.is_open()) {
            cout << "Warning: could not open final snapshot for writing: " << snap_path.string() << endl;
        }
        else {
            f << "i,K,V\n";
            for (int i = 0; i < n_k; ++i) f << i << "," << K[i] << "," << V_new[i] << "\n";
            f.close();
        }
    }

    
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
    
}


