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
#include <chrono>
#include <ctime>

using namespace std;
namespace fs = std::filesystem;

#if defined(USE_DOUBLE)
using Real = double;
constexpr const char* REAL_NAME = "double";
#else
using Real = float;
constexpr const char* REAL_NAME = "float";
#endif



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
std::tuple< int, Real, int, int, Real, Real> run_compute(int n_k) {

    //Setting up variables
    Real Kmin = 0.5; // lower bound of the state space
    Real Kmax = 100.0; // upper bound of the state space
    Real epsilon = 0.001; //tolerance of error
    Real alpha = 0.5; //capital share
    Real z = 1.0; //productivity
    Real beta = 0.96; //annual discounting
    Real delta = 0.025; //annual depreciation
    string output_dir = R"(C:\Users\Administrator\source\repos\masters-thesis\plots\out\data)"; //for plotting the results
   

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

    //consumption function (for plotting)
    auto C = [delta, z, powK](double K_i, double K_j) {return z * powK[K_i] + (1 - delta) * K_i - K_j; };


    //For plotting:
   // Ensure output directory exists so file writes succeed
    fs::path outdir = fs::path(output_dir);
    try {
        fs::create_directories(outdir);
    }
    catch (const std::exception& e) {
        cout << "Warning: failed to create output directory '" << outdir.string() << "': " << e.what() << endl;
    }
    //save V every save_every iterations (and always final)
    const int save_every = 20;
    auto save_snapshot = [&](int iter) {
        std::ostringstream fname;
        fname << outdir.string() << "/cpu_" << std::to_string(n_k) << "_vfi_iter" << setw(4) << setfill('0') << iter << ".csv";
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


    // Open diffs CSV to record diff at every iteration
    fs::path diffs_path = outdir / ("cpu_" + std::to_string(n_k) + "_vfi_diffs.csv");
    std::ofstream diffs_ofs(diffs_path.string(), std::ios::out);
    if (!diffs_ofs.is_open()) {
        cout << "Warning: could not open diffs file for writing: " << diffs_path.string() << endl;
    }
    else {
        // write header
        diffs_ofs << "iter,diff\n";
        // set precision appropriate for Real
        int prec = std::numeric_limits<Real>::digits10 + 1;
        diffs_ofs << std::setprecision(prec);
    }

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

        // record diff for this iteration (if file open)
        if (diffs_ofs.is_open()) {
            diffs_ofs << iteration << "," << diff << "\n";
        }

        if (iteration % save_every == 0) save_snapshot(iteration);


    } while (diff > epsilon && iteration < max_iter);

    // close diffs file
    if (diffs_ofs.is_open()) diffs_ofs.close();

    // Find index where K' - K changes sign: last i with K'[i] > K[i]
    auto [crossing_min, crossing_max] = find_crossing(K, n_k, policy);

    // write final CSV of policy/value
    {
        fs::path final_path = outdir / ("cpu_" + std::to_string(n_k) + "_vfi_final.csv");
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
        fs::path snap_path = outdir / ("cpu_" + std::to_string(n_k) + "_vfi_iter_final.csv");
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

    return{ iteration, diff, crossing_min, crossing_max , K[crossing_min], K[crossing_max]};
}

/*
Function to handle the running with parameters from command line
*/
void handle_running(int argc, char* argv[]) {
    cout << "Neoclassical Growth model [no GPU]" << endl;

    //results to be recorded
    Real diff, crossing_min_K, crossing_max_K;
    int iteration, crossing_min, crossing_max;
    int n_k = 145;

    //check if the user has given parameters
    if (argc > 1) {
        //assign parameters from the user
        n_k = std::atoi(argv[1]);
    }

    std::tie(iteration, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k);

    //record results to file
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    fs::path logdir = "out/log";

    try {
        fs::create_directories(logdir);
    }
    catch (const std::exception& e) {
        cout << "Warning: failed to create output directory: " << e.what() << endl;
    }

    fs::path logfile = logdir / "cpu.txt";


    std::ofstream ofs(logfile, std::ios::out | std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open file: " << logfile << '\n';
    }
    else {
        ofs << std::ctime(&now_time);
        ofs << "Using Real = " << REAL_NAME << std::endl;
        ofs << "Grid size(n_k): " << n_k << std::endl;
        ofs << "Found a solution after " << iteration << " iterations" << endl;
        ofs << "Final diff: " << diff << endl;
        ofs << "Numerical steady-state approx between K ~ " << crossing_min_K << " and K ~ " << crossing_max_K
            << ", indexes = " << crossing_min << ", " << crossing_max << endl;
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
