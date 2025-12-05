#include <cuda_runtime.h>
#include <thrust/host_vector.h>



//plotting final V & g here:
void plot() {
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


/*
* Helper function to find out if the boundaries of the state space are reached; a.k.a. if Kmax is too small
*/

void find_crossing(vector<float> K, int n_k, thrust::host_vector<int> policy) {
    int crossing = -1;
    for (int i = 0; i < n_k; ++i) {
        double Kp = K[policy[i]];
        if (Kp > K[i]) crossing = i;
    }
    if (crossing >= 0) {
        std::cout << "Numerical steady-state approx at K ~ " << K[crossing]
            << ", K' at that state = " << K[policy[crossing]] << ", index = " << crossing << std::endl;
    }
    else {
        std::cout << "No crossing found (policy never suggests K' > K)." << std::endl;
    }



 /*
* Helper function to find out if the boundaries of the state space are reached; a.k.a. if Kmax is too small
*/
void check_if_binding(const vector<int>&policy, int n_k) {

    bool isBinding = false;

    for (int idx : policy) {
        if (idx == 0 || idx == n_k - 1) {
            isBinding = true;
            break;
        }
    }
    if (isBinding) {
        std::cout << "State space is binding" << std::endl;
    }
    else {
        std::cout << "State space is not binding :) " << std::endl;
    }
}

/*
* Helper function to print out the optimal policy
*/
void print_policy(const vector<int>& policy) {
    std::cout << "Optimal policy g:" << std::endl;
    for (int x : policy) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

/*
* Helper function to print out the value function
*/
void print_V(const vector<float> V) {
    cout << "Value function V" << endl;

    for (int v : V) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

}
