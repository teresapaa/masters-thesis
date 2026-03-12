#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstdlib>


#define CUDA_CHECK(call)                                                      
do {
    cudaError_t _e = cudaMalloc(&d_K, n_k * sizeof(Real));
    if (_e != cudaSuccess) {
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__
            << "  " << cudaGetErrorString(_e) << "\n";
        std::exit(1);
    }
} while (0)


//kernel for the worker
template<typename Real>
__global__ void kernel_worker_working(
    Real* __restrict__ V_new,      
    uint16_t* __restrict__ policy,     
    Real* __restrict__ cons,       
    const Real* __restrict__ V_next,     
    const Real* __restrict__ K,         
    Real income, Real inv_1pr, Real tau_avg,
    Real T, Real P, Real beta,
    int n_k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_k) return;

    const Real c_base = K[i] + (Real(1) - tau_avg) * income - T;
    Real     best = Real(-1e10);
    uint16_t bestj = 0;
    Real     bestc = Real(0);

    for (int j = 0; j < n_k; ++j) {
        const Real c = (c_base - K[j] * inv_1pr) / P;
        const Real val = (c <= Real(0) ? Real(-1e10) : log(c)) + beta * V_next[j];
        if (val > best) { best = val; bestj = uint16_t(j); bestc = c; }
    }

    V_new[i] = best;
    policy[i] = bestj;
    cons[i] = bestc;
}


//Owns the device memory
template<typename Real>
struct GpuWorker {
    int n_k = 0, workingYears = 0;

    Real* d_K;
    Real* d_V_r;
    Real* d_V_cur;
    Real* d_V_prev; 
    Real* d_Vnew;  
    uint16_t* d_pol;  
    Real* d_cons;

    GpuWorker(int nk, int yrs, const std::vector<Real>& K_host)
        : n_k(nk), workingYears(yrs)
    {
        const std::size_t Nw = std::size_t(yrs) * std::size_t(nk); // total elements

        CUDA_CHECK(cudaMalloc(&d_K, nk * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_V_r, nk * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_V_prev, nk * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_Vnew, Nw * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_pol, Nw * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_cons, Nw * sizeof(Real)));

        // Upload K once — it never changes across iterations.
        // cudaMemcpyHostToDevice = direction: CPU RAM -> GPU RAM
        CUDA_CHECK(cudaMemcpy(d_K, K_host.data(),
            nk * sizeof(Real), cudaMemcpyHostToDevice));
    }

    void free_all() {
        cudaFree(d_K); cudaFree(d_V_r);
        cudaFree(d_V_cur); cudaFree(d_V_prev);
        cudaFree(d_Vnew); cudaFree(d_pol); cudaFree(d_cons);
        d_K = d_V_r = d_V_cur = d_V_prev = d_Vnew = d_cons = nullptr;
        d_pol = nullptr;
    }

    // Call once per bellman_one_iter(), before the year loop.
    // Uploads the current worker_r boundary (year 0 of retirement).
    void upload_boundary(const Real* V_r_host) {
        CUDA_CHECK(cudaMemcpy(d_V_r, V_r_host,
            n_k * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // Run the full backward pass. Downloads results into V_new / policy / cons
    // of the caller's BlockArrays (flat host vectors, layout [y*n_k + i]).
    void run(const std::vector<Real>& income_worker,
        Real r, Real tau_avg, Real T, Real P, Real beta,
        Real* h_V_new, uint16_t* h_policy, Real* h_cons)
    {
        const Real inv_1pr = Real(1) / (Real(1) + r);
        const int BLOCK = 256;
        const int GRID = (n_k + BLOCK - 1) / BLOCK;

        for (int y = workingYears - 1; y >= 0; --y) {

            // boundary for this year's maximisation
            const Real* d_V_next = (y == workingYears - 1) ? d_V_r : d_V_prev;

            Real* d_Vnew_y = d_Vnew + std::size_t(y) * n_k;
            uint16_t* d_pol_y = d_pol + std::size_t(y) * n_k;
            Real* d_cons_y = d_cons + std::size_t(y) * n_k;

            kernel_worker_working<Real> << <GRID, BLOCK >> > (
                d_Vnew_y, d_pol_y, d_cons_y,
                d_V_next, d_K,
                income_worker[y], inv_1pr, tau_avg, T, P, beta, n_k);
            CUDA_CHECK(cudaGetLastError());

            // make this year's result available as V_next for y-1
            CUDA_CHECK(cudaMemcpy(d_V_prev, d_Vnew_y,
                n_k * sizeof(Real), cudaMemcpyDeviceToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        const std::size_t Nw = std::size_t(workingYears) * std::size_t(n_k);
        CUDA_CHECK(cudaMemcpy(h_V_new, d_Vnew, Nw * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_policy, d_pol, Nw * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cons, d_cons, Nw * sizeof(Real), cudaMemcpyDeviceToHost));
    }
};