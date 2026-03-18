// bellman_gpu.cu  -- compile with nvcc
#include "bellman_gpu.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <memory>

// Safe CUDA check
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)           \
                      << " (" << static_cast<int>(_err) << ") at "            \
                      << __FILE__ << ":" << __LINE__ << "\n";                 \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// device log helpers
__device__ inline float dev_logf(float x) { return logf(x); }
__device__ inline double dev_logd(double x) { return log(x); }

// templated kernel: maximizes over j for each asset state i
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
    Real best = -
    uint16_t bestj = 0;
    Real bestc = Real(0);

    for (int j = 0; j < n_k; ++j) {
        const Real c = (c_base - K[j] * inv_1pr) / P;
        Real val;
        if constexpr (std::is_same_v<Real, float>) {
            val = (c <= Real(0) ? -INFINITY : dev_logf(c)) + beta * V_next[j];
        }
        else {
            val = (c <= Real(0) ? -INFINITY : dev_logd(c)) + beta * V_next[j];
        }
        if (val > best) { best = val; bestj = uint16_t(j); bestc = c; }
    }

    V_new[i] = best;
    policy[i] = bestj;
    cons[i] = bestc;
}

// RAII GPU worker (templated)
template<typename Real>
struct GpuWorkerT {
    int n_k = 0;
    int workingYears = 0;
    std::size_t Nw = 0;

    Real* d_K = nullptr;
    Real* d_V_r = nullptr;
    Real* d_V_prev = nullptr;
    Real* d_Vnew = nullptr;
    uint16_t* d_pol = nullptr;
    Real* d_cons = nullptr;

    GpuWorkerT(int nk, int yrs, const Real* K_host)
        : n_k(nk), workingYears(yrs)
    {
        Nw = std::size_t(nk) * std::size_t(yrs);
        // allocate
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_K), std::size_t(nk) * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_V_r), std::size_t(nk) * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_V_prev), std::size_t(nk) * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Vnew), Nw * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pol), Nw * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cons), Nw * sizeof(Real)));

        // copy K
        CUDA_CHECK(cudaMemcpy(d_K, K_host, std::size_t(nk) * sizeof(Real), cudaMemcpyHostToDevice));
        // zero d_V_prev/d_Vnew/pol/cons for safety
        CUDA_CHECK(cudaMemset(d_V_prev, 0, std::size_t(nk) * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_Vnew, 0, Nw * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_pol, 0, Nw * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(d_cons, 0, Nw * sizeof(Real)));
    }

    ~GpuWorkerT() { free_all(); }

    void free_all() {
        if (d_K) CUDA_CHECK(cudaFree(d_K));
        if (d_V_r) CUDA_CHECK(cudaFree(d_V_r));
        if (d_V_prev) CUDA_CHECK(cudaFree(d_V_prev));
        if (d_Vnew) CUDA_CHECK(cudaFree(d_Vnew));
        if (d_pol) CUDA_CHECK(cudaFree(d_pol));
        if (d_cons) CUDA_CHECK(cudaFree(d_cons));
        d_K = d_V_r = d_V_prev = d_Vnew = nullptr;
        d_pol = nullptr;
        d_cons = nullptr;
    }

    void upload_boundary(const Real* V_r_host) {
        CUDA_CHECK(cudaMemcpy(d_V_r, V_r_host, std::size_t(n_k) * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // income_worker must be an array of length workingYears (host)
    void run(const Real* income_worker,
        Real r, Real tau_avg, Real T, Real P, Real beta,
        Real* h_V_new, uint16_t* h_policy, Real* h_cons)
    {
        const Real inv_1pr = Real(1) / (Real(1) + r);
        const int BLOCK = 256;
        const int GRID = (n_k + BLOCK - 1) / BLOCK;

        for (int y = workingYears - 1; y >= 0; --y) {
            const Real* d_V_next = (y == workingYears - 1) ? d_V_r : d_V_prev;
            Real* d_Vnew_y = d_Vnew + std::size_t(y) * std::size_t(n_k);
            uint16_t* d_pol_y = d_pol + std::size_t(y) * std::size_t(n_k);
            Real* d_cons_y = d_cons + std::size_t(y) * std::size_t(n_k);

            // instantiate the right kernel
            kernel_worker_working<Real> << <GRID, BLOCK >> > (
                d_Vnew_y, d_pol_y, d_cons_y,
                d_V_next, d_K,
                income_worker[y], inv_1pr, tau_avg, T, P, beta, n_k);
            CUDA_CHECK(cudaGetLastError());

            // use result as next-period boundary
            CUDA_CHECK(cudaMemcpy(d_V_prev, d_Vnew_y, std::size_t(n_k) * sizeof(Real), cudaMemcpyDeviceToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // copy back entire block results
        CUDA_CHECK(cudaMemcpy(h_V_new, d_Vnew, Nw * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_policy, d_pol, Nw * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cons, d_cons, Nw * sizeof(Real), cudaMemcpyDeviceToHost));
    }
};

// Explicit instantiations
using GpuWorkerFloat = GpuWorkerT<float>;
using GpuWorkerDouble = GpuWorkerT<double>;

// C wrappers
extern "C" {

    void* gpu_worker_create_float(int nk, int yrs, const float* K_host) {
        return reinterpret_cast<void*>(new GpuWorkerFloat(nk, yrs, K_host));
    }
    void gpu_worker_free_float(void* handle) {
        delete reinterpret_cast<GpuWorkerFloat*>(handle);
    }
    void gpu_worker_run_float(void* handle,
        const float* income_worker, float r, float tau_avg, float T, float P, float beta,
        float* h_V_new, uint16_t* h_policy, float* h_cons)
    {
        auto ptr = reinterpret_cast<GpuWorkerFloat*>(handle);
        ptr->run(income_worker, r, tau_avg, T, P, beta, h_V_new, h_policy, h_cons);
    }

    void* gpu_worker_create_double(int nk, int yrs, const double* K_host) {
        return reinterpret_cast<void*>(new GpuWorkerDouble(nk, yrs, K_host));
    }
    void gpu_worker_free_double(void* handle) {
        delete reinterpret_cast<GpuWorkerDouble*>(handle);
    }
    void gpu_worker_run_double(void* handle,
        const double* income_worker, double r, double tau_avg, double T, double P, double beta,
        double* h_V_new, uint16_t* h_policy, double* h_cons)
    {
        auto ptr = reinterpret_cast<GpuWorkerDouble*>(handle);
        ptr->run(income_worker, r, tau_avg, T, P, beta, h_V_new, h_policy, h_cons);
    }

} // extern "C"