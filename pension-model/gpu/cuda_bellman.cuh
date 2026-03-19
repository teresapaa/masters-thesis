#pragma once
#include <cstdio>
#include "model_gpu.h"


#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            printf("CUDA error at %s:%d — %s\n",                             \
                   __FILE__, __LINE__, cudaGetErrorString(err));              \
        }                                                                     \
    } while (0)
#endif


struct BlockArrays;

struct DeviceBlockArrays {

    Real* V_old = nullptr;
    Real* V_new = nullptr;
    Real* cons = nullptr;
    uint16_t* policy = nullptr;

    int num_states = 0;
    int years = 0;
    int n_k = 0;

    // Total number of elements across all dimensions
    __host__ __device__
        size_t total() const {
        return size_t(num_states) * size_t(years) * size_t(n_k);
    }

    // Flat index into any of the arrays, same logic as BlockArrays::idx
    __host__ __device__
        size_t idx(int s, int y, int i) const {
        return (size_t(s) * size_t(years) + size_t(y)) * size_t(n_k) + size_t(i);
    }

    void allocate(int ns, int yrs, int nk);
    void free();
    void upload(const BlockArrays& src);
    void download(BlockArrays& dst) const;
    void swap_old_new() {
        Real* tmp = V_old;
        V_old = V_new;
        V_new = tmp;
    }
};

bool cuda_roundtrip_test(const std::vector<Real>& K,
    const std::vector<Real>& V_old);

void upload_income(Real* d_income, const Real* income, int workingYears);

void upload_income_entrep(Real* d_income_entrep, const Real* income_entrep,
    int n_types, int workingYears);

void upload_b_state(Real* d_b_worker, const Real* b_worker,
    Real* d_b_entrep, const Real* b_entrep, int n_types);

void free_device_grids(Real* d_K, Real* d_income);

void free_b_state(Real* d_b_worker, Real* d_b_entrep);
void free_entrep_grids(Real* d_income_entrep, Real* d_tau);

void init_b_state(Real** d_b_worker, Real** d_b_entrep, int n_types);

void init_device_grids(Real** d_K, const Real* K, int n_k,
    Real** d_income, int workingYears);

void init_entrep_grids(Real** d_income_entrep, int n_types, int workingYears,
    Real** d_tau, const Real* tau, int n_tau);

Real cuda_max_abs_diff(DeviceBlockArrays& d_blk);

void download_for_macros(DeviceBlockArrays& d_blk, BlockArrays& blk);

void cuda_bellman_worker_working(
        DeviceBlockArrays& d_worker_w,
        DeviceBlockArrays& d_worker_r,
        const Real* d_K,
        const Real* d_income,
        Real tau_avg, Real inv_1pr, Real P, Real beta, Real T,
        int n_k, int workingYears,
        BlockArrays& worker_w
);

void cuda_bellman_retirement(
    DeviceBlockArrays& d_ret,
    const Real* d_K,
    const Real* d_b_state,
    Real inv_1pr, Real P, Real beta,
    int n_k, int retireYears, int num_states,
    BlockArrays& ret
);

void cuda_bellman_entrep_working(
    DeviceBlockArrays& d_entrep_w,
    DeviceBlockArrays& d_entrep_r,
    const Real* d_K,
    const Real* d_income_entrep,
    const Real* d_tau,
    Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears, int retireYears, int num_states, int n_tau,
    BlockArrays& entrep_w
);