#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

    // Float API
    void* gpu_worker_create_float(int nk, int yrs, const float* K_host);
    void  gpu_worker_free_float(void* handle);
    void  gpu_worker_upload_boundary_float(void* handle, const float* V_r_host);
    void  gpu_worker_run_float(void* handle,
        const float* income_worker, float r, float tau_avg, float T, float P, float beta,
        float* h_V_new, uint16_t* h_policy, float* h_cons);

    // Double API
    void* gpu_worker_create_double(int nk, int yrs, const double* K_host);
    void  gpu_worker_free_double(void* handle);
    void  gpu_worker_upload_boundary_double(void* handle, const double* V_r_host);
    void  gpu_worker_run_double(void* handle,
        const double* income_worker, double r, double tau_avg, double T, double P, double beta,
        double* h_V_new, uint16_t* h_policy, double* h_cons);

#ifdef __cplusplus
}
#endif