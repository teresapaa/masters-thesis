
//manual compile commands
//nvcc -c cuda_bellman.cu -o cuda_bellman.obj
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "cuda_bellman.cuh"


void DeviceBlockArrays::allocate(int ns, int yrs, int nk) {
    num_states = ns;
    years = yrs;
    n_k = nk;
    const size_t N = total();
    const size_t bytes_d = N * sizeof(Real);
    const size_t bytes_p = N * sizeof(uint16_t);

    CUDA_CHECK(cudaMalloc(&V_old, bytes_d));
    CUDA_CHECK(cudaMalloc(&V_new, bytes_d));
    CUDA_CHECK(cudaMalloc(&cons, bytes_d));
    CUDA_CHECK(cudaMalloc(&policy, bytes_p));

    // zero-initialise so uninitialised reads are detectable
    CUDA_CHECK(cudaMemset(V_old, 0, bytes_d));
    CUDA_CHECK(cudaMemset(V_new, 0, bytes_d));
    CUDA_CHECK(cudaMemset(cons, 0, bytes_d));
    CUDA_CHECK(cudaMemset(policy, 0, bytes_p));
}

// Free all device arrays and null the pointers
void DeviceBlockArrays::free() {
    cudaFree(V_old);  V_old = nullptr;
    cudaFree(V_new);  V_new = nullptr;
    cudaFree(cons);   cons = nullptr;
    cudaFree(policy); policy = nullptr;
}

void DeviceBlockArrays::upload(const BlockArrays& src) {
    const size_t N = total();
    const size_t bytes_d = N * sizeof(Real);
    const size_t bytes_p = N * sizeof(uint16_t);

    CUDA_CHECK(cudaMemcpy(V_old, src.V_old.data(), bytes_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_new, src.V_new.data(), bytes_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cons, src.cons.data(), bytes_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(policy, src.policy.data(), bytes_p, cudaMemcpyHostToDevice));
}

void DeviceBlockArrays::download(BlockArrays& dst) const {
    const size_t N = total();
    const size_t bytes_d = N * sizeof(Real);
    const size_t bytes_p = N * sizeof(uint16_t);

    CUDA_CHECK(cudaMemcpy(dst.V_old.data(), V_old, bytes_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst.V_new.data(), V_new, bytes_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst.cons.data(), cons, bytes_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dst.policy.data(), policy, bytes_p, cudaMemcpyDeviceToHost));
}


__global__ void kernel_bellman_worker_working(
    const Real* K,
    const Real* V_working_old,
    const Real* V_retire_old,
    const Real* income,
    Real* V_new,
    uint16_t* policy,
    Real* cons,
    Real tau_avg, Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= workingYears * n_k) return;

    const int y = tid / n_k;
    const int i = tid % n_k;

    const Real n = K[i];
    const Real inc = income[y];
    const Real base = n + (Real(1) - tau_avg) * inc - T;

    Real     best = NEG_INF;
    uint16_t bestj = 0;
    Real     bestc = Real(0);

    for (int j = 0; j < n_k; ++j) {
        const Real Vnext = (y == workingYears - 1)
            ? V_retire_old[j]
            : V_working_old[(y + 1) * n_k + j];

        const Real c = (base - K[j] * inv_1pr) / P;
        const Real val = logf(c) + beta * Vnext;

        if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
    }

    V_new[tid] = best;
    policy[tid] = bestj;
    cons[tid] = bestc;
}



void cuda_bellman_worker_working(
    DeviceBlockArrays& d_worker_w,
    DeviceBlockArrays& d_worker_r,
    const Real* d_K,
    const Real* d_income,
    Real tau_avg, Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears,
    BlockArrays& worker_w)
{

    Real h_inc[3];
    cudaMemcpy(h_inc, d_income, 3 * sizeof(Real), cudaMemcpyDeviceToHost);

    const int total = workingYears * n_k;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    kernel_bellman_worker_working <<<gridSize, blockSize>>> (
        d_K,
        d_worker_w.V_old,
        d_worker_r.V_old,
        d_income,
        d_worker_w.V_new,
        d_worker_w.policy,
        d_worker_w.cons,
        tau_avg, inv_1pr, P, beta, T,
        n_k, workingYears
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    Real h_sample[3];
    cudaMemcpy(h_sample, d_worker_w.V_new, sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sample + 1, d_worker_w.V_new + 50, sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sample + 2, d_worker_w.V_new + (n_k - 1), sizeof(Real), cudaMemcpyDeviceToHost);

    const size_t bytes_r = total * sizeof(Real);
    const size_t bytes_p = total * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(worker_w.V_new.data(), d_worker_w.V_new, bytes_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(worker_w.policy.data(), d_worker_w.policy, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(worker_w.cons.data(), d_worker_w.cons, bytes_r, cudaMemcpyDeviceToHost));
}

void upload_income(Real* d_income, const Real* income, int workingYears) {
    CUDA_CHECK(cudaMemcpy(d_income, income, workingYears * sizeof(Real), cudaMemcpyHostToDevice));
}

void upload_b_state(Real* d_b_worker, const Real* b_worker,
    Real* d_b_entrep, const Real* b_entrep, int n_types) {
    CUDA_CHECK(cudaMemcpy(d_b_worker, b_worker, 1 * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_entrep, b_entrep, n_types * sizeof(Real), cudaMemcpyHostToDevice));
}

void upload_income_entrep(Real* d_income_entrep, const Real* income_entrep,
    int n_types, int workingYears) {
    CUDA_CHECK(cudaMemcpy(d_income_entrep, income_entrep,
        n_types * workingYears * sizeof(Real), cudaMemcpyHostToDevice));
}

void free_device_grids(Real* d_K, Real* d_income) {
    cudaFree(d_K);
    cudaFree(d_income);
}

void free_b_state(Real* d_b_worker, Real* d_b_entrep) {
    cudaFree(d_b_worker);
    cudaFree(d_b_entrep);
}

void free_entrep_grids(Real* d_income_entrep, Real* d_tau) {
    cudaFree(d_income_entrep);
    cudaFree(d_tau);
}

void init_b_state(Real** d_b_worker, Real** d_b_entrep, int n_types) {
    CUDA_CHECK(cudaMalloc(d_b_worker, 1 * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(d_b_entrep, n_types * sizeof(Real)));
}

void init_device_grids(Real** d_K, const Real* K, int n_k,
    Real** d_income, int workingYears) {
    CUDA_CHECK(cudaMalloc(d_K, n_k * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(*d_K, K, n_k * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(d_income, workingYears * sizeof(Real)));
}

void init_entrep_grids(Real** d_income_entrep, int n_types, int workingYears,
    Real** d_tau, const Real* tau, int n_tau) {
    CUDA_CHECK(cudaMalloc(d_income_entrep, n_types * workingYears * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(d_tau, n_tau * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(*d_tau, tau, n_tau * sizeof(Real), cudaMemcpyHostToDevice));
}


struct AbsDiff {
    __host__ __device__
        Real operator()(const thrust::tuple<Real, Real>& t) const noexcept {
        Real a = thrust::get<0>(t);
        Real b = thrust::get<1>(t);
        Real d = a - b;
        return d >= Real(0) ? d : -d;
    }
};

Real cuda_max_abs_diff(DeviceBlockArrays& d_blk) {

    const std::size_t N = d_blk.total();
    if (N == 0) return Real(0);

    // wrap raw device pointers
    thrust::device_ptr<Real> p_new(d_blk.V_new);
    thrust::device_ptr<Real> p_old(d_blk.V_old);

    // zip iterator [ (V_new[0], V_old[0]) ... (V_new[N-1], V_old[N-1]) ]
    auto first = thrust::make_zip_iterator(thrust::make_tuple(p_new, p_old));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(p_new + N, p_old + N));

    // initial value must be of type Real
    Real init = Real(0);

    // compute maximum absolute difference
    Real diff = thrust::transform_reduce(first, last, AbsDiff(), init, thrust::maximum<Real>());

    return diff;
}


void download_for_macros(DeviceBlockArrays& d_blk, BlockArrays& blk)
{
    const size_t N = d_blk.total();
    const size_t bytes_r = N * sizeof(Real);
    const size_t bytes_p = N * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(blk.V_old.data(), d_blk.V_old, bytes_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blk.policy.data(), d_blk.policy, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blk.cons.data(), d_blk.cons, bytes_r, cudaMemcpyDeviceToHost));
}



bool cuda_roundtrip_test(const std::vector<Real>& K,
    const std::vector<Real>& V_old)
{
    const size_t nk = K.size();
    const size_t bytes_K = nk * sizeof(Real);
    const size_t bytes_V = V_old.size() * sizeof(Real);

    // --- allocate device memory ---
    Real* d_K = nullptr;
    Real* d_V = nullptr;
    CUDA_CHECK(cudaMalloc(&d_K, bytes_K));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_V));

    // --- host -> device ---
    CUDA_CHECK(cudaMemcpy(d_K, K.data(), bytes_K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V_old.data(), bytes_V, cudaMemcpyHostToDevice));

    // --- device -> host (into fresh buffers) ---
    std::vector<Real> K_back(nk);
    std::vector<Real> V_back(V_old.size());
    CUDA_CHECK(cudaMemcpy(K_back.data(), d_K, bytes_K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V_back.data(), d_V, bytes_V, cudaMemcpyDeviceToHost));

    // --- free device memory ---
    cudaFree(d_K);
    cudaFree(d_V);

    // --- verify round-trip ---
    bool ok = true;
    for (size_t i = 0; i < nk; ++i) {
        if (K_back[i] != K[i]) {
            printf("K mismatch at i=%zu: expected %f got %f\n", i, K[i], K_back[i]);
            ok = false;
        }
    }
    for (size_t i = 0; i < V_old.size(); ++i) {
        if (V_back[i] != V_old[i]) {
            printf("V mismatch at i=%zu: expected %f got %f\n", i, V_old[i], V_back[i]);
            ok = false;
        }
    }

    if (ok) printf("CUDA round-trip test passed. K size=%zu, V size=%zu\n",
        nk, V_old.size());
    return ok;
}



__global__ void kernel_bellman_retirement(
    const Real* K,
    const Real* V_old,      // num_states * retireYears * n_k
    const Real* b_state,    // num_states (one benefit per state)
    Real* V_new,
    uint16_t* policy,
    Real* cons,
    Real inv_1pr, Real P, Real beta,
    int n_k, int retireYears, int num_states)
{
    const int total = num_states * retireYears * n_k;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // recover (sid, y, i) from flat index
    // layout: sid * retireYears * n_k + y * n_k + i
    const int sid = tid / (retireYears * n_k);
    const int rem = tid % (retireYears * n_k);
    const int y = rem / n_k;
    const int i = rem % n_k;

    const Real n = K[i];
    const Real b = b_state[sid];

    // last year: no savings decision, consume everything
    if (y == retireYears - 1) {
        const Real c = (n + b) / P;
        V_new[tid] = (c > Real(0)) ? log(c) : NEG_INF;
        policy[tid] = 0;
        cons[tid] = c;
        return;
    }

    Real     best = NEG_INF;
    uint16_t bestj = 0;
    Real     bestc = Real(0);

    for (int j = 0; j < n_k; ++j) {
        // V_old index: sid * retireYears * n_k + (y+1) * n_k + j
        const Real Vnext = V_old[sid * retireYears * n_k + (y + 1) * n_k + j];
        const Real c = (n + b - K[j] * inv_1pr) / P;
        const Real val = (c > Real(0) ? log(c) : NEG_INF) + beta * Vnext;

        if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
    }

    V_new[tid] = best;
    policy[tid] = bestj;
    cons[tid] = bestc;
}


void cuda_bellman_retirement(
    DeviceBlockArrays& d_ret,
    const Real* d_K,
    const Real* d_b_state,  // device pointer to benefit array
    Real inv_1pr, Real P, Real beta,
    int n_k, int retireYears, int num_states,
    BlockArrays& ret)        // host reference for download
{
    const int total = num_states * retireYears * n_k;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    kernel_bellman_retirement <<<gridSize, blockSize >>> (
        d_K,
        d_ret.V_old,
        d_b_state,
        d_ret.V_new,
        d_ret.policy,
        d_ret.cons,
        inv_1pr, P, beta,
        n_k, retireYears, num_states
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    // download results to host
    const size_t bytes_r = size_t(total) * sizeof(Real);
    const size_t bytes_p = size_t(total) * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(ret.V_new.data(), d_ret.V_new, bytes_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ret.policy.data(), d_ret.policy, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ret.cons.data(), d_ret.cons, bytes_r, cudaMemcpyDeviceToHost));
}



__global__ void kernel_bellman_entrep_working(
    const Real* K,
    const Real* V_working_old,  // num_states * workingYears * n_k
    const Real* V_retire_old,   // num_states * retireYears  * n_k
    const Real* income_entrep,  // num_states * workingYears
    const Real* tau,            // n_tau
    Real* V_new,
    uint16_t* policy,
    Real* cons,
    Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears, int retireYears, int num_states, int n_tau)
{
    const int total = num_states * workingYears * n_k;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // recover (sid, y, i) from flat index
    const int sid = tid / (workingYears * n_k);
    const int rem = tid % (workingYears * n_k);
    const int y = rem / n_k;
    const int i = rem % n_k;

    // tau index is the remainder of sid / n_tau
    const int it = sid % n_tau;
    const Real tau_i = tau[it];

    // income for this (sid, y)
    const Real income = income_entrep[sid * workingYears + y];

    const Real n = K[i];
    const Real base = n + (Real(1) - tau_i) * income - T;

    Real     best = NEG_INF;
    uint16_t bestj = 0;
    Real     bestc = Real(0);

    for (int j = 0; j < n_k; ++j) {
        // last working year looks into retirement y=0, same sid
        const Real Vnext = (y == workingYears - 1)
            ? V_retire_old[sid * retireYears * n_k + j]
            : V_working_old[sid * workingYears * n_k + (y + 1) * n_k + j];

        const Real c = (base - K[j] * inv_1pr) / P;
        const Real val = (c > Real(0) ? log(c) : NEG_INF) + beta * Vnext;

        if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
    }

    V_new[tid] = best;
    policy[tid] = bestj;
    cons[tid] = bestc;
}


void cuda_bellman_entrep_working(
    DeviceBlockArrays& d_entrep_w,
    DeviceBlockArrays& d_entrep_r,
    const Real* d_K,
    const Real* d_income_entrep,
    const Real* d_tau,
    Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears, int retireYears, int num_states, int n_tau,
    BlockArrays& entrep_w)
{
    const int total = num_states * workingYears * n_k;
    const int blockSize = 256;
    const int gridSize = (total + blockSize - 1) / blockSize;

    kernel_bellman_entrep_working << <gridSize, blockSize >> > (
        d_K,
        d_entrep_w.V_old,
        d_entrep_r.V_old,
        d_income_entrep,
        d_tau,
        d_entrep_w.V_new,
        d_entrep_w.policy,
        d_entrep_w.cons,
        inv_1pr, P, beta, T,
        n_k, workingYears, retireYears, num_states, n_tau
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t bytes_r = size_t(total) * sizeof(Real);
    const size_t bytes_p = size_t(total) * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(entrep_w.V_new.data(), d_entrep_w.V_new, bytes_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(entrep_w.policy.data(), d_entrep_w.policy, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(entrep_w.cons.data(), d_entrep_w.cons, bytes_r, cudaMemcpyDeviceToHost));
}