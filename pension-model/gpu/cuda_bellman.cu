
//manual compile commands
//nvcc -c cuda_bellman.cu -o cuda_bellman.obj
#include "cuda_bellman.cuh"
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>



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



__global__ void kernel_bellman_worker_working_fast(
    const Real* __restrict__ K,
    const Real* __restrict__ V_working_old,
    const Real* __restrict__ V_retire_old,
    const Real* __restrict__ income,
    Real* __restrict__ V_new,
    uint16_t* __restrict__ policy,
    Real* __restrict__ cons,
    Real tau_avg, Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears)
{
    const int bid = blockIdx.x;
    const int total = workingYears * n_k;
    if (bid >= total) return;

    const int y = bid / n_k;
    const int i = bid % n_k;

    const int thread = threadIdx.x;
    const int bDim = blockDim.x;

    const Real base = K[i] + (Real(1) - tau_avg) * income[y] - T;

    Real     local_best_v = NEG_INF;
    uint16_t local_best_j = 0;
    Real     local_best_c = Real(0);

    for (int j = thread; j < n_k; j += bDim) {
        const Real Vnext = (y == workingYears - 1)
            ? V_retire_old[j]
            : V_working_old[(y + 1) * n_k + j];

        const Real c = (base - K[j] * inv_1pr) / P;
        const Real val = (c > Real(0) ? logf(c) : NEG_INF) + beta * Vnext;

        if (val > local_best_v) {
            local_best_v = val;
            local_best_j = (uint16_t)j;
            local_best_c = c;
        }
    }

    // --- Warp shuffle reduction ---
    const unsigned FULL = 0xffffffffu;
    const int W = 32;
    const int numWarps = (bDim + W - 1) / W;
    const int lane = thread & (W - 1);
    const int warpId = thread >> 5;

    Real     rv = local_best_v;
    uint16_t rj = local_best_j;
    Real     rc = local_best_c;

    for (int offset = 16; offset > 0; offset >>= 1) {
        Real     v_other = __shfl_down_sync(FULL, rv, offset);
        uint16_t j_other = (uint16_t)__shfl_down_sync(FULL, (int)rj, offset);
        Real     c_other = __shfl_down_sync(FULL, rc, offset);
        if (v_other > rv) { rv = v_other; rj = j_other; rc = c_other; }
    }

    extern __shared__ unsigned char s_mem[];
    Real* w_vals = reinterpret_cast<Real*>(s_mem);
    int* w_js = reinterpret_cast<int*>(w_vals + numWarps);
    Real* w_cons = reinterpret_cast<Real*>(w_js + numWarps);

    if (lane == 0) {
        w_vals[warpId] = rv;
        w_js[warpId] = (int)rj;
        w_cons[warpId] = rc;
    }
    __syncthreads();

    if (warpId == 0) {
        Real vv = (thread < numWarps) ? w_vals[thread] : NEG_INF;
        int  jj = (thread < numWarps) ? w_js[thread] : 0;
        Real cc = (thread < numWarps) ? w_cons[thread] : Real(0);

        for (int offset = 16; offset > 0; offset >>= 1) {
            Real v_other = __shfl_down_sync(FULL, vv, offset);
            int  j_other = __shfl_down_sync(FULL, jj, offset);
            Real c_other = __shfl_down_sync(FULL, cc, offset);
            if (v_other > vv) { vv = v_other; jj = j_other; cc = c_other; }
        }

        if (thread == 0) {
            V_new[bid] = vv;
            policy[bid] = (uint16_t)jj;
            cons[bid] = cc;
        }
    }
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

    const int total = workingYears * n_k;
    const int blockSize = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    if (n_k <= 256) {
        const int gridSize = (total + blockSize - 1) / blockSize;
        kernel_bellman_worker_working << < gridSize, blockSize >> > (
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
    }
    else {
        // shared memory: 3 arrays of numWarps reals/ints
        const int gridSize = total;
        const int numWarps = (blockSize + 31) / 32; // = 4
        const int shmem = numWarps * (2 * sizeof(Real) + sizeof(int));
        kernel_bellman_worker_working_fast << <gridSize, blockSize, shmem >> > (
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
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("worker_working kernel: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

__global__ void kernel_bellman_retirement_fast(
    const Real* __restrict__ K,
    const Real* __restrict__ V_old,
    const Real* __restrict__ b_state,
    Real* __restrict__ V_new,
    uint16_t* __restrict__ policy,
    Real* __restrict__ cons,
    Real inv_1pr, Real P, Real beta,
    int n_k, int retireYears, int num_states)
{
    const int bid = blockIdx.x;
    const int total = num_states * retireYears * n_k;
    if (bid >= total) return;

    const int sid = bid / (retireYears * n_k);
    const int rem = bid % (retireYears * n_k);
    const int y = rem / n_k;
    const int i = rem % n_k;

    const Real b = b_state[sid];
    const Real base = K[i] + b;

    // Last year: no savings decision, only thread 0 needs to write
    if (y == retireYears - 1) {
        if (threadIdx.x == 0) {
            const Real c = base / P;
            V_new[bid] = (c > Real(0)) ? logf(c) : NEG_INF;
            policy[bid] = 0;
            cons[bid] = c;
        }
        return;
    }

    const int thread = threadIdx.x;
    const int bDim = blockDim.x;

    // Each thread finds best j in its stride
    Real     local_best_v = NEG_INF;
    uint16_t local_best_j = 0;
    Real     local_best_c = Real(0);

    for (int j = thread; j < n_k; j += bDim) {
        const Real Vnext = V_old[sid * retireYears * n_k + (y + 1) * n_k + j];
        const Real c = (base - K[j] * inv_1pr) / P;
        const Real val = (c > Real(0) ? logf(c) : NEG_INF) + beta * Vnext;

        if (val > local_best_v) {
            local_best_v = val;
            local_best_j = (uint16_t)j;
            local_best_c = c;
        }
    }

    // --- Warp shuffle reduction ---
    const unsigned FULL = 0xffffffffu;
    const int W = 32;
    const int numWarps = (bDim + W - 1) / W;
    const int lane = thread & (W - 1);
    const int warpId = thread >> 5;

    Real     rv = local_best_v;
    uint16_t rj = local_best_j;
    Real     rc = local_best_c;

    for (int offset = 16; offset > 0; offset >>= 1) {
        Real     v_other = __shfl_down_sync(FULL, rv, offset);
        uint16_t j_other = (uint16_t)__shfl_down_sync(FULL, (int)rj, offset);
        Real     c_other = __shfl_down_sync(FULL, rc, offset);
        if (v_other > rv) { rv = v_other; rj = j_other; rc = c_other; }
    }

    extern __shared__ unsigned char s_mem[];
    Real* w_vals = reinterpret_cast<Real*>(s_mem);
    int* w_js = reinterpret_cast<int*>(w_vals + numWarps);
    Real* w_cons = reinterpret_cast<Real*>(w_js + numWarps);

    if (lane == 0) {
        w_vals[warpId] = rv;
        w_js[warpId] = (int)rj;
        w_cons[warpId] = rc;
    }
    __syncthreads();

    if (warpId == 0) {
        Real vv = (thread < numWarps) ? w_vals[thread] : NEG_INF;
        int  jj = (thread < numWarps) ? w_js[thread] : 0;
        Real cc = (thread < numWarps) ? w_cons[thread] : Real(0);

        for (int offset = 16; offset > 0; offset >>= 1) {
            Real v_other = __shfl_down_sync(FULL, vv, offset);
            int  j_other = __shfl_down_sync(FULL, jj, offset);
            Real c_other = __shfl_down_sync(FULL, cc, offset);
            if (v_other > vv) { vv = v_other; jj = j_other; cc = c_other; }
        }

        if (thread == 0) {
            V_new[bid] = vv;
            policy[bid] = (uint16_t)jj;
            cons[bid] = cc;
        }
    }
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);


    if (n_k <= 256) {
        const int gridSize = (total + blockSize - 1) / blockSize;
        kernel_bellman_retirement << < gridSize, blockSize >> > (
            d_K, d_ret.V_old, d_b_state,
            d_ret.V_new, d_ret.policy, d_ret.cons,
            inv_1pr, P, beta,
            n_k, retireYears, num_states
            );
    }
    else {
        // shared memory: 3 arrays of numWarps reals/ints
        const int gridSize = total;
        const int numWarps = (blockSize + 31) / 32; // = 4
        const int shmem = numWarps * (2 * sizeof(Real) + sizeof(int));
        kernel_bellman_retirement_fast << <gridSize, blockSize, shmem >> > (
            d_K, d_ret.V_old, d_b_state,
            d_ret.V_new, d_ret.policy, d_ret.cons,
            inv_1pr, P, beta,
            n_k, retireYears, num_states
            );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("retirement kernel: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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


// One block per (sid, y, i) triple
// blockDim.x = 128 or 256 threads cover the j dimension
__global__ void kernel_bellman_entrep_working_fast(
    const Real* __restrict__ K,
    const Real* __restrict__ V_working_old,  // num_states * workingYears * n_k
    const Real* __restrict__ V_retire_old,   // num_states * retireYears  * n_k
    const Real* __restrict__ income_entrep,  // num_states * workingYears
    const Real* __restrict__ tau,            // n_tau
    Real* __restrict__ V_new,
    uint16_t* __restrict__ policy,
    Real* __restrict__ cons,
    Real inv_1pr, Real P, Real beta, Real T,
    int n_k, int workingYears, int retireYears, int num_states, int n_tau)
{
    // Grid: blockIdx.x = flat index over (sid, y, i)
    // Block: threadIdx.x covers the j dimension
    const int bid = blockIdx.x;
    const int total = num_states * workingYears * n_k;
    if (bid >= total) return;

    const int sid = bid / (workingYears * n_k);
    const int rem = bid % (workingYears * n_k);
    const int y = rem / n_k;
    const int i = rem % n_k;

    const int thread = threadIdx.x;
    const int bDim = blockDim.x;

    const int   it = sid % n_tau;
    const Real  tau_i = tau[it];
    const Real  income = income_entrep[sid * workingYears + y];
    const Real  base = K[i] + (Real(1) - tau_i) * income - T;

    // Each thread finds best j in its stride
    Real     local_best_v = NEG_INF;
    uint16_t local_best_j = 0;
    Real     local_best_c = Real(0);

    for (int j = thread; j < n_k; j += bDim) {
        const Real Vnext = (y == workingYears - 1)
            ? V_retire_old[sid * retireYears * n_k + j]
            : V_working_old[sid * workingYears * n_k + (y + 1) * n_k + j];

        const Real c = (base - K[j] * inv_1pr) / P;
        const Real val = (c > Real(0) ? logf(c) : NEG_INF) + beta * Vnext;

        if (val > local_best_v) {
            local_best_v = val;
            local_best_j = (uint16_t)j;
            local_best_c = c;
        }
    }

    // --- Warp shuffle reduction ---
    const unsigned FULL = 0xffffffffu;
    const int W = 32;
    const int numWarps = (bDim + W - 1) / W;
    const int lane = thread & (W - 1);
    const int warpId = thread >> 5;

    // Registers for reduction
    Real     rv = local_best_v;
    uint16_t rj = local_best_j;
    Real     rc = local_best_c;

    // Step 1: reduce within each warp using shuffles
    for (int offset = 16; offset > 0; offset >>= 1) {
        Real     v_other = __shfl_down_sync(FULL, rv, offset);
        uint16_t j_other = __shfl_down_sync(FULL, (int)rj, offset);
        Real     c_other = __shfl_down_sync(FULL, rc, offset);
        if (v_other > rv) {
            rv = v_other;
            rj = j_other;
            rc = c_other;
        }
    }

    // Step 2: lane 0 of each warp writes its result to shared memory
    extern __shared__ unsigned char s_mem[];
    Real* w_vals = reinterpret_cast<Real*>(s_mem);
    int* w_js = reinterpret_cast<int*>(w_vals + numWarps);
    Real* w_cons = reinterpret_cast<Real*>(w_js + numWarps);

    if (lane == 0) {
        w_vals[warpId] = rv;
        w_js[warpId] = (int)rj;
        w_cons[warpId] = rc;
    }
    __syncthreads();

    // Step 3: warp 0 reduces across warp results
    if (warpId == 0) {
        Real vv = (thread < numWarps) ? w_vals[thread] : NEG_INF;
        int  jj = (thread < numWarps) ? w_js[thread] : 0;
        Real cc = (thread < numWarps) ? w_cons[thread] : Real(0);

        for (int offset = 16; offset > 0; offset >>= 1) {
            Real v_other = __shfl_down_sync(FULL, vv, offset);
            int  j_other = __shfl_down_sync(FULL, jj, offset);
            Real c_other = __shfl_down_sync(FULL, cc, offset);
            if (v_other > vv) {
                vv = v_other;
                jj = j_other;
                cc = c_other;
            }
        }

        if (thread == 0) {
            V_new[bid] = vv;
            policy[bid] = (uint16_t)jj;
            cons[bid] = cc;
        }
    }
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


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    if (n_k <= 256) {
		const int gridSize = (total + blockSize - 1) / blockSize;
        kernel_bellman_entrep_working <<< gridSize, blockSize >>>(
            d_K, d_entrep_w.V_old, d_entrep_r.V_old,
            d_income_entrep, d_tau,
            d_entrep_w.V_new, d_entrep_w.policy, d_entrep_w.cons,
            inv_1pr, P, beta, T,
            n_k, workingYears, retireYears, num_states, n_tau
            );
    }
    else {
        // shared memory: 3 arrays of numWarps reals/ints
        const int gridSize = total;
        const int numWarps = (blockSize + 31) / 32; // = 4
        const int shmem = numWarps * (2 * sizeof(Real) + sizeof(int));
        kernel_bellman_entrep_working_fast <<<gridSize, blockSize, shmem >>> (
            d_K, d_entrep_w.V_old, d_entrep_r.V_old,
            d_income_entrep, d_tau,
            d_entrep_w.V_new, d_entrep_w.policy, d_entrep_w.cons,
            inv_1pr, P, beta, T,
            n_k, workingYears, retireYears, num_states, n_tau
            );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    //printf("entrep kernel: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


}





