
//manual compile commands
//nvcc -c cuda_bellman.cu -o cuda_bellman.obj
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
    printf("Bellman worker indeed working \n");
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
    printf("V_new[0]=%f V_new[50]=%f V_new[nk-1]=%f\n",
        (double)h_sample[0], (double)h_sample[1], (double)h_sample[2]);

    const size_t bytes_r = total * sizeof(Real);
    const size_t bytes_p = total * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(worker_w.V_new.data(), d_worker_w.V_new, bytes_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(worker_w.policy.data(), d_worker_w.policy, bytes_p, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(worker_w.cons.data(), d_worker_w.cons, bytes_r, cudaMemcpyDeviceToHost));
}

void upload_income(Real* d_income, const Real* income, int workingYears) {
    CUDA_CHECK(cudaMemcpy(d_income, income, workingYears * sizeof(Real), cudaMemcpyHostToDevice));
}


void free_device_grids(Real* d_K, Real* d_income) {
    cudaFree(d_K);
    cudaFree(d_income);
}

void init_device_grids(Real** d_K, const Real* K, int n_k,
    Real** d_income, int workingYears) {
    CUDA_CHECK(cudaMalloc(d_K, n_k * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(*d_K, K, n_k * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(d_income, workingYears * sizeof(Real)));
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

