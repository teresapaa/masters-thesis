#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>


#if defined(USE_REAL)
using Real = float;
constexpr const char* REAL_NAME = "float";
#else
using Real = double;
constexpr const char* REAL_NAME = "double";
#endif

static constexpr Real NEG_INF = Real(-1e30);

inline Real u_log(Real c) {
    if (c <= Real(0)) return NEG_INF;
    return std::log(c);
}


struct Params {
    Real beta = Real(0.98);
    Real theta = Real(5);
    Real B = Real(6000);

    int workingYears = 10;
    int retirementYears = 5;

    int n_k = 100;
    int n_tau = 15;
    int n_a = 15;

    Real Kmin = Real(0.0), Kmax = Real(50.0);
    Real tauMin = Real(0.01), tauMax = Real(0.06);
    Real amin = Real(0.5), amax = Real(5.0);

    int max_iters = 20000;
    int firm_update_every = 20;
    int check_every = 10;

    Real alpha_macro = Real(0.5);
    Real gamma_firm = Real(0.10);

    Real tol_V = Real(1e-4);
    Real tol_macro = Real(1e-4);
};


//Create and host grids for K, tau and a
struct Grids {
    std::vector<Real> K, tau, a;
    Real tau_avg = Real(0);

    explicit Grids(const Params& p) {
        K.resize(p.n_k);
        tau.resize(p.n_tau);
        a.resize(p.n_a);

        auto lin = [](std::vector<Real>& v, Real lo, Real hi) {
            const int n = (int)v.size();
            const Real step = (hi - lo) / Real(n - 1);
            for (int i = 0; i < n; ++i) v[i] = lo + Real(i) * step;
            };

        lin(K, p.Kmin, p.Kmax);
        lin(tau, p.tauMin, p.tauMax);
        lin(a, p.amin, p.amax);

        tau_avg = std::accumulate(tau.begin(), tau.end(), Real(0)) / Real(tau.size());
    }
};

inline int entrep_state_id(int a_idx, int tau_idx, int n_tau) {
    return a_idx * n_tau + tau_idx;
}

//Arrays for iteration
struct BlockArrays {
    int num_states = 0;
    int years = 0;
    int n_k = 0;

    std::vector<Real> V_old, V_new;
    std::vector<Real> S_old, S_new;
    std::vector<uint16_t> policy;
    std::vector<Real> cons;

    BlockArrays() = default;
    BlockArrays(int ns, int yrs, int nk) { reset(ns, yrs, nk); }

    void reset(int ns, int yrs, int nk) {
        num_states = ns; years = yrs; n_k = nk;
        const std::size_t N = std::size_t(ns) * std::size_t(yrs) * std::size_t(nk);
        V_old.assign(N, Real(0));
        V_new.assign(N, Real(0));
        S_old.assign(N, Real(0));
        S_new.assign(N, Real(0));
        policy.assign(N, 0);
        cons.assign(N, Real(0));
    }

    inline std::size_t idx(int s, int y, int i) const {
        return (std::size_t(s) * std::size_t(years) + std::size_t(y)) * std::size_t(n_k) + std::size_t(i);
    }

    void swap_old_new() {
        std::swap(V_old, V_new);
        std::swap(S_old, S_new);
    }
};

