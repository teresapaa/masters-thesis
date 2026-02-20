#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <numeric>
#include <cstdint>

//cl /std:c++20 /Zi /EHsc model.cpp

//Determine whether to compile as doubles or floats
#if defined(USE_REAL)
using Real = float;
constexpr const char* REAL_NAME = "float";
#else
using Real = double;
constexpr const char* REAL_NAME = "double";
#endif


// if stated as static constexpr this can be read form anywhere!xd 
static constexpr Real NEG_INF = Real(-1e30);

// Utility function with penalty for non-positive consumption
inline Real u_log(Real c) {
    // decide your penalty convention:
    if (c <= Real(0)) return NEG_INF;
    return std::log(c);
}

struct Params {
    //Model parameters 
    Real beta = Real(0.96);   
    Real theta = Real(5.0);

    // Bonds / government
    Real B = Real(200);         // debt portfolio

    // Time
    int workingYears = 10;
    int retirementYears = 5;

    // Grids
    int n_k = 100;
    int n_tau = 10;
    int n_a = 10;

    Real Kmin = Real(0.0), Kmax = Real(100.0);
    Real tauMin = Real(0.01), tauMax = Real(1.0);
    Real amin = Real(0.5), amax = Real(10.0);

    // Iteration controls
    //int max_inner_vfi = 2000;       // max number of value function iterations (inner)
    //int max_outer_iter = 200;       // max number of outer loop updates
    //Real vfi_tol = Real(1e-3);      // convergence tolerance for V
    //Real outer_tol = Real(1e-3);    // convergence tolerance for macro vars

    int max_iters = 20000;     // total Bellman sweeps allowed
    int firm_update_every = 10; // update firms every x sweeps
    int check_every = 10;       // convergence checks every N sweeps

    Real alpha_macro = Real(0.05); // damping for r,T,P,C
    Real gamma_firm = Real(0.10); // damping for firm_weight

    Real tol_V = Real(1e-6);
    Real tol_macro = Real(1e-6);

};


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


enum class Block : int {
    Worker = 0,            // working years only
    Entrep = 1,            // (a,tau) * working years
    RetWorker = 2,         // retirement years only
    RetEntrep = 3          // (a,tau) * retirement years
};

inline int entrep_state_id(int a_idx, int tau_idx, int n_tau) {
    return a_idx * n_tau + tau_idx; // 0..n_a*n_tau-1
}


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



struct Model {

    Params p;
    Grids g;


    // Blocks for all the different groups:
    // worker working: 1 x workingYears x n_k
    // entrepreneur working: (n_a*n_tau) x workingYears x n_k
    // retired worker: 1 x retirementYears x n_k
    // retired entrepreneur: (n_a*n_tau) x retirementYears x n_k
    BlockArrays worker_w;
    BlockArrays entrep_w;
    BlockArrays worker_r;
    BlockArrays entrep_r;

    std::vector<Real> income_worker;                 // size workingYears
    std::vector<Real> income_entrep;

    Real s_entry_worker = Real(0);                     // single worker type
    std::vector<Real> s_entry_entrep;

    // Entrepreneur membership weights (0/1)
    std::vector<Real> firm_weight; // size n_a*n_tau

    // Prices 
    std::vector<Real> prices;      // size n_a*n_tau

    // parameters
    Real T = Real(1);      // lump sum taxes (guess)
    Real r = Real(0.03);   // interest rate (guess)
    Real P = Real(1);      // aggregate price index (placeholder)
    Real C_agg = Real(100); // aggregate consumption (placeholder)
    Real l = Real(1);      // labor supply (placeholder)

    explicit Model(const Params& params)
        : p(params), g(p),
        worker_w(1, p.workingYears, p.n_k),
        entrep_w(p.n_a* p.n_tau, p.workingYears, p.n_k),
        worker_r(1, p.retirementYears, p.n_k),
        entrep_r(p.n_a* p.n_tau, p.retirementYears, p.n_k),
        firm_weight((std::size_t)p.n_a* p.n_tau, Real(0)),
        prices((std::size_t)p.n_a* p.n_tau, Real(0)),
        income_worker((std::size_t)p.workingYears, Real(0)),
        income_entrep((std::size_t)p.n_a* p.n_tau* p.workingYears, Real(0)),
        s_entry_entrep((std::size_t)p.n_a* p.n_tau, Real(0))
    {}


    inline std::size_t inc_idx(int sid, int y) const {
        return (std::size_t)sid * (std::size_t)p.workingYears + (std::size_t)y;
    }

    void init_educated_guess() {
        for (int it = 0; it < p.n_tau; ++it) {
            for (int ia = 0; ia < p.n_a; ++ia) {
                const bool guessFirm = (it < p.n_tau / 2) && (ia >= p.n_a / 2);
                firm_weight[(std::size_t)entrep_state_id(ia, it, p.n_tau)] = guessFirm ? Real(1) : Real(0);
            }
        }
    }

    void update_prices_from_firms() {
        // Your original: price = theta/(theta-1)/a
        // Only for entrepreneurs.
        std::fill(prices.begin(), prices.end(), Real(0));
        Real P_idx = 0;

        for (int it = 0; it < p.n_tau; ++it) {
            for (int ia = 0; ia < p.n_a; ++ia) {
                const int sid = entrep_state_id(ia, it, p.n_tau);
                if (firm_weight[(std::size_t)sid] > Real(0.5)) {
                    Real price = (p.theta / (p.theta - Real(1))) / g.a[ia];
                    prices[(std::size_t)sid] = price;
                    P_idx += std::pow(price, Real(1) - p.theta);
                }
            }
        }
        if (P_idx <= Real(0)) { P = Real(1); return; }
        P = std::pow(P_idx, Real(1) / (Real(1) - p.theta));
    
    }

    int update_entrepreneur_set(Real damping = Real(0.2)) {
        const Real Vw0 = worker_w.V_old[worker_w.idx(0, 0, 0)];

        int numFirms = 0;
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            const Real Ve0 = entrep_w.V_old[entrep_w.idx(sid, 0, 0)];
            const Real target = (Ve0 > Vw0) ? Real(1) : Real(0);

            firm_weight[(std::size_t)sid] =
                (Real(1) - damping) * firm_weight[(std::size_t)sid] + damping * target;

            if (firm_weight[(std::size_t)sid] > Real(0.5)) ++numFirms;
        }
        return numFirms;
    }

    struct ProfitCache {
        std::vector<Real> pi; // size n_a*n_tau (depends on a only here, but keep per sid for simplicity)
    };

    ProfitCache build_profit_cache() const {
        ProfitCache pc;
        pc.pi.assign((std::size_t)p.n_a * p.n_tau, Real(0));

        const Real th = p.theta;
        const Real k1 = std::pow(th - Real(1), th - Real(1)) / std::pow(th, th);
        const Real P_pow = std::pow(P, th);

        // Precompute a^(theta-1)
        std::vector<Real> a_pow(p.n_a);
        for (int ia = 0; ia < p.n_a; ++ia) a_pow[ia] = std::pow(g.a[ia], th - Real(1));

        for (int it = 0; it < p.n_tau; ++it) {
            for (int ia = 0; ia < p.n_a; ++ia) {
                const int sid = entrep_state_id(ia, it, p.n_tau);
                pc.pi[(std::size_t)sid] = k1 * a_pow[ia] * P_pow * C_agg;
            }
        }
        return pc;
    }


    void update_income_paths(const ProfitCache& pc) {
        // Worker income per year (example: l; replace with wage*l if you have wage)
        for (int y = 0; y < p.workingYears; ++y) {
            income_worker[(std::size_t)y] = l; // TODO: wage(y) * l, etc.
        }

        // Entrepreneur "income" per year: pi = base_pi * C_agg
        // If pi varies over time in your model, compute pi(y) here; otherwise constant over y.
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            const Real pi = pc.pi[(std::size_t)sid]; 
            for (int y = 0; y < p.workingYears; ++y) {
                income_entrep[inc_idx(sid, y)] = pi;
            }
        }
    }

    void bellman_worker_working() {
        const int nk = p.n_k;
        const Real inv_1pr = Real(1) / (Real(1) + r);
        const Real tau_avg = g.tau_avg;

        for (int y = p.workingYears - 1; y >= 0; --y) {

            Real income = income_worker[(std::size_t)y];
            for (int i = 0; i < nk; ++i) {
                const std::size_t cur = worker_w.idx(0, y, i);
                const Real n = g.K[i];

                Real best = NEG_INF;
                uint16_t bestj = 0;
                Real bestc = Real(0);

                for (int j = 0; j < nk; ++j) {
                    const Real Vnext = (y == p.workingYears - 1)
                        ? worker_r.V_old[worker_r.idx(0, 0, j)]
                        : worker_w.V_old[worker_w.idx(0, y + 1, j)];

                    const Real n_next = g.K[j];

                    // Worker consumption (your original structure)
                    // c = n + (1-tau_avg)*e - T - n_next/(1+r)
                    // TODO: If you have wage * l, put it here.
                    const Real c = (n + (Real(1) - tau_avg) * income - T - n_next * inv_1pr) / P;


                    const Real val = u_log(c) + p.beta * Vnext;
                    if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
                }

                worker_w.V_new[cur] = best;
                worker_w.policy[cur] = bestj;
                worker_w.cons[cur] = bestc;

            }
        }
    }





    void bellman_entrep_working(const ProfitCache& pc) {
        const int nk = p.n_k;
        const Real inv_1pr = Real(1) / (Real(1) + r);

        for (int y = p.workingYears - 1; y >= 0; --y) {
            for (int it = 0; it < p.n_tau; ++it) {
                const Real tau_i = g.tau[it];

                for (int ia = 0; ia < p.n_a; ++ia) {
                    const int sid = entrep_state_id(ia, it, p.n_tau);
                    Real income = income_entrep[inc_idx(sid, y)];

                    for (int i = 0; i < nk; ++i) {
                        const std::size_t cur = entrep_w.idx(sid, y, i);
                        const Real n = g.K[i];

                        Real best = NEG_INF;
                        uint16_t bestj = 0;
                        Real bestc = Real(0);

                        for (int j = 0; j < nk; ++j) {
                            const Real Vnext = (y == p.workingYears - 1)
                                ? entrep_r.V_old[entrep_r.idx(sid, 0, j)]
                                : entrep_w.V_old[entrep_w.idx(sid, y + 1, j)];

                            const Real n_next = g.K[j];

                            // Entrepreneur consumption (your original structure)
                            // c = n + (1-tau)*pi - T - n_next/(1+r)
                            const Real c = (n + (Real(1) - tau_i) * income - T - n_next * inv_1pr) / P;

                            const Real val = u_log(c) + p.beta * Vnext;
                            if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
                        }

                        entrep_w.V_new[cur] = best;
                        entrep_w.policy[cur] = bestj;
                        entrep_w.cons[cur] = bestc;

                    }
                }
            }
        }
    }

    int update_s_entries_and_total_b(std::vector<Real>& s_entry_entrep,
        Real& s_entry_worker,
        int& num_firms,
        int& num_workers) {

        Real total_b_per_period = 0;

        // 1) Count firms / workers (type counts)
        num_firms = 0;
        const int totalTypes = p.n_a * p.n_tau;
        for (int sid = 0; sid < totalTypes; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) ++num_firms;
        }
        num_workers = totalTypes - num_firms;

        Real total_b_worker = 0.0;

        // 2) Compute and store s_entry for worker (no asset dependence)
        {
            Real s = Real(0);
            for (int y = 0; y < p.workingYears; ++y) {
                // Your rule: s_{t+1} = (1+r)*s_t + tau_avg * income_y
                Real new_s = g.tau_avg * income_worker[(std::size_t)y];
                s = (Real(1) + r) * s + new_s;
                total_b_worker += new_s;
            }
            s_entry_worker = s;
        }

        total_b_per_period = total_b_worker * num_workers;

        // 3) Compute and store s_entry for each entrepreneur type sid (no asset dependence)
        // Also accumulate total SS inflow from firms per period (for total_b_per_period)

        for (int it = 0; it < p.n_tau; ++it) {
            const Real tau_i = g.tau[it];
            for (int ia = 0; ia < p.n_a; ++ia) {
                const int sid = entrep_state_id(ia, it, p.n_tau);

                bool is_entrep = true;
                // Only matters if this type is an entrepreneur (optional but saves work)
                if (firm_weight[(std::size_t)sid] <= Real(0.5)) {
                    is_entrep = false;
                }

                Real s = Real(0);
                for (int y = 0; y < p.workingYears; ++y) {
                    const Real inc = income_entrep[inc_idx(sid, y)];
                    Real new_s = tau_i * inc;
                    s = (Real(1) + r) * s + new_s;  // entrepreneur SS paid on income (inc)
                    if (is_entrep) { total_b_per_period += new_s; }
                }
                s_entry_entrep[(std::size_t)sid] = s;
            }
        }
        return total_b_per_period;

    }


    void build_b_state(const std::vector<Real>& s_entry_entrep,
        Real s_entry_worker,
        Real total_b,
        int num_firms,
        int num_workers,
        std::vector<Real>& b_worker_state,
        std::vector<Real>& b_entrep_state)
    {

        Real sum_entrep_s = Real(0);
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) {
                sum_entrep_s += s_entry_entrep[(std::size_t)sid];
            }
        }

        Real retiree_s = (s_entry_worker * num_workers + sum_entrep_s) * p.retirementYears;

        if (!(retiree_s > Real(0)) || !std::isfinite(retiree_s) || !std::isfinite(total_b))
        {
            b_worker_state[0] = Real(0);
            std::fill(b_entrep_state.begin(),b_entrep_state.end(),Real(0));
            return;
        }


        Real bW = s_entry_worker * total_b / retiree_s;
        if (!std::isfinite(bW) || bW < Real(0)) bW = Real(0);
        b_worker_state[0] = bW;


        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            Real bE = s_entry_entrep[(std::size_t)sid] * total_b / retiree_s;
            if (!std::isfinite(bE) || bE < Real(0)) bE = Real(0);
            b_entrep_state[(std::size_t)sid] = bE;
        }
    }



    // ------------------------------------------------------------
    // Bellman: retirement block with constant per-state benefit b
    // Consumption: c = (n + b - n_next/(1+r)) / P
    // Last period: consume everything (n + b) / P
    // ------------------------------------------------------------
    void bellman_retirement(BlockArrays& ret, const std::vector<Real>& b_state) {
        const int nk = p.n_k;
        const Real inv_1pr = Real(1) / (Real(1) + r);

        for (int y = p.retirementYears - 1; y >= 0; --y) {
            for (int sid = 0; sid < ret.num_states; ++sid) {
                const Real b = b_state[(std::size_t)sid];

                for (int i = 0; i < nk; ++i) {
                    const std::size_t cur = ret.idx(sid, y, i);
                    const Real n = g.K[i];

                    if (y == p.retirementYears - 1) {
                        const Real c = (n + b) / P;
                        ret.V_new[cur] = u_log(c);
                        ret.policy[cur] = (uint16_t)i;
                        ret.cons[cur] = c;
                        continue;
                    }

                    Real best = NEG_INF;
                    uint16_t bestj = 0;
                    Real bestc = Real(0);

                    for (int j = 0; j < nk; ++j) {
                        const Real Vnext = ret.V_old[ret.idx(sid, y + 1, j)];
                        const Real n_next = g.K[j];

                        const Real c = (n + b - n_next * inv_1pr) / P;
                        const Real val = u_log(c) + p.beta * Vnext;

                        if (val > best) { best = val; bestj = (uint16_t)j; bestc = c; }
                    }

                    ret.V_new[cur] = best;
                    ret.policy[cur] = bestj;
                    ret.cons[cur] = bestc;
                }
            }
        }
    }

    //One iteration of the vfi loops
    Real bellman_one_iter() {
        const auto pc = build_profit_cache();

        update_income_paths(pc);

        bellman_worker_working();
        bellman_entrep_working(pc);

        // retirement depends on b_state -> compute it each iteration
        std::vector<Real> s_entry_entrep((size_t)p.n_a * p.n_tau, Real(0));
        Real s_entry_worker = Real(0);
        int num_firms = 0, num_workers = 0;
        Real total_b = update_s_entries_and_total_b(s_entry_entrep, s_entry_worker, num_firms, num_workers);

        std::vector<Real> b_worker_state(1, Real(0));
        std::vector<Real> b_entrep_state((size_t)p.n_a * p.n_tau, Real(0));
        build_b_state(s_entry_entrep, s_entry_worker, total_b,
            num_firms, num_workers,
            b_worker_state, b_entrep_state);
        auto mm = std::minmax_element(b_entrep_state.begin(), b_entrep_state.end());


        bellman_retirement(worker_r, b_worker_state);
        bellman_retirement(entrep_r, b_entrep_state);

        // compute V diff and swap
        Real max_diff = Real(0);
        auto diff_and_swap = [&](BlockArrays& blk) {
            for (std::size_t k = 0; k < blk.V_old.size(); ++k) {
                max_diff = std::max(max_diff, (Real)std::abs(blk.V_new[k] - blk.V_old[k]));
            }
            blk.swap_old_new();
            };

        diff_and_swap(worker_w);
        diff_and_swap(entrep_w);
        diff_and_swap(worker_r);
        diff_and_swap(entrep_r);

        return max_diff;
    }

    struct Totals {
        Real assets;
        Real consumption;
    };

    Totals compute_totals()
    {
        Totals out{ Real(0), Real(0) };
        const int totalTypes = p.n_a * p.n_tau;

        // --------------------------------------------------
        // Count firms / workers
        // --------------------------------------------------
        int num_firms = 0;
        for (int sid = 0; sid < totalTypes; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) ++num_firms;
        }
        const int num_workers = totalTypes - num_firms;


        // --------------------------------------------------
        // 1) Workers
        // --------------------------------------------------
        {
            int i = 0;
            for (int y = 0; y < p.workingYears; ++y) {
                const std::size_t cur = worker_w.idx(0, y, i);
                out.assets += Real(num_workers) * g.K[i];
                out.consumption += Real(num_workers) * worker_w.cons[cur];
                i = (int)worker_w.policy[cur];
            }
            for (int y = 0; y < p.retirementYears; ++y) {
                const std::size_t cur = worker_r.idx(0, y, i);
                out.assets += Real(num_workers) * g.K[i];
                Real c = worker_r.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in worker_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += Real(num_workers) * c;
                i = (int)worker_r.policy[cur];
            }
        }

        // --------------------------------------------------
        // 2) Entrepreneurs
        // --------------------------------------------------
        for (int sid = 0; sid < totalTypes; ++sid) {
            if (firm_weight[(std::size_t)sid] <= Real(0.5)) continue;

            int i = 0;
            for (int y = 0; y < p.workingYears; ++y) {
                const std::size_t cur = entrep_w.idx(sid, y, i);
                out.assets += g.K[i];
                Real c = entrep_w.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in entrep_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += c;
                i = (int)entrep_w.policy[cur];
            }
            for (int y = 0; y < p.retirementYears; ++y) {
                const std::size_t cur = entrep_r.idx(sid, y, i);
                out.assets += g.K[i];
                Real c = entrep_r.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in entrep_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += c;
                i = (int)entrep_r.policy[cur];
            }
        }

        return out;
    }

    struct RUpdater {
        bool have_prev = false;
        Real r_prev = 0, F_prev = 0;

        Real max_step = Real(0.005);     // clamp dr
        Real damp = Real(0.2);       // extra damping
        Real r_min = Real(-0.95);
        Real r_max = Real(1.0);       // pick something reasonable

        Real update(Real r, Real F) {
            Real r_new = r;

            if (have_prev) {
                Real denom = (F - F_prev);
                if (std::isfinite(denom) && std::abs(denom) > Real(1e-12)) {
                    r_new = r - F * (r - r_prev) / denom;
                }
                else {
                    // fallback: tiny step in right direction
                    r_new = r - Real(1e-6) * F;
                }
            }
            else {
                // first step fallback
                r_new = r - Real(1e-6) * F;
            }

            // clamp step
            Real dr = r_new - r;
            if (dr > max_step) dr = max_step;
            if (dr < -max_step) dr = -max_step;

            r_new = r + dr;
            // damp
            r_new = (Real(1) - damp) * r + damp * r_new;

            // bounds
            r_new = std::min(std::max(r_new, r_min), r_max);

            // store
            have_prev = true;
            r_prev = r;
            F_prev = F;

            return r_new;
        }
    };
    

    void solve() {

        init_educated_guess();
        update_prices_from_firms();
        RUpdater r_updater;

        Real last_r = r, last_T = T, last_C = C_agg, last_P = P;

        for (int iter = 1; iter <= p.max_iters; ++iter) {
            // 1) One Bellman sweep (operator uses current macros + firm set)
            Real Vdiff = bellman_one_iter();

            // 2) Update macros every sweep
            //update r


            // 3) Update firms only every x sweeps (soft)
            if (iter % p.firm_update_every == 0) {
                Totals totals = compute_totals();

                Real F = totals.assets - p.B;
                
                r = r_updater.update(r, F);

                Real total_T = p.n_tau * p.n_a * (p.workingYears) * T;
                Real T_agg = (Real(1) - p.alpha_macro) * total_T + p.alpha_macro * (r * p.B);
                //T = T_agg / (p.n_tau * p.n_a * (p.workingYears));
                T = (r * p.B) / (p.n_tau * p.n_a * (p.workingYears));
                C_agg = (Real(1) - p.alpha_macro) * C_agg + p.alpha_macro * totals.consumption/ (p.retirementYears + p.workingYears);
                update_entrepreneur_set(p.gamma_firm);
                update_prices_from_firms();
            }


            std::cout << "sweep " << iter
                << " Vdiff=" << (double)Vdiff
                << " r=" << (double)r
                << " T=" << (double)T
                << " P=" << (double)P
                << " C=" << (double)C_agg
               // << " dr=" << (double)dr
                //<< " dC=" << (double)dC
                << "\n";


            // 4) Convergence checks occasionally
            if (iter % p.check_every == 0) {
                const Real dr = std::abs(r - last_r);
                const Real dT = std::abs(T - last_T);
                const Real dP = std::abs(P - last_P);
                const Real dC = std::abs(C_agg - last_C);


                // stop when BOTH V and macros are stable
                if (Vdiff < p.tol_V && dr < p.tol_macro && dT < p.tol_macro && dP < p.tol_macro && dC < p.tol_macro) {
                    std::cout << "Converged.\n";
                    break;
                }

                last_r = r; last_T = T; last_P = P; last_C = C_agg;
            }
        }
    }
};



int main() {

    Params p;
    Model m(p);
    m.solve();
    return 0;
};





/*
//Indexing: type of person (entrepreneurs (a * tau) / worker / retired) -> time -> asset grid -> potential other things
static std::size_t V_idx(std::size_t tau, std::size_t a, boolean worker, boolean retired, std::size_t time, std::size_t asset_state, boolean s, int workingYears, int retirementYears, std::size_t n_k) {
    int idx = 0;
    if (worker) {
        if (retired) { idx += (n_tau * n_a + 1) * workingYears * n_k * 2 + (n_tau * n_a) * retirementYears * n_k * 2 + time * n_k + asset_state; }
        else { idx += (n_tau * n_a) * workingYears * n_k * 2 + time * n_k * 2 + asset_state; }
    }
    else if (retired) idx += (n_tau * n_a + 1) * workingYears * n_k * 2 + (a * tau + tau) * retirementYears * n_k * 2 + time * n_k + asset_state;
    else idx += (a * tau + tau) * workingYears * n_k * 2 + time * n_k * 2 + asset_state;
    if (s) idx += 1;
    return idx;
}

static std::size_t idx_v_to_k(std::size_t idx, std::size_t tau, std::size_t a, boolean worker, boolean retired, std::size_t time, std::size_t asset_state, boolean s, int workingYears, int retirementYears, std::size_t n_k) {
    if (worker) {
        if (retired) { return idx - (n_tau * n_a + 1) * workingYears * n_k * 2 - (n_tau * n_a) * retirementYears * n_k * 2 - time * n_k - asset_state; }
        else { return idx - (n_tau * n_a) * workingYears * n_k * 2 - time * n_k * 2 - asset_state; }
    }
    else if (retired) return idx - (n_tau * n_a + 1) * workingYears * n_k * 2 - (a * tau + tau) * retirementYears * n_k * 2 - time * n_k - asset_state;
    else return idx - (a * tau + tau) * workingYears * n_k * 2 - time * n_k * 2 - asset_state;
}


static std::size_t c_polixy_idx(std::size_t tau, std::size_t a, boolean worker, boolean retired, std::size_t time, std::size_t asset_state, int workingYears, int retirementYears, std::size_t n_k) {
    int idx = 0;
    if (worker) idx += (n_tau * n_a) * workingYears * n_k + time + asset_state;
    else if (retired) idx += (n_tau * n_a + 1) * workingYears * n_k + time * n_k + asset_state;
    else idx += (n_a * n_tau + n_tau) * n_k + time * n_k + asset_state;
    return idx;
}

//Initializations
Real beta = 0.96f; //annual discounting
Real Kmin = 0.5f; // lower bound of the state space
Real Kmax = 100.0f; // upper bound of the state space
int n_k = 100; //number of grid points for capital
Real tauMin = 0.01f; // lower bound of the state space
Real tauMax = 1.0f; // upper bound of the state space
int n_tau = 10; //number of grid points for capital
Real amin = 0.5f; // lower bound of the state space
Real amax = 10.0f; // upper bound of the state space
int n_a = 10; //number of grid points for capital
Real theta = 5.0; //parameter
Real B = 500 //debt portfolio
int workingYears = 10; //number of working years
int retirementYears = 5; //number of retirement years
int numOfPersonsYearly = n_a * n_tau;
int numOfPersonsToConsider = ((n_a * n_tau + 1) * (workingYears + retirementYears)) //kaikki yrittäjät ja työntekijä kaikille työikäisille + eläkeläiset kaikille eläkeikäisille

//Initialize values for the VF iteration loop
Real diff = std::numeric_limits<Real>::max();
int iteration = 0;
const int max_iter = 20000;
const Real NEG_INF = -std::numeric_limits<Real>::max();

//a grid points
std::vector<Real> a(n_a);
Real a_step = (amax - amin) / (n_a - 1);
for (int i = 0; i < n_a; ++i)  a[i] = amin + i * a_step;

//tau grid points
std::vector<Real> tau(n_k);
Real tau_step = (tauMax - tauMin) / (n_tau - 1);
for (int i = 0; i < n_tau; ++i)  tau[i] = tauMin + i * tau_step;
Real tau_avg = std::accumulate(tau.begin(), tau.end(), 0.0) / tau.size();

//Set the asset grid points
std::vector<Real> K(n_k);
Real K_step = (Kmax - Kmin) / (n_k - 1);
for (int i = 0; i < n_k; ++i)  K[i] = Kmin + i * K_step;

//V vektori, sis kaikkien henkilöiden arvofunktiot yrittäjänä putkeen + työntekijä + eläkeläinen
std::vector<Real> V_new(numOfPersonsToConsider * n_k * 2, 0.0);

//V_old vektori, edellisen iteroinnin arvofunktiot
std::vector<Real> V_old(numOfPersonsToConsider * n_k * 2, 0.0);

//T = lump sum taxes collected by the government
Real T = 1.0; //initial guess

//r = interest rate, guess initially
Real r = 0.03;

//Policy vector, joka kertoo kunkin henkilön valinnan
std::vector<Real> policy(numOfPersonsToConsider * n_k, 0.0);

//l vektori/funktio, joka kertoo työn tarjonnan eri s arvoilla: aloietaan vakiolla
Real l = 1.0;

int numOfFirms = 0;
//guess who becomes an entrepreneur and calculate the respective price
std::vector<int> prices(numOfPersonsYearly, 0); //
for (int i_tau=0; i_tau < n_tau/2; i_tau++) {
    for (int i_a = n_a/2; i_a < n_a; i_a++) {
        prices[i_a * n_tau + i_tau] = theta / (theta - 1) / a[i_a];
        numOfFirms += 1;
    }
}


Real numOfWorkers = (numOfPersonsYearly - numOfFirms) * workingYears;

//aggregate prices
Real P = std::accumulate(prices.begin(), prices.end(), 0.0);

Real C_aggregate = 10.0; //Aggregate consumption over all times and all ppl

//aggregate consumption vector (person * possible states)
std::vector<Real> consumption(numOfPersonsToConsider, 0.0);

std::vector<Real> profits((n_a* n_tau) * workingYears, 0.0);

//Value function iteration loop
do {
    //iteroi kuluttaja joka ajassa
    for (int y = 0; y < workingYears; y++) {

        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {

            int V_idx = V_idx(n_tau, n_a, true, false, y, i, false, workingYears, retirementYears, n_k);
            int current_maxidx = 0;
            Real current_max = NEG_INF, current_maxs = 0, current_maxe = 0, current_maxc = 0;

            //Go through all the possible transitions from i
            for (int j = 0; j < n_k; j++) {

                int next_idx = 0;
                if (y = workingYears - 1) {
                    next_idx = V_idx(n_tau, n_a, true, true, 0, j, false, workingYears, retirementYears, n_k);
                }
                else {
                    next_idx = V_idx(n_tau, n_a, true, false, y + 1, j, false, workingYears, retirementYears, n_k);
                }

                //Calculate worker consumption
                Real n = V[V_idx];
                Real n_next = V[next_idx];
                Real s = V[V_idx + 1];
                Real e = l;
                Real c = n + (1 - tau_avg) * e - T - n_next / (1 + r);

                //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K <- is this still true?
                //if (c <= 0) break;

                //Update the best value found so far
                Real value = log(c) + beta * V_old[j];
                if (value > current_max) {
                    current_max = value;
                    current_maxidx = next_idx;
                    current_maxs = s;
                    current_maxe = e;
                    current_maxc = c;
                }

            }

        }
        //s update
        Real next_s = (1 + r) * current_maxs + tau_avg * current_maxe;

        //update policy and value functions
        int c_policy_idx = c_policy_idx(n_tau, n_a, true, false, y, i, workingYears, retirementYears, n_k);
        policy[c_policy_idx] = current_maxidx;
        V_new[V_idx] = current_max;
        V_new[current_maxidx + 1] = next_s;
        consumption[c_policy_idx] = current_maxc;
    }


    //calculate total C, if it's larger than our guess, update
    int idx = c_policy_idx(n_tau, n_a, true, false, 0, 0, workingYears, retirementYears, n_k);
    C_worker_agg = 0;
    for (int y = 0; y < workingYears y++) {
        C_worker_agg += consumption[idx];
        idx = policy[idx];
    }

    C_worker_agg = C_worker_agg * numOfWorkers;

    if (C_aggregate < C_worker_agg * 4 / 3) {
        C_aggregate = C_worker_agg * 4 / 3;
    }

    //iterate entrepreneurs
    for (int a = 0; a < n_a; a++) {
        for (int tau = 0; tau < n_tau; tau++) {
            //iteroi firmat joka ajassa
            for (int y = 0; y < workingYears; y++) {

                //Find the optimal state for each i
                for (int i = 0; i < n_k; i++) {

                    int V_idx = V_idx(tau, a, false, false, y, i, false, workingYears, retirementYears, n_k);
                    int current_maxidx = 0;
                    Real current_max = NEG_INF, current_maxs = 0, current_maxe = 0, current_maxc = 0;

                    //Go through all the possible transitions from i
                    for (int j = 0; j < n_k; j++) {

                        int next_idx = 0;
                        if (y = workingYears - 1) {
                            next_idx = V_idx(tau, a, false, true, 0, j, false, workingYears, retirementYears, n_k);
                        }
                        else {
                            next_idx = V_idx(tau, a, false, false, y + 1, j, false, workingYears, retirementYears, n_k);
                        }

                        //Calculate entrepreneur consumption
                        Real n = V[V_idx];
                        Real n_next = V[next_idx];
                        Real s = V[V_idx + 1];
                        Real pi = std::pow(theta - 1, theta - 1) / std::pow(theta, theta) * std::pow(a, theta - 1) * std::pow(P, theta) * C_aggregate;
                        Real c = n + (1 - tau) * pi - T - n_next / (1 + r);

                        //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K <- is this still true?
                        //if (c <= 0) break;

                        //Update the best value found so far

                        Real value = log(c) + beta * V_old[j];
                        if (value > current_max) {
                            current_max = value;
                            current_maxidx = next_idx;
                            current_maxs = s;
                            current_maxe = pi;
                            current_maxc = c;
                        }

                    }
                    //s update
                    Real next_s = (1 + r) * current_maxs + tau_avg * current_maxe;

                    //update policy and value functions
                    int c_policy_idx = c_policy_idx(tau, a, false, false, y, i, workingYears, retirementYears, n_k);
                    policy[c_policy_idx] = current_maxidx;
                    V_new[V_idx] = current_max;
                    V_new[current_maxidx + 1] = next_s;
                    consumption[c_policy_idx] = current_maxc;
                    profits[c_policy_idx] = current_maxe;
                }
            }
        }

    }
    Real sosSecFirms = 0.0;

    for (tau = 0; tau < n_tau; tau++) {
        for (a = 0; a < a_tau; a++) {
            int idx = c_policy_idx(tau, a, false, false, 0, 0, workingYears, retirementYears, n_k);
            for (int y = 0; y < workingYears; y++) {
                sosSecFirms += tau * profits[idx];
                idx = policy[idx];
            }
        }
    }

    Real total_b = tau_avg * numOfWorkers + sosSecFirms;

    //calculate total amount of s-values of retired people
    Real total_s = 0;

    //entrepreneurs
    for (tau = 0; tau < n_tau; tau++) {
        for (a = 0; a < a_tau; a++) {
            if (prices[a * n_tau + tau]) > 0 {
                int idx = c_policy_idx(tau, a, false, false, 0, 0, workingYears, retirementYears, n_k);
                for (int y = 0; y < workingYears; y++) {
                    idx = policy[idx];
                }
                total_s = V[idx + 1] * retirementYears;
            }
        }
    }

    //workers
    int idx = c_policy_idx(tau, a, true, false, 0, 0, workingYears, retirementYears, n_k);
    for (int y = 0; y < workingYears; y++) {
        idx = policy[idx];
    }
    total_s = V[idx + 1] * retirementYears * numOfWorkers;




    //TODO: iteroi kuluttajaeläkeläinen joka ajassa
    for (int y = 0; y < retiredYears; y++) {

        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {

            int V_idx = V_idx(n_tau, n_a, true, true, y, i, false, workingYears, retirementYears, n_k);
            int current_maxidx = 0;
            Real current_max = NEG_INF, current_maxs = 0, current_maxe = 0, current_maxc = 0;

            if (y = retiredYears - 1) {
                Real c = (V_old[V_idx] + V_old[V_idx + 1] / total_s * total_b) / P;
                V_new[V_idx] = log(c);
            }

            else {

                //Go through all the possible transitions from i
                for (int j = 0; j < n_k; j++) {

                    int next_idx = V_idx(n_tau, n_a, true, false, y + 1, j, false, workingYears, retirementYears, n_k);

                    //Calculate retired consumption
                    Real n = V_old[V_idx];
                    Real n_next = V_old[next_idx];
                    Real s = V_old[V_idx + 1];
                    Real b = s / total_s * total_b;
                    Real c = (n + b - n_next / (1 + r)) / P;

                    //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K <- is this still true?
                    //if (c <= 0) break;

                    //Update the best value found so far
                    Real value = log(c) + beta * V_old[j];
                    if (value > current_max) {
                        current_max = value;
                        current_maxidx = next_idx;
                        current_maxs = s;
                        current_maxe = e;
                        current_maxc = c;
                    }



                }

                //update policy and value functions
                int c_policy_idx = c_policy_idx(n_tau, n_a, true, true, y, i, workingYears, retirementYears, n_k);
                policy[c_policy_idx] = current_maxidx;
                V_new[V_idx] = current_max;
                V_new[current_maxidx + 1] = s;
                consumption[c_policy_idx] = current_maxc;

            }
        }
    }
    //iteroi entrepreneur eläkeläiset joka ajassa
        //muista että jengi delaa vikalla kiekalla
        //päivitä sama s arvo aina seuraavaan valittuun tilaan?
        //TODO: iteroi kuluttajaeläkeläinen joka ajassa
    for (int a = 0; a < n_a; a++) {
        for (int tau = 0; tau < n_tau; tau++) {
            for (int y = 0; y < retiredYears; y++) {

                //Find the optimal state for each i
                for (int i = 0; i < n_k; i++) {

                    int V_idx = V_idx(tau, a, true, true, y, i, false, workingYears, retirementYears, n_k);
                    int current_maxidx = 0;
                    Real current_max = NEG_INF, current_maxs = 0, current_maxe = 0, current_maxc = 0;

                    if (y = retiredYears - 1) {
                        Real c = (V_old[V_idx] + V_old[V_idx + 1] / total_s * total_b) / P;
                        V_new[V_idx] = log(c);
                    }

                    else {

                        //Go through all the possible transitions from i
                        for (int j = 0; j < n_k; j++) {

                            int next_idx = V_idx(tau, a, true, false, y + 1, j, false, workingYears, retirementYears, n_k);

                            //Calculate retired consumption
                            Real n = V_old[V_idx];
                            Real n_next = V_old[next_idx];
                            Real s = V_old[V_idx + 1];
                            Real b = s / total_s * total_b;
                            Real c = (n + b - n_next / (1 + r)) / P;

                            //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K <- is this still true?
                            //if (c <= 0) break;

                            //Update the best value found so far
                            Real value = log(c) + beta * V_old[n_next];
                            if (value > current_max) {
                                current_max = value;
                                current_maxidx = next_idx;
                                current_maxs = s;
                                current_maxe = e;
                                current_maxc = c;
                            }



                        }

                        //update policy and value functions
                        int c_policy_idx = c_policy_idx(tau, a, true, true, y, i, workingYears, retirementYears, n_k);
                        policy[c_policy_idx] = current_maxidx;
                        V_new[V_idx] = current_max;
                        V_new[current_maxidx + 1] = s;
                        consumption[c_policy_idx] = current_maxc;

                    }
                }
            }
        }
    }

    //loppupäivitykset arvoille:

    //päivitä r asset market ehdon mukaisesti

    //calculate total n
    //starting with entrepreneurs
    Real total_n = 0;
    for for (int a = 0; a < n_a; a++) {
        for (int tau = 0; tau < n_tau; tau++) {

            int policy_idx = c_policy_idx(tau, a, false, false, y, i, workingYears, retirementYears, n_k);
            for (int y = 0; y < workingYears; y++) {

                int n_idx =
                    total_n += K[n_idx];
                int policy_idx = policy[policy_idx];

            }

            //get the index similarly
            for (int y = 0; y < retiredYears; y++) {
                int policy_idx = c_policy_idx(tau, a, false, true, y, i, workingYears, retirementYears, n_k);
            }
        }
    }
    //then go through workers similarly

    //update r estimate, if n < B -> smaller r (maybe n/B * r)
    Real budget_ratio = total_n / B;
    r = r * budget_ratio;

    //update T estimate
    T = r * B;

    //calculate a new C aggregate based on consumption vector
    C_aggregate = C_worker_agg;

    //add entrepreneur consumption

    //add retired consumption

    //update enterpreneurs
    for (int a = 0; a < n_a; a++) {
        for (int tau = 0; tau < n_tau; tau++) {
            int worker_idx = V_idx(n_tau, n_a, true, false, 0, 0, false, workingYears, retirementYears, n_k);
            int entrepreneur_idx = V_idx(tau, a, false, false, 0, 0, false, workingYears, retirementYears, n_k);
            if (V_new[entrepreneur_idx] > V_new[worker_idx]) {
                prices[a * n_tau + tau] = theta / (theta - 1) / a[a];
            }
            else prices[a * n_tau + tau] = 0;
        }


    }


}
*/
