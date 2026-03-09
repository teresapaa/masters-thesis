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

//TODO:
// update entrepreneurs with labor/goods clearing conditions
//precaltulate (some of) the prices
// merge "update_income_vectors" somewhere else
//could profitcache be build outside one vfi round?

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
    Real beta = Real(0.98);   
    Real theta = Real(5.0);

    // Bonds / government
    Real B = Real(580);         // debt portfolio

    // Time
    int workingYears = 10;
    int retirementYears = 5;

    // Grids
    int n_k = 100;
    int n_tau = 5;
    int n_a = 5;

    Real Kmin = Real(0.0), Kmax = Real(50.0);
    Real tauMin = Real(0.01), tauMax = Real(0.1);
    Real amin = Real(0.5), amax = Real(5.0);

    int max_iters = 20000;     // total Bellman iterations allowed
    int firm_update_every = 20; // update firms every x sweeps
    int check_every = 10;       // convergence checks every N sweeps

    //damping factors
    Real alpha_macro = Real(0.5); // damping for r,T,P,C
    Real gamma_firm = Real(0.10); // damping for firm_weight

    Real tol_V = Real(1e-4);
    Real tol_macro = Real(1e-4);

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

    Real s_entry_worker = Real(0);                     
    std::vector<Real> s_entry_entrep;

    // Entrepreneur membership weights (0/1)
    std::vector<Real> firm_weight; // size n_a*n_tau

    // Prices 
    std::vector<Real> prices;      // size n_a*n_tau

    // parameters
    Real T = Real(0.001);      // lump sum taxes (guess)
    //Real T = Real(0);
    Real r = Real(0.02);   // interest rate (guess)
    Real P = Real(1);      // aggregate price index (placeholder)
    Real C_agg = Real(7); // aggregate consumption (placeholder)
    Real l = Real(1);      // labor supply (placeholder)
    Real L_agg = Real(1);
    Real A_agg = Real(1);

    int num_firms = 0;

    //helpers
    int n_types;
    int lifeYears;
    Real mass = Real(1);
    Real workerMass;

    explicit Model(const Params& params)
        : p(params), g(p),
        worker_w(1, p.workingYears, p.n_k),
        entrep_w(p.n_a* p.n_tau, p.workingYears, p.n_k),
        worker_r(1, p.retirementYears, p.n_k),
        entrep_r(p.n_a * p.n_tau, p.retirementYears, p.n_k),
        firm_weight((std::size_t)p.n_a* p.n_tau, Real(0)),
        prices((std::size_t)p.n_a* p.n_tau, Real(0)),
        income_worker((std::size_t)p.workingYears, Real(0)),
        income_entrep((std::size_t)p.n_a* p.n_tau* p.workingYears, Real(0)),
        s_entry_entrep((std::size_t)p.n_a* p.n_tau, Real(0)),
        n_types(p.n_a * p.n_tau),
        lifeYears(p.workingYears + p.retirementYears),
        //mass(Real(1) / Real(n_types * lifeYears)),
        workerMass(Real(n_types * p.workingYears) * mass)
    {}

    //index in the income vector
    inline std::size_t inc_idx(int sid, int y) const {
        return (std::size_t)sid * (std::size_t)p.workingYears + (std::size_t)y;
    }

    //initialize the enterepreneurs 
    void init_educated_guess() {
        num_firms = 0;
        for (int it = 0; it < p.n_tau; ++it) {
            for (int ia = 0; ia < p.n_a; ++ia) {
                const bool guessFirm = (it < p.n_tau / 2) && (ia >= p.n_a / 2);
                firm_weight[(std::size_t)entrep_state_id(ia, it, p.n_tau)] = guessFirm ? Real(1) : Real(0);
				num_firms += guessFirm ? 1 : 0;
            }
        }

    }


    void update_total_consumption() {
        int num_workers = n_types - num_firms;
        Real total_L = num_workers * l;
        C_agg = total_L * A_agg;
        //C_agg = 40;
        std::cout << "num_workers, total_L, C_agg: " << num_workers << ", " << total_L << ", " << C_agg << "\n";
    }

    void calculate_A_agg() {
        Real A_calc = 0;
        for (int it = 0; it < p.n_tau; ++it) {
            for (int ia = 0; ia < p.n_a; ++ia) {
                const int sid = entrep_state_id(ia, it, p.n_tau);
                if (firm_weight[(std::size_t)sid] <= Real(0.5)) continue;
                A_calc += std::pow(g.a[ia], p.theta - 1);
            }
        }
        if (A_calc <= Real(0)) { A_agg = Real(1); return; }
        A_agg = std::pow(A_calc, Real(1) / (p.theta - Real(1)));
        std::cout << "A_agg: " << A_agg << "\n";
    }

    //after new firms are chosen, calculate the respective prices and the price index
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

    //find the new entrepreneurs: TODO - take goods/labor market clearing huomioon!
    int update_entrepreneur_set(Real damping = Real(0.2)) {
        const Real Vw0 = worker_w.V_old[worker_w.idx(0, 0, 0)];
		//std::vector<std::pair<Real, int>> scores((std::size_t)p.n_a * p.n_tau, Real(0));

        num_firms = 0;
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            const Real Ve0 = entrep_w.V_old[entrep_w.idx(sid, 0, 0)];
			Real score = Ve0 - Vw0;
            const Real target = (score > 0) ? Real(1) : Real(0);

            firm_weight[(std::size_t)sid] = (Real(1) - damping) * firm_weight[(std::size_t)sid] + damping * target;

            if (firm_weight[(std::size_t)sid] > Real(0.5)) ++num_firms;
        }

        return num_firms;
    }


    struct ProfitCache {
        std::vector<Real> pi; // size n_a*n_tau
    };


    int update_entrepreneur_set_to_target(int target_num_firms, Real damping) {
        target_num_firms = std::clamp(target_num_firms, 0, n_types);

        // Baseline worker value at (age=0, asset index=0)
        const Real Vw0 = worker_w.V_old[worker_w.idx(0, 0, 0)];

        // Score each type by advantage of being entrepreneur: Ve0 - Vw0
        std::vector<std::pair<Real, int>> score;
        score.reserve((std::size_t)n_types);

        for (int sid = 0; sid < n_types; ++sid) {
            const Real Ve0 = entrep_w.V_old[entrep_w.idx(sid, 0, 0)];
            score.push_back({ Ve0 - Vw0, sid });
        }

        std::sort(score.begin(), score.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        // Softly move firm_weight toward target set: top target_num_firms => 1, rest => 0
        for (int rank = 0; rank < n_types; ++rank) {
            const int sid = score[(std::size_t)rank].second;
            const Real target = (rank < target_num_firms) ? Real(1) : Real(0);
            firm_weight[(std::size_t)sid] =
                (Real(1) - damping) * firm_weight[(std::size_t)sid] + damping * target;
        }

        // Recount
        int nf = 0;
        for (int sid = 0; sid < n_types; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) ++nf;
        }
        num_firms = nf;
        return num_firms;
    }



    //precalculate the entrepreneurs' profits
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
                Real profit = k1 * a_pow[ia] * P_pow * C_agg;
                Real alt_profit = Real(1) / p.theta * std::pow(g.a[ia] / A_agg, p.theta - Real(1)) * P * C_agg;
                pc.pi[(std::size_t)sid] = alt_profit;
                //std::cout << k1 << ", apow " << a_pow[ia] << ", P_pow " << P_pow << ", C_agg " << C_agg << " profit " << profit << "\n";

            }
        }
        return pc;
    }

    //update income vectors
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
                            const Real c = ( n + (Real(1) - tau_i) * income - T - n_next * inv_1pr) / P;

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

    //collect social security payments b from all the working age ppl and find the last year's s value.
    Real update_s_entries_and_total_b(std::vector<Real>& s_entry_entrep,
        Real& s_entry_worker) {

        Real total_b_per_period = 0;
        Real total_b_worker = 0;

        int num_workers = n_types - num_firms;
        Real worker_weight = Real(num_workers) * mass;


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

        total_b_per_period += total_b_worker * worker_weight;

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
                    if (is_entrep) { total_b_per_period += new_s * mass; }
                }
                s_entry_entrep[(std::size_t)sid] = s;
            }
        }
        
        return total_b_per_period;
        
    }

    //find the retirement payment for each retiree
    void build_b_state(const std::vector<Real>& s_entry_entrep,
        Real s_entry_worker,
        Real total_b,
        std::vector<Real>& b_worker_state,
        std::vector<Real>& b_entrep_state)
    {
        int num_workers = n_types - num_firms;
        Real sum_entrep_s = Real(0);
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) {
                sum_entrep_s += s_entry_entrep[(std::size_t)sid] * mass * p.retirementYears;
            }
        }

        Real retiree_s = (s_entry_worker * num_workers * mass * p.retirementYears + sum_entrep_s);

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

        Real paid = Real(0);

        // workers
        paid += Real(num_workers) * mass * Real(p.retirementYears) * bW;

        /*
        // entrepreneurs (only active firm types)
        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            if (firm_weight[(std::size_t)sid] > Real(0.5)) {
                paid += mass * Real(p.retirementYears) * b_entrep_state[(std::size_t)sid];
            }
        }
        */

        //std::cout << "Pension pool=" << (double)total_b
        //    << " paid=" << (double)paid
        //    << " gap=" << (double)(total_b - paid) << "\n";

    }



   
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
                        ret.policy[cur] = (uint16_t)0;
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

        //initialize values
        const auto pc = build_profit_cache();
        update_income_paths(pc);

        bellman_worker_working();
        bellman_entrep_working(pc);

        // retirement depends on b_state -> compute it each iteration
        std::vector<Real> s_entry_entrep((size_t)p.n_a * p.n_tau, Real(0));
        Real s_entry_worker = Real(0);
        Real total_b = update_s_entries_and_total_b(s_entry_entrep, s_entry_worker);

        std::vector<Real> b_worker_state(1, Real(0));
        std::vector<Real> b_entrep_state((size_t)p.n_a * p.n_tau, Real(0));
        build_b_state(s_entry_entrep, s_entry_worker, total_b, b_worker_state, b_entrep_state);

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

        const int num_workers = n_types - num_firms;

        {
            int i = 0;
            for (int y = 0; y < p.workingYears; ++y) {
                const std::size_t cur = worker_w.idx(0, y, i);
                out.assets += Real(num_workers) * mass * g.K[i];
                out.consumption += Real(num_workers) * mass * worker_w.cons[cur];
                i = (int)worker_w.policy[cur];
            }
            for (int y = 0; y < p.retirementYears; ++y) {
                const std::size_t cur = worker_r.idx(0, y, i);
                out.assets += Real(num_workers) * mass * g.K[i];
                Real c = worker_r.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in worker_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += Real(num_workers) * mass * c;
                i = (int)worker_r.policy[cur];
            }

        }

        
        for (int sid = 0; sid < totalTypes; ++sid) {
            
            if (firm_weight[(std::size_t)sid] <= Real(0.5)) continue;

            int i = 0;
            for (int y = 0; y < p.workingYears; ++y) {
                const std::size_t cur = entrep_w.idx(sid, y, i);
                out.assets += mass * g.K[i];
                Real c = entrep_w.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in entrep_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += c * mass;
                i = (int)entrep_w.policy[cur];
            }

            for (int y = 0; y < p.retirementYears; ++y) {
                const std::size_t cur = entrep_r.idx(sid, y, i);
                out.assets += g.K[i] * mass;
                Real c = entrep_r.cons[cur];
                if (!std::isfinite(c)) {
                    std::cerr << "Non-finite cons in entrep_w at y=" << y
                        << " i=" << i << " c=" << (double)c << "\n";
                    std::abort();
                }
                out.consumption += c * mass;
                i = (int)entrep_r.policy[cur];
            }
        }

        

        return out;
    }

    struct RUpdater_old {
        bool have_prev = false;
        Real r_prev = 0, F_prev = 0;


        Real max_step = Real(0.05);     // clamp dr
        Real damp = Real(0.2);       // extra damping
        Real r_min = Real(-0.95);
        Real r_max = Real(1.0);       // pick something reasonable

        Real update(Real r, Real F) {
            

            std::cout << "have prev: " << have_prev << std::endl;
            std::cout << "prev" << r_prev << std::endl;
            
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

            std::cout << "new: " << r_new << std::endl;

            // store
            have_prev = true;
            r_prev = r;
            F_prev = F;

            return r_new;
        }
    };
    
    struct RUpdater_old2 {
        Real k = Real(0.5);          // gain (0.1–2.0)
        Real max_step = Real(0.0001);  // clamp per update
        Real r_min = Real(-0.5);
        Real r_max = Real(0.5);

        Real update(Real r, Real A, Real B) {
            Real F = A - B; // >0 means too much saving
            Real scale = std::max(std::abs(B), Real(1e-6));
            Real Frel = F / scale;

            // Want: F>0 => r decreases; F<0 => r increases
            Real dr = -k * Frel;
            dr = std::clamp(dr, -max_step, max_step);

            Real r_new = std::clamp(r + dr, r_min, r_max);
            return r_new;
        }
    };

    struct RUpdater {
        Real r_lo = Real(0.0);
        Real r_hi = Real(0.5);
        bool lo_set = false;
        bool hi_set = false;

        Real update(Real r, Real assets, Real B) {
            Real gap = assets - B;

            // gap > 0: too much saving -> r too high -> update upper bound
            // gap < 0: too little saving -> r too low -> update lower bound
            if (gap > Real(0)) {
                r_hi = r;
                hi_set = true;
            }
            else {
                r_lo = r;
                lo_set = true;
            }

            // if we have a valid bracket, bisect
            if (lo_set && hi_set) {
                return (r_lo + r_hi) * Real(0.5);
            }

            // no bracket yet, fall back to a small step in the right direction
            Real step = Real(0.02);
            return r + (gap < Real(0) ? step : -step);
        }

        // how tight is the bracket
        Real bracket_width() const {
            if (lo_set && hi_set) return r_hi - r_lo;
            return Real(1.0);
        }
    };

    Real compute_total_L_demand(Real total_C) {

        Real L_coef = Real(0);
        const int num_workers = n_types - num_firms;

        for (int sid = 0; sid < p.n_a * p.n_tau; ++sid) {
            if (firm_weight[(std::size_t)sid] <= Real(0.5)) continue;
			Real price_multiplier = std::pow(prices[(std::size_t)sid] / P, p.theta);
			L_coef += price_multiplier / g.a[sid / p.n_tau];
        }

        L_coef = L_coef;

        return L_coef;
    }



    struct DiagStats {
        Real max_abs = 0;
        Real sum_abs = 0;
        std::size_t n = 0;
        void add(Real x) {
            Real ax = (Real)std::abs(x);
            if (ax > max_abs) max_abs = ax;
            sum_abs += ax;
            ++n;
        }
        Real mean_abs() const { return (n ? sum_abs / (Real)n : Real(0)); }
    };

    static inline bool in_bounds_int(int x, int lo, int hi) { return x >= lo && x <= hi; }

    void run_diagnostics() {
        std::cout << "\n=== POST-SOLVE DIAGNOSTICS ===\n";

        // Recompute totals with final policies
        Totals totals = compute_totals();
        Real bond_gap = totals.assets - p.B;

        std::cout << "Bond market gap (A - B): " << (double)bond_gap
            << " | rel=" << (double)(bond_gap / (std::max(Real(1e-12), p.B))) << "\n";

        // Labor market residuals
        Real L_supply = (n_types - num_firms) * mass * l;
        Real L_coef = compute_total_L_demand(totals.consumption);
        Real L_demand = L_coef * C_agg;
        Real L_gap = L_supply - L_demand;

        std::cout << "Labor: L_supply=" << (double)L_supply
            << " L_demand=" << (double)L_demand
            << " gap=" << (double)L_gap
            << " rel=" << (double)(L_gap / std::max(Real(1e-12), L_supply)) << "\n";

        print_worker_value_functions(/*print_to_stdout=*/true, /*write_csv=*/false);
        print_selected_entrep_value_functions(2, true, true, false);

        std::cout << "=== END DIAGNOSTICS ===\n\n";
    }


    void print_selected_entrep_value_functions(
        int max_types_to_print = 3,
        bool only_active = true,
        bool print_to_stdout = true,
        bool write_csv = true)
    {
        const int nk = p.n_k;
        int printed = 0;

        for (int sid = 0; sid < n_types && printed < max_types_to_print; ++sid) {

            if (only_active && firm_weight[(std::size_t)sid] <= Real(0.5))
                continue;

            int ia = sid / p.n_tau;
            int it = sid % p.n_tau;

            std::string tag = "entrep_sid_" + std::to_string(sid)
                + "_a" + std::to_string(ia)
                + "_tau" + std::to_string(it);

            auto print_block = [&](const BlockArrays& blk, const std::string& name)
                {
                    if (print_to_stdout) {
                        std::cout << "\n=== " << tag
                            << " | " << name << " ===\n";
                        std::cout << "rows: y, cols: (i, K, V, pol->K', c)\n";
                    }

                    if (print_to_stdout) {
                        for (int y = 0; y < blk.years; ++y) {

                            if (y % 10 != 0) continue; // avoid huge output

                            std::cout << "\n-- y=" << y << " --\n";
                            std::cout << std::setw(6) << "i"
                                << std::setw(12) << "K"
                                << std::setw(16) << "V"
                                << std::setw(8) << "pol"
                                << std::setw(12) << "K'"
                                << std::setw(16) << "c"
                                << "\n";

                            for (int i = 0; i < nk; ++i) {
                                std::size_t id = blk.idx(sid, y, i);
                                int j = (int)blk.policy[id];

                                Real K = g.K[i];
                                Real V = blk.V_old[id];
                                Real c = blk.cons[id];
                                Real Kp = (j >= 0 && j < nk) ? g.K[j] : Real(0);

                                std::cout << std::setw(6) << i
                                    << std::setw(12) << (double)K
                                    << std::setw(16) << (double)V
                                    << std::setw(8) << j
                                    << std::setw(12) << (double)Kp
                                    << std::setw(16) << (double)c
                                    << "\n";
                            }
                        }
                    }

                    if (write_csv) {
                        std::string fname = tag + "_" + name + "_VF.csv";
                        std::ofstream out(fname);

                        out << "sid,y,i,K,V,policy_j,K_next,cons\n";

                        for (int y = 0; y < blk.years; ++y) {
                            for (int i = 0; i < nk; ++i) {
                                std::size_t id = blk.idx(sid, y, i);
                                int j = (int)blk.policy[id];

                                Real K = g.K[i];
                                Real V = blk.V_old[id];
                                Real c = blk.cons[id];
                                Real Kp = (j >= 0 && j < nk) ? g.K[j] : Real(0);

                                out << sid << "," << y << "," << i << ","
                                    << (double)K << "," << (double)V << ","
                                    << j << "," << (double)Kp << ","
                                    << (double)c << "\n";
                            }
                        }
                    }
                };

            print_block(entrep_w, "working");
            print_block(entrep_r, "retirement");

            //printed++;
        }
    }

 
   
    void debug_print_entrepreneur_economics_grid(
        bool only_active = false,
        bool print_price_terms = true)
    {
        std::cout << "\n=== Entrepreneur Economics Debug (a x tau grid) ===\n";

        // Profit cache from current P and C_agg
        const auto pc = build_profit_cache();

        std::cout << "Globals: P=" << (double)P
            << " C_agg=" << (double)C_agg
            << " r=" << (double)r
            << " l(worker income)=" << (double)l
            << "\n";

        auto pick3 = [](int n) {
            // returns indices: low, mid, high (unique if possible)
            std::vector<int> idx;
            if (n <= 0) return idx;
            idx.push_back(0);
            if (n > 2) idx.push_back(n / 2);
            if (n > 1) idx.push_back(n - 1);
            // ensure uniqueness
            std::sort(idx.begin(), idx.end());
            idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
            return idx;
            };

        const auto a_idx = pick3(p.n_a);
        const auto tau_idx = pick3(p.n_tau);

        if (print_price_terms) {
            std::cout << "Note: theta=" << (double)p.theta
                << " P^theta=" << (double)std::pow(P, p.theta)
                << "\n";
        }

        std::cout << "\nColumns: sid | ia it | a tau | active | price | pi | pi/l\n";

        for (int ia : a_idx) {
            for (int it : tau_idx) {
                const int sid = entrep_state_id(ia, it, p.n_tau);

                const bool active = firm_weight[(std::size_t)sid] > Real(0.5);
                if (only_active && !active) continue;

                const Real a = g.a[ia];
                const Real tau_i = g.tau[it];
                const Real price = prices[(std::size_t)sid];      // 0 if not active in your current price update
                const Real pi = pc.pi[(std::size_t)sid];

                std::cout << "sid=" << sid
                    << " | " << ia << " " << it
                    << " | a=" << (double)a
                    << " tau=" << (double)tau_i
                    << " | " << (active ? "Y" : "N")
                    << " | price=" << (double)price
                    << " | pi=" << (double)pi
                    << " | pi/l=" << (double)(pi / std::max(Real(1e-12), l))
                    << "\n";
            }
        }

        std::cout << "=== End Entrepreneur Economics Debug ===\n\n";
    }

   

    void print_worker_value_functions(bool print_to_stdout = true, bool write_csv = true) {
        const int nk = p.n_k;

        auto print_block = [&](const BlockArrays& blk, const std::string& name) {
            if (print_to_stdout) {
                std::cout << "\n=== Worker Value Function: " << name << " ===\n";
                std::cout << "rows: year y, cols: asset grid i (showing K, V, pol->K', c)\n";
            }

            // stdout: print each year as a small table
            if (print_to_stdout) {
                for (int y = 0; y < blk.years; ++y) {
                    if (y % 10 != 0) continue;

                    std::cout << "\n-- y=" << y << " --\n";
                    std::cout << std::setw(6) << "i"
                        << std::setw(12) << "K"
                        << std::setw(16) << "V"
                        << std::setw(8) << "pol"
                        << std::setw(12) << "K'"
                        << std::setw(16) << "c"
                        << "\n";

                    for (int i = 0; i < nk; ++i) {
                        const std::size_t id = blk.idx(0, y, i);
                        const int j = (int)blk.policy[id];
                        const Real K = g.K[i];
                        const Real V = blk.V_old[id];
                        const Real c = blk.cons[id];
                        const Real Kp = (j >= 0 && j < nk) ? g.K[j] : Real(0);

                        std::cout << std::setw(6) << i
                            << std::setw(12) << (double)K
                            << std::setw(16) << (double)V
                            << std::setw(8) << j
                            << std::setw(12) << (double)Kp
                            << std::setw(16) << (double)c
                            << "\n";
                    }
                }
            }

            // CSV: long format (one row per (y,i))
            if (write_csv) {
                std::string fname = "worker_" + name + "_VF.csv";
                std::ofstream out(fname);
                if (!out) {
                    std::cerr << "Could not open " << fname << " for writing.\n";
                    return;
                }
                out << "block,y,i,K,V,policy_j,K_next,cons\n";
                for (int y = 0; y < blk.years; ++y) {
                    for (int i = 0; i < nk; ++i) {
                        const std::size_t id = blk.idx(0, y, i);
                        const int j = (int)blk.policy[id];
                        const Real K = g.K[i];
                        const Real V = blk.V_old[id];
                        const Real c = blk.cons[id];
                        const Real Kp = (j >= 0 && j < nk) ? g.K[j] : Real(0);

                        out << name << "," << y << "," << i << ","
                            << (double)K << "," << (double)V << ","
                            << j << "," << (double)Kp << ","
                            << (double)c << "\n";
                    }
                }
                std::cout << "Wrote " << fname << "\n";
            }
            };

        // worker working + worker retirement
        print_block(worker_w, "working");
        print_block(worker_r, "retirement");
    }

    void solve() {

        init_educated_guess();
        update_prices_from_firms();
        calculate_A_agg();
        update_total_consumption();


        RUpdater r_updater;

        Real last_r = r, last_T = T, last_C = C_agg, last_P = P;

        Real assets = 0;


        for (int iter = 1; iter <= p.max_iters; ++iter) {

            // 1) One Bellman sweep (operator uses current macros + firm set)
            Real Vdiff = bellman_one_iter();

            // 2) Update macros every sweep

            // 3) Update firms only every x sweeps (soft)
            if (iter % p.firm_update_every == 0) {
                Totals totals = compute_totals();

                Real F = totals.assets - p.B;
                
                if (iter % p.firm_update_every == 0 && Vdiff < p.tol_V) {
                    r = r_updater.update(r, totals.assets, p.B);
                    //r = r_updater.update(r, totals.assets, p.B);
                    //r = r_updater_old.update(r, F);
                    Real working_age_population = Real(n_types) * Real(p.workingYears);
                    Real T_new = r * p.B / working_age_population;               // per-person tax among workers
                    //T = (1 - p.alpha_macro) * T + p.alpha_macro * T_new;
                    T = T_new;

                    std::cout << "RUPDATE: assets=" << totals.assets
                        << " B=" << p.B
                        << " gap=" << (totals.assets - p.B)
                        << " lo=" << r_updater.r_lo
                        << " hi=" << r_updater.r_hi
                        << " lo_set=" << r_updater.lo_set
                        << " hi_set=" << r_updater.hi_set << "\n";
                }

                Real L_supply = (n_types - num_firms) * mass * l;
                //Real L_coef = compute_total_L_demand(totals.consumption);

                Real L_coef = compute_total_L_demand(C_agg);
                Real L_demand = L_coef * L_supply;

                //C_agg = L_supply / L_coef;

                //C_agg = totals.consumption;

                int target = n_types - std::round(L_demand / L_supply);

                //int nf = update_entrepreneur_set_to_target(target, p.gamma_firm);
                std::cout << "number of firms: " << num_firms << "\n";

                //update_prices_from_firms();

                std::cout << "consumption: " << totals.consumption << "\n";
                std::cout << "assets: " << totals.assets << "\n";

                std::cout << "L supply and demand: " << L_supply << ", " << L_coef << "\n";

                //Old total consumption update
                //C_agg = (Real(1) - p.alpha_macro) * C_agg + p.alpha_macro * new_C_xd;

                //debug_print_entrepreneur_economics_grid(/*only_active=*/true);
                assets = totals.assets;

                std::cout << "Asset diff: " << p.B - assets << "\n";

            }

           /*
            std::cout << "sweep " << iter
                << " Vdiff=" << (double)Vdiff
                << " r=" << (double)r
                << " T=" << (double)T
                << " P=" << (double)P
                << " C=" << (double)C_agg
               // << " dr=" << (double)dr
                //<< " dC=" << (double)dC
                << "\n";
                */


            // 4) Convergence checks occasionally
            if (iter % p.check_every == 0) {
                const Real dr = std::abs(r - last_r);
                const Real dT = std::abs(T - last_T);
                const Real dP = std::abs(P - last_P);
                const Real dC = std::abs(C_agg - last_C);

                const Real d_am = std::abs(assets - p.B);


                // stop when BOTH V and macros are stable
                if (Vdiff < p.tol_V && dr < p.tol_macro && dT < p.tol_macro && dP < p.tol_macro && dC < p.tol_macro && d_am < 1 && r_updater.bracket_width() < 0.001) {
                    std::cout << "Converged.\n";
                    run_diagnostics();
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

