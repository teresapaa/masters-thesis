//release: cl /std:c++17 /O2 /EHsc /DUSE_DOUBLE=1 rl-solution.cpp /Fe:rl-solution.exe
//debug: cl /std:c++17 /Zi /Od /EHsc rl-solution.cpp /Fe:rl-solution.exe

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

#if defined(USE_DOUBLE)
using Real = double;
#else
using Real = float;
#endif

//calculate the index of a state-action pair in the flattened Q-table
static inline std::size_t idx(std::size_t s, std::size_t a, std::size_t n_actions) {
	return s * n_actions + a;
}


Real max_abs_difference(const std::vector<Real>& V0, const std::vector<Real>& V1) {
	Real d = 0.0;
	for (size_t i = 0; i < V0.size(); ++i) {
		Real diff = std::abs(V0[i] - V1[i]);
		if (diff > d) d = diff;
	}
	return d;
}

/*
* A helper function to find index where K' - K changes sign: last i with K'[i] > K[i]
*/
void find_crossing(std::vector<Real> K, int n_k, std::vector<size_t> policy) {

	int crossing_min = -1;
	int crossing_max = -1;
	for (int i = 0; i < n_k; ++i) {
		Real Kp = K[policy[i]];
		if (Kp > K[i]) crossing_min = i;
		if (Kp >= K[i]) crossing_max = i;
	}
	if (crossing_min >= 0) {
		std::cout << "Numerical steady-state approx between K ~ " << K[crossing_min] << " and K ~ " << K[crossing_max]
			<< ", K' at the max state = " << K[policy[crossing_max]] << ", indexes = " << crossing_min << ", " << crossing_max << std::endl;
	}
	else {
		std::cout << "No crossing found (policy never suggests K' > K)." << std::endl;
	}
}

void find_convergence_idx(std::vector<Real> diffs, int iters, Real epsilon) {

	int iteration = -1;
	for (int i = 0; i < iters; ++i) {
		if (diffs[i] > epsilon) iteration = i;
	}
	if (iteration >= 0) {
		std::cout << "Converged to " << epsilon << " after " << iteration << " iterations." << std::endl;
	}
	else {
		std::cout << "No convergence index found." << std::endl;
	}
}



void run_compute(int argc, char* argv[]) {
	std::cout << "Neoclassical Growth model [rl]" << std::endl;
	auto start_time = std::chrono::steady_clock::now();

	//Setting up variables
	std::size_t n_k = 145; // number of grid points
	Real Kmin = 0.5f; // lower bound of the state space
	Real Kmax = 100.0f; // upper bound of the state space
	Real epsilon = 0.001f; //tolerance of error
	Real alpha = 0.5f; //capital share
	Real z = 1.0f; //productivity
	Real beta = 0.96f; //annual discounting
	Real delta = 0.025f; //annual depreciation
	int explr_limits = 100;

	//Q-learning parameters
	Real explr_multiplier = 0.995f;
	Real learning_rate = 0.7f;
	Real exploration_prob = 0.9;
	bool decay_epsilon = true;
	int max_iter = 10000;
	int steps_per_iter = 5000;
	Real diff = 100.0f;
	int iters = 5000;

	//parameters from the user
	if (argc > 5) {
		n_k = std::atoi(argv[1]);
		steps_per_iter = std::atoi(argv[2]);
		exploration_prob = std::atof(argv[3]);
		explr_multiplier = std::atof(argv[4]);
		iters = std::atoi(argv[5]);
		explr_limits = std::atoi(argv[6]);
	}


	//Set the grid points
	std::vector<Real> K(n_k);
	Real step = (Kmax - Kmin) / (n_k - 1);
	for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

	//flattened Q-table - maybe restrict the search in a smaller space?
	std::vector<Real> Q(n_k * n_k, 0);

	//flattened current best value + index table
	std::vector<Real> bestV(n_k, 0.0f);
	std::vector<Real> bestV_old(n_k, 0.0f);
	std::vector<int> bestA(n_k, 0);

	//collect the differences for convergence analysis
	std::vector<Real> diffs(iters, 0.0f);


	std::cout << "n_k: " << n_k << std::endl;
	std::cout << "steps_per_iter: " << steps_per_iter << std::endl;
	std::cout << "exploration_prob: " << exploration_prob << std::endl;

	//set up the probability distributions
	std::mt19937_64 gen(2002);
	std::uniform_real_distribution<Real> uniform01(0.0, 1.0);
	std::uniform_int_distribution<int> rand_state(0, n_k-1);
	//std::uniform_int_distribution<int> rand_action(0, n_k - 1);

	size_t current_state, action;



	for (int iter = 0; iter < iters; ++iter) {

		//start with a random state
		current_state = rand_state(gen);

		for (int t = 0; t < steps_per_iter; ++t) {

			//epsilon-greedy action selection -> explore with probability exploration_prob		
			if (uniform01(gen) < exploration_prob) {
				//randomly choose next action
				int s = static_cast<int>(current_state);
				int lower = s - explr_limits;
				if (lower < 0) lower = 0;
				int upper = s + explr_limits;
				if (upper >= n_k) upper = n_k - 1;
				std::uniform_int_distribution<int> rand_action(lower, upper);
				action = rand_action(gen);
			}

			else {
				action = bestA[current_state];

			}
			
			// calculate reward for the Q table update
			Real Kpow = std::pow(K[current_state], alpha);
			Real consumption = z * Kpow + (1.0 - delta) * K[current_state] - K[action];
			Real reward = consumption <= 0.0 ? -1e3f : std::log(consumption);

			// compute max over actions in next state
			Real max_next = bestV[action];

			//find current index and value in flattened Q
			std::size_t qIndex = idx(current_state, action, n_k);
			Real currentQ = Q[qIndex];

			//Q-learning update
			auto update = currentQ + learning_rate * (reward + beta * max_next - currentQ);
			Q[qIndex] = update;

			if (update > bestV[current_state]) {
				bestV[current_state] = update;
				bestA[current_state] =  action;
			}

			//initialize next round
			current_state = action;

			}

		// decay exploration slowly
		if (decay_epsilon) {
			exploration_prob *= explr_multiplier;
			if (exploration_prob < 0.01) exploration_prob = 0.01;
			decay_epsilon = false;
		}

		diff = max_abs_difference(bestV_old, bestV);
		diffs[iter] = diff;
		bestV_old = bestV;

	};


	auto end_time = std::chrono::steady_clock::now();
	std::chrono::duration<Real, std::milli> time = end_time - start_time;
	std::cout << "End-to-end host wall-clock time: " << time.count() << " ms\n";

	//Find out policy function (argmax action for each state)
	std::vector<std::size_t> policy(n_k);
	for (std::size_t s = 0; s < n_k; ++s) {
		auto row_begin = Q.begin() + static_cast<std::ptrdiff_t>(idx(s, 0, n_k));
		auto row_end = row_begin + static_cast<std::ptrdiff_t>(n_k);
		auto it = std::max_element(row_begin, row_end);
		policy[s] = static_cast<std::size_t>(std::distance(row_begin, it));
	}

	/*

	//Print first 20 entries of policy and value (for inspection)
	std::cout << "\nSample policy (state K, chosen K'):\n";
	std::cout << " K      K'     (index) \n";
	for (std::size_t s = 0; s < std::min<std::size_t>(100, n_k); ++s) {
		std::size_t a = policy[s];
		std::cout << std::fixed << std::setprecision(10)
			<< K[s] << " -> " << K[a] << "   (" << a << ")\n";
	}
	*/
	
	// Find index where K' - K changes sign: last i with K'[i] > K[i]
	//std::cout << "Found a solution after " << iteration << " iterations" << std::endl;
	std::cout << "Final diff: " << diff << std::endl;
	find_crossing(K, n_k, policy);
	find_convergence_idx(diffs, iters, epsilon);
	
}



int main(int argc, char* argv[])
{
	std::cout << "masters_thesis: starting compute\n";
	run_compute(argc, argv);
	std::cout << "masters_thesis: finished\n";
	return 0;
}
