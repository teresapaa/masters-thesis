#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

//calculate the index of a state-action pair in the flattened Q-table
static inline std::size_t idx(std::size_t s, std::size_t a, std::size_t n_actions) {
	return s * n_actions + a;
}

void run_compute() {
	std::cout << "Neoclassical Growth model [rl]" << std::endl;
	auto start_time = std::chrono::steady_clock::now();

	//Setting up variables
	std::size_t n_k = 1000; // number of grid points
	float Kmin = 0.5f; // lower bound of the state space
	float Kmax = 100.0f; // upper bound of the state space
	float epsilon = 0.001f; //tolerance of error
	float alpha = 0.5f; //capital share
	float z = 1.0f; //productivity
	float beta = 0.96f; //annual discounting
	float delta = 0.025f; //annual depreciation


	//Set the grid points
	std::vector<float> K(n_k);
	float step = (Kmax - Kmin) / (n_k - 1);
	for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

	//flattened Q-table
	std::vector<float> Q(n_k * n_k, 0);

	//Q-learning parameters
	float learning_rate = 0.7f;
	double exploration_prob = 0.2;
	bool decay_epsilon = true;
	int epochs = 200;
	int steps_per_epoch = 1000;

	//set up the probability distributions
	std::mt19937_64 gen(std::random_device{}());
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);
	std::uniform_int_distribution<int> rand_state(0, n_k-1);
	std::uniform_int_distribution<int> rand_action(0, n_k - 1);

	size_t current_state, action;

	for (int ep = 0; ep < epochs; ++ep) {

		//start with a random state
		current_state = rand_state(gen);

		for (int t = 0; t < steps_per_epoch; ++t) {

			//epsilon-greedy action selection -> explore with probability exploration_prob		
			if (uniform01(gen) < exploration_prob) {
				//randomly choose next action
				action = rand_action(gen);
			}

			else {
				//define next sction as argmax over actions for state s
				auto row_begin = Q.begin() + static_cast<std::ptrdiff_t>(idx(current_state, 0, n_k));
				auto row_end = row_begin + static_cast<std::ptrdiff_t>(n_k);
				auto maxIterator = std::max_element(row_begin, row_end);
				action = static_cast<std::size_t>(std::distance(row_begin, maxIterator));
			}
			
			// calculate reward for the Q table update
			float Kpow = std::pow(K[current_state], alpha);
			float consumption = z * Kpow + (1.0 - delta) * K[current_state] - K[action];
			float reward = consumption <= 0.0 ? -1e3f : std::log(consumption);

			// compute max over actions in next state
			auto next_row_begin = Q.begin() + static_cast<std::ptrdiff_t>(idx(action, 0, n_k));
			auto next_row_end = next_row_begin + static_cast<std::ptrdiff_t>(n_k);
			double max_next = *std::max_element(next_row_begin, next_row_end);

			//find current index and value in flattened Q
			std::size_t qIndex = idx(current_state, action, n_k);
			double currentQ = Q[qIndex];

			//Q-learning update
			Q[qIndex] = currentQ + learning_rate * (reward + beta * max_next - currentQ);

			//initialize next round
			current_state = action;
			}

		// decay exploration slowly
		if (decay_epsilon) {
			exploration_prob *= 0.995;
			if (exploration_prob < 0.01) exploration_prob = 0.01;
		}

		}

	auto end_time = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> time = end_time - start_time;
	std::cout << "End-to-end host wall-clock time: " << time.count() << " ms\n";

	//Find out policy function (argmax action for each state)
	std::vector<std::size_t> policy(n_k);
	for (std::size_t s = 0; s < n_k; ++s) {
		auto row_begin = Q.begin() + static_cast<std::ptrdiff_t>(idx(s, 0, n_k));
		auto row_end = row_begin + static_cast<std::ptrdiff_t>(n_k);
		auto it = std::max_element(row_begin, row_end);
		policy[s] = static_cast<std::size_t>(std::distance(row_begin, it));
	}

	// Print first 20 entries of policy and value (for inspection)
	std::cout << "\nSample policy (state K, chosen K'):\n";
	std::cout << " K      K'     (index) \n";
	for (std::size_t s = 0; s < std::min<std::size_t>(100, n_k); ++s) {
		std::size_t a = policy[s];
		std::cout << std::fixed << std::setprecision(3)
			<< K[s] << " -> " << K[a] << "   (" << a << ")\n";
	}
}



int main()
{
	std::cout << "masters_thesis: starting compute\n";
	run_compute();
	std::cout << "masters_thesis: finished\n";
	return 0;
}
