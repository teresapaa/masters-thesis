//release double: cl /std:c++17 /O2 /EHsc /DUSE_DOUBLE=1 rl-plots.cpp /Fe:rl-plots-double.exe
//release float: cl /std:c++17 /O2 /EHsc rl-plots.cpp /Fe:rl-plots-float.exe
//debug double: cl /std:c++17 /Zi /Od /EHsc /DUSE_DOUBLE=1 rl-plots.cpp /Fe:rl-plots-double-d.exe
//debug float: cl /std:c++17 /Zi /Od /EHsc rl-plots.cpp /Fe:rl-plots-float-d.exe
//parameters for grid 1 000: 1000 50000 0.99 0.999 0.95 38 20 0.001 5
//parameters for grid 10 000: 10000 125000 0.99 0.999 0.99 458 200 0.0006 5
//parametrit komentoriville: n_k steps_per_iter exploration_prob explr_multiplier learning_rate iters explr_limits epsilon rounds warmups

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <limits>
#include <cstdint>
#include <ctime>


namespace fs = std::filesystem;

#if defined(USE_DOUBLE)
using Real = double;
constexpr const char* REAL_NAME = "double";
#else
using Real = float;
constexpr const char* REAL_NAME = "float";
#endif

/*
Helper function to calculate the index of a state-action pair in the flattened Q-table
*/
static inline std::size_t idx(std::size_t s, std::size_t a, std::size_t n_actions) {
	return s * n_actions + a;
}


/*
* Helper function to calculate medians of the running times
*/

template<typename T>
Real median_of_vector(const std::vector<T>& input) {
	if (input.empty()) throw std::domain_error("median of empty vector");
	std::vector<T> v = input;            // make a copy if we must preserve original
	size_t n = v.size();
	size_t mid = n / 2;
	std::nth_element(v.begin(), v.begin() + mid, v.end());
	if (n % 2 == 1) {
		return static_cast<Real>(v[mid]);
	}
	else {
		// v[mid] is the upper middle. Find the maximum in the lower partition [0, mid).
		T upper = v[mid];
		T lower = *std::max_element(v.begin(), v.begin() + mid);
		return (static_cast<Real>(lower) + static_cast<Real>(upper)) * 0.5;
	}
}

/*
* Helper function to calculate the maximum pointwise distance of two vectors
*/
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
std::tuple<int, int> find_crossing(std::vector<Real> K, int n_k, std::vector<size_t> policy) {

	int crossing_min = -1;
	int crossing_max = -1;
	for (int i = 0; i < n_k; ++i) {
		Real Kp = K[policy[i]];
		if (Kp > K[i]) crossing_min = i;
		if (Kp >= K[i]) crossing_max = i;
	}
	if (crossing_min < 0) {
		std::cout << "No crossing found (policy never suggests K' > K)." << std::endl;
		
		//to make sure the program won't crash		
		crossing_min = 0;
		crossing_max = 0;
	}
	return { crossing_min, crossing_max };
}

/*
A helper function to find the index after which abs(V_old - V_neew) < epsilon
*/
int find_convergence_idx(std::vector<Real> diffs, int iters, Real epsilon) {

	int iteration = -1;
	for (int i = 0; i < iters; ++i) {
		if (diffs[i] > epsilon) iteration = i;
	}
	if (iteration >= 0) {
		//std::cout << "Converged to " << epsilon << " after " << iteration << " iterations." << std::endl;
	}
	else {
		std::cout << "No convergence index found." << std::endl;
	}
	return iteration;
}


/*
Where the actual calculation happens
*/
std::tuple<int, Real, int, int, Real, Real> run_compute(int n_k, int steps_per_iter, Real exploration_prob, Real explr_multiplier, Real learning_rate, int iters, int explr_limits, Real epsilon, int save_every) {

	//start measuring time and CPU cycles
	auto start_time = std::chrono::steady_clock::now();
	HANDLE hProcess = GetCurrentProcess();
	ULONG64 startCycles = 0, endCycles = 0;
	QueryProcessCycleTime(hProcess, &startCycles);

	//Setting up variables
	Real Kmin = 0.5f; // lower bound of the state space
	Real Kmax = 100.0f; // upper bound of the state space
	Real alpha = 0.5f; //capital share
	Real z = 1.0f; //productivity
	Real beta = 0.96f; //annual discounting
	Real delta = 0.025f; //annual depreciation
	std::string output_dir = R"(out\data)"; //for plotting the results


	//Q-learning parameters
	bool decay_epsilon = true; //whether to decay exploration probability
	int max_iter = 10000; //max number of iterations to make sure we don't run forever
	Real diff = 100.0f; //initial difference for convergence check


	//Set the grid points
	std::vector<Real> K(n_k);
	Real step = (Kmax - Kmin) / (n_k - 1);
	for (int i = 0; i < n_k; ++i) K[i] = Kmin + i * step;

	//flattened Q-table
	std::vector<Real> Q(n_k * n_k, 0);

	//flattened current best value + index table
	std::vector<Real> bestV(n_k, 0.0f);
	std::vector<Real> bestV_old(n_k, 0.0f);
	std::vector<int> bestA(n_k, 0);

	//collect the differences for convergence analysis
	std::vector<Real> diffs(iters, 0.0f);

	//set up the probability distributions (seeded for reproducibility
	std::mt19937_64 gen(2002);
	std::uniform_real_distribution<Real> uniform01(0.0, 1.0);
	std::uniform_int_distribution<int> rand_state(0, n_k-1);

	//Declare variables for the main loop
	size_t current_state, action;


	//For plotting:
   // Ensure output directory exists so file writes succeed
	fs::path outdir = fs::path(output_dir);
	try {
		fs::create_directories(outdir);
	}
	catch (const std::exception& e) {
		std::cout << "Warning: failed to create output directory '" << outdir.string() << "': " << e.what() << std::endl;
	}
	//save V every save_every iterations (and always final)
	auto save_snapshot = [&](int iter) {
		std::ostringstream fname;
		fname << outdir.string() << "/rl_" << std::to_string(n_k) << "_vfi_iter" << std::setw(4) << std::setfill('0') << iter << ".csv";
		std::ofstream f(fname.str());
		if (!f.is_open()) {
			std::cout << "Warning: could not open snapshot file for writing: " << fname.str() << std::endl;
			return;
		}
		f << "i,K,V\n";
		for (int i = 0; i < n_k; ++i) {
			f << i << "," << K[i] << "," << bestV_old[i] << "\n";
		}
		f.close();
		};

	// Open diffs CSV to record diff at every iteration
	fs::path diffs_path = outdir / ("rl_" + std::to_string(n_k) + "_vfi_diffs.csv");
	std::ofstream diffs_ofs(diffs_path.string(), std::ios::out);
	if (!diffs_ofs.is_open()) {
		std::cout << "Warning: could not open diffs file for writing: " << diffs_path.string() << std::endl;
	}
	else {
		// write header
		diffs_ofs << "iter,diff\n";
		// set precision appropriate for Real
		int prec = std::numeric_limits<Real>::digits10 + 1;
		diffs_ofs << std::setprecision(prec);
	}

	//precompute K^alpha
	std::vector<Real> powK(n_k);
	std::transform(K.begin(), K.end(), powK.begin(),
		[alpha](Real k) { return pow(k, alpha); });

	//The main Q-learning loop
	for (int iter = 0; iter < iters; ++iter) {

		//start with a random state
		current_state = rand_state(gen);

		for (int t = 0; t < steps_per_iter; ++t) {

			//epsilon-greedy action selection -> explore with probability exploration_prob		
			if (uniform01(gen) < exploration_prob) {

				//randomly choose next action whithin the exploration limits
				int s = static_cast<int>(current_state);
				int lower = s - explr_limits;
				if (lower < 0) lower = 0;
				int upper = s + explr_limits;
				if (upper >= n_k) upper = n_k - 1;
				std::uniform_int_distribution<int> rand_action(lower, upper);
				action = rand_action(gen);
			}

			else {
				//choose best action according to current policy
				action = bestA[current_state];

			}
			
			// calculate reward for the Q table update
			Real consumption = z * powK[current_state] + (1.0 - delta) * K[current_state] - K[action];
			Real reward = consumption <= 0.0 ? -1e3f : std::log(consumption);

			// compute max over actions in next state
			Real max_next = bestV[action];

			//find current index and value in flattened Q
			std::size_t qIndex = idx(current_state, action, n_k);
			Real currentQ = Q[qIndex];

			//Q-learning update
			auto update = currentQ + learning_rate * (reward + beta * max_next - currentQ);
			Q[qIndex] = update;

			//Update best value and action if needed
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

		//to record convergence
		diff = max_abs_difference(bestV_old, bestV);
		diffs[iter] = diff;
		bestV_old = bestV;

		// record diff for this iteration (if file open)
		if (diffs_ofs.is_open()) {
			diffs_ofs << iter << "," << diff << "\n";
		}

		if (iter % save_every == 0) save_snapshot(iter);

	};

	//stop measuring time and CPU cycles
	QueryProcessCycleTime(hProcess, &endCycles);
	auto end_time = std::chrono::steady_clock::now();
	std::uint64_t CPU_cycles = (endCycles - startCycles);
	std::chrono::duration<Real, std::milli> time = end_time - start_time;


	//Find out policy function (argmax action for each state)
	std::vector<std::size_t> policy(n_k);
	for (std::size_t s = 0; s < n_k; ++s) {
		auto row_begin = Q.begin() + static_cast<std::ptrdiff_t>(idx(s, 0, n_k));
		auto row_end = row_begin + static_cast<std::ptrdiff_t>(n_k);
		auto it = std::max_element(row_begin, row_end);
		policy[s] = static_cast<std::size_t>(std::distance(row_begin, it));
	}

	// Find index where K' - K changes sign: last i with K'[i] > K[i]
	auto [crossing_min, crossing_max] = find_crossing(K, n_k, policy);

	//Find the index of convergence
	auto conv_idx = find_convergence_idx(diffs, iters, epsilon);


	// write final CSV of policy/value
	{
		fs::path final_path = outdir / ("rl_" + std::to_string(n_k) + "_vfi_final.csv");
		std::ofstream fout(final_path.string());
		if (!fout.is_open()) {
			std::cout << "Warning: could not open final CSV for writing: " << final_path.string() << std::endl;
		}
		else {
			fout << "i,K,V,Kp_index,Kp,c\n";
			for (int i = 0; i < n_k; ++i) {
				int j = policy[i];
				double Ki = K[i];
				double Kj = K[j];
				double c = z * powK[Ki] + (1 - delta) * Ki - Kj;
				fout << i << "," << K[i] << "," << bestV[i] << "," << j << "," << Kj << "," << c << "\n";
			}
			fout.close();
		}
	}


	// save final V snapshot too
	{
		fs::path snap_path = outdir / ("rl_" + std::to_string(n_k) + "_vfi_iter_final.csv");
		std::ofstream f(snap_path.string());
		if (!f.is_open()) {
			std::cout << "Warning: could not open final snapshot for writing: " << snap_path.string() << std::endl;
		}
		else {
			f << "i,K,V\n";
			for (int i = 0; i < n_k; ++i) f << i << "," << K[i] << "," << bestV[i] << "\n";
			f.close();
		}
	}

	return{conv_idx, diff, crossing_min, crossing_max , K[crossing_min], K[crossing_max]};
}

/*
Function to handle the running with parameters from command line
*/
void handle_running(int argc, char* argv[]) {

	std::cout << "Neoclassical Growth model [no GPU]" << std::endl;

	//results to be recorded
	Real diff, crossing_min_K, crossing_max_K;
	int iteration, crossing_min, crossing_max, conv_idx;

	//parameters with default values
	int n_k = 145; // number of grid points
	int steps_per_iter = 5000; // steps per iteration
	Real exploration_prob = 0.9; //initial exploration probability
	Real explr_multiplier = 0.995f; //exploration probability multiplier per iteration
	Real learning_rate = 0.7f; //learning rate
	int iters = 5000; //number of iterations
	int explr_limits = 100; //exploration limits
	Real epsilon = 0.001; //tolerance of error
	int save_every = 10; //how often to save snapshots


	//check if the user has given parameters
	if (argc > 9) {
		//assign parameters from the user
		n_k = std::atoi(argv[1]);
		steps_per_iter = std::atoi(argv[2]);
		exploration_prob = std::atof(argv[3]);
		explr_multiplier = std::atof(argv[4]);
		learning_rate = std::atof(argv[5]);
		iters = std::atoi(argv[6]);
		explr_limits = std::atoi(argv[7]);
		epsilon = std::atof(argv[8]);
		save_every = std::atoi(argv[9]);
	}

	std::tie(conv_idx, diff, crossing_min, crossing_max, crossing_min_K, crossing_max_K) = run_compute(n_k, steps_per_iter, exploration_prob, explr_multiplier, learning_rate, iters, explr_limits, epsilon, save_every);
	

	//record results to file
	auto now = std::chrono::system_clock::now();
	std::time_t now_time = std::chrono::system_clock::to_time_t(now);
	fs::path logdir = "out\\log";

	try {
		fs::create_directories(logdir);
	}
	catch (const std::exception& e) {
		std::cout << "Warning: failed to create output directory: " << e.what() << std::endl;
	}

	fs::path logfile = logdir / "rl.txt";

	std::ofstream ofs(logfile, std::ios::out | std::ios::app);
	if (!ofs) {
		std::cerr << "Failed to open file: " << logfile << '\n';
	}

	else {
		ofs << std::ctime(&now_time);
		ofs << "Using Real = " << REAL_NAME << std::endl;
		ofs << "Grid size(n_k): " << n_k << std::endl;
		ofs << "Iterations, steps per iteration: " << iters << ", " << steps_per_iter << std::endl;
		ofs << "Exploration probability, exploration multiplier, learning rate: " << exploration_prob << ", " << explr_multiplier << ", " << learning_rate << std::endl;
		ofs << "Exploration limit: " << explr_limits << std::endl;
		ofs << "Converged to error limit " << epsilon << " in " << conv_idx << " iterations" << std::endl;
		ofs << "Final diff: " << diff << std::endl;
		ofs << "Numerical steady-state approx between K ~ " << crossing_min_K << " and K ~ " << crossing_max_K
			<< ", indexes = " << crossing_min << ", " << crossing_max << std::endl;
		ofs << "\n";
	}

}


int main(int argc, char* argv[])
{
	std::cout << "masters_thesis: starting compute\n";
	handle_running(argc, argv);
	std::cout << "masters_thesis: finished\n";
	return 0;
}
