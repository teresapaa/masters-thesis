#include <iostream>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <windows.h>
#include <numeric>






//Determine whether to compile as doubles or floats
#if defined(USE_DOUBLE)
using Real = double;
constexpr const char* REAL_NAME = "double";
#else
using Real = float;
constexpr const char* REAL_NAME = "float";
#endif

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
                        Real pi = std::pow(theta - 1, theta -1)/std::pow(theta, theta) * std::pow(a, theta - 1) * std::pow(P, theta) * C_aggregate;
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
    Real total_n = 0;
    for for (int a = 0; a < n_a; a++) {
        for (int tau = 0; tau < n_tau; tau++) {
            for (int y = 0; y < workingYears; y++) {
                int policy_idx = c_policy_idx(tau, a, true, true, y, i, workingYears, retirementYears, n_k);
            }

            for (int y = 0; y < retiredYears; y++) {
                int policy_idx = c_policy_idx(tau, a, true, true, y, i, workingYears, retirementYears, n_k);
            }
        }
    }

    //update r estimate, if n < B -> smaller r (maybe n/B * r) 
     
     
    
    //jos goods market ei clearaa, päivitä arvausta kokonaiskulutuksesta
    //päivitä yrittäjät aina välillä? 

   
    }
    //ehkä päivitä kokonaiskulutusta?

    //iteroi firmat joka ajassa

    //laske sos sec 

    //iteroi retired (ehkä yksi aika riittää?)

    //update r estimate

    //update T estimate

    //update firms

    //update C aggregate


/*
        //Find the optimal state for each i
        for (int i = 0; i < n_k; i++) {

            int current_maxidx = 0;
            Real current_max = NEG_INF;

            //Go through all the possible transitions from i
            for (int j = 0; j < n_k; j++) {

                //calculate all the entrepreneurs
                for (int k = 0; i < n_a * n_tau * n_k) {

                    //Calculate entrepreneur consumption
                    //Real c = ...

                    //If consumption is nonpositive, break the loop since C is a decreasing function for increasing K <- is this still true?
                    //if (c <= 0) break; 

                    //Update the best value found so far
                    //else {
                        //Real value = log(c) + beta * V_old[j];
                        //if (value > current_max) {
                        //    current_max = value;
                        //    current_maxidx = j;
                    //    }
                    }
                }
                //update worker and retired
                //Real c_worker = z * powK[i] + (1 - delta) * K[i] - K[j];
                //Real c_retired = z * powK[i] + (1 - delta) * K[i] - K[j];

            }
            //update policy and value functions 
            policy[i] = current_maxidx;
            V_new[i] = current_max;
        }

        diff = max_abs_difference(V_old, V_new);
        V_old = V_new;
        ++iteration;
   

} while (diff > epsilon && iteration < max_iter);

*/

//vfi; joka kierroksella:

//päivitä kaikki henkilöiden arvofunktiot, aloittaen eläkeikäisistä ja päätyen työikäisiin, koska eläkeikäisten arvofunktio ei riipu muiden arvofunktioista, mutta työikäisten arvofunktio riippuu muiden arvofunktioista
//kysymys on kulutuksen määrittämisestä eri ryhmille
	// kulutus työntekijälle: hintaindeksi + määritettyjen arvojen funktio
	// kulutus yrittäjälle = firman voitot + hintaindeksi + määritettyjen arvojen funktio
	// firman voitot (tallenna myös jonnekin erikseen) <- arvio kokonaiskulutuksesta yhteiskunnasssa (arvaa) + fixed stuff
	// laske sosiaalituen määrä ja jaa se eläkeläisten s arvojen suhteessa, jotta saadaan eläketulot
	// kulutus eläkeikäiselle: pääomatulot + eläketulot + kulutukseen allokoitava osa varallisuudesta <- eläketulot saadaan, kun lasketaan kaikki tukina jaettava ja suhteutetaan se eläkeläisten s arvoon
//päivitä arvio kokonaiskulutuksesta aggregoimalla
//päivitä arvio r:stä asset market clearaa ehdollla
//ehkä tarkista clearaako labor ja good marketit?

//tee arvofunktioiden perusteella uudet valinnat yrittäjiksi ja laske tuotteiden hinnat ja P tän perusteella
//aja johonkin asti ja viim sitten tutki kuinka hyvin eri ehdot toteutuu?


//miten parallelisoituu? Laitetaan kerneli hakemaan tietty arvofunktio/tyyppi ja laskemaan se. Jossain vaiheessa täytyy tosin odottaa että kaikki threadit on valmiita, ennen ku siirtyy koko yhteiskunnan laajuisiin juttuihin




//make initial guesses for prices (interest rate r, prices p and aggregate prices P)
//compute value functions for the employee and the different firm owners, moving from the working age VF definition to the retirement one
//calculate resulting aggregates on capital, labour etc
//find out prices that satisfy the conditions set for the economy
// -> iterate until prices stable

//set r, p, P

//initialise all vfs 

//for each entrepreneur + the employee:
	//for each i:
		//for each j, k :
			// if emloyee: C =
			// if retired: C = 
			// if entrepreneur: C = 
			// V(i) = u(c) + V_old(j)
		//choose best j and update it
	




//OLD:
//make initial guesses of the steady state values of the aggregate capital stock K and employment N

//Compute the values w, r, and tau, which solve the firms's euler equations and the gorvernment budget

//Compute the optimal path for consumption, savings, and employment for the new-born generation by backward induction given the initial capital stock k1 = 0

//Compute the aggregate capital stock K and employment N

//Update K and N and return to step 2 until convergence