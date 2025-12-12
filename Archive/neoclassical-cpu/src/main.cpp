#include <iostream>

// run_compute() is provided by either the CPU or GPU implementation at link-time.
extern "C" void run_compute(int argc, char* argv[]);

int main(int argc, char* argv[]) {
    std::cout << "masters_thesis: starting compute\n";
    run_compute(argc, argv);
    std::cout << "masters_thesis: finished\n";
    return 0;
}