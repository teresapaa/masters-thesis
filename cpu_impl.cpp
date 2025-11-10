#include <iostream>
extern "C" void run_compute() {
    std::cout << "[CPU] running simple compute (no CUDA)\n";
    const int N = 8;
    float a[N];
    for (int i = 0; i < N; ++i) a[i] = i;
    for (int i = 0; i < N; ++i) a[i] += 1.0f;
    std::cout << "[CPU] result: ";
    for (int i = 0; i < N; ++i) std::cout << a[i] << " ";
    std::cout << "\n";
}