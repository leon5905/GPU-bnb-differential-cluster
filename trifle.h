#ifndef TRIFLE_GUARD
#define TRIFLE_GUARD 

#include <chrono>
#include "kernel_trifle.cuh"


//Configuration set by kernel_trifle.cuh
void trifle_diff_cluster_cpu_only(double& cluster_prob, long long& cluster_size);
void MITM_trifle_diff_cluster_cpu_only(double& cluster_prob, long long& cluster_size);

// void trifle_diff_cluster_recursion(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, double cur_prob);
void trifle_diff_cluster_gpu(double& cluster_prob, long long& cluster_size, std::chrono::steady_clock::time_point &end);

void trifle_diff_cluster_r10_search();
void trifle_diff_dispersion_search();

void trifle_init();

#endif