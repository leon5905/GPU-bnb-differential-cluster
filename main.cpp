#include <iostream>
#include <chrono>
#include <thread>
#include "common.h"

#include "trifle.h"

//Note that GPU_Bundled_Launch_is_not_complete, ie work in progress.

//Important the program does not perform any permutation at final round
//To obtain the desirable DY, do a permutation on the dy provided in the program.

void investigate_r10();

int main(){
    std::cout << "Starting TRIFLE Differential Path Computation\n";

    double double_after_reduction=0;
    long long cluster_num_gpu_after_reduction=0;

    // const unsigned char sbox[16] = {0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2}; //PRESENT SBOX
    // const unsigned char sbox[16] = {0x0, 0xC, 0x9, 0x7, 0x3, 0x5, 0xE, 0x4, 0x6, 0xB, 0xA, 0x2, 0xD, 0x1, 0x8, 0xF}; //TRIFLE_SBOX
    // unsigned int freq_table[16][8]={0};
    // unsigned int diff_table[16][8] = {0};
    // unsigned int table_dy_length[16]= {0};
    // wz::diff_table_4bit_compact_reversed(sbox,diff_table,freq_table,table_dy_length);

    trifle_init();

    std::chrono::steady_clock::time_point end_premature = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //Call Trifle Differential Solver
    //trifle_diff_cluster_cpu_only(double_after_reduction,cluster_num_gpu_after_reduction);
    // trifle_diff_cluster_gpu(double_after_reduction, cluster_num_gpu_after_reduction, end_premature);
    MITM_trifle_diff_cluster_cpu_only(double_after_reduction,cluster_num_gpu_after_reduction); //Despite the name, it does search for gpu or cpu depending on parameter
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //Print Output
	std::cout << "\n\nCluster Probabilities: log2: " << std::dec << log2(double_after_reduction) << " , base_10 : " << double_after_reduction;
	std::cout << "\nNumber of Cluster Trails:" << cluster_num_gpu_after_reduction;

    std::cout << "\nTime difference (s ) = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
	std::cout << "\nTime difference (us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "\nTime difference (us) discount reduction = " << std::chrono::duration_cast<std::chrono::microseconds>( (end_premature) - begin).count();
	std::cout << "\nTime difference (ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count();

    std::cout <<std::endl;

    return 0;
}