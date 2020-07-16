#ifndef TRIFLE_KERNEL_GUARD
#define TRIFLE_KERNEL_GUARD 

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
* CUDA specific configuration
*/

//WARNING: MUST BE >=32 , prefer 128
//2432 Threads per kernel launches
#define THREAD_PER_BLOCK 128
//#define THREAD_PER_BLOCK 256 
//Number of thread per block, multiple of 32, 128-512
//Must be >128 because of initialization in shared block...

#define BLOCK_NUM 19
#define BLOCK_NUM_EXACT_BUNDLE 18
//#define BLOCK_NUM 10
// Must be >9
//2560 threads ... only 2401 is used...
// 7* 7 ... * 7 ((7^4)^2) = 5,764,801‬ // 26,785,714, 20% occupancy of memory (can store 2 rounds which uses up 40% memory)

//128bit/4 = 32
#define MAX_SBOX 32

//NOTE: edit both at the same time
//KINDA wanted to set it to 7 or 6 to improve memory usage....
//TODO something is wrong with AS6
//TODO something is wrong with AS4!?!?
#define MAX_AS_USED 4

#ifndef CUDA_ERROR_PRINT
#define CUDA_ERROR_PRINT
#endif

#define PATTERN_ROUND_MITM_FORWARD 10
#define PATTERN_ROUND_MITM_BACKWARD 10
#define PATTERN_ROUND (PATTERN_ROUND_MITM_FORWARD+PATTERN_ROUND_MITM_BACKWARD)

//TODO not used yet, can use to minimize ram usage but goes for fewer rounds
//TODO i need this to do concurrent launch to maximize the performance because of bandwidth limit
#define MAX_POSSIBLE_ROUND_MINUS_ONE 1 

//7^8
#define MAX_BRANCH_PER_ROUND 5764801

//pow(2, -2), 1/4, 0.25
//pow(2, -3), 1/8, 0.125

//Upper limit bounded single paths
//Need to copy this to GPU device
//21 is the 10s one with 27 cluster size
//31 is the 331s one with 159 cluster size (AS 4 with same probabilities)
//-32 seems more paths
//NOTE: all pattern round need to be modified as well...
const float CLUSTER_PROB_BOUND = (pow(pow(2, -3), PATTERN_ROUND_MITM_FORWARD - 2) * pow(2, -2) * pow(2, -2) * pow(2, -21)); 
//21

const float CLUSTER_PROB_BOUND_PURE = (pow(pow(2, -3), PATTERN_ROUND - 2) * pow(2, -2) * pow(2, -2) * pow(2, -21)); 

const float CLUSTER_PROB_BOUND_LOG2 = log2(CLUSTER_PROB_BOUND);

//Used for round AS estimation
#define CLUSTER_1AS_BEST_PROB 0.25f
#define CLUSTER_PROB_INDIV 0.25f

namespace TRIFLE{
    void init();

    /*
	* BC specific permutation and DTT
	*/
    //Contains configuration (macro / c++ global variable) intended to be used across different translation unit
    extern unsigned char perm_host[128]; 
    extern unsigned char perm_host_reversed[128]; 

    //[0] front  ||   [1] back
    extern unsigned long long perm_lookup_host[32][16][2];
    extern unsigned long long perm_lookup_host_reversed[32][16][2];

    extern unsigned int diff_table_host[][8];
    extern unsigned int diff_table_host_reversed[][8];

    extern float prob_table_host[16][8];
    extern unsigned int freq_table_host[][8];
    extern unsigned int diff_table_size_host[16]; //ONLY USED in main class || not used here

    /*
    * DX and DY changes
    */
    //NOTE: change the following
    //Refernce Value for PATTERN_ROUND = 10
    //Dx -> Dy : 7000 0000 0000 0000 0000 0000 0000 0000   ->   4000 0000 4000 0000 0000 0000 0000 0000 (before permutation is line below)
    //DY_below is 0c00 0000 0000 0000 0000 0000 0000 0000
    //Cluster Probabilities:-27.9928
    //Number of Cluster Trails : 31
    //HACK: CHANGE the folloig value together with final_dy
    //NOTE: IT IS DY_B4Permutation... (save performance on GPU)

    extern unsigned char final_dy_host[32];
    extern unsigned char ref_dx_host[32];
};

//Meant for different gpu/threads
struct Kernel_TRIFLE_t{
    //Output (intermediate)
    float* device_prob_arr; //Float * MAX_BRANCH_PER_ROUND 
    unsigned char* device_dy_arr; //unsigned char * 32 * MAX_BRANCH_PER_ROUND
    int* device_sbox_num_and_index_arr; //int * 9 * MAX_BRANCH_PER_ROUND 

    //Input
    unsigned char* device_dx; //single unsigned char*32
    int* device_sbox_index; //int * 8
    
    //Final Output (Need to be reduced)
    unsigned int* device_cluster_size_final; //unsigned int * thread num * thread block 
    float* device_prob_final;         //float * thread_num * thread_block

    //MITM Forward Output
    //Size of 3 Sbox with 32 position information
    //134217728
    static const int MITM_size = 134217728;
    float* MITM_prob_interm_global;
    int* MITM_size_interm_global;
    
    //MITM Backward Output (Final)
    double* MITM_prob_final_global; // double * THREAD_PER_BLOCK * BLOCK_NUM
    long long* MITM_size_final_global;

    //Others
    cudaStream_t stream_obj;

    Kernel_TRIFLE_t(int thread_id, unsigned char *& pinned_host_dx_rounds, float *&cur_round_prob_pinned, int *&next_round_sbox_num_and_index,
        unsigned char *&pinned_input_dx, int *&pinned_input_sbox_index);

    void change_parameter(unsigned char* new_dy, unsigned char* new_dx=nullptr);
    
    void kernel_compute_1round(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r,
        unsigned char* next_round_dx_array_pinned, float* cur_round_prob_pinned, int* next_round_sbox_num_and_index);

    void kernel_compute_1round_last(int offset_index);

    //MITM Specific
    void kernel_MITM_intermediate_reduction(float* prob_interm, int* size_interm); //Will modify Both gpu memory and cpu memory

    void kernel_compute_1round_backward(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r,
        unsigned char* next_round_dx_array_pinned, float* cur_round_prob_pinned, int* next_round_sbox_num_and_index);
    void MITM_kernel_compute_1round_last_forward(int offset_index);
    void MITM_kernel_compute_1round_last_backward(int offset_index);

    //TODO task 3
    void kernel_compute_bundled();
    void kernel_compute_bundled_last();

    void kernel_reduction(long long& gpu_cluster_size, double &gpu_cluster_prob);
};

#endif