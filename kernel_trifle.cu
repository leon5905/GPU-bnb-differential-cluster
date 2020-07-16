#include "kernel_trifle.cuh"
#include <iostream>
#include <cstring>

namespace TRIFLE{
    /*
	* BC specific permutation and DTT
	*/
    //Contains configuration (macro / c++ global variable) intended to be used across different translation unit
    __shared__ unsigned long long perm_lookup_shared[32][16][2]; 
    __device__ unsigned long long perm_lookup_global[32][16][2];
    __device__ unsigned long long perm_lookup_global_reversed[32][16][2];
    
    unsigned char perm_host[128];
    unsigned char perm_host_reversed[128];

    unsigned long long perm_lookup_host[32][16][2]; //8192 bytes, 8KB, one SM can have 49KB should be fine
    unsigned long long perm_lookup_host_reversed[32][16][2];
    //_host  //ONLY USED in main class || not used here

    //NOTE: _host have no uses inside this class
    __shared__ unsigned int diff_table_shared[16][8];  //NOTE: init in kernel by 1st thread of the block.
    __device__ unsigned int diff_table_global[][8] = {
        {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
        {0xc, 0x6, 0x7, 0x8, 0xa, 0xd, 0xe, 0x0},
        {0x9, 0x1, 0x5, 0xb, 0xc, 0xd, 0xe, 0x0},
        {0x7, 0x1, 0x2, 0x4, 0x5, 0x9, 0xb, 0x0},
        {0x3, 0x2, 0x7, 0x9, 0xa, 0xb, 0xd, 0x0},
        {0x5, 0x6, 0x7, 0x9, 0xa, 0xd, 0xf, 0x0},
        {0xe, 0x2, 0x3, 0x4, 0x7, 0x8, 0xa, 0x0},
        {0x4, 0x2, 0x3, 0x9, 0xb, 0xc, 0xf, 0x0},
        {0x6, 0x3, 0x4, 0x5, 0x7, 0xb, 0xe, 0x0},
        {0xb, 0x1, 0x2, 0x8, 0xa, 0xc, 0xd, 0x0},
        {0xa, 0x3, 0x5, 0xb, 0xc, 0xe, 0xf, 0x0},
        {0x2, 0x1, 0x6, 0x9, 0xc, 0xd, 0xf, 0x0},
        {0xd, 0x1, 0x4, 0x5, 0x6, 0x8, 0xe, 0x0},
        {0x1, 0x3, 0x6, 0x8, 0xc, 0xe, 0xf, 0x0},
        {0x8, 0x3, 0x4, 0x6, 0x7, 0x9, 0xf, 0x0},
        {0xf, 0x1, 0x2, 0x4, 0x5, 0x8, 0xa, 0x0},
    };
    __device__ unsigned int diff_table_global_reversed[][8] = {
        { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
        { 0xd, 0x2, 0x3, 0x9, 0xb, 0xc, 0xf, 0x0},
        { 0xb, 0x3, 0x4, 0x6, 0x7, 0x9, 0xf, 0x0},
        { 0x4, 0x6, 0x7, 0x8, 0xa, 0xd, 0xe, 0x0},
        { 0x7, 0x3, 0x6, 0x8, 0xc, 0xe, 0xf, 0x0},
        { 0x5, 0x2, 0x3, 0x8, 0xa, 0xc, 0xf, 0x0},
        { 0x8, 0x1, 0x5, 0xb, 0xc, 0xd, 0xe, 0x0},
        { 0x3, 0x1, 0x4, 0x5, 0x6, 0x8, 0xe, 0x0},
        { 0xe, 0x1, 0x6, 0x9, 0xc, 0xd, 0xf, 0x0},
        { 0x2, 0x3, 0x4, 0x5, 0x7, 0xb, 0xe, 0x0},
        { 0xa, 0x1, 0x4, 0x5, 0x6, 0x9, 0xf, 0x0},
        { 0x9, 0x2, 0x3, 0x4, 0x7, 0x8, 0xa, 0x0},
        { 0x1, 0x2, 0x7, 0x9, 0xa, 0xb, 0xd, 0x0},
        { 0xc, 0x1, 0x2, 0x4, 0x5, 0x9, 0xb, 0x0},
        { 0x6, 0x1, 0x2, 0x8, 0xa, 0xc, 0xd, 0x0},
        { 0xf, 0x5, 0x7, 0xa, 0xb, 0xd, 0xe, 0x0}
    };
    unsigned int diff_table_host[][8] = {
        {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
        {0xc, 0x6, 0x7, 0x8, 0xa, 0xd, 0xe, 0x0},
        {0x9, 0x1, 0x5, 0xb, 0xc, 0xd, 0xe, 0x0},
        {0x7, 0x1, 0x2, 0x4, 0x5, 0x9, 0xb, 0x0},
        {0x3, 0x2, 0x7, 0x9, 0xa, 0xb, 0xd, 0x0},
        {0x5, 0x6, 0x7, 0x9, 0xa, 0xd, 0xf, 0x0},
        {0xe, 0x2, 0x3, 0x4, 0x7, 0x8, 0xa, 0x0},
        {0x4, 0x2, 0x3, 0x9, 0xb, 0xc, 0xf, 0x0},
        {0x6, 0x3, 0x4, 0x5, 0x7, 0xb, 0xe, 0x0},
        {0xb, 0x1, 0x2, 0x8, 0xa, 0xc, 0xd, 0x0},
        {0xa, 0x3, 0x5, 0xb, 0xc, 0xe, 0xf, 0x0},
        {0x2, 0x1, 0x6, 0x9, 0xc, 0xd, 0xf, 0x0},
        {0xd, 0x1, 0x4, 0x5, 0x6, 0x8, 0xe, 0x0},
        {0x1, 0x3, 0x6, 0x8, 0xc, 0xe, 0xf, 0x0},
        {0x8, 0x3, 0x4, 0x6, 0x7, 0x9, 0xf, 0x0},
        {0xf, 0x1, 0x2, 0x4, 0x5, 0x8, 0xa, 0x0},
    };
    //Init by init
    unsigned int diff_table_host_reversed[][8] = {
        { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
        { 0xd, 0x2, 0x3, 0x9, 0xb, 0xc, 0xf, 0x0},
        { 0xb, 0x3, 0x4, 0x6, 0x7, 0x9, 0xf, 0x0},
        { 0x4, 0x6, 0x7, 0x8, 0xa, 0xd, 0xe, 0x0},
        { 0x7, 0x3, 0x6, 0x8, 0xc, 0xe, 0xf, 0x0},
        { 0x5, 0x2, 0x3, 0x8, 0xa, 0xc, 0xf, 0x0},
        { 0x8, 0x1, 0x5, 0xb, 0xc, 0xd, 0xe, 0x0},
        { 0x3, 0x1, 0x4, 0x5, 0x6, 0x8, 0xe, 0x0},
        { 0xe, 0x1, 0x6, 0x9, 0xc, 0xd, 0xf, 0x0},
        { 0x2, 0x3, 0x4, 0x5, 0x7, 0xb, 0xe, 0x0},
        { 0xa, 0x1, 0x4, 0x5, 0x6, 0x9, 0xf, 0x0},
        { 0x9, 0x2, 0x3, 0x4, 0x7, 0x8, 0xa, 0x0},
        { 0x1, 0x2, 0x7, 0x9, 0xa, 0xb, 0xd, 0x0},
        { 0xc, 0x1, 0x2, 0x4, 0x5, 0x9, 0xb, 0x0},
        { 0x6, 0x1, 0x2, 0x8, 0xa, 0xc, 0xd, 0x0},
        { 0xf, 0x5, 0x7, 0xa, 0xb, 0xd, 0xe, 0x0}
    };

    __shared__ float prob_table_shared[16][8];  //NOTE: init in kernel by 1st thread of the block.
    float prob_table_host[16][8]={
        {1, 1, 1, 1, 1, 1, 1, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
        {0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 1},
    };
    __device__ unsigned int freq_table_global[][8] = {
        {16, 16, 16, 16, 16, 16, 16, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
    };
    unsigned int freq_table_host[][8] = {
        {16, 16, 16, 16, 16, 16, 16, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
        {4, 2, 2, 2, 2, 2, 2, 16},
    };

    __shared__ unsigned int diff_table_size_shared[16];  //NOTE: init in kernel by 1st thread of the block.
    __device__ unsigned int diff_table_size_global[16] = {1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    unsigned int diff_table_size_host[16] = {1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};  //ONLY USED in main class || not used here

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

	//Constant memory because it is accessed by the same warp @ the same addresses. (broadcasting) else request will be serialized
    __constant__ unsigned char final_dy_constant[32] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
	};
    unsigned char final_dy_host[32] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x6, 0x0, 0x6, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
	};

    unsigned char ref_dx_host[32] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0xb,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
    };

    __constant__ float CLUSTER_PROB_BOUND_const = 0; 
};

void TRIFLE::init(){
    std::cout <<"\nInit Trifle Reverse Differential Table:{\n";

    std::cout <<"\nTrifle Permutation:{\n";
    for (int i = 0; i < 128; i++) {
        TRIFLE::perm_host[i] = (i / 4) + ((i % 4) * 32);
        std::cout << (int) perm_host[i]<< ",";
    }
    std::cout << "\n}\n" ;

    std::cout <<"\nTrifle Permutation Reversed:{\n";
    for (int i=0;i<128;i++){
        TRIFLE::perm_host_reversed[perm_host[i]] = i;
    }
    for (int i=0;i<128;i++){
        std::cout << (int) perm_host_reversed[i]<< ",";
    }
    std::cout << "}\n" ;

    //--
    std::cout <<"\n4bit Permutation LUTable * 32 (Size is 32*16*16 is 8192Bytes) :{\n";
    for (int sbox_pos=0;sbox_pos<32;sbox_pos++){
        for (int sbox_val=0;sbox_val<16;sbox_val++){
            unsigned char dx[32] = {0};
            dx[sbox_pos] = sbox_val;

            //Do permutation
            unsigned long long front_64 = 0, back_64 = 0, front_64_reversed=0, back_64_reversed=0;
			for (int i = 0; i < 32; i++) {
				if (dx[i] > 0) {
					for (int j = 0; j < 4; j++) {
                        //Actually filtered_bit
						unsigned long long filtered_word = ((dx[i] & (0x1 << j)) >> j) & 0x1;
						if (filtered_word == 0) continue; //no point continue if zero, go to next elements

                        int bit_pos = (TRIFLE::perm_host[((31 - i) * 4) + j]);
                        int bit_pos_reversed = (TRIFLE::perm_host_reversed[((31 - i) * 4) + j]);

						if ((bit_pos / 64) == 1) {  //Front
							bit_pos -= 64;
							front_64 |= (filtered_word << bit_pos);
						}
						else {  //Back
							back_64 |= (filtered_word << bit_pos);
                        }
                        
                        if ((bit_pos_reversed / 64) == 1) {  //Front
							bit_pos_reversed -= 64;
							front_64_reversed |= (filtered_word << bit_pos_reversed);
						}
						else {  //Back
							back_64_reversed |= (filtered_word << bit_pos_reversed);
                        }
					}
				}
			}
            
            //Front 64, 0-15, Back64 - 16-31
            perm_lookup_host[sbox_pos][sbox_val][0]=front_64;
            perm_lookup_host[sbox_pos][sbox_val][1]=back_64;

            perm_lookup_host_reversed[sbox_pos][sbox_val][0]=front_64_reversed;
            perm_lookup_host_reversed[sbox_pos][sbox_val][1]=back_64_reversed;
        }
    }
    std::cout << "}\n" ;
    
    std::cout << "\nTransfered constant matsui bound from host to device";
    auto cudaStatus = cudaMemcpyToSymbol(TRIFLE::CLUSTER_PROB_BOUND_const, &CLUSTER_PROB_BOUND, sizeof(float));
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol CLUSTER_PROB_BOUND_const failed!");
		goto Error;
	}

    std::cout << "\nTransfered perm_LUhost from host to device";
    cudaStatus = cudaMemcpyToSymbol(TRIFLE::perm_lookup_global, TRIFLE::perm_lookup_host, sizeof(unsigned long long)*32*16*2);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy perm_LUhost failed!");
		goto Error;
    }
    
    std::cout << "\nTransfered perm_LUhost Reversed from host to device";
    cudaStatus = cudaMemcpyToSymbol(TRIFLE::perm_lookup_global_reversed, TRIFLE::perm_lookup_host_reversed, sizeof(unsigned long long)*32*16*2);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy perm_LUhost failed!");
		goto Error;
	}

    std::cout <<"\n----\n";
    return;

Error:
    std::cout << "\nCritical Error. Aborting Program";
    if (cudaStatus != cudaSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cout << "\nCRITICAL ERROR from TRIFLE init...";
        fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
        std::cout << "\nPress any key to continue...";
        getchar();
    }
};

//TODO refactor cuda error check to inline function
Kernel_TRIFLE_t::Kernel_TRIFLE_t(int thread_id, unsigned char *& pinned_host_dx_rounds, float *&cur_round_prob_pinned, int *&next_round_sbox_num_and_index,
    unsigned char *&pinned_input_dx, int *&pinned_input_sbox_index){
    
    cudaStreamCreate( &(this->stream_obj) );

    int round_to_allocate = PATTERN_ROUND_MITM_FORWARD > PATTERN_ROUND_MITM_BACKWARD? PATTERN_ROUND_MITM_FORWARD : PATTERN_ROUND_MITM_BACKWARD;
    if (round_to_allocate < 20)
        round_to_allocate = 20;

    //CUDA specific init
    //Should use thread id to select devices...
    auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

    //CUDA ALLOC (Internal Use)
    //Round Output
    cudaStatus = cudaMalloc((void**)& device_prob_arr, sizeof(float)*MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_prob_arr @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_prob_arr, 0, sizeof(float) * MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_prob_arr failed!");
		goto Error;
	}
    
    cudaStatus = cudaMalloc((void**)& device_dy_arr, sizeof(unsigned char)* 32 * MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_dy_arr @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_dy_arr, 0, sizeof(unsigned char)* 32 * MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_dy_arr failed!");
		goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_sbox_num_and_index_arr, sizeof(int)* 9 * MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_sbox_num_and_index_arr @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_sbox_num_and_index_arr, 0, sizeof(int)* 9 * MAX_BRANCH_PER_ROUND);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_sbox_num_and_index_arr failed!");
		goto Error;
    }

    //Round Input
    cudaStatus = cudaMalloc((void**)& device_dx, sizeof(unsigned char)* 32);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_dx @init failed!");
		goto Error;
    }

    cudaStatus = cudaMalloc((void**)& device_sbox_index, sizeof(int)* 8);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_sbox_index @init failed!");
		goto Error;
    }

    //Reduction Output
    cudaStatus = cudaMalloc((void**)& device_cluster_size_final, sizeof(unsigned int)* THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_cluster_size_final @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_cluster_size_final, 0, sizeof(unsigned int)* THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_cluster_size_final failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)& device_prob_final, sizeof(float)*  THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  device_prob_final @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(device_prob_final, 0, sizeof(float)*  THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_prob_final failed!");
		goto Error;
    }

    //MITM 
    cudaStatus = cudaMalloc((void**)& MITM_prob_interm_global, sizeof(float)*  Kernel_TRIFLE_t::MITM_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  MITM_prob_interm_global @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(MITM_prob_interm_global, 0, sizeof(float)*  Kernel_TRIFLE_t::MITM_size );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
		goto Error;
    }

    cudaStatus = cudaMalloc((void**)& MITM_size_interm_global, sizeof(int)*  Kernel_TRIFLE_t::MITM_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  MITM_prob_interm_global @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(MITM_size_interm_global, 0, sizeof(int)*  Kernel_TRIFLE_t::MITM_size );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
		goto Error;
    }

    cudaStatus = cudaMalloc((void**)& MITM_prob_final_global, sizeof(double)*  THREAD_PER_BLOCK * BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  MITM_prob_interm_global @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(MITM_prob_final_global, 0, sizeof(double)*  THREAD_PER_BLOCK * BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
		goto Error;
    }

    cudaStatus = cudaMalloc((void**)& MITM_size_final_global, sizeof(long long)*  THREAD_PER_BLOCK * BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc  MITM_size_final_global @init failed!");
		goto Error;
    }
    cudaStatus = cudaMemset(MITM_size_final_global, 0, sizeof(long long)*  THREAD_PER_BLOCK * BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_size_final_global failed!");
		goto Error;
    }

    //CUDA HOST ALLOC (External Coordination)
    //22 rounds maximum (LAST ROUND IS NOT STORED but reduction)

    //Pinned Device->Host output
    cudaStatus = cudaHostAlloc((void**)& pinned_host_dx_rounds, sizeof(unsigned char)*32*MAX_BRANCH_PER_ROUND*(round_to_allocate-1), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudahostMalloc failed!");
		goto Error;
    }
    
    cudaStatus = cudaHostAlloc((void**)& cur_round_prob_pinned, sizeof(float)*MAX_BRANCH_PER_ROUND*(round_to_allocate-1), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudahostMalloc failed!");
		goto Error;
    }

    cudaStatus = cudaHostAlloc((void**)& next_round_sbox_num_and_index, sizeof(int)*9*MAX_BRANCH_PER_ROUND*(round_to_allocate-1), cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudahostMalloc failed!");
		goto Error;
    }
    memset(next_round_sbox_num_and_index,0, sizeof(int)*9*MAX_BRANCH_PER_ROUND*(round_to_allocate-1));

    //Pinned Host->Device Input
    cudaStatus = cudaHostAlloc((void**)& pinned_input_dx, sizeof(unsigned char)*32, cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudahostMalloc failed!");
		goto Error;
    }
    cudaStatus = cudaHostAlloc((void**)& pinned_input_sbox_index, sizeof(int)*8, cudaHostAllocDefault);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudahostMalloc failed!");
		goto Error;
    }

    return;

    Error:
    std::cout << "\nCritical Error. Aborting Program";
	if (cudaStatus != cudaSuccess) {
		cudaError_t err = cudaGetLastError();
		std::cout << "\nCRITICAL ERROR...";
		fprintf(stderr, "\nError Code %d : %s: %s .", cudaStatus, cudaGetErrorName(err), cudaGetErrorString(err));
		std::cout << "\nPress any key to continue...";
		getchar();
	}
};

//One round kernel compute for 0 -> n-1, Last 2 parameter is both hostalloc and cudalloc
__global__ void kernel_trifle_n(unsigned char* dx, int* sbox_num, int* sbox_index, float* cur_prob, float* dev_new_prob_arr,  unsigned int* dev_cluster_size_arr){
     //Number of threads must be >=32
    if (threadIdx.x <32){
        if (threadIdx.x<16){
            TRIFLE::diff_table_size_shared[threadIdx.x] = TRIFLE::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                TRIFLE::diff_table_shared[threadIdx.x][j] = TRIFLE::diff_table_global[threadIdx.x][j];
                TRIFLE::prob_table_shared[threadIdx.x][j] = TRIFLE::freq_table_global[threadIdx.x][j]/16.0; 
            }
        }

        for (int j=0;j<16;j++){
            TRIFLE::perm_lookup_shared[threadIdx.x][j][0]= TRIFLE::perm_lookup_global[threadIdx.x][j][0];
            TRIFLE::perm_lookup_shared[threadIdx.x][j][1]= TRIFLE::perm_lookup_global[threadIdx.x][j][1];
        }
    }

    __syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    int thread_id_global = (blockIdx.x * THREAD_PER_BLOCK) + threadIdx.x; 
	float prob_thread = (*cur_prob); //Caching into local memory
    int sbox_num_local = (*sbox_num); //NOTE: difference from n-1, where parameter is not pointer
    
    //Start Range
	// 2560 different threads  1-2559
	//Each thread is responsible for 2401 array entry in result array (n-1 rounds 2 rounds results)
    //There will be 256*10 = 2560 threads handling 2401 

    //0 - 2400
    int thread_id_workforce = 2401;
	if (thread_id_global >= thread_id_workforce) {
		return;
    } 

	//1 Round for 4AS and above
	unsigned char cur_thread_partial_dy[32] = { 0 };
	//Subs for 4 AS
    int divide_factor = 1;
    int diff_freq_index;
    int remaining_value = thread_id_global; //0 - 2400

    //NOTE: Only support sbox>=4
    for (int i = 0; i < 4; i++) {
        diff_freq_index = (remaining_value / divide_factor) % 7; 

        unsigned char cur_val = dx[sbox_index[i]];

        cur_thread_partial_dy[sbox_index[i]] = TRIFLE::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy

        prob_thread *= (TRIFLE::prob_table_shared[cur_val][diff_freq_index]);

        divide_factor *= 7;
    }
    
    if (sbox_num_local >  4){
        int s6,s7,s8;
        if (sbox_num_local == 5){
            s6 = 0;
            s7 = 0; 
            s8 = 0;
        }
        else if (sbox_num_local == 6){
            s6 = 6;
            s7 = 0; 
            s8 = 0;
        }
        else if (sbox_num_local==7){
            s6 = 6;
            s7 = 6; 
            s8 = 0;
        }
        else{
            s6 = 6;
            s7 = 6; 
            s8 = 6;
        }

        for (int s8_loop_i=0; s8_loop_i <= s8; s8_loop_i++){
            for (int i=4;i<sbox_num_local;i++){
                cur_thread_partial_dy[sbox_index[i]] = dx[sbox_index[i]];
            }    

            float cur_prob_final_fourth = prob_thread;
            unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
            memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy, 32);
            cur_thread_partial_dy_RN_fourth[32] = 0;
    
            unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[7]];
            cur_thread_partial_dy_RN_fourth[sbox_index[7]] = TRIFLE::diff_table_shared[cur_val][s8_loop_i];
            cur_prob_final_fourth *= (TRIFLE::prob_table_shared[cur_val][s8_loop_i]);
        
            for (int s7_loop_i=0; s7_loop_i <= s7; s7_loop_i++){
                float cur_prob_final_third = cur_prob_final_fourth;
                unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
                memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
                //cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over
    
                cur_val = cur_thread_partial_dy_RN_third[sbox_index[6]];
                cur_thread_partial_dy_RN_third[sbox_index[6]] = TRIFLE::diff_table_shared[cur_val][s7_loop_i];
                cur_prob_final_third *= (TRIFLE::prob_table_shared[cur_val][s7_loop_i]);

                for (int s6_loop_i=0; s6_loop_i <= s6; s6_loop_i++){
                    float cur_prob_final_second = cur_prob_final_third;
                    unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
                    memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
    
                    cur_val = cur_thread_partial_dy_RN_second[sbox_index[5]];
                    cur_thread_partial_dy_RN_second[sbox_index[5]] = TRIFLE::diff_table_shared[cur_val][s6_loop_i];
                    cur_prob_final_second *= (TRIFLE::prob_table_shared[cur_val][s6_loop_i]);

                    for (int s5_loop_i=0;s5_loop_i<7;s5_loop_i++){
                        float cur_prob_final = cur_prob_final_second;
                        unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
                        memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
    
                        //Substitution final
                        cur_val = cur_thread_partial_dy_RN[sbox_index[4]];
                        cur_thread_partial_dy_RN[sbox_index[4]] = TRIFLE::diff_table_shared[cur_val][s5_loop_i];
                        cur_prob_final *= (TRIFLE::prob_table_shared[cur_val][s5_loop_i] );    

                        bool is_same = true;
                        for (int i=0;i<32;i++){
                            if (TRIFLE::final_dy_constant[i] != cur_thread_partial_dy_RN[i]){
                                is_same= false;
                                break;
                            }
                        }
                    
                        if (is_same){
                            dev_new_prob_arr[thread_id_global] += cur_prob_final;
                            dev_cluster_size_arr[thread_id_global] += 1;
                        }
                    }
                }
            }
        }
    }
    else{
        //Calculate whether to add to final dx dy
        bool is_same = true;
        for (int i=0;i<32;i++){
            if (TRIFLE::final_dy_constant[i] != cur_thread_partial_dy[i]){
                is_same= false;
                break;
            }
        }

        if (is_same){
            dev_new_prob_arr[thread_id_global] += prob_thread;
            dev_cluster_size_arr[thread_id_global] += 1;
        }
    }
};

__global__ void kernel_trifle_n_forward(unsigned char* dx, int* sbox_num, int* sbox_index, float* cur_prob, float* dev_new_prob_arr,  unsigned int* dev_cluster_size_arr,
    float* MITM_prob_interm_global, int* MITM_size_interm_global){
    //Number of threads must be >=32
   if (threadIdx.x <32){
       if (threadIdx.x<16){
           TRIFLE::diff_table_size_shared[threadIdx.x] = TRIFLE::diff_table_size_global[threadIdx.x];

           for (int j = 0; j < 8; j++) {
               TRIFLE::diff_table_shared[threadIdx.x][j] = TRIFLE::diff_table_global[threadIdx.x][j];
               TRIFLE::prob_table_shared[threadIdx.x][j] = TRIFLE::freq_table_global[threadIdx.x][j]/16.0; 
           }
       }

       for (int j=0;j<16;j++){
           TRIFLE::perm_lookup_shared[threadIdx.x][j][0]= TRIFLE::perm_lookup_global[threadIdx.x][j][0];
           TRIFLE::perm_lookup_shared[threadIdx.x][j][1]= TRIFLE::perm_lookup_global[threadIdx.x][j][1];
       }
   }

   __syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

   //Computing target array index (id and output_range)
   //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
   int thread_id_global = (blockIdx.x * THREAD_PER_BLOCK) + threadIdx.x; 
   float prob_thread = (*cur_prob); //Caching into local memory
   int sbox_num_local = (*sbox_num); //NOTE: difference from n-1, where parameter is not pointer
   
   //Start Range
   // 2560 different threads  1-2559
   //Each thread is responsible for 2401 array entry in result array (n-1 rounds 2 rounds results)
   //There will be 256*10 = 2560 threads handling 2401 

   //0 - 2400
   int thread_id_workforce = 2401;
   if (thread_id_global >= thread_id_workforce) {
       return;
   } 

   //1 Round for 4AS and above
   unsigned char cur_thread_partial_dy[32] = { 0 };
   //Subs for 4 AS
   int divide_factor = 1;
   int diff_freq_index;
   int remaining_value = thread_id_global; //0 - 2400

   //NOTE: Only support sbox>=4
   for (int i = 0; i < 4; i++) {
       diff_freq_index = (remaining_value / divide_factor) % 7; 

       unsigned char cur_val = dx[sbox_index[i]];

       cur_thread_partial_dy[sbox_index[i]] = TRIFLE::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy

       prob_thread *= (TRIFLE::prob_table_shared[cur_val][diff_freq_index]);

       divide_factor *= 7;
   }
   
   if (sbox_num_local >  4){
       int s6,s7,s8;
       if (sbox_num_local == 5){
           s6 = 0;
           s7 = 0; 
           s8 = 0;
       }
       else if (sbox_num_local == 6){
           s6 = 6;
           s7 = 0; 
           s8 = 0;
       }
       else if (sbox_num_local==7){
           s6 = 6;
           s7 = 6; 
           s8 = 0;
       }
       else{
           s6 = 6;
           s7 = 6; 
           s8 = 6;
       }

       for (int s8_loop_i=0; s8_loop_i <= s8; s8_loop_i++){
           for (int i=4;i<sbox_num_local;i++){
               cur_thread_partial_dy[sbox_index[i]] = dx[sbox_index[i]];
           }    

           float cur_prob_final_fourth = prob_thread;
           unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
           memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy, 32);
           cur_thread_partial_dy_RN_fourth[32] = 0;
   
           unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[7]];
           cur_thread_partial_dy_RN_fourth[sbox_index[7]] = TRIFLE::diff_table_shared[cur_val][s8_loop_i];
           cur_prob_final_fourth *= (TRIFLE::prob_table_shared[cur_val][s8_loop_i]);
       
           for (int s7_loop_i=0; s7_loop_i <= s7; s7_loop_i++){
               float cur_prob_final_third = cur_prob_final_fourth;
               unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
               memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
               //cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over
   
               cur_val = cur_thread_partial_dy_RN_third[sbox_index[6]];
               cur_thread_partial_dy_RN_third[sbox_index[6]] = TRIFLE::diff_table_shared[cur_val][s7_loop_i];
               cur_prob_final_third *= (TRIFLE::prob_table_shared[cur_val][s7_loop_i]);

               for (int s6_loop_i=0; s6_loop_i <= s6; s6_loop_i++){
                   float cur_prob_final_second = cur_prob_final_third;
                   unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
                   memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
   
                   cur_val = cur_thread_partial_dy_RN_second[sbox_index[5]];
                   cur_thread_partial_dy_RN_second[sbox_index[5]] = TRIFLE::diff_table_shared[cur_val][s6_loop_i];
                   cur_prob_final_second *= (TRIFLE::prob_table_shared[cur_val][s6_loop_i]);

                   for (int s5_loop_i=0;s5_loop_i<7;s5_loop_i++){
                       float cur_prob_final = cur_prob_final_second;
                       unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
                       memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
   
                       //Substitution final
                       cur_val = cur_thread_partial_dy_RN[sbox_index[4]];
                       cur_thread_partial_dy_RN[sbox_index[4]] = TRIFLE::diff_table_shared[cur_val][s5_loop_i];
                       cur_prob_final *= (TRIFLE::prob_table_shared[cur_val][s5_loop_i] );    

                        //Permutation
                        unsigned char new_partial_dy[32] = { 0 };
                        unsigned long long front_64 = 0, back_64 = 0;

                        for (int i = 0; i < 32; i++) {
                            if (cur_thread_partial_dy_RN[i] > 0) {
                                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][0];
                                back_64  |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][1];
                            }
                        }
                        for (int i = 0; i < 16; i++) {
                            new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
                        }
                        for (int i = 16; i < 32; i++) {
                            new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
                        }

                        {
                            //Check for condition
                            int sbox_num=0;
                            int sbox_index[32]={0};
                            for (int i=0;i<32;i++){
                                if (new_partial_dy[i] !=0){
                                    sbox_index[sbox_num] = i;
                                    sbox_num+=1;
                                }
                            }

                            if (sbox_num <=3){ //Possible to store three only...
                                //Computing appropriate index
                                int index=0;
                                for (int i=0;i<sbox_num;i++){
                                    index|= ( ( (sbox_index[i]&0b11111) | ( (new_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                                }

                                atomicAdd( MITM_size_interm_global+index, 1);
                                atomicAdd( MITM_prob_interm_global+index, cur_prob_final);
                            }
                        }
                   }
               }
           }
       }
   }
   else{
        //Permutation
        unsigned char new_partial_dy[32] = { 0 };
        unsigned long long front_64 = 0, back_64 = 0;

        for (int i = 0; i < 32; i++) {
            if (cur_thread_partial_dy[i] > 0) {
                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][0];
                back_64  |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][1];
            }
        }
        for (int i = 0; i < 16; i++) {
            new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
        }
        for (int i = 16; i < 32; i++) {
            new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
        }

        {
            //Check for condition
            int sbox_num=0;
            int sbox_index[32]={0};
            for (int i=0;i<32;i++){
                if (new_partial_dy[i] !=0){
                    sbox_index[sbox_num] = i;
                    sbox_num+=1;
                }
            }

            if (sbox_num <=3){ //Possible to store three only...
                //Computing appropriate index
                int index=0;
                for (int i=0;i<sbox_num;i++){
                    index|= ( ( (sbox_index[i]&0b11111) | ( (new_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                }

                atomicAdd( MITM_size_interm_global+index, 1);
                atomicAdd( MITM_prob_interm_global+index, prob_thread);
            }
        }
   }
};

__global__ void kernel_trifle_n_backward(unsigned char* dx, int* sbox_num, int* sbox_index, float* cur_prob, float* dev_new_prob_arr,  unsigned int* dev_cluster_size_arr,
    float* MITM_prob_interm_global, int* MITM_size_interm_global, double* MITM_prob_final_global, long long* MITM_size_final_global){
    //Number of threads must be >=32
    if (threadIdx.x <32){
        if (threadIdx.x<16){
            TRIFLE::diff_table_size_shared[threadIdx.x] = TRIFLE::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                //NOTE: Reversed
                TRIFLE::diff_table_shared[threadIdx.x][j] = TRIFLE::diff_table_global_reversed[threadIdx.x][j];
                TRIFLE::prob_table_shared[threadIdx.x][j] = TRIFLE::freq_table_global[threadIdx.x][j]/16.0; 
            }
        }

        for (int j=0;j<16;j++){
            //NOTE: Reversed
            TRIFLE::perm_lookup_shared[threadIdx.x][j][0]= TRIFLE::perm_lookup_global_reversed[threadIdx.x][j][0];
            TRIFLE::perm_lookup_shared[threadIdx.x][j][1]= TRIFLE::perm_lookup_global_reversed[threadIdx.x][j][1];
        }
    }

   __syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

   //Computing target array index (id and output_range)
   //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
   int thread_id_global = (blockIdx.x * THREAD_PER_BLOCK) + threadIdx.x; 
   float prob_thread = (*cur_prob); //Caching into local memory
   int sbox_num_local = (*sbox_num); //NOTE: difference from n-1, where parameter is not pointer
   
   //Start Range
   // 2560 different threads  1-2559
   //Each thread is responsible for 2401 array entry in result array (n-1 rounds 2 rounds results)
   //There will be 256*10 = 2560 threads handling 2401 

   //0 - 2400
   int thread_id_workforce = 2401;
   if (thread_id_global >= thread_id_workforce) {
       return;
   } 

   //1 Round for 4AS and above
   unsigned char cur_thread_partial_dy[32] = { 0 };
   //Subs for 4 AS
   int divide_factor = 1;
   int diff_freq_index;
   int remaining_value = thread_id_global; //0 - 2400

   //NOTE: Only support sbox>=4
   for (int i = 0; i < 4; i++) {
       diff_freq_index = (remaining_value / divide_factor) % 7; 

       unsigned char cur_val = dx[sbox_index[i]];

       cur_thread_partial_dy[sbox_index[i]] = TRIFLE::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy

       prob_thread *= (TRIFLE::prob_table_shared[cur_val][diff_freq_index]);

       divide_factor *= 7;
   }
   
   if (sbox_num_local >  4){
       int s6,s7,s8;
       if (sbox_num_local == 5){
           s6 = 0;
           s7 = 0; 
           s8 = 0;
       }
       else if (sbox_num_local == 6){
           s6 = 6;
           s7 = 0; 
           s8 = 0;
       }
       else if (sbox_num_local==7){
           s6 = 6;
           s7 = 6; 
           s8 = 0;
       }
       else{
           s6 = 6;
           s7 = 6; 
           s8 = 6;
       }

       for (int s8_loop_i=0; s8_loop_i <= s8; s8_loop_i++){
           for (int i=4;i<sbox_num_local;i++){
               cur_thread_partial_dy[sbox_index[i]] = dx[sbox_index[i]];
           }    

           float cur_prob_final_fourth = prob_thread;
           unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
           memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy, 32);
           cur_thread_partial_dy_RN_fourth[32] = 0;
   
           unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[7]];
           cur_thread_partial_dy_RN_fourth[sbox_index[7]] = TRIFLE::diff_table_shared[cur_val][s8_loop_i];
           cur_prob_final_fourth *= (TRIFLE::prob_table_shared[cur_val][s8_loop_i]);
       
           for (int s7_loop_i=0; s7_loop_i <= s7; s7_loop_i++){
               float cur_prob_final_third = cur_prob_final_fourth;
               unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
               memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
               //cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over
   
               cur_val = cur_thread_partial_dy_RN_third[sbox_index[6]];
               cur_thread_partial_dy_RN_third[sbox_index[6]] = TRIFLE::diff_table_shared[cur_val][s7_loop_i];
               cur_prob_final_third *= (TRIFLE::prob_table_shared[cur_val][s7_loop_i]);

               for (int s6_loop_i=0; s6_loop_i <= s6; s6_loop_i++){
                   float cur_prob_final_second = cur_prob_final_third;
                   unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
                   memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
   
                   cur_val = cur_thread_partial_dy_RN_second[sbox_index[5]];
                   cur_thread_partial_dy_RN_second[sbox_index[5]] = TRIFLE::diff_table_shared[cur_val][s6_loop_i];
                   cur_prob_final_second *= (TRIFLE::prob_table_shared[cur_val][s6_loop_i]);

                   for (int s5_loop_i=0;s5_loop_i<7;s5_loop_i++){
                       float cur_prob_final = cur_prob_final_second;
                       unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
                       memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
   
                       //Substitution final
                       cur_val = cur_thread_partial_dy_RN[sbox_index[4]];
                       cur_thread_partial_dy_RN[sbox_index[4]] = TRIFLE::diff_table_shared[cur_val][s5_loop_i];
                       cur_prob_final *= (TRIFLE::prob_table_shared[cur_val][s5_loop_i] );    

                        {
                            //Check for condition
                            int sbox_num=0;
                            int sbox_index[32]={0};
                            for (int i=0;i<32;i++){
                                if (cur_thread_partial_dy_RN[i] !=0){
                                    sbox_index[sbox_num] = i;
                                    sbox_num+=1;
                                }
                            }

                            if (sbox_num <=3){ //Possible to store three only...
                                //Computing appropriate index
                                int index=0;
                                for (int i=0;i<sbox_num;i++){
                                    index|= ( ( (sbox_index[i]&0b11111) | ( (cur_thread_partial_dy_RN[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                                }

                                int target_size =  MITM_size_interm_global[index];
                                if(target_size > 0){ //Exist connection
                                    double target_prob = ( (double) cur_prob_final * MITM_prob_interm_global[index]);

                                    //Add to collection
                                    MITM_prob_final_global[thread_id_global] += target_prob;
                                    MITM_size_final_global[thread_id_global] += target_size;
                                }
                            }
                        }
                   }
               }
           }
       }
   }
   else{
        {
            //Check for condition
            int sbox_num=0;
            int sbox_index[32]={0};
            for (int i=0;i<32;i++){
                if (cur_thread_partial_dy[i] !=0){
                    sbox_index[sbox_num] = i;
                    sbox_num+=1;
                }
            }

            if (sbox_num <=3){ //Possible to store three only...
                //Computing appropriate index
                int index=0;
                for (int i=0;i<sbox_num;i++){
                    index|= ( ( (sbox_index[i]&0b11111) | ( (cur_thread_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
                }

                int target_size =  MITM_size_interm_global[index];
                if(target_size > 0){ //Exist connection
                    double target_prob = ( (double) prob_thread * MITM_prob_interm_global[index]);

                    //Add to collection
                    MITM_prob_final_global[thread_id_global] += target_prob;
                    MITM_size_final_global[thread_id_global] += target_size;
                }
            }
        }
   }
};

/*One round kernel compute for 0-n (not last round)
* Number of AS should be >=4 for optimal performance (AS with smaller than 4 is undefined behvaior)
* Input - DX, Sbox_num, Sbox_index, Cur_prob
* Output - Dev_dy_arr, dev_new_prob_arr
*/
__global__ void kernel_trifle_n_minus_one(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r, 
    unsigned char* dev_dy, float* dev_new_prob_arr, int* dev_sbox_num_and_index){
    //Number of threads must be >=32
    if (threadIdx.x <32){
        if (threadIdx.x<16){
            TRIFLE::diff_table_size_shared[threadIdx.x] = TRIFLE::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                TRIFLE::diff_table_shared[threadIdx.x][j] = TRIFLE::diff_table_global[threadIdx.x][j];
                TRIFLE::prob_table_shared[threadIdx.x][j] = TRIFLE::freq_table_global[threadIdx.x][j]/16.0; 
            }
        }

        for (int j=0;j<16;j++){
            TRIFLE::perm_lookup_shared[threadIdx.x][j][0]= TRIFLE::perm_lookup_global[threadIdx.x][j][0];
            TRIFLE::perm_lookup_shared[threadIdx.x][j][1]= TRIFLE::perm_lookup_global[threadIdx.x][j][1];
        }
    }

	__syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    int thread_id_global = (blockIdx.x * THREAD_PER_BLOCK) + threadIdx.x; 
	float prob_thread = cur_prob; //Caching into local memory
	int sbox_num_local = sbox_num;

    int thread_process_amount = 1; //Amount of data to be processed by one thread
    int s5,s6,s7,s8;
    if (sbox_num_local == 4){
        thread_process_amount = 1;
    }
    else if (sbox_num_local == 5){
        thread_process_amount = 7;
        s5 = 6;
        s6 = 0;
        s7 = 0; 
        s8 = 0;
    }
    else if (sbox_num_local == 6){
        thread_process_amount = 49;
        s5 = 6;
        s6 = 6;
        s7 = 0; 
        s8 = 0;
    }
    else if (sbox_num_local==7){
        thread_process_amount = 343;
        s5 = 6;
        s6 = 6;
        s7 = 6; 
        s8 = 0;
    }
    else{
        thread_process_amount = 2401;
        s5 = 6;
        s6 = 6;
        s7 = 6; 
        s8 = 6;
    }

    unsigned char* output_dy = dev_dy + (thread_id_global*thread_process_amount*32);
    float* output_prob = dev_new_prob_arr + (thread_id_global*thread_process_amount*1);
    int* output_sbox_num = dev_sbox_num_and_index + (thread_id_global*thread_process_amount*9);
    int* output_sbox_index = output_sbox_num + 1; 

    //Start Range
	// 2560 different threads  1-2559
	//Each thread is responsible for 2401 array entry in result array (n-1 rounds 2 rounds results)
    //There will be 256*10 = 2560 threads handling 2401 

    //0 - 2400
    int thread_id_workforce = 2401;
	if (thread_id_global >= thread_id_workforce) {
		return;
    } 

	//1 Round for 4AS and above
	unsigned char cur_thread_partial_dy[32] = { 0 };
	//Subs for 4 AS
	{
		int divide_factor = 1;
		int diff_freq_index;
		int remaining_value = thread_id_global; //0 - 2400

        //NOTE: Only support sbox>=4
		for (int i = 0; i < 4; i++) {
			diff_freq_index = (remaining_value / divide_factor) % 7; 

			unsigned char cur_val = dx[sbox_index[i]];

			cur_thread_partial_dy[sbox_index[i]] = TRIFLE::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy

			prob_thread *= (TRIFLE::prob_table_shared[cur_val][diff_freq_index]);

			divide_factor *= 7;
		}
    }

    if (sbox_num_local >  4){
        for (int i=4;i<sbox_num_local;i++){
            cur_thread_partial_dy[sbox_index[i]] = dx[sbox_index[i]];
        }

        for (int s8_loop_i=0; s8_loop_i <= s8; s8_loop_i++){
            float cur_prob_final_fourth = prob_thread;
            unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
            memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy, 32);
            cur_thread_partial_dy_RN_fourth[32] = 0;
    
            unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[7]];
            cur_thread_partial_dy_RN_fourth[sbox_index[7]] = TRIFLE::diff_table_shared[cur_val][s8_loop_i];
            cur_prob_final_fourth *= (TRIFLE::prob_table_shared[cur_val][s8_loop_i]);
        
            for (int s7_loop_i=0; s7_loop_i <= s7; s7_loop_i++){
                float cur_prob_final_third = cur_prob_final_fourth;
                unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
                memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
                //cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over
    
                cur_val = cur_thread_partial_dy_RN_third[sbox_index[6]];
                cur_thread_partial_dy_RN_third[sbox_index[6]] = TRIFLE::diff_table_shared[cur_val][s7_loop_i];
                cur_prob_final_third *= (TRIFLE::prob_table_shared[cur_val][s7_loop_i]);

                for (int s6_loop_i=0; s6_loop_i <= s6; s6_loop_i++){
                    float cur_prob_final_second = cur_prob_final_third;
                    unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
                    memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
    
                    cur_val = cur_thread_partial_dy_RN_second[sbox_index[5]];
                    cur_thread_partial_dy_RN_second[sbox_index[5]] = TRIFLE::diff_table_shared[cur_val][s6_loop_i];
                    cur_prob_final_second *= (TRIFLE::prob_table_shared[cur_val][s6_loop_i]);

                    for (int s5_loop_i=0;s5_loop_i<7;s5_loop_i++){
                        float cur_prob_final = cur_prob_final_second;
                        unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
                        memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
    
                        //Substitution final
                        cur_val = cur_thread_partial_dy_RN[sbox_index[4]];
                        cur_thread_partial_dy_RN[sbox_index[4]] = TRIFLE::diff_table_shared[cur_val][s5_loop_i];
                        cur_prob_final *= (TRIFLE::prob_table_shared[cur_val][s5_loop_i] );    

                        //Permutation
                        unsigned long long front_64 = 0, back_64 = 0;
                        for (int i = 0; i < 32; i++) {
                            if ( cur_thread_partial_dy_RN[i] > 0) {
                                //Permutation LUTable
                                //25% less running time compared to normal computation
                                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][0];
                                back_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][1];
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            cur_thread_partial_dy_RN[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
                        }
                        #pragma unroll
                        for (int i = 16; i < 32; i++) {
                            cur_thread_partial_dy_RN[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
                        }
                        //cur_thread_partial_dy is already subbed and permutated..
                        //prob_thread records its probabilities

                        //Calculte sbox index and sbox number
                        int cur_sbox_num = 0;
                        int cur_sbox_index[32]; //Will point to non existance 32 array entry (see substitution below)
                        #pragma unroll
                        for (int i=0;i<8;i++){
                            cur_sbox_index[i] = 32;
                        }
                        #pragma unroll
                        for (int i = 0; i < 32; i++) {
                            if ((cur_thread_partial_dy_RN[i] & 0xf) > 0) {
                                cur_sbox_index[cur_sbox_num] = i;
                                cur_sbox_num++;
                            }
                        }

                        //Calculate Matsui Bound
                        if (cur_sbox_num <= MAX_AS_USED)  //If only next round AS <= 8
                        {
                            //MATSUI BOUND
                            float estimated_com_prob = ( powf(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_FORWARD - cur_r - 2)) * powf(CLUSTER_1AS_BEST_PROB, cur_sbox_num) );
                            if ((estimated_com_prob * cur_prob_final) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
                            // if ((estimated_com_prob * cur_prob_final) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
                            // if (true) {
                                //Save everything
                                memcpy(output_dy,cur_thread_partial_dy_RN,32);
                                *output_prob = cur_prob_final;
                                *output_sbox_num = cur_sbox_num;
                                memcpy(output_sbox_index, cur_sbox_index, sizeof(int) * 8 );
                            } else{
                                *output_sbox_num = 0; //Indicate jump over this
                            }
                        }
                        else{
                            *output_sbox_num = 0; //Indicate jump over this
                        }

                        //Calculate the next set of address to save to
                        output_dy = output_dy + 32;
                        output_prob = output_prob + 1;
                        output_sbox_num = output_sbox_num + 9;
                        output_sbox_index = output_sbox_index + 9;
                    }
                }
            }
        }
    }
    else{
        //Permutation
        unsigned long long front_64 = 0, back_64 = 0;
        for (int i = 0; i < 32; i++) {
            if ( cur_thread_partial_dy[i] > 0) {
                //Permutation LUTable
                //25% less running time compared to normal computation
                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][0];
                back_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][1];
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            cur_thread_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
        }
        #pragma unroll
        for (int i = 16; i < 32; i++) {
            cur_thread_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
        }
        //cur_thread_partial_dy is already subbed and permutated..
        //prob_thread records its probabilities
        
        //Calculte sbox index and sbox number
        int cur_sbox_num = 0;
        int cur_sbox_index[32]; //Will point to non existance 32 array entry (see substitution below)
        for (int i=0;i<8;i++){
            cur_sbox_index[i] = 32;
        }
        for (int i = 0; i < 32; i++) {
            if ((cur_thread_partial_dy[i] & 0xf) > 0) {
                cur_sbox_index[cur_sbox_num] = i;
                cur_sbox_num++;
            }
        }

        //Calculate Matsui Bound
        if (cur_sbox_num <= MAX_AS_USED)  //If only next round AS <= 8
        {
            //MATSUI BOUND
            float estimated_com_prob = ( powf(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_FORWARD - cur_r - 2)) * powf(CLUSTER_1AS_BEST_PROB, cur_sbox_num) );
            if ((estimated_com_prob * prob_thread) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
            // if (true) {
                //Save everything
                memcpy(output_dy,cur_thread_partial_dy,32);
                *output_prob = prob_thread;
                *output_sbox_num = cur_sbox_num;
                memcpy(output_sbox_index, cur_sbox_index, sizeof(int) * 8 );
            } else{
                *output_sbox_num = 0; //Indicate jump over this
            }
        }
        else{
            *output_sbox_num = 0; //Indicate jump over this
        }
    }
};

__global__ void kernel_trifle_n_minus_one_backward(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r, 
    unsigned char* dev_dy, float* dev_new_prob_arr, int* dev_sbox_num_and_index){
    //NOTE: Different between this and normal one is only during copying to shared variable

    //Number of threads must be >=32
    if (threadIdx.x <32){
        if (threadIdx.x<16){
            TRIFLE::diff_table_size_shared[threadIdx.x] = TRIFLE::diff_table_size_global[threadIdx.x];

            for (int j = 0; j < 8; j++) {
                //NOTE: Reversed
                TRIFLE::diff_table_shared[threadIdx.x][j] = TRIFLE::diff_table_global_reversed[threadIdx.x][j];
                TRIFLE::prob_table_shared[threadIdx.x][j] = TRIFLE::freq_table_global[threadIdx.x][j]/16.0; 
            }
        }

        for (int j=0;j<16;j++){
            //NOTE: Reversed
            TRIFLE::perm_lookup_shared[threadIdx.x][j][0]= TRIFLE::perm_lookup_global_reversed[threadIdx.x][j][0];
            TRIFLE::perm_lookup_shared[threadIdx.x][j][1]= TRIFLE::perm_lookup_global_reversed[threadIdx.x][j][1];
        }
    }

	__syncthreads(); //wait for init to be finished, sync up all threads within a block... shared memory lies within each block.

	//Computing target array index (id and output_range)
    //I - THREAD ID / total thread (including all block) - Used to coordinate splitting of tasks
    int thread_id_global = (blockIdx.x * THREAD_PER_BLOCK) + threadIdx.x; 
	float prob_thread = cur_prob; //Caching into local memory
	int sbox_num_local = sbox_num;

    int thread_process_amount = 1; //Amount of data to be processed by one thread
    int s5,s6,s7,s8;
    if (sbox_num_local == 4){
        thread_process_amount = 1;
    }
    else if (sbox_num_local == 5){
        thread_process_amount = 7;
        s5 = 6;
        s6 = 0;
        s7 = 0; 
        s8 = 0;
    }
    else if (sbox_num_local == 6){
        thread_process_amount = 49;
        s5 = 6;
        s6 = 6;
        s7 = 0; 
        s8 = 0;
    }
    else if (sbox_num_local==7){
        thread_process_amount = 343;
        s5 = 6;
        s6 = 6;
        s7 = 6; 
        s8 = 0;
    }
    else{
        thread_process_amount = 2401;
        s5 = 6;
        s6 = 6;
        s7 = 6; 
        s8 = 6;
    }

    unsigned char* output_dy = dev_dy + (thread_id_global*thread_process_amount*32);
    float* output_prob = dev_new_prob_arr + (thread_id_global*thread_process_amount*1);
    int* output_sbox_num = dev_sbox_num_and_index + (thread_id_global*thread_process_amount*9);
    int* output_sbox_index = output_sbox_num + 1; 

    //Start Range
	// 2560 different threads  1-2559
	//Each thread is responsible for 2401 array entry in result array (n-1 rounds 2 rounds results)
    //There will be 256*10 = 2560 threads handling 2401 

    //0 - 2400
    int thread_id_workforce = 2401;
	if (thread_id_global >= thread_id_workforce) {
		return;
    } 

	//1 Round for 4AS and above
	unsigned char cur_thread_partial_dy[32] = { 0 };
	//Subs for 4 AS
	{
		int divide_factor = 1;
		int diff_freq_index;
		int remaining_value = thread_id_global; //0 - 2400

        //NOTE: Only support sbox>=4
		for (int i = 0; i < 4; i++) {
			diff_freq_index = (remaining_value / divide_factor) % 7; 

			unsigned char cur_val = dx[sbox_index[i]];

			cur_thread_partial_dy[sbox_index[i]] = TRIFLE::diff_table_shared[cur_val][diff_freq_index]; //Assigning target val to partial_dy

			prob_thread *= (TRIFLE::prob_table_shared[cur_val][diff_freq_index]);

			divide_factor *= 7;
		}
    }

    if (sbox_num_local >  4){
        for (int i=4;i<sbox_num_local;i++){
            cur_thread_partial_dy[sbox_index[i]] = dx[sbox_index[i]];
        }

        for (int s8_loop_i=0; s8_loop_i <= s8; s8_loop_i++){
            float cur_prob_final_fourth = prob_thread;
            unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
            memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy, 32);
            cur_thread_partial_dy_RN_fourth[32] = 0;
    
            unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[7]];
            cur_thread_partial_dy_RN_fourth[sbox_index[7]] = TRIFLE::diff_table_shared[cur_val][s8_loop_i];
            cur_prob_final_fourth *= (TRIFLE::prob_table_shared[cur_val][s8_loop_i]);
        
            for (int s7_loop_i=0; s7_loop_i <= s7; s7_loop_i++){
                float cur_prob_final_third = cur_prob_final_fourth;
                unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
                memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
                //cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over
    
                cur_val = cur_thread_partial_dy_RN_third[sbox_index[6]];
                cur_thread_partial_dy_RN_third[sbox_index[6]] = TRIFLE::diff_table_shared[cur_val][s7_loop_i];
                cur_prob_final_third *= (TRIFLE::prob_table_shared[cur_val][s7_loop_i]);

                for (int s6_loop_i=0; s6_loop_i <= s6; s6_loop_i++){
                    float cur_prob_final_second = cur_prob_final_third;
                    unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
                    memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
    
                    cur_val = cur_thread_partial_dy_RN_second[sbox_index[5]];
                    cur_thread_partial_dy_RN_second[sbox_index[5]] = TRIFLE::diff_table_shared[cur_val][s6_loop_i];
                    cur_prob_final_second *= (TRIFLE::prob_table_shared[cur_val][s6_loop_i]);

                    for (int s5_loop_i=0;s5_loop_i<7;s5_loop_i++){
                        float cur_prob_final = cur_prob_final_second;
                        unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
                        memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
    
                        //Substitution final
                        cur_val = cur_thread_partial_dy_RN[sbox_index[4]];
                        cur_thread_partial_dy_RN[sbox_index[4]] = TRIFLE::diff_table_shared[cur_val][s5_loop_i];
                        cur_prob_final *= (TRIFLE::prob_table_shared[cur_val][s5_loop_i] );    

                        //Permutation
                        unsigned long long front_64 = 0, back_64 = 0;
                        for (int i = 0; i < 32; i++) {
                            if ( cur_thread_partial_dy_RN[i] > 0) {
                                //Permutation LUTable
                                //25% less running time compared to normal computation
                                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][0];
                                back_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy_RN[i]][1];
                            }
                        }
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            cur_thread_partial_dy_RN[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
                        }
                        #pragma unroll
                        for (int i = 16; i < 32; i++) {
                            cur_thread_partial_dy_RN[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
                        }
                        //cur_thread_partial_dy is already subbed and permutated..
                        //prob_thread records its probabilities

                        //Calculte sbox index and sbox number
                        int cur_sbox_num = 0;
                        int cur_sbox_index[32]; //Will point to non existance 32 array entry (see substitution below)
                        #pragma unroll
                        for (int i=0;i<8;i++){
                            cur_sbox_index[i] = 32;
                        }
                        #pragma unroll
                        for (int i = 0; i < 32; i++) {
                            if ((cur_thread_partial_dy_RN[i] & 0xf) > 0) {
                                cur_sbox_index[cur_sbox_num] = i;
                                cur_sbox_num++;
                            }
                        }

                        //Calculate Matsui Bound
                        if (cur_sbox_num <= MAX_AS_USED)  //If only next round AS <= 8
                        {
                            //MATSUI BOUND
                            float estimated_com_prob = ( powf(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_BACKWARD - cur_r - 2)) * powf(CLUSTER_1AS_BEST_PROB, cur_sbox_num) );
                            if ((estimated_com_prob * cur_prob_final) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
                            // if ((estimated_com_prob * cur_prob_final) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
                            // if (true) {
                                //Save everything
                                memcpy(output_dy,cur_thread_partial_dy_RN,32);
                                *output_prob = cur_prob_final;
                                *output_sbox_num = cur_sbox_num;
                                memcpy(output_sbox_index, cur_sbox_index, sizeof(int) * 8 );
                            } else{
                                *output_sbox_num = 0; //Indicate jump over this
                            }
                        }
                        else{
                            *output_sbox_num = 0; //Indicate jump over this
                        }

                        //Calculate the next set of address to save to
                        output_dy = output_dy + 32;
                        output_prob = output_prob + 1;
                        output_sbox_num = output_sbox_num + 9;
                        output_sbox_index = output_sbox_index + 9;
                    }
                }
            }
        }
    }
    else{
        //Permutation
        unsigned long long front_64 = 0, back_64 = 0;
        for (int i = 0; i < 32; i++) {
            if ( cur_thread_partial_dy[i] > 0) {
                //Permutation LUTable
                //25% less running time compared to normal computation
                front_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][0];
                back_64 |= TRIFLE::perm_lookup_shared[i][cur_thread_partial_dy[i]][1];
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            cur_thread_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
        }
        #pragma unroll
        for (int i = 16; i < 32; i++) {
            cur_thread_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
        }
        //cur_thread_partial_dy is already subbed and permutated..
        //prob_thread records its probabilities
        
        //Calculte sbox index and sbox number
        int cur_sbox_num = 0;
        int cur_sbox_index[32]; //Will point to non existance 32 array entry (see substitution below)
        for (int i=0;i<8;i++){
            cur_sbox_index[i] = 32;
        }
        for (int i = 0; i < 32; i++) {
            if ((cur_thread_partial_dy[i] & 0xf) > 0) {
                cur_sbox_index[cur_sbox_num] = i;
                cur_sbox_num++;
            }
        }

        //Calculate Matsui Bound
        if (cur_sbox_num <= MAX_AS_USED)  //If only next round AS <= 8
        {
            //MATSUI BOUND
            float estimated_com_prob = ( powf(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_BACKWARD - cur_r - 2)) * powf(CLUSTER_1AS_BEST_PROB, cur_sbox_num) );
            if ((estimated_com_prob * prob_thread) >= TRIFLE::CLUSTER_PROB_BOUND_const) {
            // if (true) {
                //Save everything
                memcpy(output_dy,cur_thread_partial_dy,32);
                *output_prob = prob_thread;
                *output_sbox_num = cur_sbox_num;
                memcpy(output_sbox_index, cur_sbox_index, sizeof(int) * 8 );
            } else{
                *output_sbox_num = 0; //Indicate jump over this
            }
        }
        else{
            *output_sbox_num = 0; //Indicate jump over this
        }
    }
};

void Kernel_TRIFLE_t::kernel_compute_1round(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r,
    unsigned char* next_round_dx_array_pinned, float* cur_round_prob_pinned, int* next_round_sbox_num_and_index){
    cudaError_t cudaStatus;

    //Input Copy
    cudaStatus = cudaMemcpyAsync(device_dx, dx, sizeof(unsigned char) * 32, cudaMemcpyHostToDevice, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (next_round_dx_array_pinned) failed!");
        }
    #endif

    cudaStatus = cudaMemcpyAsync(device_sbox_index, sbox_index, sizeof(int) * 8, cudaMemcpyHostToDevice, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (sbox_index) failed!");
        }
    #endif

    //Grid Size (Num Block), Block Size, Shared Memory dynamically allocated, stream 
    kernel_trifle_n_minus_one<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, (this->stream_obj)>>> (this->device_dx, this->device_sbox_index, sbox_num, cur_prob, cur_r,
        this->device_dy_arr, this->device_prob_arr, this->device_sbox_num_and_index_arr);

    cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error in kernel launch: %s - %s\n",  cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
      exit(-1);
    }
    
    //Output Copy
    //How much to copy out depends on sbox_num

    // int total_branch = std::pow(7, sbox_num);

    // cudaStatus = cudaMemcpyAsync(next_round_dx_array_pinned, device_dy_arr, sizeof(unsigned char) * 32 * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    // #ifdef CUDA_ERROR_PRINT
    //     if (cudaStatus != cudaSuccess) {
    //         fprintf(stderr, "cudaMemcpyAsync (next_round_dx_array_pinned) failed!");
    //     }
    // #endif

    // cudaStatus = cudaMemcpyAsync(cur_round_prob_pinned, device_prob_arr, sizeof(float) * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    // #ifdef CUDA_ERROR_PRINT
    //     if (cudaStatus != cudaSuccess) {
    //         fprintf(stderr, "cudaMemcpyAsync (cur_round_prob_pinned) failed!");
    //     }
    // #endif

    // cudaStatus = cudaMemcpyAsync(next_round_sbox_num_and_index, device_sbox_num_and_index_arr, sizeof(int) * 9 * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    // #ifdef CUDA_ERROR_PRINT
    //     if (cudaStatus != cudaSuccess) {
    //         fprintf(stderr, "cudaMemcpyAsync (next_round_sbox_num_and_index) failed!");
    //     }
    // #endif


    //Wait until complete
    cudaStatus = cudaStreamSynchronize(this->stream_obj);
    if(cudaStatus != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
      exit(-1);
    }
};

void Kernel_TRIFLE_t::kernel_compute_1round_backward(unsigned char* dx, int* sbox_index, int sbox_num, float cur_prob, int cur_r,
    unsigned char* next_round_dx_array_pinned, float* cur_round_prob_pinned, int* next_round_sbox_num_and_index){
    cudaError_t cudaStatus;

    //Input Copy
    cudaStatus = cudaMemcpyAsync(device_dx, dx, sizeof(unsigned char) * 32, cudaMemcpyHostToDevice, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (next_round_dx_array_pinned) failed!");
        }
    #endif

    cudaStatus = cudaMemcpyAsync(device_sbox_index, sbox_index, sizeof(int) * 8, cudaMemcpyHostToDevice, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (sbox_index) failed!");
        }
    #endif

    //Grid Size (Num Block), Block Size, Shared Memory dynamically allocated, stream 
    kernel_trifle_n_minus_one_backward<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, (this->stream_obj)>>> (this->device_dx, this->device_sbox_index, sbox_num, cur_prob, cur_r,
        this->device_dy_arr, this->device_prob_arr, this->device_sbox_num_and_index_arr);

    cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error in kernel launch: %s - %s\n",  cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
      exit(-1);
    }
    
    //Output Copy
    //How much to copy out depends on sbox_num
    int total_branch = std::pow(7, sbox_num);

    cudaStatus = cudaMemcpyAsync(next_round_dx_array_pinned, device_dy_arr, sizeof(unsigned char) * 32 * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (next_round_dx_array_pinned) failed!");
        }
    #endif

    cudaStatus = cudaMemcpyAsync(cur_round_prob_pinned, device_prob_arr, sizeof(float) * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (cur_round_prob_pinned) failed!");
        }
    #endif

    cudaStatus = cudaMemcpyAsync(next_round_sbox_num_and_index, device_sbox_num_and_index_arr, sizeof(int) * 9 * total_branch, cudaMemcpyDeviceToHost, (this->stream_obj) );
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyAsync (next_round_sbox_num_and_index) failed!");
        }
    #endif

    //Wait until complete
    cudaStatus = cudaStreamSynchronize(this->stream_obj);
    if(cudaStatus != cudaSuccess)
    {
      // print the CUDA error message and exit
      printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
      exit(-1);
    }
};

void Kernel_TRIFLE_t::kernel_compute_1round_last(int offset_index){
    //unsigned char* dx, int* sbox_num, int* sbox_index, float* cur_prob, float* dev_new_prob_arr, int* dev_cluster_size_arr
    // ( device_dy_arr+(offset_index * 32), device_sbox_num_and_index_arr + (offset_index * 9) 
    // , device_sbox_num_and_index_arr+1+(offset_index*9), device_prob_arr+(offset_index)
    // , this->device_prob_final ,this->device_cluster_size_final );

    kernel_trifle_n<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, (this->stream_obj)>>> 
        ( device_dy_arr+(offset_index * 32), device_sbox_num_and_index_arr + (offset_index * 9) 
            , device_sbox_num_and_index_arr+(offset_index*9)+1, device_prob_arr+(offset_index)
            , this->device_prob_final ,this->device_cluster_size_final );

    auto cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error in kernel launch: %s - %s\n",  cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
        exit(-1);
    }
};

void Kernel_TRIFLE_t::MITM_kernel_compute_1round_last_forward(int offset_index){

    kernel_trifle_n_forward<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, (this->stream_obj)>>> 
        ( device_dy_arr+(offset_index * 32), device_sbox_num_and_index_arr + (offset_index * 9) 
            , device_sbox_num_and_index_arr+(offset_index*9)+1, device_prob_arr+(offset_index)
            , this->device_prob_final ,this->device_cluster_size_final ,MITM_prob_interm_global,MITM_size_interm_global 
            );
        
    auto cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error in kernel launch: %s - %s\n",  cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
        exit(-1);
    }
}

void Kernel_TRIFLE_t::MITM_kernel_compute_1round_last_backward(int offset_index)
{
    kernel_trifle_n_backward<<<BLOCK_NUM, THREAD_PER_BLOCK, 0, (this->stream_obj)>>> 
        ( device_dy_arr+(offset_index * 32), device_sbox_num_and_index_arr + (offset_index * 9) 
            , device_sbox_num_and_index_arr+(offset_index*9)+1, device_prob_arr+(offset_index)
            , this->device_prob_final ,this->device_cluster_size_final
            , MITM_prob_interm_global,MITM_size_interm_global 
            , MITM_prob_final_global, MITM_size_final_global );

    auto cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error in kernel launch: %s - %s\n",  cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
        exit(-1);
    }
}

void Kernel_TRIFLE_t::kernel_MITM_intermediate_reduction(float* prob_interm, int* size_interm){
    float* prob_staging = new float[MITM_size];
    int* size_staging = new int[MITM_size];

    auto cudaStatus = cudaMemcpy(prob_staging, MITM_prob_interm_global, sizeof(float)* MITM_size, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (next_round_dx_array_pinned) failed!");\
            getchar();
        }
    #endif

    cudaStatus = cudaMemcpy(size_staging, MITM_size_interm_global, sizeof(int) * MITM_size, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (cur_round_prob_pinned) failed!");
            getchar();
        }
    #endif

    for (int i=0;i< MITM_size;i++){ //Automatically modify back to the source
        prob_interm[i] += prob_staging[i];
        size_interm[i] += size_staging[i];
    }

    cudaStatus = cudaMemcpy(MITM_prob_interm_global, prob_interm, sizeof(float) * MITM_size, cudaMemcpyHostToDevice);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (cur_round_prob_pinned) failed!");
            getchar();
        }
    #endif

    cudaStatus = cudaMemcpy(MITM_size_interm_global, size_interm, sizeof(int) * MITM_size, cudaMemcpyHostToDevice);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (cur_round_prob_pinned) failed!");
            getchar();
        }
    #endif

    delete[] prob_staging;
    delete[] size_staging; 
}


void Kernel_TRIFLE_t::kernel_reduction(long long& gpu_cluster_size, double &gpu_cluster_prob){
    unsigned int device_cluster_size_final_arr[THREAD_PER_BLOCK * BLOCK_NUM]; //unsigned int * thread num * thread block 
    float device_prob_final_arr[THREAD_PER_BLOCK * BLOCK_NUM];       //float * thread_num * thread_block

    long long MITM_final_size_arr[THREAD_PER_BLOCK * BLOCK_NUM];
    double MITM_final_prob_arr[THREAD_PER_BLOCK * BLOCK_NUM];

    auto cudaStatus = cudaMemcpy(device_cluster_size_final_arr, device_cluster_size_final, sizeof(unsigned int)* THREAD_PER_BLOCK*BLOCK_NUM, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (next_round_dx_array_pinned) failed!");\
            getchar();
        }
    #endif

    cudaStatus = cudaMemcpy(device_prob_final_arr, device_prob_final, sizeof(float) * THREAD_PER_BLOCK*BLOCK_NUM, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (cur_round_prob_pinned) failed!");
            getchar();
        }
    #endif

    for (int i=0;i<(THREAD_PER_BLOCK*BLOCK_NUM); i++ ){
        gpu_cluster_size += device_cluster_size_final_arr[i];
        gpu_cluster_prob += device_prob_final_arr[i];
    }

    cudaStatus = cudaMemcpy(MITM_final_size_arr, MITM_size_final_global, sizeof(long long)* THREAD_PER_BLOCK*BLOCK_NUM, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (next_round_dx_array_pinned) failed!");\
            getchar();
        }
    #endif

    cudaStatus = cudaMemcpy(MITM_final_prob_arr, MITM_prob_final_global, sizeof(double) * THREAD_PER_BLOCK*BLOCK_NUM, cudaMemcpyDeviceToHost);
    #ifdef CUDA_ERROR_PRINT
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy (cur_round_prob_pinned) failed!");
            getchar();
        }
    #endif

    for (int i=0;i<(THREAD_PER_BLOCK*BLOCK_NUM); i++ ){
        gpu_cluster_size += MITM_final_size_arr[i];
        gpu_cluster_prob += MITM_final_prob_arr[i];
    }

}   

void Kernel_TRIFLE_t::change_parameter(unsigned char* new_dy, unsigned char* new_dx){
    //Set DX and DY
    auto cudaStatus = cudaMemcpyToSymbol(TRIFLE::final_dy_constant, new_dy, sizeof(unsigned char)*32);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpyToSymbol final_dy_constant failed!");
		goto Error;
    }
    
    memcpy(TRIFLE::final_dy_host,new_dy,32);
    if ( new_dx!=nullptr ){
        memcpy(TRIFLE::ref_dx_host,new_dx,32);
    }

    //Reset Result
    cudaStatus = cudaMemset(device_prob_final, 0, sizeof(float)*  THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_prob_final failed!");
		goto Error;
    }

    cudaStatus = cudaMemset(device_cluster_size_final, 0, sizeof(unsigned int)* THREAD_PER_BLOCK * BLOCK_NUM);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset device_cluster_size_final failed!");
		goto Error;
    }

    cudaStatus = cudaMemset(MITM_prob_interm_global, 0, sizeof(float)*  Kernel_TRIFLE_t::MITM_size );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_prob_interm_global failed!");
		goto Error;
    }

    cudaStatus = cudaMemset(MITM_size_interm_global, 0, sizeof(int)*  Kernel_TRIFLE_t::MITM_size );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_size_interm_global failed!");
		goto Error;
    }

    cudaStatus = cudaMemset(MITM_prob_final_global, 0, sizeof(double)*  THREAD_PER_BLOCK*BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_prob_final_global failed!");
		goto Error;
    }

    cudaStatus = cudaMemset(MITM_size_final_global, 0, sizeof(long long)*  THREAD_PER_BLOCK*BLOCK_NUM );
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset MITM_size_final_global failed!");
		goto Error;
    }
    
    return;

    Error:
    return;
}
