#include "trifle.h"
#include "common.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_profiler_api.h>

#include <chrono>
#include <thread>

//Bundle launch feature referenced in this section is incomplete. 

// //  GPU with cpu only
//  Rounds Trails 0 : 7
//  Rounds Trails 1 : 2905
//  Rounds Trails 2 : 141085
//  Rounds Trails 3 : 4405569
//  Rounds Trails 4 : 18994976
//  Rounds Trails 5 : 50225840
//  Rounds Trails 6 : 67951681
//  Rounds Trails 7 : 79148797
//  Rounds Trails 8 : 52947118
//  Rounds Trails 9 : 38605875

//  CPU
//  Rounds Trails 0 : 7
//  Rounds Trails 1 : 2905
//  Rounds Trails 2 : 141085
//  Rounds Trails 3 : 4405569
//  Rounds Trails 4 : 18994976
//  Rounds Trails 5 : 50225840
//  Rounds Trails 6 : 67951681
//  Rounds Trails 7 : 79148797
//  Rounds Trails 8 : 52947118
//  Rounds Trails 9 : 38605875

// GPU CUDA
//  Rounds Trails 0 : 7
//  Rounds Trails 1 : 2905
//  Rounds Trails 2 : 141085
//  Rounds Trails 3 : 4405569
//  Rounds Trails 4 : 18994976
//  Rounds Trails 5 : 50225840
//  Rounds Trails 6 : 67951681
//  Rounds Trails 7 : 79148797
//  Rounds Trails 8 : 52947118
//  Rounds Trails 9 : 38605875

// #define RECORD_EVERYTHING
#define PRINT_DIFFERENCE_BETWEEN_CPU_GPU

//Code path of GPU 1-4
//#define GPU_BUNDLED_LAUNCH

unsigned long long last_round_trails[22]={0};

//This can be one struct if needed to be multithreaded
struct TRIFLE_GPU_manager_t{
	//Cluster Output (Partial)
	//For last rounds
	long long cluster_size=0;
	double cluster_prob=0.0;

	//CPU
	int cluster_size_cpu_only=0;
	double cluster_prob_cpu_only=0.0;

	//MITM
	const int mitm_cache_array_length = 134217728; //3 Sbox * 9 (5pos + 4 val) bit = 27 bit = this value
	int* mitm_cluster_size_cache;
	float* mitm_prob_cache;

	//Output pinned
	//21/MAX_ROUND_MINUS_ONE Rounds allocated...
	unsigned char *next_round_dx_array_pinned; //sizeof(unsigned char)*32*MAX_BRANCH_PER_ROUND*21
	float *cur_round_prob_pinned; //sizeof(float)*MAX_BRANCH_PER_ROUND*21
	int* next_round_sbox_num_and_index; //sizeof(int)*9 *MAX_BRANCH_PER_ROUND*21 -> memsetted to 0

	//Input pinned
	unsigned char* dx_pinned;//unsigned char * 32
	int* sbox_index_pinned; //int * 8

	//GPU Bundled Launch (have to sort 1, 2, 3)
	int* index_sorting_bundled; //sizeof (int) * 21 rounds * 8 * MAX_BRANCH_PER_ROUND

	TRIFLE_GPU_manager_t(){
		//Bucket Sort (one layer)
		index_sorting_bundled = new int[MAX_POSSIBLE_ROUND_MINUS_ONE*8*MAX_BRANCH_PER_ROUND]; //21? (MAX) Rounds, 8 Sbox 

		for (int i=0;i< MAX_POSSIBLE_ROUND_MINUS_ONE*8*MAX_BRANCH_PER_ROUND;i++){
			index_sorting_bundled[i] = 0;
		}

		//MITM - CPU Portion
		mitm_cluster_size_cache = new int[mitm_cache_array_length];
		mitm_prob_cache = new float[mitm_cache_array_length];

		for (int i=0;i< mitm_cache_array_length;i++){
			mitm_cluster_size_cache[i] = 0;
			mitm_prob_cache[i] = 0;
		}
	}

	~TRIFLE_GPU_manager_t(){
		delete[] index_sorting_bundled;
		delete[] mitm_cluster_size_cache;
		delete[] mitm_prob_cache;
	}

	void reduction(){
		long long gpu_size=0;
		double gpu_prob=0;

		this->trifle_manager->kernel_reduction(gpu_size, gpu_prob);

		//Combined CPU (if any) with gpu
		this->cluster_size += gpu_size;
		this->cluster_prob += gpu_prob;
	}

	void reset(){
		for (int i=0;i< mitm_cache_array_length;i++){
			mitm_cluster_size_cache[i] = 0;
			mitm_prob_cache[i] = 0;
		}

		this->cluster_size = 0;
		this->cluster_prob = 0;
	}

	Kernel_TRIFLE_t* trifle_manager;
};
const int cpu_thread_size=1;
TRIFLE_GPU_manager_t trifle_gpu_manager_arr[cpu_thread_size];

//Enum
enum PROCESSING_METHOD {CPU, GPU};

void trifle_init(){
	TRIFLE::init();

	std::cout << "\nPattern ROund Forward : " << PATTERN_ROUND_MITM_FORWARD;
	std::cout << "\nPattern ROund Backward : " << PATTERN_ROUND_MITM_BACKWARD; 
	std::cout << "\nPATTERN_ROUND : " << PATTERN_ROUND;
	std::cout << "\nMaximum Prob: - " << CLUSTER_PROB_BOUND_LOG2;

	//TODO temprorary init one thread
	std::cout << "\n Init GPU resources";
	std::cout.flush();
	int thread_id=0;
	trifle_gpu_manager_arr[thread_id].trifle_manager = new Kernel_TRIFLE_t(thread_id,trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned
		,trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned, trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index,
		trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned);
	// thread_id=1;
	// trifle_gpu_manager_arr[thread_id].trifle_manager = new Kernel_TRIFLE_t(thread_id,trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned
	// 	,trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned, trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index,
	// 	trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned);
	std::cout << "\nInit GPU completed\n----\n";
};

//Support Sbox<=4 (Forward Search)
void trifle_diff_cluster_gpu_using_cpu(unsigned char* output_dy, float* output_prob, 
	int* output_sbox_num, int* output_sbox_index,unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index, float cur_prob){
	int second_level_limit = (cur_num_sbox >= 2 ? 6 : 0);
	int third_level_limit = (cur_num_sbox >= 3 ? 6 : 0);
	int fourth_level_limit = (cur_num_sbox >= 4 ? 6 : 0);

	// int five_level_limit = (cur_num_sbox >= 5 ? 6 : 0);
	// int six_level_limit = (cur_num_sbox >= 6 ? 6 : 0);
	// int seven_level_limit = (cur_num_sbox >= 7 ? 6 : 0);
	// int eight_level_limit = (cur_num_sbox >= 8 ? 6 : 0);

	// //int output_size_all_local=0;

	// //Emulating branching
	// for (int i = 0; i <= eight_level_limit; i++) {
	// 	float cur_prob_final_eight = cur_prob;
	// 	unsigned char cur_thread_partial_dy_RN_eight[33]; //The [32] or size 33 is for fast procesing only
	// 	memcpy(cur_thread_partial_dy_RN_eight, cur_partial_dy, 32);
	// 	cur_thread_partial_dy_RN_eight[32] = 0;

	// 	unsigned char cur_val = cur_thread_partial_dy_RN_eight[sbox_index[7]];
	// 	cur_thread_partial_dy_RN_eight[sbox_index[7]] = TRIFLE::diff_table_host[cur_val][i];
	// 	cur_prob_final_eight *= (TRIFLE::prob_table_host[cur_val][i]);

	// for (int j = 0; j <= seven_level_limit; j++) {
	// 	float cur_prob_final_seven = cur_prob_final_eight;
	// 	unsigned char cur_thread_partial_dy_RN_seven[33]; //The [32] or size 33 is for fast procesing only
	// 	memcpy(cur_thread_partial_dy_RN_seven, cur_thread_partial_dy_RN_eight, 33);

	// 	unsigned char cur_val = cur_thread_partial_dy_RN_seven[sbox_index[6]];
	// 	cur_thread_partial_dy_RN_seven[sbox_index[6]] = TRIFLE::diff_table_host[cur_val][i];
	// 	cur_prob_final_seven *= (TRIFLE::prob_table_host[cur_val][i]);

	// for (int j = 0; j <= six_level_limit; j++) {
	// 	float cur_prob_final_six = cur_prob_final_seven;
	// 	unsigned char cur_thread_partial_dy_RN_six[33]; //The [32] or size 33 is for fast procesing only
	// 	memcpy(cur_thread_partial_dy_RN_six, cur_thread_partial_dy_RN_seven, 33);

	// 	unsigned char cur_val = cur_thread_partial_dy_RN_six[sbox_index[5]];
	// 	cur_thread_partial_dy_RN_six[sbox_index[5]] = TRIFLE::diff_table_host[cur_val][i];
	// 	cur_prob_final_six *= (TRIFLE::prob_table_host[cur_val][i]);

	// for (int j = 0; j <= five_level_limit; j++) {
	// 	float cur_prob_final_five = cur_prob_final_six;
	// 	unsigned char cur_thread_partial_dy_RN_five[33]; //The [32] or size 33 is for fast procesing only
	// 	memcpy(cur_thread_partial_dy_RN_five, cur_thread_partial_dy_RN_six, 33);

	// 	unsigned char cur_val = cur_thread_partial_dy_RN_five[sbox_index[4]];
	// 	cur_thread_partial_dy_RN_five[sbox_index[4]] = TRIFLE::diff_table_host[cur_val][i];
	// 	cur_prob_final_five *= (TRIFLE::prob_table_host[cur_val][i]);

	//Emulating branching
	for (int i = 0; i <= fourth_level_limit; i++) {
		float cur_prob_final_fourth = cur_prob;
		// float cur_prob_final_fourth = cur_prob_final_five;
		unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
		memcpy(cur_thread_partial_dy_RN_fourth, cur_partial_dy, 32);
		// memcpy(cur_thread_partial_dy_RN_fourth, cur_thread_partial_dy_RN_five, 33);
		cur_thread_partial_dy_RN_fourth[32] = 0;

		unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[3]];
		cur_thread_partial_dy_RN_fourth[sbox_index[3]] = TRIFLE::diff_table_host[cur_val][i];
		cur_prob_final_fourth *= (TRIFLE::prob_table_host[cur_val][i]);

		for (int j = 0; j <= third_level_limit; j++) {
			float cur_prob_final_third = cur_prob_final_fourth;
			unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
			memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
			//cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over

			cur_val = cur_thread_partial_dy_RN_third[sbox_index[2]];
			cur_thread_partial_dy_RN_third[sbox_index[2]] = TRIFLE::diff_table_host[cur_val][j];
			cur_prob_final_third *= (TRIFLE::prob_table_host[cur_val][j]);

			for (int k = 0; k <= second_level_limit; k++) {
				float cur_prob_final_second = cur_prob_final_third;
				unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
				memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);

				cur_val = cur_thread_partial_dy_RN_second[sbox_index[1]];
				cur_thread_partial_dy_RN_second[sbox_index[1]] = TRIFLE::diff_table_host[cur_val][k];
				cur_prob_final_second *= (TRIFLE::prob_table_host[cur_val][k]);

				#pragma unroll
				for (int l = 0; l <= 6; l++) {
					float cur_prob_final = cur_prob_final_second;
					unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
					memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);

					//Substitution final
					cur_val = cur_thread_partial_dy_RN[sbox_index[0]];
					cur_thread_partial_dy_RN[sbox_index[0]] = TRIFLE::diff_table_host[cur_val][l];
					cur_prob_final *= (TRIFLE::prob_table_host[cur_val][l] );

					//Permutation
					unsigned long long front_64 = 0, back_64 = 0;
					for (int i = 0; i < 32; i++) {
						if (cur_thread_partial_dy_RN[i] > 0) {
							//Permutation LUTable
							//25% less running time compared to normal computation
							front_64 |= TRIFLE::perm_lookup_host[i][cur_thread_partial_dy_RN[i]][0];
							back_64 |= TRIFLE::perm_lookup_host[i][cur_thread_partial_dy_RN[i]][1];
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
					//NOW cur_thread_partial_dy_RN permutated

					//Check for condition
					int weight = 0;
					wz::hw_word_u4(cur_thread_partial_dy_RN, 32, weight);
					
					int sbox_index_ptr =0;
					int sbox_index_saved[8] = {32,32,32,32,32,32,32,32}; 
					if (weight <= MAX_AS_USED)  //If only next round AS <= 8
					{
						//MATSUI BOUND
						float estimated_com_prob = ( pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_FORWARD - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight) );
						if ((estimated_com_prob * cur_prob_final) >= CLUSTER_PROB_BOUND) {
							//Save everything
							for (int i=0;i<32;i++){
								if (cur_thread_partial_dy_RN[i]>0){
									sbox_index_saved[sbox_index_ptr] = i;
									sbox_index_ptr+=1;
								}
							}
							memcpy(output_dy,cur_thread_partial_dy_RN,32); //TODO output_dy ???
							*output_prob = cur_prob_final;
							*output_sbox_num = sbox_index_ptr;
							memcpy(output_sbox_index,sbox_index_saved, sizeof(int) * 8 );
						} else{
							#ifdef RECORD_EVERYTHING
							memcpy(output_dy,cur_thread_partial_dy_RN,32);
							*output_prob = cur_prob_final;
							*output_sbox_num = sbox_index_ptr;
							memcpy(output_sbox_index, sbox_index_saved, sizeof(int) * 8 );
							#endif
							*output_sbox_num = 0; //Indicate jump over this
						}
					}
					else{
						#ifdef RECORD_EVERYTHING
						memcpy(output_dy,cur_thread_partial_dy_RN,32);
						*output_prob = cur_prob_final;
						*output_sbox_num = sbox_index_ptr;
						memcpy(output_sbox_index,sbox_index_saved, sizeof(int) * 8 );
						#endif
						*output_sbox_num = 0; //Indicate jump over this
					}

					//last_round_trails[cur_r] +=1;
					//output_size_all_local+=1;

					//This structure mimic the gpu one
					//Calculate the next set of address to save to
					output_dy = output_dy + 32;
					output_prob = output_prob + 1;
					output_sbox_num = output_sbox_num + 9;
					output_sbox_index = output_sbox_index + 9;
				}
			}
		}
	}	

	// }}}}

	//output_size_all = output_size_all_local;
};

//Backward Search
void trifle_diff_cluster_gpu_using_cpu_backward(unsigned char* output_dy, float* output_prob, 
	int* output_sbox_num, int* output_sbox_index,unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index, float cur_prob){
	int second_level_limit = (cur_num_sbox >= 2 ? 6 : 0);
	int third_level_limit = (cur_num_sbox >= 3 ? 6 : 0);
	int fourth_level_limit = (cur_num_sbox >= 4 ? 6 : 0);

	//int output_size_all_local=0;

	//Emulating branching
	for (int i = 0; i <= fourth_level_limit; i++) {
		float cur_prob_final_fourth = cur_prob;
		unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
		memcpy(cur_thread_partial_dy_RN_fourth, cur_partial_dy, 32);
		cur_thread_partial_dy_RN_fourth[32] = 0;

		unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[3]];
		cur_thread_partial_dy_RN_fourth[sbox_index[3]] = TRIFLE::diff_table_host_reversed[cur_val][i];
		cur_prob_final_fourth *= (TRIFLE::prob_table_host[cur_val][i]);

		for (int j = 0; j <= third_level_limit; j++) {
			float cur_prob_final_third = cur_prob_final_fourth;
			unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
			memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
			//cur_thread_partial_dy_RN_third[32] = 0; // No need as already copied over

			cur_val = cur_thread_partial_dy_RN_third[sbox_index[2]];
			cur_thread_partial_dy_RN_third[sbox_index[2]] = TRIFLE::diff_table_host_reversed[cur_val][j];
			cur_prob_final_third *= (TRIFLE::prob_table_host[cur_val][j]);

			for (int k = 0; k <= second_level_limit; k++) {
				float cur_prob_final_second = cur_prob_final_third;
				unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
				memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);

				cur_val = cur_thread_partial_dy_RN_second[sbox_index[1]];
				cur_thread_partial_dy_RN_second[sbox_index[1]] = TRIFLE::diff_table_host_reversed[cur_val][k];
				cur_prob_final_second *= (TRIFLE::prob_table_host[cur_val][k]);

				#pragma unroll
				for (int l = 0; l <= 6; l++) {
					float cur_prob_final = cur_prob_final_second;
					unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
					memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);

					//Substitution final
					cur_val = cur_thread_partial_dy_RN[sbox_index[0]];
					cur_thread_partial_dy_RN[sbox_index[0]] = TRIFLE::diff_table_host_reversed[cur_val][l];
					cur_prob_final *= (TRIFLE::prob_table_host[cur_val][l] );

					//Permutation
					unsigned long long front_64 = 0, back_64 = 0;
					for (int i = 0; i < 32; i++) {
						if (cur_thread_partial_dy_RN[i] > 0) {
							front_64 |= TRIFLE::perm_lookup_host_reversed[i][cur_thread_partial_dy_RN[i]][0];
							back_64 |= TRIFLE::perm_lookup_host_reversed[i][cur_thread_partial_dy_RN[i]][1];
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
					//NOW cur_thread_partial_dy_RN permutated

					//Check for condition
					int weight = 0;
					wz::hw_word_u4(cur_thread_partial_dy_RN, 32, weight);
					
					int sbox_index_ptr =0;
					int sbox_index_saved[8] = {32,32,32,32,32,32,32,32}; 
					if (weight <= MAX_AS_USED)  //If only next round AS <= 8
					{
						//MATSUI BOUND
						float estimated_com_prob = ( pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND_MITM_BACKWARD - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight) );
						if ((estimated_com_prob * cur_prob_final) >= CLUSTER_PROB_BOUND) {
							//Save everything
							for (int i=0;i<32;i++){
								if (cur_thread_partial_dy_RN[i]>0){
									sbox_index_saved[sbox_index_ptr] = i;
									sbox_index_ptr+=1;
								}
							}
							memcpy(output_dy,cur_thread_partial_dy_RN,32); //TODO output_dy ???
							*output_prob = cur_prob_final;
							*output_sbox_num = sbox_index_ptr;
							memcpy(output_sbox_index,sbox_index_saved, sizeof(int) * 8 );
						} else{
							#ifdef RECORD_EVERYTHING
							memcpy(output_dy,cur_thread_partial_dy_RN,32);
							*output_prob = cur_prob_final;
							*output_sbox_num = sbox_index_ptr;
							memcpy(output_sbox_index, sbox_index_saved, sizeof(int) * 8 );
							#endif
							*output_sbox_num = 0; //Indicate jump over this
						}
					}
					else{
						#ifdef RECORD_EVERYTHING
						memcpy(output_dy,cur_thread_partial_dy_RN,32);
						*output_prob = cur_prob_final;
						*output_sbox_num = sbox_index_ptr;
						memcpy(output_sbox_index,sbox_index_saved, sizeof(int) * 8 );
						#endif
						*output_sbox_num = 0; //Indicate jump over this
					}

					//last_round_trails[cur_r] +=1;
					//output_size_all_local+=1;

					//This structure mimic the gpu one
					//Calculate the next set of address to save to
					output_dy = output_dy + 32;
					output_prob = output_prob + 1;
					output_sbox_num = output_sbox_num + 9;
					output_sbox_index = output_sbox_index + 9;
				}
			}
		}
	}	

	//output_size_all = output_size_all_local;
};

//Support Sbox<=4
void trifle_diff_cluster_gpu_using_cpu_final_r(unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index, float cur_prob,
	long long* cluster_size, double* cluster_prob){
	int second_level_limit = (cur_num_sbox >= 2 ? 6 : 0);
	int third_level_limit = (cur_num_sbox >= 3 ? 6 : 0);
	int fourth_level_limit = (cur_num_sbox >= 4 ? 6 : 0);

	//Emulating branching
	for (int i = 0; i <= fourth_level_limit; i++) {
		float cur_prob_final_fourth = cur_prob;
		unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
		memcpy(cur_thread_partial_dy_RN_fourth, cur_partial_dy, 32);
		cur_thread_partial_dy_RN_fourth[32] = 0;

		unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[3]];
		cur_thread_partial_dy_RN_fourth[sbox_index[3]] = TRIFLE::diff_table_host[cur_val][i];
		cur_prob_final_fourth *= (TRIFLE::prob_table_host[cur_val][i]);

		for (int j = 0; j <= third_level_limit; j++) {
			float cur_prob_final_third = cur_prob_final_fourth;
			unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
			memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
			//cur_thread_partial_dy_RN_third[32] = 0;

			cur_val = cur_thread_partial_dy_RN_third[sbox_index[2]];
			cur_thread_partial_dy_RN_third[sbox_index[2]] = TRIFLE::diff_table_host[cur_val][j];
			cur_prob_final_third *= (TRIFLE::prob_table_host[cur_val][j]);

			for (int k = 0; k <= second_level_limit; k++) {
				float cur_prob_final_second = cur_prob_final_third;
				unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
				memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
				//cur_thread_partial_dy_RN_third[32] = 0;

				cur_val = cur_thread_partial_dy_RN_second[sbox_index[1]];
				cur_thread_partial_dy_RN_second[sbox_index[1]] = TRIFLE::diff_table_host[cur_val][k];
				cur_prob_final_second *= (TRIFLE::prob_table_host[cur_val][k]);

				#pragma unroll
				for (int l = 0; l <= 6; l++) {
					float cur_prob_final = cur_prob_final_second;
					unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
					memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
					//cur_thread_partial_dy_RN[32] = 0;

					//Substitution final
					cur_val = cur_thread_partial_dy_RN[sbox_index[0]];
					cur_thread_partial_dy_RN[sbox_index[0]] = TRIFLE::diff_table_host[cur_val][l];
					cur_prob_final *= (TRIFLE::prob_table_host[cur_val][l] );

					last_round_trails[PATTERN_ROUND-1]+=1;

					//Check for condition
					bool is_same=true;
					for (int i=0;i<32;i++){
						if (TRIFLE::final_dy_host[i] != cur_thread_partial_dy_RN[i]){
							is_same = false;
							break;
						}
					}

					if (is_same){
						(*cluster_prob) += cur_prob_final;
						(*cluster_size) += 1;
					}
					
				}
			}
		}
	}	
};

void MITM_trifle_diff_cluster_gpu_using_cpu_final_r_forward(unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index, float cur_prob,
	int thread_id){
	int second_level_limit = (cur_num_sbox >= 2 ? 6 : 0);
	int third_level_limit = (cur_num_sbox >= 3 ? 6 : 0);
	int fourth_level_limit = (cur_num_sbox >= 4 ? 6 : 0);

	//Emulating branching
	for (int i = 0; i <= fourth_level_limit; i++) {
		float cur_prob_final_fourth = cur_prob;
		unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
		memcpy(cur_thread_partial_dy_RN_fourth, cur_partial_dy, 32);
		cur_thread_partial_dy_RN_fourth[32] = 0;

		unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[3]];
		cur_thread_partial_dy_RN_fourth[sbox_index[3]] = TRIFLE::diff_table_host[cur_val][i];
		cur_prob_final_fourth *= (TRIFLE::prob_table_host[cur_val][i]);

		for (int j = 0; j <= third_level_limit; j++) {
			float cur_prob_final_third = cur_prob_final_fourth;
			unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
			memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
			//cur_thread_partial_dy_RN_third[32] = 0;

			cur_val = cur_thread_partial_dy_RN_third[sbox_index[2]];
			cur_thread_partial_dy_RN_third[sbox_index[2]] = TRIFLE::diff_table_host[cur_val][j];
			cur_prob_final_third *= (TRIFLE::prob_table_host[cur_val][j]);

			for (int k = 0; k <= second_level_limit; k++) {
				float cur_prob_final_second = cur_prob_final_third;
				unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
				memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
				//cur_thread_partial_dy_RN_third[32] = 0;

				cur_val = cur_thread_partial_dy_RN_second[sbox_index[1]];
				cur_thread_partial_dy_RN_second[sbox_index[1]] = TRIFLE::diff_table_host[cur_val][k];
				cur_prob_final_second *= (TRIFLE::prob_table_host[cur_val][k]);

				#pragma unroll
				for (int l = 0; l <= 6; l++) {
					float cur_prob_final = cur_prob_final_second;
					unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
					memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
					//cur_thread_partial_dy_RN[32] = 0;

					//Substitution final
					cur_val = cur_thread_partial_dy_RN[sbox_index[0]];
					cur_thread_partial_dy_RN[sbox_index[0]] = TRIFLE::diff_table_host[cur_val][l];
					cur_prob_final *= (TRIFLE::prob_table_host[cur_val][l] );

					last_round_trails[PATTERN_ROUND_MITM_FORWARD-1]+=1;

					//Permutation
					unsigned char new_partial_dy[32] = { 0 };
					unsigned long long front_64 = 0, back_64 = 0;

					for (int i = 0; i < 32; i++) {
						if (cur_thread_partial_dy_RN[i] > 0) {
							front_64 |= TRIFLE::perm_lookup_host[i][cur_thread_partial_dy_RN[i]][0];
							back_64  |= TRIFLE::perm_lookup_host[i][cur_thread_partial_dy_RN[i]][1];
						}
					}
					for (int i = 0; i < 16; i++) {
						new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
					}
					for (int i = 16; i < 32; i++) {
						new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
					}

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

						trifle_gpu_manager_arr[thread_id].mitm_prob_cache[index] += cur_prob_final;
						trifle_gpu_manager_arr[thread_id].mitm_cluster_size_cache[index] += 1;
					}
				}
			}
		}
	}	
};

void MITM_trifle_diff_cluster_gpu_using_cpu_final_r_backward(unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index, float cur_prob,
	int thread_id){
	int second_level_limit = (cur_num_sbox >= 2 ? 6 : 0);
	int third_level_limit = (cur_num_sbox >= 3 ? 6 : 0);
	int fourth_level_limit = (cur_num_sbox >= 4 ? 6 : 0);

	//Emulating branching
	for (int i = 0; i <= fourth_level_limit; i++) {
		float cur_prob_final_fourth = cur_prob;
		unsigned char cur_thread_partial_dy_RN_fourth[33]; //The [32] or size 33 is for fast procesing only
		memcpy(cur_thread_partial_dy_RN_fourth, cur_partial_dy, 32);
		cur_thread_partial_dy_RN_fourth[32] = 0;

		unsigned char cur_val = cur_thread_partial_dy_RN_fourth[sbox_index[3]];
		cur_thread_partial_dy_RN_fourth[sbox_index[3]] = TRIFLE::diff_table_host_reversed[cur_val][i];
		cur_prob_final_fourth *= (TRIFLE::prob_table_host[cur_val][i]);

		for (int j = 0; j <= third_level_limit; j++) {
			float cur_prob_final_third = cur_prob_final_fourth;
			unsigned char cur_thread_partial_dy_RN_third[33]; //The [32] or size 33 is for fast procesing only
			memcpy(cur_thread_partial_dy_RN_third, cur_thread_partial_dy_RN_fourth, 33);
			//cur_thread_partial_dy_RN_third[32] = 0;

			cur_val = cur_thread_partial_dy_RN_third[sbox_index[2]];
			cur_thread_partial_dy_RN_third[sbox_index[2]] = TRIFLE::diff_table_host_reversed[cur_val][j];
			cur_prob_final_third *= (TRIFLE::prob_table_host[cur_val][j]);

			for (int k = 0; k <= second_level_limit; k++) {
				float cur_prob_final_second = cur_prob_final_third;
				unsigned char cur_thread_partial_dy_RN_second[33]; //The [32] or size 33 is for fast procesing only
				memcpy(cur_thread_partial_dy_RN_second, cur_thread_partial_dy_RN_third, 33);
				//cur_thread_partial_dy_RN_third[32] = 0;

				cur_val = cur_thread_partial_dy_RN_second[sbox_index[1]];
				cur_thread_partial_dy_RN_second[sbox_index[1]] = TRIFLE::diff_table_host_reversed[cur_val][k];
				cur_prob_final_second *= (TRIFLE::prob_table_host[cur_val][k]);

				#pragma unroll
				for (int l = 0; l <= 6; l++) {
					float cur_prob_final = cur_prob_final_second;
					unsigned char cur_thread_partial_dy_RN[33]; //The [32] or size 33 is for fast procesing only
					memcpy(cur_thread_partial_dy_RN, cur_thread_partial_dy_RN_second, 33);
					//cur_thread_partial_dy_RN[32] = 0;

					//Substitution final
					cur_val = cur_thread_partial_dy_RN[sbox_index[0]];
					cur_thread_partial_dy_RN[sbox_index[0]] = TRIFLE::diff_table_host_reversed[cur_val][l];
					cur_prob_final *= (TRIFLE::prob_table_host[cur_val][l] );

					last_round_trails[PATTERN_ROUND_MITM_BACKWARD-1]+=1;
					
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

						int target_size =  trifle_gpu_manager_arr[thread_id].mitm_cluster_size_cache[index];
						if(target_size > 0){ //Exist connection
							double target_prob = ( (double) cur_prob_final * trifle_gpu_manager_arr[thread_id].mitm_prob_cache[index]);

							//Add to collection
							trifle_gpu_manager_arr[thread_id].cluster_prob+= target_prob;
							trifle_gpu_manager_arr[thread_id].cluster_size+= target_size;
						}
					}
				}
			}
		}
	}	
};


//Cache 22Rounds Maximum of 128bit input, maybe not recursion
//cur_num_sbox help to pick <2 cpu >2 gpu
void trifle_diff_cluster_gpu_recursion(int& thread_id, unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index ,float cur_prob, int i_limit,
	bool is_bundled=false, int bundled_amount=0, int* bundled_item_index=nullptr){
	//Working out the address of heap
	//maximum of 17* 21 * MAX_BRANCH so it is still 32bit integer signed
	unsigned char* next_dy = trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned + (cur_r * 32ull * MAX_BRANCH_PER_ROUND);
	float* next_round_prob = trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned  + (cur_r * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_num = trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index + (cur_r * 9 * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_index= next_round_sbox_num+1;

	#ifdef GPU_BUNDLED_LAUNCH	
	//Saved start of next_dy
	unsigned char* next_dy_bundle_normal = next_dy;
	float* next_round_prob_bundle_normal = next_round_prob;
	int* next_round_sbox_num_bundle_normal = next_round_sbox_num;
	int* next_round_sbox_index_bundle_normal = next_round_sbox_index;

	//Bundled GPU
	int* cur_index_bundled = trifle_gpu_manager_arr[thread_id].index_sorting_bundled + (cur_r* 8 * MAX_BRANCH_PER_ROUND);
	#endif

	//Progress Tracking..
	if (cur_r == 1) {
		std::cerr << "\n"
				<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
		std::cerr.flush();
	}

	//Launch Current Level using CPU, GPU / GPU Bundled
	PROCESSING_METHOD method = PROCESSING_METHOD::GPU; //current level processing using cpu or gpu

	#ifdef GPU_BUNDLED_LAUNCH	
	if (is_bundled){
		int previous_r = cur_r -1;
		unsigned char* previous_dy = trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned + (previous_r * 32ull * MAX_BRANCH_PER_ROUND);
		float* previous_round_prob = trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned + (previous_r * MAX_BRANCH_PER_ROUND);
		int* previous_round_sbox_num = trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index + (previous_r * 9 * MAX_BRANCH_PER_ROUND);
		int* previous_round_sbox_index = previous_round_sbox_num + 1;

		//TODO do bundle launch here
		//TODO task 1

		//Copy all item pointed by index to pinned memory

	}
	else{ //Not bundled
	
	#endif
		for (int i=0;i<32;i++){
			cur_partial_dy[i] = 0;
		}

		cur_num_sbox = 4;
		for (int i=0;i<cur_num_sbox;i++){
			cur_partial_dy[i] = 1;
			sbox_index[i] = i;
		}

		// Used to profile performance
		// cudaProfilerStart();
		// for (int i=0;i<100;i++){
		// std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		// for (int i=0;i<10;i++){
		// trifle_diff_cluster_gpu_using_cpu(next_dy, next_round_prob, next_round_sbox_num, next_round_sbox_index,cur_partial_dy, cur_r, cur_num_sbox, sbox_index, cur_prob);
		// method = PROCESSING_METHOD::CPU;
		// }
		// std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		// // std::cout << "\nTime difference (s ) = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
		// std::cout << "\n\nHost Time difference (us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/10.0;
		// // std::cout << "\nTime difference (ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count()/10.0;

		// std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
		// for (int i=0;i<1;i++){
		// memcpy(trifle_gpu_manager_arr[thread_id].sbox_index_pinned, sbox_index, sizeof(int)*8);
		// memcpy(trifle_gpu_manager_arr[thread_id].dx_pinned, cur_partial_dy, 32);
		// trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round(trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned,
		// cur_num_sbox, cur_prob, cur_r, next_dy, next_round_prob, next_round_sbox_num); 
		// }
		// std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
		// std::cout << "\nDevice: Time difference (us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count()/1.0;
		// // std::cout << "\nDevice: Time difference (ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end2 - begin2).count()/10.0;
		// // getchar();
		// }	
		// cudaProfilerStop();
		// return;

		// 5188176.554 CPU enum r25
		// 2190112.038

		if (cur_num_sbox<=3){ //<4 do CPU
			//First round and other round where bundle amount left out some unbundlable one
			trifle_diff_cluster_gpu_using_cpu(next_dy, next_round_prob, next_round_sbox_num, next_round_sbox_index,cur_partial_dy, cur_r, cur_num_sbox, sbox_index, cur_prob);
			method = PROCESSING_METHOD::CPU;
		} 
		else{ // if (cur_num_sbox<=8){
			//copy input
			memcpy(trifle_gpu_manager_arr[thread_id].sbox_index_pinned, sbox_index, sizeof(int)*8);
			memcpy(trifle_gpu_manager_arr[thread_id].dx_pinned, cur_partial_dy, 32);

			trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round(trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned,
				cur_num_sbox, cur_prob, cur_r, next_dy, next_round_prob, next_round_sbox_num); 
		}
	#ifdef GPU_BUNDLED_LAUNCH		
	}
	#endif

	#ifdef GPU_BUNDLED_LAUNCH
	//NOTE: combined to all is bundled launch
	//After computation, recursively let the computation to take place
	// if (i_limit <=7){ //7 Branches do it normally, NO BUNDLE
	// 	for (int i=0;i<i_limit;i++){
	// 		if ( (*next_round_sbox_num)!=0){
	// 			//NOTE: Code Duplication ID 1, No 1
	// 			//Not Last Round continue deeper, n, .., n-2 -> n-1
	// 			if (cur_r!=PATTERN_ROUND-2){
	// 				int next_round_i_limit = std::pow(7, (*next_round_sbox_num));
	// 				trifle_diff_cluster_gpu_recursion(thread_id,next_dy, cur_r+1, next_round_sbox_num, next_round_sbox_index, next_round_prob, next_round_i_limit ,false);
	// 			}
	// 			//Last Round, n-1 -> n
	// 			else{
	// 				if ( (*cur_num_sbox) >=4  &&  method==PROCESSING_METHOD::GPU ){
	// 					trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round_last(i); 
	// 				}
	// 				else{
	// 					//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
	// 					trifle_diff_cluster_gpu_using_cpu_final_r(next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob,
	// 						&trifle_gpu_manager_arr[thread_id].cluster_size, &trifle_gpu_manager_arr[thread_id].cluster_prob);
	// 				}
	// 			}
	// 		}

	// 		//Prepare for the next one
	// 		next_dy = next_dy + 32;
	// 		next_round_prob = next_round_prob + 1;
	// 		next_round_sbox_num = next_round_sbox_num + 9;
	// 		next_round_sbox_index = next_round_sbox_index + 9;
	// 	}
    // }
	if(true) { //Do bundled launches
		int sbox_num_count[8]; //Final number is the number of element, to access element at last pos decrement (-1) 1st...
		int sbox_num_count_total =0;
		//int* cur_index_bundled_temp =  cur_index_bundled; //Contain the starting location of index[cur_r][0-8][0-MAX_BRANCH_NUM]
		int* branch_entry_seperated_by_sbox_num[8];
		int* branch_entry_seperated_by_sbox_num_ori[8];
		for (int i=0;i<8;i++){ //Calculate the starting location of eight sboxes index
			branch_entry_seperated_by_sbox_num[i] = cur_index_bundled + (i*MAX_BRANCH_PER_ROUND);
			branch_entry_seperated_by_sbox_num_ori[i] = branch_entry_seperated_by_sbox_num[i];
		}

		//Get Information of Bundling 1,2,3 sboxes record down, 4-8 sboxes launch directly from here
		for (int i=0;i<i_limit;i++){
			last_round_trails[cur_r] += 1; //HACK: logistic record

			int sbox_num_temp_per = *next_round_sbox_num; //Get the value out as cache
			if (sbox_num_temp_per == 0 ){ 
				//If condition not met by algorithm, skip
			}
			else if (sbox_num_temp_per>=4){ //4 Sboxes, Launch normally
				if (cur_r!=PATTERN_ROUND-2){
					int next_round_i_limit = std::pow(7, (sbox_num_temp_per));
					trifle_diff_cluster_gpu_recursion(thread_id,next_dy, cur_r+1, sbox_num_temp_per, next_round_sbox_index, *next_round_prob, next_round_i_limit ,false);
				}
				//Last Round, n-1 -> n
				else{
					if ( method==PROCESSING_METHOD::GPU ){
						trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round_last(i); 
					}
					else{
						//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
						trifle_diff_cluster_gpu_using_cpu_final_r(next_dy, cur_r+1, sbox_num_temp_per, next_round_sbox_index, *next_round_prob,
							&trifle_gpu_manager_arr[thread_id].cluster_size, &trifle_gpu_manager_arr[thread_id].cluster_prob);
					}
				}
			}
			else{ //If Sbox_num <4, ie 1, 2, 3
				sbox_num_temp_per -=1; //Make indexing easier, does not modify original sbox num value

				*(branch_entry_seperated_by_sbox_num[sbox_num_temp_per]) = i; //Record down the index (when looping from i-i_limit)
				branch_entry_seperated_by_sbox_num[sbox_num_temp_per]+= 1; //Move the pointer to next empty slot
				sbox_num_count_total += 1;
				sbox_num_count[sbox_num_temp_per] +=1; //Increase element count by one
			}

			//Advance to next entry
			next_dy = next_dy + 32;
			next_round_prob = next_round_prob + 1;
			next_round_sbox_num = next_round_sbox_num + 9;
			next_round_sbox_index = next_round_sbox_index + 9;
		}

		// //Finished launching the remaining 1,2,3 sboxes in bundle. or solo cpu for remaining stuff
		int max_target_val = THREAD_PER_BLOCK * BLOCK_NUM_EXACT_BUNDLE;
		while(sbox_num_count_total > 0){
			int cur_bundled_branch_val = 0;
			int cur_bundled_amount = 0;
		
			//Take as many from highest number first, best-fit, Packing
			int max_item_3box = max_target_val/343; //343 is branch number
			max_item_3box = sbox_num_count[2]<max_item_3box? sbox_num_count[2] : max_item_3box;
			cur_bundled_amount += max_item_3box;
			cur_bundled_branch_val += max_item_3box*343;

			//2sbox
			int remaining_val = max_target_val - (max_item_3box*343);
			int max_item_2box = remaining_val/49;
			max_item_2box = sbox_num_count[1]<max_item_2box? sbox_num_count[1] : max_item_2box;
			cur_bundled_amount += max_item_2box;
			cur_bundled_branch_val += max_item_2box*49;

			//1Sbox
			remaining_val -= (max_item_2box*49);
			int max_item_1box = remaining_val/7;
			max_item_1box = sbox_num_count[0]<max_item_1box? sbox_num_count[0] : max_item_1box;
			cur_bundled_amount += max_item_1box;
			cur_bundled_branch_val += max_item_1box*7;

			//Finished initial packing estimation, Begginging Launching Bundle
			if (cur_bundled_branch_val>=144 && cur_r!=PATTERN_ROUND-2){ //TODO adjust here as well
				//Bundling
				int* bundle_index = new int[cur_bundled_amount];
				int bundle_index_arr_ptr=0;

				//3Sbox
				for (int i=0;i< max_item_3box;i++){
					bundle_index[bundle_index_arr_ptr] = *(branch_entry_seperated_by_sbox_num_ori[2]);
					branch_entry_seperated_by_sbox_num_ori[2] += 1;
					sbox_num_count[2] 	-=1;
					bundle_index_arr_ptr+=1;
				}
				//2Sbox
				for (int i=0;i< max_item_2box;i++){
					bundle_index[bundle_index_arr_ptr] = *(branch_entry_seperated_by_sbox_num_ori[1]);
					branch_entry_seperated_by_sbox_num_ori[1] += 1;
					sbox_num_count[1] 	-=1;
					bundle_index_arr_ptr+=1;
				}
				//1Sbox
				for (int i=0;i< max_item_1box;i++){
					bundle_index[bundle_index_arr_ptr] = *(branch_entry_seperated_by_sbox_num_ori[0]);
					branch_entry_seperated_by_sbox_num_ori[0] += 1;
					sbox_num_count[0] 	-=1;
					bundle_index_arr_ptr+=1;
				}

				//Bundled Launch
				if (cur_r!=PATTERN_ROUND-2){
					//Recursively call this
					trifle_diff_cluster_gpu_recursion(thread_id,nullptr, cur_r+1, 0, nullptr, 0, cur_bundled_branch_val, true, cur_bundled_amount, bundle_index);
				}
				else{
					//TODO temperory only do GPU for n-?? -> n-2
					//n-1 -> n and GPU is used at n-1
					// if ( method==PROCESSING_METHOD::GPU ){
					// 	trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_bundled_last(i); 
					// } 
					// else{ // n-1 > n and CPU CPU is used at n-1
					// 	trifle_diff_cluster_gpu_using_cpu_final_r(next_dy, cur_r+1, sbox_num_temp_per, next_round_sbox_index, *next_round_prob,
					// 		&trifle_gpu_manager_arr[thread_id].cluster_size, &trifle_gpu_manager_arr[thread_id].cluster_prob);
					// }
				}

				delete[] bundle_index;
			}
			else{
				break;
			}

			//Post processing
			sbox_num_count[2] -= max_item_3box;
			sbox_num_count[1] -= max_item_2box;
			sbox_num_count[0] -= max_item_1box;

			sbox_num_count_total -= cur_bundled_branch_val;
		}
		
		for (int i=0; i < 4; i++){
			for (int j=0;j < sbox_num_count[i];j++){
				int index = *(branch_entry_seperated_by_sbox_num[i]);

				unsigned char* next_dy_bundle_norma_indi = next_dy_bundle_normal + (32*index);
				float* next_round_prob_bundle_normal_indi = next_round_prob_bundle_normal + (index);
				int* next_round_sbox_num_bundle_normal_indi = next_round_sbox_num_bundle_normal + (index*9) ;
				int* next_round_sbox_index_bundle_normal_indi = next_round_sbox_index_bundle_normal + (index*9);

				int next_round_i_limit_indi = pow(7, *next_round_sbox_num_bundle_normal_indi);

				if (cur_r!=PATTERN_ROUND-2){
					//Recursively call this
					trifle_diff_cluster_gpu_recursion(thread_id, next_dy_bundle_norma_indi, cur_r+1, *next_round_sbox_num_bundle_normal_indi, next_round_sbox_index, 
					*next_round_prob_bundle_normal_indi, next_round_i_limit_indi ,false);
				}
				else{
					//Using GPU on normal launch of 1,2,3 does not make performance difference
					// if ( method==PROCESSING_METHOD::GPU ){
					// 	trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round_last(i); 
					// }
					if(true){
						//TODO fix this as well...
						//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
						trifle_diff_cluster_gpu_using_cpu_final_r(next_dy, cur_r+1, sbox_num_temp_per, next_round_sbox_index, *next_round_prob,
							&trifle_gpu_manager_arr[thread_id].cluster_size, &trifle_gpu_manager_arr[thread_id].cluster_prob);
					}
				}
			}
		}
		//End of launching
	}
	#endif

	#ifndef GPU_BUNDLED_LAUNCH		
	//Support Sbox <=4
	for (int i=0;i<i_limit;i++){
		last_round_trails[cur_r] += 1;

		if ( (*next_round_sbox_num)!=0){
			int next_round_i_limit = std::pow(7, (*next_round_sbox_num));

			//Not Last Round continue deeper, 1, .., n-2 -> n-1
			if (cur_r != PATTERN_ROUND-2){
				trifle_diff_cluster_gpu_recursion(thread_id,next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob, next_round_i_limit ,false);
			}
			//Last Round, n-1 -> n
			else{
				if ( (*next_round_sbox_num)>=4 && method==PROCESSING_METHOD::GPU ){
					trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round_last(i); 
					last_round_trails[PATTERN_ROUND-1]+= pow(7,*next_round_sbox_num);
				}
				else{
					//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
					trifle_diff_cluster_gpu_using_cpu_final_r(next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob,
						&trifle_gpu_manager_arr[thread_id].cluster_size, &trifle_gpu_manager_arr[thread_id].cluster_prob);
				}
			}
		}

		//Prepare for the next one
		next_dy = next_dy + 32;
		next_round_prob = next_round_prob + 1;
		next_round_sbox_num = next_round_sbox_num + 9;
		next_round_sbox_index = next_round_sbox_index + 9;
	}
	#endif
};

//MITM GPU Forward
void MITM_trifle_diff_cluster_gpu_recursion_forward(int& thread_id, unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index ,float cur_prob, int i_limit,
	bool is_bundled=false, int bundled_amount=0, int* bundled_item_index=nullptr){

	//Working out the address of heap
	//maximum of 17* 21 * MAX_BRANCH so it is still 32bit integer signed
	unsigned char* next_dy = trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned + (cur_r * 32ull * MAX_BRANCH_PER_ROUND);
	float* next_round_prob = trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned  + (cur_r * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_num = trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index + (cur_r * 9 * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_index= next_round_sbox_num+1;

	#ifdef GPU_BUNDLED_LAUNCH	
	//EMPTY
	#endif

	//Progress Tracking..
	if (cur_r == 1) {
		std::cerr << "\n"
				<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
		std::cerr.flush();
	}

	//Launch Current Level using CPU, GPU / GPU Bundled
	PROCESSING_METHOD method = PROCESSING_METHOD::GPU; //current level processing using cpu or gpu

	#ifdef GPU_BUNDLED_LAUNCH	
	//EMPTY
	#endif
		if (cur_num_sbox<=3){ //<4 do CPU
			//First round and other round where bundle amount left out some unbundlable one
			trifle_diff_cluster_gpu_using_cpu(next_dy, next_round_prob, next_round_sbox_num, next_round_sbox_index,cur_partial_dy, cur_r, cur_num_sbox, sbox_index, cur_prob);
			method = PROCESSING_METHOD::CPU;
		} 
		else{ // if (cur_num_sbox<=8){
			//copy input
			memcpy(trifle_gpu_manager_arr[thread_id].sbox_index_pinned, sbox_index, sizeof(int)*8);
			memcpy(trifle_gpu_manager_arr[thread_id].dx_pinned, cur_partial_dy, 32);

			trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round(trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned,
				cur_num_sbox, cur_prob, cur_r, next_dy, next_round_prob, next_round_sbox_num); 
		}
	#ifdef GPU_BUNDLED_LAUNCH		
	}
	#endif

	#ifdef GPU_BUNDLED_LAUNCH
	//EMPTY
	#endif

	#ifndef GPU_BUNDLED_LAUNCH		
	//Support Sbox <=4
	for (int i=0;i<i_limit;i++){
		last_round_trails[cur_r] += 1;

		if ( (*next_round_sbox_num)!=0){
			int next_round_i_limit = std::pow(7, (*next_round_sbox_num));

			//Not Last Round continue deeper, 1, .., n-2 -> n-1
			if (cur_r != PATTERN_ROUND_MITM_FORWARD-2){
				MITM_trifle_diff_cluster_gpu_recursion_forward(thread_id,next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob, next_round_i_limit ,false);
			}
			//Last Round, n-1 -> n
			else{
				//if (false){
				if ( (*next_round_sbox_num)>=4 && method==PROCESSING_METHOD::GPU ){
					trifle_gpu_manager_arr[thread_id].trifle_manager->MITM_kernel_compute_1round_last_forward(i); 
					last_round_trails[PATTERN_ROUND-1]+= pow(7,*next_round_sbox_num);
				}
				else{
					//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
					MITM_trifle_diff_cluster_gpu_using_cpu_final_r_forward(next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob,
						thread_id);
				}
			}
		}

		//Prepare for the next one
		next_dy = next_dy + 32;
		next_round_prob = next_round_prob + 1;
		next_round_sbox_num = next_round_sbox_num + 9;
		next_round_sbox_index = next_round_sbox_index + 9;
	}
	#endif
}

//MITM GPU Backward
void MITM_trifle_diff_cluster_gpu_recursion_backward(int& thread_id, unsigned char* cur_partial_dy, int cur_r, int cur_num_sbox, int* sbox_index ,float cur_prob, int i_limit,
	bool is_bundled=false, int bundled_amount=0, int* bundled_item_index=nullptr){

	//Working out the address of heap
	//maximum of 17* 21 * MAX_BRANCH so it is still 32bit integer signed
	unsigned char* next_dy = trifle_gpu_manager_arr[thread_id].next_round_dx_array_pinned + (cur_r * 32ull * MAX_BRANCH_PER_ROUND);
	float* next_round_prob = trifle_gpu_manager_arr[thread_id].cur_round_prob_pinned  + (cur_r * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_num = trifle_gpu_manager_arr[thread_id].next_round_sbox_num_and_index + (cur_r * 9 * MAX_BRANCH_PER_ROUND);
	int* next_round_sbox_index= next_round_sbox_num+1;

	#ifdef GPU_BUNDLED_LAUNCH	
	//EMPTY
	#endif

	//Progress Tracking..
	if (cur_r == 1) {
		std::cerr << "\n"
				<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
		std::cerr.flush();
	}

	//Launch Current Level using CPU, GPU / GPU Bundled
	PROCESSING_METHOD method = PROCESSING_METHOD::GPU; //current level processing using cpu or gpu

	#ifdef GPU_BUNDLED_LAUNCH	
	//EMPTY
	#endif
		if (cur_num_sbox<=3){ //<4 do CPU
			//First round and other round where bundle amount left out some unbundlable one
			trifle_diff_cluster_gpu_using_cpu_backward(next_dy, next_round_prob, next_round_sbox_num, next_round_sbox_index,cur_partial_dy, cur_r, cur_num_sbox, sbox_index, cur_prob);
			method = PROCESSING_METHOD::CPU;
		} 
		else{ // if (cur_num_sbox<=8){
			//copy input
			memcpy(trifle_gpu_manager_arr[thread_id].sbox_index_pinned, sbox_index, sizeof(int)*8);
			memcpy(trifle_gpu_manager_arr[thread_id].dx_pinned, cur_partial_dy, 32);

			trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_compute_1round_backward(trifle_gpu_manager_arr[thread_id].dx_pinned, trifle_gpu_manager_arr[thread_id].sbox_index_pinned,
				cur_num_sbox, cur_prob, cur_r, next_dy, next_round_prob, next_round_sbox_num); 
		}
	#ifdef GPU_BUNDLED_LAUNCH		
	}
	#endif

	#ifdef GPU_BUNDLED_LAUNCH
	//EMPTY
	#endif

	#ifndef GPU_BUNDLED_LAUNCH		
	//Support Sbox <=4
	for (int i=0;i<i_limit;i++){
		last_round_trails[cur_r] += 1;

		if ( (*next_round_sbox_num)!=0){
			int next_round_i_limit = std::pow(7, (*next_round_sbox_num));

			//Not Last Round continue deeper, 1, .., n-2 -> n-1
			if (cur_r != PATTERN_ROUND_MITM_BACKWARD-2){
				MITM_trifle_diff_cluster_gpu_recursion_backward(thread_id,next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob, next_round_i_limit ,false);
			}
			//Last Round, n-1 -> n
			else{
				// if (false){
				if ( (*next_round_sbox_num)>=4 && method==PROCESSING_METHOD::GPU ){
					trifle_gpu_manager_arr[thread_id].trifle_manager->MITM_kernel_compute_1round_last_backward(i); 
					last_round_trails[PATTERN_ROUND-1]+= pow(7,*next_round_sbox_num);
				}
				else{
					//NOTE: Only bundled launches can use gpu for <=3 sboxes, if processing method for cur_r is cpu, then call this as well (becuase gpu does have info)
					MITM_trifle_diff_cluster_gpu_using_cpu_final_r_backward(next_dy, cur_r+1, *next_round_sbox_num, next_round_sbox_index, *next_round_prob,
						thread_id);
				}
			}
		}

		//Prepare for the next one
		next_dy = next_dy + 32;
		next_round_prob = next_round_prob + 1;
		next_round_sbox_num = next_round_sbox_num + 9;
		next_round_sbox_index = next_round_sbox_index + 9;
	}
	#endif
}


/*
NOTE: CPU ONLY Code Region
*/
void trifle_diff_cluster_recursion_cpu(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, float cur_prob) {
	int cur_val, weight;
	unsigned int loop_length;
	if (cur_sbox >= MAX_SBOX) { //Special case for after last sbox get subsituted
		goto NEXT; //If last sbox is empty this will not be here but straight to NEXT from the same recursion level
	}
	cur_val = cur_partial_dy[cur_sbox]; //Special Init 
	while (cur_val == 0) { //If empty
		cur_sbox++; //inspect the next sbox
		if (cur_sbox >= MAX_SBOX) { //If next sbox is out of bounds (if last sbox is empty straight to NEXT)
			goto NEXT;
		}
		cur_val = cur_partial_dy[cur_sbox];
	}

	//cur_partial_dy[x] is not empty do
	//Relevant substitution
	//Get from DDT (branching here)
	//Check Continue? (Early abortion not implemented yet)
	loop_length = TRIFLE::diff_table_size_host[cur_val];
	for (unsigned int i = 0; i < loop_length; i++) { //Check all possible value here
		unsigned char target_val = TRIFLE::diff_table_host[cur_val][i];

		// cout << "\nP: " << log2(target_p) << "\tVal:" << (int)target_val << "\ti:" << i << "\tdiff_table_size_host[cur_val]:" << diff_table_size_host[cur_val];
		//TODO update this to prob_table_host
		
		float target_p = (TRIFLE::prob_table_host[cur_val][i] );

		unsigned char cur_partial_dy_new[32];
		memcpy(cur_partial_dy_new, cur_partial_dy, 32);
		cur_partial_dy_new[cur_sbox] = target_val;

		trifle_diff_cluster_recursion_cpu(cur_partial_dy_new, cur_r, cur_sbox + 1, cur_prob * target_p); //Branch to it
	}

	return;

	NEXT:
	;
	//Last Sbox (Proceed to next round, terminate current branch, or save as best prob)
	if (true) { //(cur_sbox >= MAX_SBOX)
		//Last round or next round
		if (cur_r != PATTERN_ROUND - 1) { //Second last round 10 if PATTERN_R = 10, then Round 8 (or 7 zero-indexed)

			//Permutation (modify cur_partial_dy and save to new_partial_dy)
			unsigned char new_partial_dy[32] = { 0 };
			unsigned long long front_64 = 0, back_64 = 0;

			for (int i = 0; i < 32; i++) {
				if (cur_partial_dy[i] > 0) {
					//Permutation LUTable
					//25% less running time compared to normal computation
					front_64 |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][0];
					back_64  |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][1];

					//Permutation Normal Computation
					// for (int j = 0; j < 4; j++) {
					// 	unsigned long long filtered_word = ((cur_partial_dy[i] & (0x1 << j)) >> j) & 0x1;
					// 	if (filtered_word == 0) continue; //no point continue if zero, go to next elements

					// 	int bit_pos = (TRIFLE::perm_host[((31 - i) * 4) + j]); //THIS is correct
					// 	// std::cout << bit_pos << " check ";

					// 	if ((bit_pos / 64) == 1) {  //Front
					// 		bit_pos -= 64;
					// 		front_64 |= (filtered_word << bit_pos);
					// 		// std::cout << bit_pos << " frontcheck " << filtered_word << " " << std::hex <<  ( (unsigned long long) filtered_word << bit_pos ) << std::dec ;
					// 	}
					// 	else {  //Back
					// 		back_64 |= (filtered_word << bit_pos);
					// 		// std::cout << bit_pos << " backcheck " << filtered_word << " " << std::hex <<  ( (unsigned long long) filtered_word << bit_pos ) << std::dec;
					// 	}
					// 	// getchar();
					// }
				}
			}
			for (int i = 0; i < 16; i++) {
				new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
			}
			for (int i = 16; i < 32; i++) {
				new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
			}

			//----
			//NOT Last Round but last sbox
			// Proceed to next round
			last_round_trails[cur_r] += 1;
			if (cur_r == 1) {
				std::cerr << "\n"
							<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
				std::cerr.flush();
			}

            weight = 0;
			wz::hw_word_u4(new_partial_dy, 32, weight);
			//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
			if (weight <= MAX_AS_USED) //If only next round AS <= 8, weight <9, weight<5 (target)
			{
				//HACK: MATSUI BOUND
				float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
				//For example 10 round, final round is 9 (it wont reach here) so is cur_r max is 8, 10-8-2 = 0
				//So next round + all round remaining = (2^-2) weight number of time (meanns optimistics) * (2^-2)(reamining round assumed to be 1 AS per round) rounds times
				if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND_PURE) {
					trifle_diff_cluster_recursion_cpu(new_partial_dy, cur_r + 1, 0, cur_prob);
				}
				else
				{
					// cout << "\nEsimated Com " << log2(estimated_com_prob) << "\tcur_prob:" << log2(cur_prob) << "\tesimte*cur=" << log2(estimated_com_prob*cur_prob) << "\tR:"<<cur_r+1;
					return;
				}
			}
		}
		else {
			last_round_trails[PATTERN_ROUND-1]+=1;

			bool is_same= true;
			for (int i=0;i<32;i++){
				if (TRIFLE::final_dy_host[i] != cur_partial_dy[i]){ //Comparing non-permutated last round result
					is_same= false;
					break;
				}
			}

			if (is_same){
				//Debug infor
				// float debug_prob = log2(cur_prob);
				// std::cout << "\nProb: " << debug_prob;

				trifle_gpu_manager_arr[0].cluster_prob+= cur_prob;
				trifle_gpu_manager_arr[0].cluster_size++;
			}
		}
	}
};

//Backward Search Verification - Only search backwards like forward in reverse
void trifle_diff_cluster_recursion_cpu_backward(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, float cur_prob){
	int cur_val, weight;
	unsigned int loop_length;
	if (cur_sbox >= MAX_SBOX) { //Special case for after last sbox get subsituted
		goto NEXT; //If last sbox is empty this will not be here but straight to NEXT from the same recursion level
	}
	cur_val = cur_partial_dy[cur_sbox]; //Special Init 
	while (cur_val == 0) { //If empty
		cur_sbox++; //inspect the next sbox
		if (cur_sbox >= MAX_SBOX) { //If next sbox is out of bounds (if last sbox is empty straight to NEXT)
			goto NEXT;
		}
		cur_val = cur_partial_dy[cur_sbox];
	}

	//cur_partial_dy[x] is not empty do
	//Relevant substitution
	//Get from DDT (branching here)
	//Check Continue? (Early abortion not implemented yet)
	loop_length = TRIFLE::diff_table_size_host[cur_val];
	for (unsigned int i = 0; i < loop_length; i++) { //Check all possible value here
		unsigned char target_val = TRIFLE::diff_table_host_reversed[cur_val][i];

		// cout << "\nP: " << log2(target_p) << "\tVal:" << (int)target_val << "\ti:" << i << "\tdiff_table_size_host[cur_val]:" << diff_table_size_host[cur_val];
		//TODO update this to prob_table_host
		
		float target_p = (TRIFLE::prob_table_host[cur_val][i] );

		unsigned char cur_partial_dy_new[32];
		memcpy(cur_partial_dy_new, cur_partial_dy, 32);
		cur_partial_dy_new[cur_sbox] = target_val;

		trifle_diff_cluster_recursion_cpu_backward(cur_partial_dy_new, cur_r, cur_sbox + 1, cur_prob * target_p); //Branch to it
	}

	return;

	NEXT:
	;
	//Last Sbox (Proceed to next round, terminate current branch, or save as best prob)
	if (true) { //(cur_sbox >= MAX_SBOX)
		//Last round or next round
		if (cur_r != PATTERN_ROUND - 1) { //Second last round 10 if PATTERN_R = 10, then Round 8 (or 7 zero-indexed)

			//Permutation (modify cur_partial_dy and save to new_partial_dy)
			unsigned char new_partial_dy[32] = { 0 };
			unsigned long long front_64 = 0, back_64 = 0;

			for (int i = 0; i < 32; i++) {
				if (cur_partial_dy[i] > 0) {
					//Permutation LUTable
					//25% less running time compared to normal computation
					front_64 |= TRIFLE::perm_lookup_host_reversed[i][cur_partial_dy[i]][0];
					back_64  |= TRIFLE::perm_lookup_host_reversed[i][cur_partial_dy[i]][1];
				}
			}
			for (int i = 0; i < 16; i++) {
				new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
			}
			for (int i = 16; i < 32; i++) {
				new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
			}

			//----
			//NOT Last Round but last sbox
			//Proceed to next round
			last_round_trails[cur_r] += 1;
			// if (cur_r == 1) {
			// 	std::cerr << "\n"
			// 				<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
			// 	std::cerr.flush();
			// }

            weight = 0;
			wz::hw_word_u4(new_partial_dy, 32, weight);
			//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
			if (weight <= MAX_AS_USED) //If only next round AS <= 8, weight <9, weight<5 (target)
			{
				//HACK: MATSUI BOUND
				float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
				//For example 10 round, final round is 9 (it wont reach here) so is cur_r max is 8, 10-8-2 = 0
				//So next round + all round remaining = (2^-2) weight number of time (meanns optimistics) * (2^-2)(reamining round assumed to be 1 AS per round) rounds times
				if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND) {
					trifle_diff_cluster_recursion_cpu_backward(new_partial_dy, cur_r + 1, 0, cur_prob);
				}
				else
				{
					// cout << "\nEsimated Com " << log2(estimated_com_prob) << "\tcur_prob:" << log2(cur_prob) << "\tesimte*cur=" << log2(estimated_com_prob*cur_prob) << "\tR:"<<cur_r+1;
					return;
				}
			}
		}
		else {
			last_round_trails[PATTERN_ROUND-1]+=1;

			bool is_same= true;
			for (int i=0;i<32;i++){
				if (TRIFLE::ref_dx_host[i] != cur_partial_dy[i]){ //Comparing non-permutated last round result
					is_same= false;
					break;
				}
			}

			if (is_same){
				//Debug infor
				float debug_prob = log2(cur_prob);
				std::cout << "\nProb: " << debug_prob;

				trifle_gpu_manager_arr[0].cluster_prob+= cur_prob;
				trifle_gpu_manager_arr[0].cluster_size++;
			}
		}
	}
}

//MITM forward search
void MITM_trifle_diff_cluster_recursion_cpu_forward(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, float cur_prob, double* mitm_cache_write,
	int* mitm_cache_size_write){
	int cur_val, weight;
	unsigned int loop_length;
	if (cur_sbox >= MAX_SBOX) { //Special case for after last sbox get subsituted
		goto NEXT; //If last sbox is empty this will not be here but straight to NEXT from the same recursion level
	}
	cur_val = cur_partial_dy[cur_sbox]; //Special Init 
	while (cur_val == 0) { //If empty
		cur_sbox++; //inspect the next sbox
		if (cur_sbox >= MAX_SBOX) { //If next sbox is out of bounds (if last sbox is empty straight to NEXT)
			goto NEXT;
		}
		cur_val = cur_partial_dy[cur_sbox];
	}

	//cur_partial_dy[x] is not empty do
	//Relevant substitution
	//Get from DDT (branching here)
	//Check Continue? (Early abortion not implemented yet)
	loop_length = TRIFLE::diff_table_size_host[cur_val];
	for (unsigned int i = 0; i < loop_length; i++) { //Check all possible value here
		unsigned char target_val = TRIFLE::diff_table_host[cur_val][i];

		// cout << "\nP: " << log2(target_p) << "\tVal:" << (int)target_val << "\ti:" << i << "\tdiff_table_size_host[cur_val]:" << diff_table_size_host[cur_val];
		
		float target_p = (TRIFLE::prob_table_host[cur_val][i] );

		unsigned char cur_partial_dy_new[32];
		memcpy(cur_partial_dy_new, cur_partial_dy, 32);
		cur_partial_dy_new[cur_sbox] = target_val;

		MITM_trifle_diff_cluster_recursion_cpu_forward(cur_partial_dy_new, cur_r, cur_sbox + 1, cur_prob * target_p, mitm_cache_write, mitm_cache_size_write); //Branch to it
	}

	return;

	NEXT:
	;
	//Last Sbox (Proceed to next round, terminate current branch, or save as best prob)
	if (true) { //(cur_sbox >= MAX_SBOX)
		//Permutation (modify cur_partial_dy and save to new_partial_dy)
		unsigned char new_partial_dy[32] = { 0 };
		unsigned long long front_64 = 0, back_64 = 0;

		for (int i = 0; i < 32; i++) {
			if (cur_partial_dy[i] > 0) {
				//Permutation LUTable
				//25% less running time compared to normal computation
				front_64 |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][0];
				back_64  |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][1];
			}
		}
		for (int i = 0; i < 16; i++) {
			new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
		}
		for (int i = 16; i < 32; i++) {
			new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
		}

		//Last round or next round
		if (cur_r != PATTERN_ROUND_MITM_FORWARD - 1) { //Second last round 10 if PATTERN_R = 10, then Round 8 (or 7 zero-indexed)

			//----
			//NOT Last Round but last sbox
			//Proceed to next round
			last_round_trails[cur_r] += 1;
			if (cur_r == 1) {
				std::cerr << "\n"
							<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
				std::cerr.flush();
			}

            weight = 0;
			wz::hw_word_u4(new_partial_dy, 32, weight);
			//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
			if (weight <= MAX_AS_USED) //If only next round AS <= 8, weight <9, weight<5 (target)
			{
				//HACK: MATSUI BOUND
				float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
				//For example 10 round, final round is 9 (it wont reach here) so is cur_r max is 8, 10-8-2 = 0
				//So next round + all round remaining = (2^-2) weight number of time (meanns optimistics) * (2^-2)(reamining round assumed to be 1 AS per round) rounds times
				if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND) {
					MITM_trifle_diff_cluster_recursion_cpu_forward(new_partial_dy, cur_r + 1, 0, cur_prob, mitm_cache_write, mitm_cache_size_write);
				}
				else
				{
					// cout << "\nEsimated Com " << log2(estimated_com_prob) << "\tcur_prob:" << log2(cur_prob) << "\tesimte*cur=" << log2(estimated_com_prob*cur_prob) << "\tR:"<<cur_r+1;
					return;
				}
			}
		}
		else {
			last_round_trails[cur_r]+=1;

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

				mitm_cache_write[index] += cur_prob;
				mitm_cache_size_write[index] += 1;
			}

			// if (is_same){
			// 	//Debug infor
			// 	float debug_prob = log2(cur_prob);
			// 	std::cout << "\nProb: " << debug_prob;

			// 	trifle_gpu_manager_arr[0].cluster_prob+= cur_prob;
			// 	trifle_gpu_manager_arr[0].cluster_size++;
			// }
		}
	}
};

//MITM backward search
void MITM_trifle_diff_cluster_recursion_cpu_backward(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, float cur_prob,  double* mitm_cache_read, int* mitm_size_cache_read){
	int cur_val, weight;
	unsigned int loop_length;
	if (cur_sbox >= MAX_SBOX) { //Special case for after last sbox get subsituted
		goto NEXT; //If last sbox is empty this will not be here but straight to NEXT from the same recursion level
	}
	cur_val = cur_partial_dy[cur_sbox]; //Special Init 
	while (cur_val == 0) { //If empty
		cur_sbox++; //inspect the next sbox
		if (cur_sbox >= MAX_SBOX) { //If next sbox is out of bounds (if last sbox is empty straight to NEXT)
			goto NEXT;
		}
		cur_val = cur_partial_dy[cur_sbox];
	}

	//cur_partial_dy[x] is not empty do
	//Relevant substitution
	//Get from DDT (branching here)
	//Check Continue? (Early abortion not implemented yet)
	loop_length = TRIFLE::diff_table_size_host[cur_val];
	for (unsigned int i = 0; i < loop_length; i++) { //Check all possible value here
		unsigned char target_val = TRIFLE::diff_table_host_reversed[cur_val][i];

		// cout << "\nP: " << log2(target_p) << "\tVal:" << (int)target_val << "\ti:" << i << "\tdiff_table_size_host[cur_val]:" << diff_table_size_host[cur_val];
	
		float target_p = (TRIFLE::prob_table_host[cur_val][i] );

		unsigned char cur_partial_dy_new[32];
		memcpy(cur_partial_dy_new, cur_partial_dy, 32);
		cur_partial_dy_new[cur_sbox] = target_val;

		MITM_trifle_diff_cluster_recursion_cpu_backward(cur_partial_dy_new, cur_r, cur_sbox + 1, cur_prob * target_p, mitm_cache_read, mitm_size_cache_read); //Branch to it
	}

	return;

	NEXT:
	;
	//Last Sbox (Proceed to next round, terminate current branch, or save as best prob)
	if (true) { //(cur_sbox >= MAX_SBOX)
		//Last round or next round
		if (cur_r != PATTERN_ROUND_MITM_BACKWARD - 1) { //Second last round 10 if PATTERN_R = 10, then Round 8 (or 7 zero-indexed)

			//Permutation (modify cur_partial_dy and save to new_partial_dy)
			unsigned char new_partial_dy[32] = { 0 };
			unsigned long long front_64 = 0, back_64 = 0;

			for (int i = 0; i < 32; i++) {
				if (cur_partial_dy[i] > 0) {
					//Permutation LUTable
					//25% less running time compared to normal computation
					front_64 |= TRIFLE::perm_lookup_host_reversed[i][cur_partial_dy[i]][0];
					back_64  |= TRIFLE::perm_lookup_host_reversed[i][cur_partial_dy[i]][1];
				}
			}
			for (int i = 0; i < 16; i++) {
				new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
			}
			for (int i = 16; i < 32; i++) {
				new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
			}

			//----
			//NOT Last Round but last sbox
			//Proceed to next round
			last_round_trails[cur_r] += 1;
			// if (cur_r == 1) {
			// 	std::cerr << "\n"
			// 				<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
			// 	std::cerr.flush();
			// }

            weight = 0;
			wz::hw_word_u4(new_partial_dy, 32, weight);
			//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
			if (weight <= MAX_AS_USED) //If only next round AS <= 8, weight <9, weight<5 (target)
			{
				//HACK: MATSUI BOUND
				float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
				//For example 10 round, final round is 9 (it wont reach here) so is cur_r max is 8, 10-8-2 = 0
				//So next round + all round remaining = (2^-2) weight number of time (meanns optimistics) * (2^-2)(reamining round assumed to be 1 AS per round) rounds times
				if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND) {
					MITM_trifle_diff_cluster_recursion_cpu_backward(new_partial_dy, cur_r + 1, 0, cur_prob, mitm_cache_read, mitm_size_cache_read);
				}
				else
				{
					// cout << "\nEsimated Com " << log2(estimated_com_prob) << "\tcur_prob:" << log2(cur_prob) << "\tesimte*cur=" << log2(estimated_com_prob*cur_prob) << "\tR:"<<cur_r+1;
					return;
				}
			}
		}
		else {
			last_round_trails[cur_r]+=1;

			int sbox_num=0;
			int sbox_index[32]={0};
			for (int i=0;i<32;i++){
				if (cur_partial_dy[i] !=0){
					sbox_index[sbox_num] = i;
					sbox_num+=1;
				}
			}

			if (sbox_num <=3){ //Possible to store three only...
				//Computing appropriate index
				int index=0;
				for (int i=0;i<sbox_num;i++){
					index|= ( ( (sbox_index[i]&0b11111) | ( (cur_partial_dy[sbox_index[i]]&0b1111) << 5) ) << (i * 9) ); 
				}

				int target_size =  mitm_size_cache_read[index];
				if(target_size > 0){ //Exist connection
					float target_prob = ( cur_prob * mitm_cache_read[index]);

					// if ( target_prob >= CLUSTER_PROB_BOUND) {
					if (true){
						//Add to collection
						trifle_gpu_manager_arr[0].cluster_prob+= target_prob;
						trifle_gpu_manager_arr[0].cluster_size+= target_size;

						//Debug Information
						// float debug_prob = log2(target_prob);
						// std::cout << "\nProb: " << debug_prob;
					}
				}
			}
		}
	}
};

//Top X amount of Weight Y
//HARDCODED to AS4
const int MIN_WEIGHT_TOPX = 1;
void trifle_diff_cluster_recursion_cpu_topx(unsigned char* cur_partial_dy, int cur_r, int cur_sbox, float cur_prob, differential_128b_4s_t* top_x_arr, int top_x_num, unsigned char* ori_dx) {
	int cur_val, weight;
	unsigned int loop_length;
	if (cur_sbox >= MAX_SBOX) { //Special case for after last sbox get subsituted
		goto NEXT; //If last sbox is empty this will not be here but straight to NEXT from the same recursion level
	}
	cur_val = cur_partial_dy[cur_sbox]; //Special Init 
	while (cur_val == 0) { //If empty
		cur_sbox++; //inspect the next sbox
		if (cur_sbox >= MAX_SBOX) { //If next sbox is out of bounds (if last sbox is empty straight to NEXT)
			goto NEXT;
		}
		cur_val = cur_partial_dy[cur_sbox];
	}

	loop_length = TRIFLE::diff_table_size_host[cur_val];
	for (unsigned int i = 0; i < loop_length; i++) { //Check all possible value here
		unsigned char target_val = TRIFLE::diff_table_host[cur_val][i];
		if ( cur_r != PATTERN_ROUND-1){ //If not last round prune bad branch
			weight = 0;
            wz::hw_bit_u4(&target_val, 1, weight);
			//HACK: use 1 for quick single path
            if (weight > 1) {  //HACK: Weight is set as 3
                continue;
            }
        }
		
		float target_p = (TRIFLE::prob_table_host[cur_val][i] );

		unsigned char cur_partial_dy_new[32];
		memcpy(cur_partial_dy_new, cur_partial_dy, 32);
		cur_partial_dy_new[cur_sbox] = target_val;

		trifle_diff_cluster_recursion_cpu_topx(cur_partial_dy_new, cur_r, cur_sbox + 1, cur_prob * target_p, top_x_arr, top_x_num, ori_dx); //Branch to it
	}

	return;

	NEXT:
	;
	//Last Sbox (Proceed to next round, terminate current branch, or save as best prob)
	if (true) {
		unsigned char new_partial_dy[32] = { 0 };
		unsigned long long front_64 = 0, back_64 = 0;

		for (int i = 0; i < 32; i++) {
			if (cur_partial_dy[i] > 0) {
				front_64 |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][0];
				back_64  |= TRIFLE::perm_lookup_host[i][cur_partial_dy[i]][1];
			}
		}
		for (int i = 0; i < 16; i++) {
			new_partial_dy[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
		}
		for (int i = 16; i < 32; i++) {
			new_partial_dy[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
		}

		//HACK: remember to -1 if not using advanced culling, else use -2 for advanced culling
		if (cur_r != PATTERN_ROUND - 1) { 

			//----
			//NOT Last Round but last sbox
			//Proceed to next round
			last_round_trails[cur_r] += 1;
			if (cur_r == 1) {
				std::cerr << "\n"
							<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
				std::cerr.flush();
			}

            weight = 0;
			wz::hw_word_u4(new_partial_dy, 32, weight);
			//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
			if (weight <= 4) //If only next round AS <= 8, weight <9, weight<5 (target)
			{
				float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
				if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND_PURE) {

					trifle_diff_cluster_recursion_cpu_topx(new_partial_dy, cur_r + 1, 0, cur_prob, top_x_arr, top_x_num, ori_dx);
				}
				else
				{
					return;
				}
			}
		}
		//Last Rounds - 2
		// else{
		// 	last_round_trails[cur_r] += 1;
		// 	if (cur_r == 1) {
		// 		std::cerr << "\n"
		// 					<< last_round_trails[1] << " per 2905 - " << last_round_trails[1] / 2905.0 * 100 << "% completed";
		// 		std::cerr.flush();
		// 	}

        //     weight = 0;
		// 	wz::hw_word_u4(new_partial_dy, 32, weight);
		// 	//HACK: the ACTIVE AS can be increase further for LARGER ROUND because of the effect...
		// 	if (weight <= 4) //If only next round AS <= 8, weight <9, weight<5 (target)
		// 	{
		// 		float estimated_com_prob = pow(CLUSTER_PROB_INDIV, (PATTERN_ROUND - cur_r - 2)) * pow(CLUSTER_1AS_BEST_PROB, weight);
		// 		if ((estimated_com_prob * cur_prob) >= CLUSTER_PROB_BOUND_PURE) {
		// 			//LAST ROUND IS HERE... Only do one subs because other sub is kinda the same.. cluster wise
		// 			//But it need at least three to cover all range nw @ only two...
		// 			for (int range=0;range<3;range++){
		// 				unsigned char last_round_char[32] = {0};
		// 				unsigned char last_round_char_permutated[32]= {0};
		// 				memcpy(last_round_char, new_partial_dy,32);
		// 				float new_prob = cur_prob;

		// 				for (int i=0;i<32;i++){ //Subs
		// 					if (last_round_char[i]!=0){
		// 						unsigned char target_val = TRIFLE::diff_table_host[last_round_char[i]][range];
		// 						float target_prob = TRIFLE::prob_table_host[last_round_char[i]][range];

		// 						last_round_char[i] = target_val;
		// 						new_prob *= target_prob;
		// 					}
		// 				}

		// 				for (int i = 0; i < 32; i++) { //Permutate
		// 					if (last_round_char[i] > 0) {
		// 						front_64 |= TRIFLE::perm_lookup_host[i][last_round_char[i]][0];
		// 						back_64  |= TRIFLE::perm_lookup_host[i][last_round_char[i]][1];
		// 					}
		// 				}
		// 				for (int i = 0; i < 16; i++) {
		// 					last_round_char_permutated[i] = (front_64 >> ((15 - i) * 4)) & 0xf;
		// 				}
		// 				for (int i = 16; i < 32; i++) {
		// 					last_round_char_permutated[i] = (back_64 >> ((31 - i) * 4)) & 0xf;
		// 				}

		// 				//Normal else routine
		// 				last_round_trails[PATTERN_ROUND-1]+=1;
		// 				weight = 0;
		// 				wz::hw_word_u4(last_round_char, 32, weight); //Before permutation must be >1
		// 				if (weight>=MIN_WEIGHT_TOPX){
		// 					for (int i=0;i<top_x_num;i++){
		// 						//HACK: >3 need >4 sboxes to work
		// 						if (new_prob > top_x_arr[i].p){
		// 							//Empty the spot
		// 							for (int j=top_x_num-1;j>i;j--){
		// 								top_x_arr[j] = top_x_arr[j-1]; //invoke copy assignement
		// 							}

		// 							differential_128b_4s_t diff;
		// 							memcpy(diff.dx, ori_dx ,32);
		// 							memcpy(diff.dy, last_round_char_permutated,32);
		// 							memcpy(diff.dy_b4p, last_round_char, 32);
		// 							diff.p = new_prob;
		// 							top_x_arr[i] = diff; //invoke copy assignement

		// 							break;
		// 						} 
		// 					}
		// 				}
		// 			}
		// 		}
		// 		else
		// 		{
		// 			return;
		// 		}
		// 	}
		// }

		else { //LAst Round
			last_round_trails[PATTERN_ROUND-1]+=1;
			weight = 0;
			wz::hw_word_u4(cur_partial_dy, 32, weight); //Before permutation must be >1
			if (weight>=MIN_WEIGHT_TOPX){
				for (int i=0;i<top_x_num;i++){
					//HACK: >3 need >4 sboxes to work
					if (cur_prob > top_x_arr[i].p){
						//Empty the spot
						for (int j=top_x_num-1;j>i;j--){
							top_x_arr[j] = top_x_arr[j-1]; //invoke copy assignement
						}

						differential_128b_4s_t diff;
						memcpy(diff.dx, ori_dx ,32);
						memcpy(diff.dy, new_partial_dy,32);
						memcpy(diff.dy_b4p, cur_partial_dy, 32);
						diff.p = cur_prob;
						top_x_arr[i] = diff; //invoke copy assignement

						break;
					} 
				}
			}
		}
			//End of Last Rounds.

	}
};
/*
* End of Only CPU region
*/


/*
* Region : Called from Main
*/
void trifle_diff_cluster_gpu(double& cluster_prob, long long& cluster_size, std::chrono::steady_clock::time_point &end){
	int thread_id =0;
	int sbox_index[32];
	for (int i=0;i<32;i++){
		sbox_index[i] = 32;
	}
	int sbox_index_ptr =0;
	for (int i=0;i<32;i++){
		if (TRIFLE::ref_dx_host[i] > 0){
			sbox_index[sbox_index_ptr] = i;
			sbox_index_ptr++;
		}
	}

	int sbox_num = sbox_index_ptr;
	float cur_prob = 1.0f;
    trifle_diff_cluster_gpu_recursion(thread_id, TRIFLE::ref_dx_host, 0, sbox_num, sbox_index, cur_prob ,7,false); //ref_dx from kernel_trifle.cuh

	end = std::chrono::steady_clock::now();

	//for (int i=0;i<cpu_thread_size;i++){ //Havent init probably
	for (int i=0;i<1;i++){
		trifle_gpu_manager_arr[i].reduction();
		cluster_prob += trifle_gpu_manager_arr[i].cluster_prob;
		cluster_size += trifle_gpu_manager_arr[i].cluster_size;
	}

	for (int i=0;i<10;i++){
		std::cout << "\n Rounds Trails " << i << " : " << last_round_trails[i] ;
	}
	std::cout << std::endl;
};

void trifle_diff_cluster_cpu_only(double& cluster_prob, long long& cluster_size){
    trifle_diff_cluster_recursion_cpu(TRIFLE::ref_dx_host, 0, 0, 1.0); //ref_dx from kernel_trifle.cuh

	cluster_prob = trifle_gpu_manager_arr[0].cluster_prob;
	cluster_size = trifle_gpu_manager_arr[0].cluster_size;

	for (int i=0;i<10;i++){
		std::cout << "\n Rounds Trails " << i << " : " << last_round_trails[i] ;
	}
	std::cout << std::endl;
};

void MITM_trifle_diff_cluster_cpu_only(double& cluster_prob, long long& cluster_size){
	//Verification of backwards search
	// R10
	// Forwards 
	// Cluster Probabilities: log2: -35.9714 , base_10 : 1.48431e-11
	// Number of Cluster Trails:17

	//Backwards - Cuz Edge case -50 (>49) etc is very edge case that gotten slipped through because final round sbox is all assumed to be best case -2 (but they are not)
	//Cluster Probabilities: log2: -35.9716 , base_10 : 1.48415e-11
	//Number of Cluster Trails:11

	//MITM - CPU
	// Cluster Probabilities: log2: -35.9708 , base_10 : 1.48499e-11
	// Number of Cluster Trails:6435
	// Time difference (s ) = 7

	//MITM - GPU
	// Cluster Probabilities: log2: -35.9708 , base_10 : 1.48499e-11
	// Number of Cluster Trails:6435
	// Time difference (s ) = 5

	//58777 cluster
	//35.9707
	// unsigned char dx[32] = {
	// 	0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0xb,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x6, 0x0, 0x6, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x6, 0x0, 0x6, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	//-33.9725 ish 
	//Cluster Probabilities: log2: -33.9722 , base_10 : 5.93403e-11
	// Number of Cluster Trails:10216
	// unsigned char dx[32] = {
	// 	0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0xb,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x0, 0x0, 0x0, 0x0,
    //     0x6, 0x0, 0x6, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x6, 0x0, 0x6, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	//GPU MITM Testing
	//R10
	//Cluster Probabilities: log2: -37.2747 , base_10 : 6.01427e-12
	//Number of Cluster Trails:369

	//R20 - -79
	//Mostly* ALL GPU MITM
	//Cluster Probabilities: log2: -57.9743 , base_10 : 3.53189e-18
	//Number of Cluster Trails:33972408
	//Time : Time difference (s ) = 343
	// ??



	//R30 - 
	// Cluster Probabilities: log2: -87.9533 , base_10 : 3.33753e-27
	// Number of Cluster Trails:9428161536
	// Time difference (s ) = 13366
	//Time : Time difference (s ) = 343
	// unsigned char dx[32] = {
	// 	0x0, 0xb, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x3,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// R40 - 
	// unsigned char dx[32] = {
	// 	0x7, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x3,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	//R10 - 3Sbox
	// unsigned char dx[32] = {
	// 	0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0xb, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x9, 0x0, 0x9, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x9, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	//R22 - 3AS using 4AS
	unsigned char dx[32] = {
		0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0xb, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
	};

	unsigned char dy[32] = {
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x3, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0
	};

	//Investigate GPU AS 6... R5
	// unsigned char dx[32] = {
	// 	0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x3, 0x0
	// };

	// unsigned char dy[32] = {
    //     0x5, 0x3, 0x5, 0x3,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0xc, 0x3,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //   	0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0,
    //     0x0, 0x0, 0x0, 0x0
	// };

	//0000 6060 0000 6060 0000 0000 0000 0000

	trifle_gpu_manager_arr[0].trifle_manager->change_parameter(dy, dx);

	//trifle_diff_cluster_cpu_only(cluster_prob, cluster_size);
	//trifle_diff_cluster_recursion_cpu_backward(TRIFLE::final_dy_host, 0, 0, 1.0);

	double* mitm_cache = new double[134217728]; //1GB-ish
	int* mitm_size_cache = new int[134217728];
	for (int i=0;i< 134217728; i++){
		mitm_cache[i]=  0;
		mitm_size_cache[i] = 0;
	}

	bool is_MITM = true;
	bool is_GPU = true;
	if (is_MITM){
		if (is_GPU){
			//GPU
			int thread_id =0;
			{
				int sbox_index[32];
				for (int i=0;i<32;i++){
					sbox_index[i] = 32;
				}
				int sbox_index_ptr =0;
				for (int i=0;i<32;i++){
					if (TRIFLE::ref_dx_host[i] > 0){
						sbox_index[sbox_index_ptr] = i;
						sbox_index_ptr++;
					}
				}
				float cur_prob = 1.0f;
				int sbox_num = sbox_index_ptr;
				MITM_trifle_diff_cluster_gpu_recursion_forward(thread_id, TRIFLE::ref_dx_host, 0, sbox_num, sbox_index, cur_prob ,pow(7,sbox_num),false); //ref_dx from kernel_trifle.cuh
			}

			trifle_gpu_manager_arr[thread_id].trifle_manager->kernel_MITM_intermediate_reduction
				(trifle_gpu_manager_arr[thread_id].mitm_prob_cache, trifle_gpu_manager_arr[thread_id].mitm_cluster_size_cache);
			{
				int sbox_index[32];
				for (int i=0;i<32;i++){
					sbox_index[i] = 32;
				}
				int sbox_index_ptr =0;
				for (int i=0;i<32;i++){
					if (TRIFLE::final_dy_host[i] > 0){
						sbox_index[sbox_index_ptr] = i;
						sbox_index_ptr++;
					}
				}
				float cur_prob = 1.0f;
				int sbox_num = sbox_index_ptr;

				//NOTE: backward gpu last round n is temporary disblaed because have not been verified
				MITM_trifle_diff_cluster_gpu_recursion_backward(thread_id, TRIFLE::final_dy_host, 0, sbox_num, sbox_index, cur_prob ,pow(7,sbox_num),false); //ref_dx from kernel_trifle.cuh

				trifle_gpu_manager_arr[thread_id].reduction();
			}
		}
		else{
			MITM_trifle_diff_cluster_recursion_cpu_forward(TRIFLE::ref_dx_host, 0, 0, 1.0, mitm_cache, mitm_size_cache); //ref_dx from kernel_trifle.cuh
			MITM_trifle_diff_cluster_recursion_cpu_backward(TRIFLE::final_dy_host, 0,0,1.0,mitm_cache, mitm_size_cache);
		}
	}
	else{
		if (is_GPU){
			//GPU
			int thread_id =0;
			int sbox_index[32];
			for (int i=0;i<32;i++){
				sbox_index[i] = 32;
			}
			int sbox_index_ptr =0;
			for (int i=0;i<32;i++){
				if (TRIFLE::ref_dx_host[i] > 0){
					sbox_index[sbox_index_ptr] = i;
					sbox_index_ptr++;
				}
			}

			float cur_prob = 1.0f;
			int sbox_num = sbox_index_ptr;
			trifle_diff_cluster_gpu_recursion(thread_id, TRIFLE::ref_dx_host, 0, sbox_num, sbox_index, cur_prob ,7,false); //ref_dx from kernel_trifle.cuh

			trifle_gpu_manager_arr[thread_id].reduction();
		}
		else{
			//CPU
			trifle_diff_cluster_cpu_only(cluster_prob, cluster_size);
		}
	}

	End:
	cluster_prob = trifle_gpu_manager_arr[0].cluster_prob;
	cluster_size = trifle_gpu_manager_arr[0].cluster_size;

	for (int i=0;i<10;i++){
		std::cout << "\n Rounds Trails " << i << " : " << last_round_trails[i] ;
	}
	std::cout << std::endl;
};

//NOTE: Search for Top x
//32(diff sbox position)*15 (1-16) *11 (top10) including original * 5s (GPU) = 8 hours...
void trifle_diff_cluster_r10_search(){
	//Generate Input
	//TODO todo
	unsigned char dx[32]={0};

	//TODO maybe do 2 active sbox!?
	//One active sbox at a time, 32 / 996 (31 times) if two active sboxes.
	// int sbox_starting = 0;
	// int sbox_ending = 32;
	int sbox_starting = 0;
	int sbox_ending = 32; //Change back to 32

	int dx_val_start = 1;
	int dx_val_end = 16;

	int dx_val_ref[4];
	//Do 8,11,15 cuz it cover all range
	// dx_val_ref[0] = 0x8;
	// dx_val_ref[1] = 0xb;
	// dx_val_ref[2] = 0xf;

	//Use the foolinwg: 1st paths will leads to better one...
	dx_val_ref[0] = 0x7;
	dx_val_ref[1] = 0xb;
	dx_val_ref[2] = 0xd;
	dx_val_ref[3] = 0xe;

	// dx_val_ref[0] = 0x4;
	// dx_val_ref[1] = 0x1;
	// dx_val_ref[2] = 0x2;
	// dx_val_ref[3] = 0x8;


	double* mitm_cache = new double[134217728]; //1GB-ish
	int* mitm_size_cache = new int[134217728];
	
	//TODO remember replace back to 0 to 32
	for (int sbox=sbox_starting;sbox<sbox_ending;sbox++){ //Front to Back
	// for (int sbox=sbox_ending-1;sbox>=sbox_starting;sbox--){ //Back to Front
		//15 * 15 (225) or  2*2(4) - 56 times ??
		//TODO remember replace back to dx_val_start and end

		//dx_val = 0; dx_val < 3; dx_val++
		for (int dx_val=0;dx_val<4;dx_val++){
			//Reset DX
			for (int sbox_clear=0;sbox_clear<32;sbox_clear++){
				dx[sbox_clear] = 0;
			}
			//Process DX
			dx[sbox] = dx_val_ref[dx_val];

			//Get Top 10 Paths from DX -> DY
			const int XAMOUNT = 1;
			const int START = 0;
			// differential_128b_4s_t topx[XAMOUNT];
			differential_128b_4s_t* topx = new differential_128b_4s_t[XAMOUNT];
			trifle_diff_cluster_recursion_cpu_topx(dx,0,0,1.0f,topx,XAMOUNT, dx);

			for (int i=START;i<XAMOUNT;i++){
				// printf("\n----\n%i :\nDX: %s \nDY: %s\nDy_B4P: %s\nProb:%.17g\nProb (Log2):%.17g\n----\n",i,topx[i].dx_str().c_str(),topx[i].dy_str().c_str(),topx[i].dy_b4p_str().c_str()
				// ,topx[i].p,log2(topx[i].p));

				trifle_gpu_manager_arr[0].trifle_manager->change_parameter(topx[i].dy_b4p, dx);

				//Cluster them
				double cluster_prob = 0;
				unsigned long long cluster_size= 0;

				//CPU
				// trifle_diff_cluster_recursion_cpu(dx, 0, 0, 1.0); //ref_dx from kernel_trifle.cuh
				// cluster_prob = trifle_gpu_manager_arr[0].cluster_prob;
				// cluster_size = trifle_gpu_manager_arr[0].cluster_size;
				// trifle_gpu_manager_arr[0].cluster_size= 0 ;
				// trifle_gpu_manager_arr[0].cluster_prob= 0 ;

				//GPU
				// int sbox_num = 1;
				// int sbox_index[8]={32};
				// for (int sbox_i=0;sbox_i<8;sbox_i++){
				// 	sbox_index[sbox_i] = 32;
				// }
				// sbox_index[0] = sbox; 
				// int thread_id = 0;
				// trifle_diff_cluster_gpu_recursion(thread_id, dx, 0, sbox_num, sbox_index, 1.0f, 7, false);  //ref_dx from kernel_trifle.cuh

				// trifle_gpu_manager_arr[0].reduction(); //First combine gpu with cpu first
				// cluster_prob += trifle_gpu_manager_arr[0].cluster_prob; //extract the result
				// cluster_size += trifle_gpu_manager_arr[0].cluster_size;

				// trifle_gpu_manager_arr[0].cluster_size=  0;
				// trifle_gpu_manager_arr[0].cluster_prob=  0;

				//CPU MITM
				// for (int i=0;i< 134217728; i++){
				// 	mitm_cache[i]=  0;
				// 	mitm_size_cache[i] = 0;
				// }

				// MITM_trifle_diff_cluster_recursion_cpu_forward(TRIFLE::ref_dx_host, 0, 0, 1.0, mitm_cache, mitm_size_cache); //ref_dx from kernel_trifle.cuh
				// MITM_trifle_diff_cluster_recursion_cpu_backward(TRIFLE::final_dy_host, 0,0,1.0,mitm_cache, mitm_size_cache);

				// cluster_prob = trifle_gpu_manager_arr[0].cluster_prob;
				// cluster_size = trifle_gpu_manager_arr[0].cluster_size;

				// trifle_gpu_manager_arr[0].cluster_size=  0;
				// trifle_gpu_manager_arr[0].cluster_prob=  0;

				//GPU MITM
				int thread_id =0;
				{
					int sbox_index[32];
					for (int i=0;i<32;i++){
						sbox_index[i] = 32;
					}
					int sbox_index_ptr =0;
					for (int i=0;i<32;i++){
						if (TRIFLE::ref_dx_host[i] > 0){
							sbox_index[sbox_index_ptr] = i;
							sbox_index_ptr++;
						}
					}
					float cur_prob = 1.0f;
					int sbox_num = sbox_index_ptr;
					MITM_trifle_diff_cluster_gpu_recursion_forward(thread_id, TRIFLE::ref_dx_host, 0, sbox_num, sbox_index, cur_prob ,pow(7,sbox_num),false); //ref_dx from kernel_trifle.cuh
				}
				{
					int sbox_index[32];
					for (int i=0;i<32;i++){
						sbox_index[i] = 32;
					}
					int sbox_index_ptr =0;
					for (int i=0;i<32;i++){
						if (TRIFLE::final_dy_host[i] > 0){
							sbox_index[sbox_index_ptr] = i;
							sbox_index_ptr++;
						}
					}
					float cur_prob = 1.0f;
					int sbox_num = sbox_index_ptr;
					MITM_trifle_diff_cluster_gpu_recursion_backward(thread_id, TRIFLE::final_dy_host, 0, sbox_num, sbox_index, cur_prob ,pow(7,sbox_num),false); //ref_dx from kernel_trifle.cuh

					trifle_gpu_manager_arr[thread_id].reduction();
				}
				
				cluster_prob = trifle_gpu_manager_arr[0].cluster_prob;
				cluster_size = trifle_gpu_manager_arr[0].cluster_size;
				trifle_gpu_manager_arr[0].reset();

            	//Save the result onto file //Currently output to stdout
				printf("\n----\n%i :\nDX: %s \nDY: %s\nDy_B4P: %s\nProb:%.17g\nProb (Log2):%.17g",i,topx[i].dx_str().c_str(),topx[i].dy_str().c_str(),topx[i].dy_b4p_str().c_str()
					   ,topx[i].p,log2(topx[i].p));
				std::cout << "\nCluster Probabilities: log2: " << std::dec << log2(cluster_prob) << " , base_10 : " << cluster_prob;
				std::cout << "\nNumber of Cluster Trails:" << cluster_size;
				std::cout << "\n----\n";
				std::cout.flush();
			}

			delete[] topx;
		}
	}


}
