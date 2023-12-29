#include <curand.h>
#include <curand_kernel.h>

#include <iomanip>
#include <iostream>
#include <chrono>
#include <vector>
#define ATOMIC_THREAD_CNT 1024
#define ATOMIC_BLOCK_CNT 32768


__global__ void setup(curandStateXORWOW_t  *state)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(123456, id, 0, &state[id]);
}

__global__ void generateRand(curandStateXORWOW_t  *state, float* results, int runs)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  curandStateXORWOW_t  localState = state[id];
  for (int i = 0; i < runs; ++i)
  {
    results[id + i * ATOMIC_BLOCK_CNT] = curand_uniform(&localState);
  }
  state[id] = localState;
}
__global__ void testAtomic(const float *in, float *out)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;

  for (int c=0; c<3; c++){
      atomicAdd(&(out[blockIdx.x * 4 + c]), in[id * 4+c] * in[id*4+3]);
  }
  atomicAdd(&(out[blockIdx.x * 4 + 3]), in[id*4+3]);
}

void validate(const std::vector<float> &in, std::vector<float> &out)
{
  for (int i=0; i<ATOMIC_BLOCK_CNT; i++){
    for (int j=0; j<ATOMIC_THREAD_CNT; j++){
      int id = i*ATOMIC_THREAD_CNT + j;
      for (int c=0; c<3; c++){
          out[i * 4 + c] += in[id * 4+c] * in[id*4+3];
      }
      out[i*4+3] += in[id * 4 + 3];
    }
  }
}
int main(int argc, char* argv[])
{
  curandStateXORWOW_t * d_states;
  cudaMalloc((void **) &d_states, sizeof(curandStateXORWOW_t ) * ATOMIC_BLOCK_CNT);
  setup<<<32, ATOMIC_BLOCK_CNT/32>>>(d_states);

  float* d_results;
  cudaMalloc((void **) &d_results, sizeof(float) * ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);
  cudaMemset(d_results, 0, sizeof(float) * ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);
  std::vector<float> hostResults(ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);

  for (int i=0; i<4; i++)
    generateRand<<<32, ATOMIC_BLOCK_CNT/32>>>(d_states, d_results+ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*i, ATOMIC_THREAD_CNT);
  // Copy results
  cudaMemcpy(hostResults.data(), d_results, hostResults.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // Output number 12345
  ::std::cout << "end is " << hostResults[hostResults.size()-1] << std::endl;
  std::cout << "red: " << hostResults[0] << " alpha: " << hostResults[3] << std::endl;

  float *d_atomic_test;
  
  cudaMalloc(&d_atomic_test, sizeof(float)*ATOMIC_BLOCK_CNT*4);
  
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
	std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

	start_ct1 = std::chrono::steady_clock::now();

  cudaMemset(d_atomic_test, 0, sizeof(float)*ATOMIC_BLOCK_CNT*4);
  testAtomic<<<ATOMIC_BLOCK_CNT, ATOMIC_THREAD_CNT>>>(d_results,d_atomic_test);
  cudaStreamSynchronize(0);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    std::cout << "Error: " << cudaGetErrorString(status) << std::endl;


	stop_ct1 = std::chrono::steady_clock::now();
	float atomic_time =
		std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
			.count();


  std::vector<float> h_atomic_test(ATOMIC_BLOCK_CNT*4);
  status = cudaMemcpy(h_atomic_test.data(), d_atomic_test, sizeof(float)*ATOMIC_BLOCK_CNT*4, cudaMemcpyDeviceToHost);
  
  if (status != cudaSuccess)
    std::cout << "Error memcpy d_atomic_test: " << cudaGetErrorString(status) << std::endl;

  std::vector<float> validate_atomic(ATOMIC_BLOCK_CNT*4, 0);
  validate(hostResults, validate_atomic);
  int cnt = 0;
  for (int i=0; i<validate_atomic.size(); i++){
    if (std::abs(h_atomic_test[i] - validate_atomic[i]) > 1e-3){
      if (cnt < 5)
       std::cout << i << " " << h_atomic_test[i] << " " << validate_atomic[i] << " " << std::abs(h_atomic_test[i] - validate_atomic[i]) << std::endl;
      cnt++;
    }
  }
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "val: " << h_atomic_test[0] << " kernel_time (cudaEventElapsedTime) " << std::setw(5) << atomic_time << "ms" << std::endl;
  std::cout << "Number of invalid results: " << cnt << std::endl;
  return 0;
}