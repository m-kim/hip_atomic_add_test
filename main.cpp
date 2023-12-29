#include <hip/hip_runtime.h>
#include <hiprand/hiprand.hpp>

#include <iomanip>
#include <iostream>
#define ATOMIC_THREAD_CNT 1024
#define ATOMIC_BLOCK_CNT 32768


__global__ void setup(hiprandStateXORWOW_t *state)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  hiprand_init(123456, id, 0, &state[id]);
}

__global__ void generateRand(hiprandStateXORWOW_t *state, float* results, int runs)
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  hiprandStateXORWOW_t localState = state[id];
  for (int i = 0; i < runs; ++i)
  {
    results[id + i * ATOMIC_BLOCK_CNT] = hiprand_uniform(&localState);
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

int main(int argc, char* argv[])
{
  hiprandStateXORWOW_t* d_states;
  hipMalloc((void **) &d_states, sizeof(hiprandStateXORWOW_t) * ATOMIC_BLOCK_CNT);
  setup<<<32, ATOMIC_BLOCK_CNT/32>>>(d_states);

  float* d_results;
  hipMalloc((void **) &d_results, sizeof(float) * ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);
  hipMemset(d_results, 0, sizeof(float) * ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);
  std::vector<float> hostResults(ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4);

  for (int i=0; i<4; i++)
    generateRand<<<32, ATOMIC_BLOCK_CNT/32>>>(d_states, d_results+ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*i, ATOMIC_THREAD_CNT);
  // Copy results
  hipMemcpy(hostResults.data(), d_results, hostResults.size() * sizeof(float), hipMemcpyDeviceToHost);

  // Output number 12345
  ::std::cout << "end is " << hostResults[hostResults.size()-1] << std::endl;
  std::cout << "red: " << hostResults[0] << " alpha: " << hostResults[3] << std::endl;

  float *d_atomic_test;
  
  hipMalloc(&d_atomic_test, sizeof(float)*ATOMIC_BLOCK_CNT*sizeof(float)*4);
  
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
	std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

	start_ct1 = std::chrono::steady_clock::now();

  hipMemset(d_atomic_test, 0, sizeof(float)*ATOMIC_BLOCK_CNT*sizeof(float)*4);
  testAtomic<<<ATOMIC_BLOCK_CNT, ATOMIC_THREAD_CNT>>>(d_results,d_atomic_test);
  hipStreamSynchronize(0);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess)
    std::cout << "Error: " << hipGetErrorString(status) << std::endl;


	stop_ct1 = std::chrono::steady_clock::now();
	float atomic_time =
		std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
			.count();


  float val;
  status = hipMemcpy(&val, d_atomic_test, sizeof(float), hipMemcpyDeviceToHost);
  if (status != hipSuccess)
    std::cout << "Error memcpy d_atomic_test: " << hipGetErrorString(status) << std::endl;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "val: " << val << " kernel_time (hipEventElapsedTime) " << std::setw(5) << atomic_time << "ms" << std::endl;
  return 0;
}