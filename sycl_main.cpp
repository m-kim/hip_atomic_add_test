#include <sycl/sycl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <iomanip>
#include <iostream>
#define ATOMIC_THREAD_CNT 1024
#define ATOMIC_BLOCK_CNT 65536


void atomicAdd(float *out, float val){sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> aref(out[0]); aref.fetch_add(val);}
void testAtomic(const float *in, float *out, sycl::nd_item<1> &item_ct1)
{
  auto id = item_ct1.get_global_id(0);

  for (int c=0; c<3; c++){
      atomicAdd(&(out[item_ct1.get_group(0) * 4 + c]), in[id * 4+c] * in[id*4+3]);

  }
  atomicAdd(&(out[item_ct1.get_group(0) * 4 + 3]), in[id*4+3]);
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
  sycl::queue q(sycl::gpu_selector_v);
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl; 

  std::vector<float> h_results(ATOMIC_THREAD_CNT * ATOMIC_BLOCK_CNT*4, 0.0f);
  sycl::buffer<float, 1> d_results(h_results);
  
  
  // Submit a kernel to generate on device
  q.submit([&](sycl::handler& cgh) {
      auto results_acc = d_results.template get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for<>(sycl::range<1>(d_results.size()), [=](sycl::item<1> item) {
          // Create an engine object
          oneapi::mkl::rng::device::philox4x32x10<> engine(123456, item.get_id(0));
          // Create a distribution object
          oneapi::mkl::rng::device::uniform<> distr;
          // Call generate function to obtain scalar random number
          float res = oneapi::mkl::rng::device::generate(distr, engine);


          results_acc[item.get_id(0)] = res;
      });
  }).wait();

  // Output number 12345
  ::std::cout << "end is " << h_results[h_results.size()-1] << std::endl;
  std::cout << "red: " << h_results[0] << " alpha: " << h_results[3] << std::endl;

//   float *d_atomic_test;
  
//   hipMalloc(&d_atomic_test, sizeof(float)*ATOMIC_BLOCK_CNT*4);
  std::vector<float> h_atomic_test(ATOMIC_BLOCK_CNT*4);
  sycl::buffer<float,1> d_atomic_test(h_atomic_test);

  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
	std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

	start_ct1 = std::chrono::steady_clock::now();

//   hipMemset(d_atomic_test, 0, sizeof(float)*ATOMIC_BLOCK_CNT*4);
  try{
    q.submit([&](sycl::handler& cgh) {
        auto results_acc = d_results.template get_access<sycl::access::mode::read>(cgh);
        auto atomic_acc = d_atomic_test.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<>(sycl::nd_range<1>(sycl::range<1>(ATOMIC_BLOCK_CNT),sycl::range<1>(ATOMIC_THREAD_CNT)), [=](sycl::nd_item<1> item_ct1) {

          testAtomic(results_acc.get_multi_ptr<sycl::access::decorated::yes>().get(),atomic_acc.get_multi_ptr<sycl::access::decorated::yes>().get(), item_ct1);
        });
    }).wait();
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }



	stop_ct1 = std::chrono::steady_clock::now();
	float atomic_time =
		std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
			.count();


std::vector<float> validate_atomic(ATOMIC_BLOCK_CNT*4, 0);
  validate(h_atomic_test, validate_atomic);
  int cnt = 0;
  for (int i=0; i<validate_atomic.size(); i++){
    if (std::abs(h_atomic_test[i] - validate_atomic[i]) > 1e-3){
      if (cnt < 5)
       std::cout << i << " " << h_atomic_test[i] << " " << validate_atomic[i] << " " << std::abs(h_atomic_test[i] - validate_atomic[i]) << std::endl;
      cnt++;
    }
  }
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "val: " << h_atomic_test[0] << " kernel_time (hipEventElapsedTime) " << std::setw(5) << atomic_time << "ms" << std::endl;
  std::cout << "Number of invalid results: " << cnt << std::endl;
  return 0;
}