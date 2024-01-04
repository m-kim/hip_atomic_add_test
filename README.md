HIP/ROCm AtomicAdd testing.

I'm getting poor performance with OneAPI icpx and AdaptiveCPP acpp compilers. In particular, the floating atomic performance is abysmal (over 8x slower than Nvidia). So, I wrote a quick test harness in hip to compare my AMD hardware with my Nvidia hardware.

compile CUDA:
nvcc main.cu

compile HIP:
hipcc hip_main.cpp

compile SYCL:
icx-cl /EHa -fsycl -I 'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include\sycl\' .\sycl_main.cpp

compile Vulkan:
cl.exe /std:c++20 .\vlk_main.cpp  /I C:\VulkanSDK\1.3.268.0\Include\ /link /LIBPATH:"C:\VulkanSDK\1.3.268.0\Lib\" "vulkan-1.lib"
glslc.exe .\shader.comp -o shader.spv