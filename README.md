HIP/ROCm AtomicAdd testing.

I'm getting poor performance with OneAPI icpx and AdaptiveCPP acpp compilers. In particular, the floating atomic performance is abysmal (over 8x slower than Nvidia). So, I wrote a quick test harness in hip to compare my AMD hardware with my Nvidia hardware.