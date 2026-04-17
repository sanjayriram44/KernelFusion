# KernelFusion

KernelFusion is an exploration and benchmarking project investigating the performance characteristics of custom CUDA kernels, Triton kernels, and native PyTorch operations. By fusing operations like element-wise addition and ReLU into a single kernel, we can significantly reduce memory bandwidth overhead and improve overall throughput on the GPU.

## Benchmarking Results

The following profiles were generated using NVIDIA Nsight Systems on our element-wise Add + ReLU operation. We analyze the runtime characteristics across OS usage, CUDA APIs, and most importantly, GPU kernel execution times.

### OS Runtime Summary (Top Contributors)

| Name | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) |
|------|----------|-----------------|-----------|----------|----------|
| poll | 26.2 | 13,197,441,696 | 137 | 96,331,691.2 | 100,143,992.0 |
| pthread_cond_timedwait | 23.8 | 12,001,997,845 | 8 | 1,500,249,730.6 | 49,102.0 |
| read | 18.7 | 9,441,746,245 | 9,735 | 969,876.3 | 3,548.0 |
| pthread_cond_wait | 12.0 | 6,055,828,730 | 3 | 2,018,609,576.7 | 521,822.0 |
| waitpid | 7.8 | 3,919,939,765 | 45 | 87,109,772.6 | 2,858.0 |
| sem_clockwait | 7.4 | 3,730,757,281 | 1 | 3,730,757,281.0 | 3,730,757,281.0 |
| clock_nanosleep | 2.5 | 1,267,948,357 | 30 | 42,264,945.2 | 50,093,399.0 |

### CUDA API Summary

| Name | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Med (ns) |
|------|----------|-----------------|-----------|----------|----------|
| cudaDeviceSynchronize | 71.9 | 380,655,204 | 9 | 42,295,022.7 | 12,538,104.0 |
| cudaLaunchKernel | 23.9 | 126,500,028 | 363 | 348,484.9 | 6,153.0 |
| cudaGetDeviceProperties_v12000 | 1.5 | 7,734,138 | 4 | 1,933,534.5 | 1,882,897.0 |
| cudaStreamSynchronize | 1.2 | 6,102,882 | 2 | 3,051,441.0 | 3,051,441.0 |

### CUDA GPU Kernel Summary (Performance Evaluation)

This section demonstrates where we calculate and display our performance improvements. By comparing the `Avg (ns)` and `Total Time (ns)` execution times in this table, you can see how the custom Add+ReLU implementations stack up against native PyTorch execution.

| Name | Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) |
|------|----------|-----------------|-----------|----------|----------|
| at::native::vectorized_elementwise_kernel<4, ...add...> | 21.7 | 92,764,215 | 113 | 820,922.3 | 820,792.0 |
| **add_relu_kernel** (Custom CUDA) | 21.3 | 91,050,255 | 111 | 820,272.6 | 820,344.0 |
| **add_relu_kernel dispatcher** | 20.1 | 86,281,565 | 111 | 777,311.4 | 777,400.0 |
| **triton_poi_fused_add_relu_0** (torch.compile) | 19.7 | 84,372,204 | 110 | 767,020.0 | 767,096.5 |
| at::native::vectorized_elementwise_kernel<4, ...clamp...> | 14.6 | 62,353,982 | 111 | 561,747.6 | 561,562.0 |

*Note: PyTorch native non-compiled operations execute two sequential kernels (one for add, followed by one for clamp/relu). This requires independent memory reads and writes to global memory for the intermediate tensors. Fused approaches like our custom CUDA and Triton kernels perform both steps in a single memory pass.*

### CUDA GPU MemOps Summary

| Operation | Time (%) | Total Time (ns) | Count | Avg (ns) | Med (ns) |
|-----------|----------|-----------------|-------|----------|----------|
| [CUDA memcpy Device-to-Host] | 70.7 | 4,320 | 2 | 2,160.0 | 2,160.0 |
| [CUDA memset] | 29.3 | 1,792 | 2 | 896.0 | 896.0 |

## Triton and torch.compile

The kernel summary table highlights a critical insight: the `triton_poi_fused_add_relu_0` kernel operates faster on average than the normal PyTorch native calls, and is highly competitive with our handwritten CUDA kernel. 

This occurs because `torch.compile` leverages OpenAI's Triton as its backend engine. When you invoke `torch.compile`, it dynamically analyzes the PyTorch computational graph ahead of time. Instead of executing the standard element-wise operation sequence—which writes an intermediate addition output to VRAM and then reads it back to run the ReLU—it writes dynamic Triton code that effectively merges the operations.

Triton generates a heavily optimized CUDA kernel designed entirely around block-level parallelism. It reads the inputs exactly once into fast SRAM (shared memory), calculates the addition, clamps values below zero for the ReLU, and only then writes the result out to global VRAM. Because it bypasses the massive global memory bandwidth bottleneck entirely, `torch.compile` generates kernels that dramatically outperform standard sequential execution and rival expertly written custom CUDA functions.

## Replicating the Results

Included in this repository is the Colab runner file. If you wish to replicate these benchmark profiles, evaluate the custom CUDA extensions, or examine the Python benchmarking logic firsthand, simply open the runner within Google Colab, connect to an NVIDIA GPU runtime, and run the notebook cells.
