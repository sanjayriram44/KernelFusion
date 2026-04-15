#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// High-performance constant for sqrt(2/pi)
#define SQRT_2_OVER_PI 0.79788456f

__global__ void add_gelu_kernel(const float* x, const float* bias, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        
        float val = x[idx] + bias[idx];
        
        
        float cube = 0.044715f * val * val * val;
        float inner = SQRT_2_OVER_PI * (val + cube);
        out[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

torch::Tensor fused_add_gelu(torch::Tensor x, torch::Tensor bias) {
    auto out = torch::empty_like(x);
    int n = x.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        n
    );

    return out;
}