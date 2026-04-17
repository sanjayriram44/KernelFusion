#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void add_relu_kernel(const float* x, const float* bias, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = x[idx] + bias[idx];
        out[idx] = fmaxf(0.0f, val);
    }
}

torch::Tensor fused_add_relu(torch::Tensor x, torch::Tensor bias) {
    auto out = torch::empty_like(x);
    int n = x.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        n
    );

    return out;
}
