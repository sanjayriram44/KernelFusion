#pragma once
#include <torch/extension.h>

// This prototype must match the signature in your .cu file exactly
torch::Tensor fused_add_gelu(torch::Tensor x, torch::Tensor bias);