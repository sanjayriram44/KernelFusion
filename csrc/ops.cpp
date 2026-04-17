#include <torch/extension.h>
#include "include/fused_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_add_relu",        
        &fused_add_relu,           
        "Fused Add and ReLU activation (CUDA)" 
    );
}