#include <torch/extension.h>
#include "include/fused_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_add_gelu",        
        &fused_add_gelu,           
        "Fused Add and GELU activation (CUDA)" 
    );
}