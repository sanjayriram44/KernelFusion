import torch
import torch.nn.functional as F
import time


if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    backend = "cuda"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple M3 Pro (MPS)"
    backend = "mps"
else:
    device = torch.device("cpu")
    device_name = "CPU"
    backend = "cpu"

print(f"🚀 Running on: {device_name}")


N = 1024 * 1024 * 16 
x = torch.randn(N, device=device, dtype=torch.float32)
bias = torch.randn(N, device=device, dtype=torch.float32)

def eager_add_gelu(x, bias):
    return F.gelu(x + bias)

@torch.compile
def compiled_add_gelu(x, bias):
    return F.gelu(x + bias)


def benchmark(func, x, bias, label):
    print(f"--- Benchmarking {label} ---")
    
    # Warmup
    for _ in range(10):
        func(x, bias)
    
    if backend == "cuda":

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(100):
            _ = func(x, bias)
        end_event.record()
        torch.cuda.synchronize()
        avg_ms = start_event.elapsed_time(end_event) / 100
    else:

        if backend == "mps":
            torch.mps.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(100):
            _ = func(x, bias)
        
        if backend == "mps":
            torch.mps.synchronize()
        
        avg_ms = (time.perf_counter() - start_time) * 1000 / 100

    print(f"Average Time: {avg_ms:.4f} ms\n")
    return avg_ms


eager_ms = benchmark(eager_add_gelu, x, bias, "Eager Mode")

compiled_ms = benchmark(compiled_add_gelu, x, bias, "Compiled Mode")