import os
import time
import gc
import tracemalloc
import threading

import jax
from sigkerax.sigkernel import SigKernel

# Ensure we use GPU-0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N = 10
d = 5
M = 5

n_components_rff = 100
n_components_proj = 100


################################################################################
def monitor_time(function, X, *args, **kwargs):
    # Just use CPU time, they are the same
    start_cpu = time.perf_counter()

    kernel = function(X, *args, **kwargs)
    K = kernel.kernel_matrix(X, X)
    _ = jax.block_until_ready(K)

    end_cpu = time.perf_counter()
    t_cpu = end_cpu - start_cpu

    del kernel
    del K
    gc.collect()

    return t_cpu, t_cpu


def monitor_cpu_memory(function, X, *args, **kwargs):
    tracemalloc.start()
    tracemalloc.reset_peak()

    kernel = function(X, *args, **kwargs)
    K = kernel.kernel_matrix(X, X)
    _ = jax.block_until_ready(K)

    current, peak = tracemalloc.get_traced_memory()

    cpu_current = current
    cpu_peak = peak

    tracemalloc.stop()

    del kernel
    del K
    gc.collect()

    return cpu_current, cpu_peak


def gpu_mem_used(id):
    return int(jax.local_devices()[0].memory_stats()["peak_bytes_in_use"])

def gpu_mem_used_no_cache(id):
    # Need to do jax.block_until_ready?
    return gpu_mem_used(id)

def peak_monitor_start():
    global peak_monitoring
    peak_monitoring = True

    # this thread samples RAM usage as long as the current epoch of the fit loop is running
    peak_monitor_thread = threading.Thread(target=peak_monitor_func)
    peak_monitor_thread.daemon = True
    peak_monitor_thread.start()

def peak_monitor_stop():
    global peak_monitoring
    peak_monitoring = False

def peak_monitor_func():
    global nvml_peak, peak_monitoring
    nvml_peak = 0
    id = 0

    while True:
        nvml_peak = max(gpu_mem_used(id), nvml_peak)
        if not peak_monitoring: break
        time.sleep(0.001) # 1msec


def monitor_gpu_memory(function, X, *args, **kwargs):
    global peak_monitoring, nvml_peak
    # Global variables
    peak_monitoring = False
    nvml_peak = 0

    id = 0
    nvml_before = gpu_mem_used_no_cache(id)

    peak_monitor_start()

    kernel = function(X, *args, **kwargs)
    K = kernel.kernel_matrix(X, X)
    jax.block_until_ready(K)

    # code finished
    peak_monitor_stop()
    nvml_after = gpu_mem_used_no_cache(id)

    gpu_current = nvml_after
    gpu_peak = nvml_peak-nvml_before

    del kernel
    del K
    gc.collect()

    return gpu_current, gpu_peak

################################################################################
def sigkerax(X, **kwargs):
    signature_kernel = SigKernel(static_kernel_kind="rbf")
    return signature_kernel
################################################################################