import os
import time
import gc
import tracemalloc
import threading

import cupy as cp
import ksig

# Ensure we use GPU-0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EMBEDDING_ORDER = 1
N = 10
d = 5

M = 5
n_components_rff = 100
n_components_proj = 100


################################################################################
def monitor_time(function, X, *args, **kwargs):
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    start_cpu = time.perf_counter()

    kernel = function(X, *args, **kwargs)
    K = kernel(X)

    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    t_cpu = end_cpu - start_cpu

    del kernel
    del K
    gc.collect()

    return t_gpu, t_cpu


def monitor_cpu_memory(function, X, *args, **kwargs):

    cp.get_default_memory_pool().free_all_blocks()
    tracemalloc.start()
    tracemalloc.reset_peak()

    kernel = function(X, *args, **kwargs)
    K = kernel(X)

    current, peak = tracemalloc.get_traced_memory()

    cpu_current = current
    cpu_peak = peak

    tracemalloc.stop()

    del kernel
    del K
    gc.collect()

    return cpu_current, cpu_peak


def gpu_mem_used(id):
    return int(cp.get_default_memory_pool().used_bytes())

def gpu_mem_used_no_cache(id):
    cp.get_default_memory_pool().free_all_blocks()
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
    id = cp.cuda.runtime.getDevice()

    while True:
        nvml_peak = max(gpu_mem_used(id), nvml_peak)
        if not peak_monitoring: break
        time.sleep(0.001) # 1msec


def monitor_gpu_memory(function, X, *args, **kwargs):
    global peak_monitoring, nvml_peak
    # Global variables
    peak_monitoring = False
    nvml_peak = 0

    id = cp.cuda.runtime.getDevice()
    nvml_before = gpu_mem_used_no_cache(id)

    peak_monitor_start()

    kernel = function(X, *args, **kwargs)
    K = kernel(X)

    # code finished
    peak_monitor_stop()
    nvml_after = gpu_mem_used_no_cache(id)

    gpu_current = nvml_after
    gpu_peak = nvml_peak-nvml_before

    del kernel
    del K
    gc.collect()

    return gpu_current, gpu_peak


def monitor_accuracy(function, X, *args, **kwargs):
    exact_kernel = lifted_kt(X, *args, **kwargs)
    K_exact = exact_kernel(X)

    kernel = function(X, *args, **kwargs)
    K = kernel(X)

    num_entries = N * (N-1) // 2

    error = cp.triu(K_exact - K, k=1)
    rmse = cp.sqrt((error ** 2).sum() / num_entries)

    ratio = cp.triu(K / K_exact - 1)
    mape = cp.abs(ratio).sum() / num_entries

    del kernel
    del K

    del exact_kernel
    del K_exact

    gc.collect()

    return rmse, mape


def monitor_accuracy_by_level(function, X, *args, **kwargs):

    N = 20

    exact_kernel = lifted_kt(X, *args, **kwargs)
    exact_levels = exact_kernel._compute_kernel(X)
    diag = cp.diagonal(exact_levels, axis1=1, axis2=2)
    normalised_levels = exact_levels / cp.sqrt(
        diag[:, :, None] * diag[:, None, :])

    kernel = function(X, *args, **kwargs)
    K = kernel(X)

    num_entries = N * (N-1) // 2

    error = cp.triu(K_exact - K, k=1)
    rmse = cp.sqrt((error ** 2).sum() / num_entries)

    ratio = cp.triu(K / K_exact - 1)
    mape = cp.abs(ratio).sum() / num_entries

    del kernel
    del K

    del exact_kernel
    del K_exact

    gc.collect()

    return rmse, mape

################################################################################
def rfsf_trp(X, n_levels, n_components_rff, n_components_proj, **kwargs):
    static_features = ksig.static.features.RandomFourierFeatures(n_components=n_components_rff)
    projection = ksig.projections.TensorizedRandomProjection(n_components=n_components_proj)

    FM = ksig.kernels.SignatureFeatures(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_features=static_features,
        projection=projection,
    )
    FM.fit(X)

    return FM


def rfsf_dp2(X, n_levels, n_components_rff, n_components_proj, **kwargs):
    static_features = ksig.static.features.RandomFourierFeatures(n_components=n_components_rff)
    projection = ksig.projections.DiagonalProjection(internal_size=2)

    FM = ksig.kernels.SignatureFeatures(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_features=static_features,
        projection=projection,
    )
    FM.fit(X)

    return FM


def rfsf_dp1(X, n_levels, n_components_rff, n_components_proj, **kwargs):
    static_features = ksig.static.features.RandomFourierFeatures1D(n_components=n_components_rff)
    projection = ksig.projections.DiagonalProjection(internal_size=1)

    FM = ksig.kernels.SignatureFeatures(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_features=static_features,
        projection=projection,
    )
    FM.fit(X)

    return FM


def rfsf_cs(X, n_levels, n_components_rff, n_components_proj, **kwargs):
    static_features = ksig.static.features.RandomFourierFeatures(n_components=n_components_rff)
    projection = ksig.projections.CountSketchRandomProjection(n_components=n_components_proj)

    FM = ksig.kernels.SignatureFeatures(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_features=static_features,
        projection=projection,
    )
    FM.fit(X)

    return FM

def rfsf_vsp(X, n_levels, n_components_rff, n_components_proj, **kwargs):
    static_features = ksig.static.features.RandomFourierFeatures(n_components=n_components_rff)
    projection = ksig.projections.VerySparseRandomProjection(n_components=n_components_proj)

    FM = ksig.kernels.SignatureFeatures(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_features=static_features,
        projection=projection,
    )
    FM.fit(X)

    return FM

def lifted_kt(X, n_levels, **kwargs):
    static_kernel = ksig.static.kernels.RBFKernel()

    Ksig = ksig.kernels.SignatureKernel(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_kernel=static_kernel
    )

    return Ksig


def lifted_pde(X, **kwargs):
    static_kernel = ksig.static.kernels.RBFKernel()

    Ksig = ksig.kernels.SignaturePDEKernel(
        static_kernel=static_kernel
    )

    return Ksig
################################################################################