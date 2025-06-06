import numpy as np
import gc
import pickle

import ksig
import cupy as cp
from math import ceil


OUTPUT_DIR = "results/truncation_feature_accuracy/"

EMBEDDING_ORDER = 1

N = 20
d = 5
L = 100

###############################################################################
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

def lifted_kt(X, n_levels, **kwargs):
    static_kernel = ksig.static.kernels.RBFKernel()

    Ksig = ksig.kernels.SignatureKernel(
        n_levels=n_levels, order=EMBEDDING_ORDER,
        static_kernel=static_kernel
    )

    return Ksig


###############################################################################

# Generate X, Y
X = cp.random.randn(N, L, d)
X = cp.cumsum(X, axis=1) / cp.sqrt(L)

exponents = np.linspace(2, 4, num=2*10 + 1)
D_values = [int(10 ** x) for x in exponents][:-6]
M_values = [m for m in range(1, 10+1)]


### TRP
print("TRP")
results = dict()

# Change these to have lower resolution
for i in range(len(D_values)):
    for j in range(len(M_values)):
        f = D_values[i]
        n = f

        M = M_values[j]

        print(f, M)
        fm = rfsf_trp(X, M, f, n)
        K = fm(X)

        exact_kernel = lifted_kt(X, M)
        K_exact = exact_kernel(X)

        # Only take upper triangular to avoid duplicates
        # Ignore diagonal which should just be 0
        num_entries = N * (N-1) // 2

        error = cp.triu(K_exact - K, k=1)
        rmse = cp.sqrt((error ** 2).sum() / num_entries)

        ratio = cp.triu(K / K_exact - 1)
        mape = cp.abs(ratio).sum() / num_entries

        # Ensure these are floats
        rmse = rmse.item()
        mape = mape.item()

        F = f*M + 1
        results[(F, M)] = (rmse, mape)

        print(rmse, mape)
        print()


        del K
        del fm
        gc.collect()

print(results)
print()


with open(f"{OUTPUT_DIR}/trp.pkl", "wb") as f:
    pickle.dump(results, f)


### CS
print("TS")
results = dict()

# Change these to have lower resolution
for i in range(len(D_values)):
    for j in range(len(M_values)):
        f = D_values[i]
        n = f

        M = M_values[j]

        print(f, M)
        fm = rfsf_cs(X, M, f, n)
        K = fm(X)

        exact_kernel = lifted_kt(X, M)
        K_exact = exact_kernel(X)

        # Only take upper triangular to avoid duplicates
        # Ignore diagonal which should just be 0
        num_entries = N * (N-1) // 2

        error = cp.triu(K_exact - K, k=1)
        rmse = cp.sqrt((error ** 2).sum() / num_entries)

        ratio = cp.triu(K / K_exact - 1)
        mape = cp.abs(ratio).sum() / num_entries

        # Ensure these are floats
        rmse = rmse.item()
        mape = mape.item()

        F = f*M + 1
        results[(F, M)] = (rmse, mape)

        print(rmse, mape)
        print()


        del K
        del fm
        gc.collect()

print(results)
print()


with open(f"{OUTPUT_DIR}/ts.pkl", "wb") as f:
    pickle.dump(results, f)


### DP1
print("DP1")
results = dict()

# Change these to have lower resolution
for i in range(len(D_values)):
    for j in range(len(M_values)):
        f = D_values[i]
        n = f

        M = M_values[j]

        print(f, M)
        fm = rfsf_dp1(X, M, f, n)
        K = fm(X)

        exact_kernel = lifted_kt(X, M)
        K_exact = exact_kernel(X)

        # Only take upper triangular to avoid duplicates
        # Ignore diagonal which should just be 0
        num_entries = N * (N-1) // 2

        error = cp.triu(K_exact - K, k=1)
        rmse = cp.sqrt((error ** 2).sum() / num_entries)

        ratio = cp.triu(K / K_exact - 1)
        mape = cp.abs(ratio).sum() / num_entries

        # Ensure these are floats
        rmse = rmse.item()
        mape = mape.item()

        F = f*M + 1
        results[(F, M)] = (rmse, mape)

        print(rmse, mape)
        print()


        del K
        del fm
        gc.collect()

print(results)


with open(f"{OUTPUT_DIR}/dp.pkl", "wb") as f:
    pickle.dump(results, f)


### DP2
results = dict()

# Change these to have lower resolution
for i in range(len(D_values)):
    for j in range(len(M_values)):
        M = M_values[j]
        f = ceil(D_values[i] * M / (2**(M+1) - 1))
        n = f

        print(f, M)
        fm = rfsf_dp2(X, M, f, n)
        K = fm(X)

        exact_kernel = lifted_kt(X, M)
        K_exact = exact_kernel(X)

        # Only take upper triangular to avoid duplicates
        # Ignore diagonal which should just be 0
        num_entries = N * (N-1) // 2

        error = cp.triu(K_exact - K, k=1)
        rmse = cp.sqrt((error ** 2).sum() / num_entries)

        ratio = cp.triu(K / K_exact - 1)
        mape = cp.abs(ratio).sum() / num_entries

        # Ensure these are floats
        rmse = rmse.item()
        mape = mape.item()

        F = f*M + 1
        results[(F, M)] = (rmse, mape)

        print(rmse, mape)
        print()


        del K
        del fm
        gc.collect()

print(results)


with open(f"{OUTPUT_DIR}/dp2.pkl", "wb") as f:
    pickle.dump(results, f)