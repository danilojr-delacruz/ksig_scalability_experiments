import ksig
import cupy as cp
import numpy as np
import gc
import pickle


OUTPUT_DIR = "results/projection_accuracy/"

EMBEDDING_ORDER = 1

N = 20
d = 5
M = 5
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

exact_kernel = lifted_kt(X, M)
K_exact = exact_kernel(X)

exponents = np.linspace(2, 4, num=2*10 + 1)
values = [int(10 ** x) for x in exponents][:-5]

q = len(values)
results = dict()

# Change these to have lower resolution
for i in range(q):
    for j in range(q):
        f = values[i]
        n = values[j]

        print(f, n)
        fm = rfsf_cs(X, M, f, n)
        K = fm(X)

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

        results[(f, n)] = (rmse, mape)

        print(rmse, mape)
        print()


        del K
        del fm
        gc.collect()

print(results)


with open(f"{OUTPUT_DIR}/ts.pkl", "wb") as f:
    pickle.dump(results, f)