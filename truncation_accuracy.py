import cupy as cp
import numpy as np
import time
import gc
import pickle
import subprocess

import ksig

from math import ceil

EMBEDDING_ORDER = 1
OUTPUT_DIR = "results/truncation_accuracy/"

# In seconds, 10 minutes
MAX_RUN_TIME = 600


###############################################################################
functions_to_call = [
    "rfsf_trp", "rfsf_dp2", "rfsf_dp1",
    "rfsf_cs", "rfsf_vsp"
]

blacklisted = dict()
for function in functions_to_call:
    blacklisted[function] = False

results = dict()

def get_function_call_code(M, name):
    n = 1000
    if name == "rfsf_dp2":
        n = ceil(1000 * M / (2**(M+1) - 1))
    return f"""L = 100
N=100
# Generate X, Y
cp.random.seed(123)
X = cp.random.randn(N, L, d)
X = cp.cumsum(X, axis=1) / cp.sqrt(L)

function = {name}

M = {M}
n = {n}

rmse, mape = monitor_accuracy(function, X, n_levels=M,
                n_components_rff={n},
                n_components_proj={n}
            )

print(rmse, mape)
"""


with open("base_script.py", "r") as f:
    base_script = f.read()

for M in range(1, 10):
    print(M)
    for name in functions_to_call:

        function_call_code = get_function_call_code(M, name)
        script = base_script + "\n" + function_call_code
        print(name)

        results[(name, M)] = None
        times = []
        if blacklisted[name]:
            print("blacklisted")
            results[(name, M)] = None
            continue

        try:
            result = subprocess.run(['python3', '-c', script],
                                    capture_output=True, text=True,
                                    timeout=MAX_RUN_TIME)
            # print(result)
            # exit()
            if result.returncode == 0:
                rmse, mape = map(
                    float, result.stdout.strip().split(" ")
                )
                results[(name, M)] = (rmse, mape)
                print(rmse, mape)
            else:
                print("Script failed")
                # Don't try it for larger cases
                blacklisted[name] = True
                results[(name, M)] = None
        except subprocess.TimeoutExpired:
            print("Timeout")
            blacklisted[name] = True
            results[(name, M)] = "Timeout"
        print()

    print()

print(results)

with open(f"{OUTPUT_DIR}/brownian.pkl", "wb") as f:
    pickle.dump(results, f)