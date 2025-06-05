import cupy as cp
import numpy as np
import time
import gc
import pickle
import subprocess

import ksig

from math import ceil

EMBEDDING_ORDER = 1
OUTPUT_DIR = "results/truncation_time/"
NUM_RUNS = 11

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
# Generate X, Y
cp.random.seed(123)
X = cp.random.randn(N, L, d)
X = cp.cumsum(X, axis=1) / cp.sqrt(L)

function = {name}

M = {M}
n = {n}

cpu_time, gpu_time = monitor_time(function, X, n_levels=M,
                n_components_rff={n},
                n_components_proj={n}
            )

print(cpu_time, gpu_time)
"""


with open("base_script.py", "r") as f:
    base_script = f.read()

for M in range(1, 10):
    print(M)
    for name in functions_to_call:

        function_call_code = get_function_call_code(M, name)
        script = base_script + "\n" + function_call_code
        print(name)
        results[("cpu", name, M)] = []
        results[("gpu", name, M)] = []

        for i in range(NUM_RUNS):
            if blacklisted[name]:
                print(i, "blacklisted")
                results[("cpu", name, M)].append(None)
                results[("gpu", name, M)].append(None)
                continue

            try:
                result = subprocess.run(['python3', '-c', script],
                                        capture_output=True, text=True,
                                        timeout=MAX_RUN_TIME)
                if result.returncode == 0:
                    cpu_time, gpu_time = map(
                    float, result.stdout.strip().split(" ")
                )
                    print(i, "ran in", cpu_time, gpu_time)
                    results[("cpu", name, M)].append(cpu_time)
                    results[("gpu", name, M)].append(gpu_time)
                else:
                    print(i, "Script failed")
                    # Don't try it for larger cases
                    blacklisted[name] = True
                    results[("cpu", name, M)].append(None)
                    results[("gpu", name, M)].append(None)
            except subprocess.TimeoutExpired:
                print(i, "Timeout")
                blacklisted[name] = True
                results[("cpu", name, M)].append("Timeout")
                results[("gpu", name, M)].append("Timeout")
        print()

    print()

print(results)

with open(f"{OUTPUT_DIR}/brownian.pkl", "wb") as f:
    pickle.dump(results, f)