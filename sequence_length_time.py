import cupy as cp
import numpy as np
import time
import gc
import pickle
import subprocess

import ksig

EMBEDDING_ORDER = 1
OUTPUT_DIR = "results/sequence_length_time/"
NUM_RUNS = 11

# In seconds, 10 minutes
MAX_RUN_TIME = 600


###############################################################################
functions_to_call = [
    "rfsf_trp", "rfsf_dp2", "rfsf_dp1",
    "rfsf_cs", "rfsf_vsp",
    "lifted_kt", "lifted_pde",
]

blacklisted = dict()
for function in functions_to_call:
    blacklisted[function] = False

results = dict()

def get_function_call_code(L, name):
    return f"""L = {L}
# Generate X, Y
cp.random.seed(123)
X = cp.random.randn(N, L, d)
X = cp.cumsum(X, axis=1) / cp.sqrt(L)

function = {name}

cpu_time, gpu_time = monitor_time(function, X, n_levels=M,
                n_components_rff=n_components_rff,
                n_components_proj=n_components_proj
            )

print(cpu_time, gpu_time)
"""


with open("base_script.py", "r") as f:
    base_script = f.read()

for x in np.linspace(2, 6, num=4*5 + 1):
    L = int(10 ** x)
    print(L)
    for name in functions_to_call:

        function_call_code = get_function_call_code(L, name)
        script = base_script + "\n" + function_call_code
        print(name)
        results[("cpu", name, L)] = []
        results[("gpu", name, L)] = []

        for i in range(NUM_RUNS):
            if blacklisted[name]:
                print(i, "blacklisted")
                results[("cpu", name, L)].append(None)
                results[("gpu", name, L)].append(None)
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
                    results[("cpu", name, L)].append(cpu_time)
                    results[("gpu", name, L)].append(gpu_time)
                else:
                    print(i, "Script failed")
                    # Don't try it for larger cases
                    blacklisted[name] = True
                    results[("cpu", name, L)].append(None)
                    results[("gpu", name, L)].append(None)
            except subprocess.TimeoutExpired:
                print(i, "Timeout")
                blacklisted[name] = True
                results[("cpu", name, L)].append("Timeout")
                results[("gpu", name, L)].append("Timeout")
        print()

    print()

print(results)

with open(f"{OUTPUT_DIR}/brownian.pkl", "wb") as f:
    pickle.dump(results, f)