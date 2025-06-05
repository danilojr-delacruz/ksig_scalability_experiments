import numpy as np
import time
import gc
import pickle
import subprocess

import ksig
import cupy as cp

OUTPUT_DIR = "results/sequence_length_memory/"

# In seconds
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

cpu_current, cpu_peak = monitor_cpu_memory(function, X, n_levels=M,
                n_components_rff=n_components_rff,
                n_components_proj=n_components_proj
            )
gpu_current, gpu_peak = monitor_gpu_memory(function, X, n_levels=M,
                n_components_rff=n_components_rff,
                n_components_proj=n_components_proj
            )

print(cpu_current, cpu_peak, gpu_current, gpu_peak)
"""


with open("truncated_signature/base_script.py", "r") as f:
    base_script = f.read()

for x in np.linspace(2, 6, num=4*5 + 1):
    L = int(10 ** x)
    print(L)
    for name in functions_to_call:

        function_call_code = get_function_call_code(L, name)
        script = base_script + "\n" + function_call_code
        print(name)

        results[(name, L)] = None
        times = []
        if blacklisted[name]:
            print("blacklisted")
            results[(name, L)] = None
            continue

        try:
            result = subprocess.run(['python3', '-c', script],
                                    capture_output=True, text=True,
                                    timeout=MAX_RUN_TIME)
            if result.returncode == 0:
                cpu_current, cpu_peak, gpu_current, gpu_peak = map(
                    int, result.stdout.strip().split(" ")
                )
                results[(name, L)] = (cpu_current, cpu_peak, gpu_current, gpu_peak)
                print(cpu_current, cpu_peak, gpu_current, gpu_peak)
            else:
                print("Script failed")
                # Don't try it for larger cases
                blacklisted[name] = True
                results[(name, L)] = None
        except subprocess.TimeoutExpired:
            print("Timeout")
            blacklisted[name] = True
            results[(name, L)] = "Timeout"
        print()

    print()

print(results)

with open(f"{OUTPUT_DIR}/brownian_pM.pkl", "wb") as f:
    pickle.dump(results, f)