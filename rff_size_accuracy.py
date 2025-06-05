import cupy as cp
import numpy as np
import time
import gc
import pickle
import subprocess

import ksig

EMBEDDING_ORDER = 1
OUTPUT_DIR = "results/rff_size_accuracy/"

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

def get_function_call_code(n, name):
    return f"""L = 100
N = 20
# Generate X, Y
cp.random.seed(123)
X = cp.random.randn(N, L, d)
X = cp.cumsum(X, axis=1) / cp.sqrt(L)

function = {name}

rmse, mape = monitor_accuracy(function, X, n_levels=M,
                n_components_rff={n},
                n_components_proj={n}
            )

print(rmse, mape)
"""


with open("base_script.py", "r") as f:
    base_script = f.read()

x_values = np.concatenate([
    np.array([1]),
    np.linspace(2, 6, num=4*5 + 1)
])

for x in x_values:
    n = int(10 ** x)
    print(n)
    for name in functions_to_call:

        function_call_code = get_function_call_code(n, name)
        script = base_script + "\n" + function_call_code
        print(name)

        results[(name, n)] = None
        times = []
        if blacklisted[name]:
            print("blacklisted")
            results[(name, n)] = None
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
                results[(name, n)] = (rmse, mape)
                print(rmse, mape)
            else:
                print("Script failed")
                # Don't try it for larger cases
                blacklisted[name] = True
                results[(name, n)] = None
        except subprocess.TimeoutExpired:
            print("Timeout")
            blacklisted[name] = True
            results[(name, n)] = "Timeout"
        print()

    print()

print(results)

with open(f"{OUTPUT_DIR}/brownian.pkl", "wb") as f:
    pickle.dump(results, f)