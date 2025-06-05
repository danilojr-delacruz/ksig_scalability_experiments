import subprocess

print("Running Sigkerax Experiments")


for file in [
    "sigkerax/sequence_length_memory",
    "sigkerax/sequence_length_time"
]:
    try:
        print(file)
        subprocess.run(["python3", f"{file}.py"])
    except Exception as e:
        print(e)
    print()