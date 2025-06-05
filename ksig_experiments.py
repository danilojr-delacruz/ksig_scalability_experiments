import subprocess

print("Running KSig Experiments")


for file in [
    "sequence_length_memory",
    "sequence_length_time",
    "rff_size_memory",
    "rff_size_time",
    "rff_size_accuracy",
    "truncation_memory",
    "truncation_time",
    "truncation_accuracy",
    "projection_accuracy_trp",
    "projection_accuracy_ts",
    "truncation_feature_accuracy",
    "truncated_signature/sequence_length_memory",
    "truncated_signature/sequence_length_time"
]:
    try:
        print(file)
        subprocess.run(["python3", f"{file}.py"])
    except Exception as e:
        print(e)
    print()