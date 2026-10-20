# driver.py
import subprocess
import sys

configs = [
    ["--algorithm", "quicksort", "--n", "1000"],
    ["--algorithm", "quicksort", "--n", "100000"],
    ["--algorithm", "mergesort", "--n", "1000"],
    ["--algorithm", "mergesort", "--n", "100000"],
]

with open("results.txt", "a") as log:
    for args in configs:
        result = subprocess.run(
            [sys.executable, "test_algo.py", *args],
            capture_output=True, text=True
        )
        log.write(f"{' '.join(args)}: {result.stdout.strip()}\n")
        log.flush()