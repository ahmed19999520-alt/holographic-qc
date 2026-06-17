import subprocess
import sys
from pathlib import Path


notebooks = [
    "notebooks/01_virasoro_algebra.py",
    "notebooks/02_holographic_protection.py",
    "notebooks/03_shor_algorithm.py",
    "notebooks/04_grover_algorithm.py",
    "notebooks/05_materials_benchmarks.py",
]


def main():
    print("Running all notebooks...")
    all_passed = True
    for nb in notebooks:
        path = Path(nb)
        if not path.exists():
            print(f"  SKIP {nb} (not found)")
            continue
        print(f"  Running {nb}...")
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print(f"  PASS {nb}")
        else:
            print(f"  FAIL {nb}")
            print(result.stderr[-500:] if result.stderr else "")
            all_passed = False
    print("\n" + ("All notebooks passed." if all_passed else "Some notebooks failed."))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()