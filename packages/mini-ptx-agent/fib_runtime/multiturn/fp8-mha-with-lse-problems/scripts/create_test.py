import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SUITE_DIR = SCRIPT_DIR.parent
TEMPLATE_PATH = SUITE_DIR.parent / "template_compile_measure_cuda.txt"
CSV_PATH = SUITE_DIR / "problems.csv"

with open(TEMPLATE_PATH) as f:
    COMPILE_SCRIPT_CONTENT = f.read()

generated = 0
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        definition_name = row["definition_name"]
        workload_uuid = row["workload_uuid"]
        out_path = SUITE_DIR / f"{definition_name}_{workload_uuid}.py"
        with open(out_path, "w") as out:
            out.write(
                COMPILE_SCRIPT_CONTENT.replace("<definition_name>", definition_name).replace(
                    "<workload_uuid>", workload_uuid
                )
            )
        generated += 1

if generated != 8:
    raise SystemExit(f"expected to generate 8 tests, generated {generated}")

print(f"Generated {generated} tests in {SUITE_DIR}")
