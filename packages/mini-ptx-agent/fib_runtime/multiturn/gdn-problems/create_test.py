import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR.parent / "template_compile_measure_cuda.txt"
CSV_PATH = SCRIPT_DIR / "gdn_problems.csv"

with open(TEMPLATE_PATH) as f:
    COMPILE_SCRIPT_CONTENT = f.read()

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        definition_name = row["definition_name"]
        workload_uuid = row["workload_uuid"]
        out_path = SCRIPT_DIR / f"{definition_name}_{workload_uuid}.py"
        with open(out_path, "w") as out:
            out.write(COMPILE_SCRIPT_CONTENT.replace("<definition_name>", definition_name).replace("<workload_uuid>", workload_uuid))
