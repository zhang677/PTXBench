import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MULTITURN_DIR = SCRIPT_DIR.parent
TEMPLATE_PATH = MULTITURN_DIR / "template_compile_measure_cuda.txt"
CSV_PATH = SCRIPT_DIR / "problems.csv"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def write_test_files(rows: list[dict[str, str]]) -> int:
    compile_script_content = TEMPLATE_PATH.read_text()
    generated = 0

    for row in rows:
        definition_name = row["definition"]
        workload_uuid = row["workload_uuid"]
        out_path = SCRIPT_DIR / f"{definition_name}_{workload_uuid}.py"
        out_path.write_text(
            compile_script_content.replace("<definition_name>", definition_name).replace(
                "<workload_uuid>", workload_uuid
            )
        )
        generated += 1

    return generated


def main() -> None:
    rows = load_rows()
    generated = write_test_files(rows)
    print(f"Wrote {generated} test files under {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
