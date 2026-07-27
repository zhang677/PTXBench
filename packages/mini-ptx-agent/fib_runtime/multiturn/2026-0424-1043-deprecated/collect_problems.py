SERVICE_URL = "http://localhost:10000"

import csv
import requests

MHA_NAMES = ["mha_h48_d128", "mha_h48_d128_causal"]


def list_mha_definitions():
    resp = requests.get(f"{SERVICE_URL}/definitions")
    resp.raise_for_status()
    available = {d["name"] for d in resp.json()}
    return [n for n in MHA_NAMES if n in available]


def get_workloads(definition_name):
    resp = requests.get(f"{SERVICE_URL}/definitions/{definition_name}/workloads")
    resp.raise_for_status()
    return resp.json()


def main():
    definitions = list_mha_definitions()
    print(f"Found {len(definitions)} MHA definitions")

    rows = []
    for defn_name in definitions:
        workloads = get_workloads(defn_name)
        workloads = sorted(workloads, key=lambda w: w["axes"]["S"])
        print(f"  {defn_name}: {len(workloads)} workloads")

        for w in workloads:
            rows.append((defn_name, w["uuid"], w["axes"]["S"]))
            print(f"    S={w['axes']['S']:>4d}  uuid={w['uuid']}")

    out_path = "mha_problems.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["definition_name", "workload_uuid", "S"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
