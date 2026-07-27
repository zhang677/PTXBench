SERVICE_URL = "http://localhost:10000"

import csv
import math
import requests

GEMM_PREFIXES = ["gemm_n"]  # exclude kb_gemm_* fused ops
M_MIN = 64   # exclusive lower bound
M_MAX = 10000  # exclusive upper bound
NUM_WORKLOADS = 5


def list_gemm_definitions():
    resp = requests.get(f"{SERVICE_URL}/definitions")
    resp.raise_for_status()
    return [d["name"] for d in resp.json() if any(d["name"].startswith(p) for p in GEMM_PREFIXES)]


def get_workloads(definition_name):
    resp = requests.get(f"{SERVICE_URL}/definitions/{definition_name}/workloads")
    resp.raise_for_status()
    return resp.json()


def pick_diverse(workloads, n):
    """Pick n workloads with M values spread across the log-scale range."""
    if len(workloads) <= n:
        return workloads

    # Sort by M
    workloads = sorted(workloads, key=lambda w: w["axes"]["M"])
    ms = [w["axes"]["M"] for w in workloads]

    # Use log-spaced targets between min and max M
    log_min = math.log(ms[0])
    log_max = math.log(ms[-1])
    targets = [math.exp(log_min + i * (log_max - log_min) / (n - 1)) for i in range(n)]

    picked = []
    used = set()
    for t in targets:
        best_idx = None
        best_dist = float("inf")
        for i, m in enumerate(ms):
            if i in used:
                continue
            dist = abs(math.log(m) - math.log(t))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        used.add(best_idx)
        picked.append(workloads[best_idx])

    return picked


def main():
    definitions = sorted(list_gemm_definitions())
    print(f"Found {len(definitions)} GEMM definitions")

    rows = []
    for defn_name in definitions:
        workloads = get_workloads(defn_name)
        filtered = [w for w in workloads if M_MIN < w["axes"]["M"] < M_MAX]
        print(f"  {defn_name}: {len(workloads)} total, {len(filtered)} with {M_MIN}<M<{M_MAX}")

        if not filtered:
            print(f"    WARNING: no workloads in range, skipping")
            continue

        selected = pick_diverse(filtered, NUM_WORKLOADS)
        for w in selected:
            rows.append((defn_name, w["uuid"], w["axes"]["M"]))
            print(f"    M={w['axes']['M']:>6d}  uuid={w['uuid']}")

    out_path = "gemm_problems.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["definition_name", "workload_uuid", "M"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
