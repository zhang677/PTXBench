import os
import requests
import json
import time

# Sync with the latest flashinfer-trace: python accrl/scripts/download_traces.py
# Launch the modal server with: modal serve modal_serve.py
# Run this PROFILE_BASE_URL="https://kunleslab--flashinfer-bench-serve-serve-dev.modal.run" python accrl/scripts/test_persistent_profile.py

PROFILE_BASE_URL = os.environ.get("PROFILE_BASE_URL", "http://localhost:10000")
# Submit
solution_paths = [
    "examples/solutions/triton_solution_example.json",
    "examples/solutions/cuda_solution_example.json",
    "examples/solutions/python_solution_example.json"
]

for solution_path in solution_paths:
    with open(solution_path) as f:
        json_body = json.load(f)

    definition_name = json_body["definition"]
    print(f"Solution: {solution_path} (definition: {definition_name})")

    # Fetch workloads for this definition
    wl_resp = requests.get(f"{PROFILE_BASE_URL}/definitions/{definition_name}/workloads")
    wl_resp.raise_for_status()
    workloads = wl_resp.json()
    print(f"  Found {len(workloads)} workloads")

    # Pick the last workload
    last_workload = workloads[-1]
    last_uuid = last_workload["uuid"]
    print(f"  Using last workload UUID: {last_uuid}")

    resp = requests.post(f"{PROFILE_BASE_URL}/evaluate", json={
        "solution": json_body,
        "workload_uuids": [last_uuid],
    })
    task_id = resp.json()["task_id"]

    # Poll with long-polling (wait up to 60s)
    while True:
        result = requests.get(f"{PROFILE_BASE_URL}/tasks/{task_id}?timeout=60")
        if result.json()["status"] == "completed":
            print(result.json())
            print("="*60)
            break
        time.sleep(30)
    