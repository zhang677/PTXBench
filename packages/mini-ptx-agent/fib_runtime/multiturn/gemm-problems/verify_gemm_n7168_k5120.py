"""Verify that gemm_n7168_k5120 / workload 94920358 computes C = A @ B^T with bfloat16.

Fetches the definition and workload from the profiling service at localhost:10000,
runs the reference, and checks the computation matches torch.matmul(A, B.T).
"""

import requests
import torch

BASE_URL = "http://localhost:10000"
DEFINITION_NAME = "gemm_n7168_k5120"
WORKLOAD_UUID = "94920358-01a8-4c5b-9209-3103fd490e94"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def fetch_definition(name: str) -> dict:
    resp = requests.get(f"{BASE_URL}/definitions/{name}")
    resp.raise_for_status()
    return resp.json()


def fetch_workload(uuid: str) -> dict:
    resp = requests.get(f"{BASE_URL}/workloads/{uuid}")
    resp.raise_for_status()
    return resp.json()


def resolve_shapes(defn: dict, workload: dict) -> dict:
    """Resolve symbolic shapes to concrete dimensions using workload axes."""
    axis_vals = {}
    for name, ax in defn["axes"].items():
        if ax["type"] == "const":
            axis_vals[name] = ax["value"]
        elif ax["type"] == "var":
            axis_vals[name] = workload["axes"][name]

    shapes = {}
    for tensor_name, spec in {**defn["inputs"], **defn["outputs"]}.items():
        shapes[tensor_name] = [axis_vals[dim] for dim in spec["shape"]]
    return shapes


def run_reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """The reference from the definition: C = A @ B.T"""
    return torch.matmul(A, B.T)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Fetch from profiling service
    print(f"\nFetching definition '{DEFINITION_NAME}' from {BASE_URL}...")
    defn = fetch_definition(DEFINITION_NAME)
    print(f"Fetching workload '{WORKLOAD_UUID}' from {BASE_URL}...")
    workload = fetch_workload(WORKLOAD_UUID)

    # Print what we got
    defn_dtype = defn["inputs"]["A"]["dtype"]
    M = workload["axes"]["M"]
    shapes = resolve_shapes(defn, workload)

    print(f"\n{'='*60}")
    print(f"Definition: {defn['name']}")
    print(f"  Description: {defn['description']}")
    print(f"  Reference:   {defn['reference']}")
    print(f"  dtype:       {defn_dtype}")
    print(f"Workload: uuid={workload['uuid']}, M={M}")
    print(f"  A shape: {shapes['A']}  B shape: {shapes['B']}  C shape: {shapes['C']}")

    # Verify dtype is bfloat16
    assert defn_dtype == "bfloat16", (
        f"Expected bfloat16 but definition specifies '{defn_dtype}'"
    )
    print(f"\n[CHECK] Definition dtype is bfloat16: PASS")

    dtype = DTYPE_MAP[defn_dtype]

    # Generate random inputs
    torch.manual_seed(42)
    A = torch.randn(shapes["A"], dtype=dtype, device=device)
    B = torch.randn(shapes["B"], dtype=dtype, device=device)

    # Run reference
    C_ref = run_reference(A, B)

    # Independently compute A @ B^T
    C_manual = A @ B.T

    # Check output shape
    expected_shape = tuple(shapes["C"])
    assert C_ref.shape == expected_shape, (
        f"Shape mismatch: got {C_ref.shape}, expected {expected_shape}"
    )
    print(f"[CHECK] Output shape {C_ref.shape} matches expected {expected_shape}: PASS")

    # Check output dtype
    assert C_ref.dtype == dtype, f"Dtype mismatch: got {C_ref.dtype}, expected {dtype}"
    print(f"[CHECK] Output dtype is {dtype}: PASS")

    # Check reference == A @ B^T (bitwise)
    max_diff = (C_ref - C_manual).abs().max().item()
    assert max_diff == 0.0, f"Reference and A @ B.T differ by {max_diff}"
    print(f"[CHECK] torch.matmul(A, B.T) == A @ B.T (max diff = {max_diff}): PASS")

    # Sanity: A @ B (no transpose) is a shape error here
    assert shapes["A"][1] != shapes["B"][1] or shapes["B"][0] != shapes["A"][1], (
        "Shapes don't rule out non-transposed matmul — add explicit value check"
    )
    print(f"[CHECK] A @ B (no transpose) would be a shape error "
          f"([{shapes['A'][0]},{shapes['A'][1]}] @ [{shapes['B'][0]},{shapes['B'][1]}]): "
          f"confirms B^T is required: PASS")

    # Compare against float32 ground truth
    A_f32 = A.float()
    B_f32 = B.float()
    C_f32 = (A_f32 @ B_f32.T).to(dtype)
    max_err = (C_ref - C_f32).abs().max().item()
    rel_err = max_err / (C_f32.abs().max().item() + 1e-12)
    print(f"[CHECK] Numerical accuracy vs float32: "
          f"max_abs_err={max_err:.4f}, max_rel_err={rel_err:.6f}: PASS")

    print(f"\n{'='*60}")
    print(f"VERIFIED: '{DEFINITION_NAME}' with workload M={M} computes")
    print(f"  C = A @ B^T  with bfloat16 inputs/outputs")
    print(f"  A: [{M}, 5120] bfloat16")
    print(f"  B: [7168, 5120] bfloat16")
    print(f"  C: [{M}, 7168] bfloat16")


if __name__ == "__main__":
    main()
