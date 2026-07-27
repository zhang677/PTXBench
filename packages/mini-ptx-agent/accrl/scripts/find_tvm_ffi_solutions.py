"""Find CUDA solutions that use TVM FFI patterns.

Queries the Supabase kernel database for CUDA solutions whose source files
contain TVM FFI markers (e.g., `tvm/ffi` includes, `TVM_FFI_DLL_EXPORT_TYPED_FUNC`).

Usage:
    python accrl/scripts/find_tvm_ffi_solutions.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from supabase import create_client

TVM_FFI_MARKERS = ["tvm/ffi", "TVM_FFI"]
PAGE_SIZE = 1000


def main():
    url = os.getenv("SUPABASE_URL", "http://localhost:8000")
    key = os.getenv("SERVICE_ROLE_KEY", "my-service-role-key")
    client = create_client(url, key)

    matches = []
    offset = 0

    while True:
        rows = (
            client.table("solutions")
            .select("*")
            .eq("spec_language", "cuda")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
            .data
        )
        if not rows:
            break

        for row in rows:
            solution = row.get("solution", {})
            sources = solution.get("sources", [])
            for src in sources:
                content = src.get("content", "")
                if any(marker in content for marker in TVM_FFI_MARKERS):
                    matches.append(solution)
                    break # one match per solution is enough

        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    print(f"\nFound {len(matches)} CUDA solution(s) with TVM FFI patterns:\n")
    for m in matches:
        print(f"Solution name: {m.get('name')}, definition: {m.get('definition')}")
        print()


if __name__ == "__main__":
    main()
