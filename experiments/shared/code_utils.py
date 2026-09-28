"""Utility functions for code parsing, hashing, and validation.

Adapted from /root/data/AccelOpt-RL/accrl/utils/code_utils.py with
support for Python, Triton, and CUDA (torch binding) kernels.
"""

import hashlib
import re
from typing import Optional


def _is_python_code(code: str) -> bool:
    """Check if code is a valid Python kernel solution.

    A Python solution just needs a ``def run(`` entry point.
    It does NOT require ``@triton.jit``.
    """
    if not code or code.strip() == "":
        return False
    return "def run(" in code


def _is_triton_code(code: str) -> bool:
    """Check if code is a valid Triton kernel.

    Requires both ``@triton.jit`` and ``def run(``.
    """
    if not code or code.strip() == "":
        return False
    if "@triton.jit" not in code:
        return False
    if "def run(" not in code:
        return False
    return True


def _is_cuda_code(code: str) -> bool:
    """Check if code is a valid CUDA kernel.

    Supports two binding styles:
    - **torch**: ``PYBIND11_MODULE`` or ``torch/extension.h``
    - **TVM FFI**: ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``

    Also requires evidence of CUDA work (kernel launch ``<<<>>>``
    or ``__global__``).
    """
    if not code or code.strip() == "":
        return False

    # Detect binding
    has_torch_binding = (
        "PYBIND11_MODULE" in code
        or "torch/extension.h" in code
        or "torch::extension.h" in code
    )
    has_tvm_binding = "TVM_FFI_DLL_EXPORT_TYPED_FUNC" in code

    if not (has_torch_binding or has_tvm_binding):
        return False

    # Must have evidence of a CUDA kernel
    has_kernel_launch = "<<<" in code and ">>>" in code
    has_global = "__global__" in code
    has_tensorview = "TensorView" in code

    return has_kernel_launch or has_global or has_tensorview


def _validate_code_for_language(code: str, language: str) -> bool:
    """Validate code for a specific language."""
    if language == "triton":
        return _is_triton_code(code)
    elif language == "python":
        return _is_python_code(code)
    elif language in ("cuda", "cpp"):
        return _is_cuda_code(code)
    return False


def extract_code_block(
    text: str,
    languages: Optional[list[str]] = None,
    keep_separators: bool = True,
    last_response_strict: bool = True,
) -> str:
    """Extract the last valid code block from markdown text.

    Searches for code blocks in reverse order and returns the last one that
    passes validation for the target language. If no valid code is found,
    falls back to returning the last code block.

    Args:
        text: Text to parse for code blocks.
        languages: List of language identifiers to look for
            (default: ['python', 'triton', 'cuda', 'cpp']).
        keep_separators: If True, return code with ```language wrapper.
        last_response_strict: If True, return empty string for invalid code.

    Returns:
        Extracted code block or empty string based on strict mode.
    """
    if not isinstance(text, str):
        return ""

    if languages is None:
        languages = ["python", "triton", "cuda", "cpp"]

    languages_pattern = "|".join(map(re.escape, languages))
    codeblock_start = f"```({languages_pattern})"
    pattern = re.compile(
        codeblock_start + r"\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL
    )
    matches = list(pattern.finditer(text))

    if matches:
        # Try to find the last valid code block
        for match in reversed(matches):
            language = match.group(1)
            code_content = match.group(2).rstrip()

            if not code_content or code_content.strip() == "":
                continue

            # Validate based on detected language
            if _validate_code_for_language(code_content, language):
                if keep_separators:
                    return f"```{language}\n{code_content}\n```"
                else:
                    return code_content

        # Fallback: no valid code found, use last non-empty code block
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()

        if not code_content or code_content.strip() == "":
            return "" if last_response_strict else text

        if keep_separators:
            return f"```{language}\n{code_content}\n```"
        else:
            return code_content
    else:
        return "" if last_response_strict else text


def hash_code(code: str) -> str:
    """Generate stable hash for code deduplication.

    Args:
        code: Code string to hash.

    Returns:
        SHA256 hash as hex string.
    """
    normalized = " ".join(code.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_code(code: str, language: str = "triton") -> tuple[bool, str]:
    """Validate that code is a valid kernel for the given language.

    Args:
        code: Code string to validate.
        language: Target language ('triton', 'python', 'cuda', 'cpp').

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Extract actual code if wrapped in markdown
    if "```" in code:
        lang_list = (
            ["python", "triton"] if language in ("triton", "python") else ["cuda", "cpp"]
        )
        code = extract_code_block(code, languages=lang_list, keep_separators=False)

    if not code or code.strip() == "":
        return False, "Empty code"

    if language == "triton":
        if "@triton.jit" not in code:
            return False, "Missing @triton.jit decorator"
        if "def run(" not in code:
            return False, "Missing 'def run(' function definition"
        return True, ""
    elif language == "python":
        if "def run(" not in code:
            return False, "Missing 'def run(' function definition"
        return True, ""
    elif language in ("cuda", "cpp"):
        has_torch = (
            "PYBIND11_MODULE" in code
            or "torch/extension.h" in code
            or "torch::extension.h" in code
        )
        has_tvm = "TVM_FFI_DLL_EXPORT_TYPED_FUNC" in code
        if not (has_torch or has_tvm):
            return False, "Missing binding: need PYBIND11_MODULE or TVM_FFI_DLL_EXPORT_TYPED_FUNC"
        has_kernel = ("<<<" in code and ">>>" in code) or "__global__" in code
        if not has_kernel:
            return False, "Missing CUDA kernel (__global__ or <<<>>> launch)"
        return True, ""
    else:
        return False, f"Unsupported language: {language}"
