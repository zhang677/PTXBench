"""Configuration for benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Per-definition eval parameters. All fields Optional; None means inherit from parent layer."""

    warmup_runs: Optional[int] = Field(default=None, ge=0)
    """Warmup iterations before timing. `None` means inherit."""
    iterations: Optional[int] = Field(default=None, gt=0)
    """Timed iterations per trial. `None` means inherit."""
    num_trials: Optional[int] = Field(default=None, gt=0)
    """Number of benchmark trials. `None` means inherit."""
    rtol: Optional[float] = Field(default=None, gt=0)
    """Relative tolerance for numerical checks. `None` means inherit."""
    atol: Optional[float] = Field(default=None, gt=0)
    """Absolute tolerance for numerical checks. `None` means inherit."""
    required_matched_ratio: Optional[float] = Field(default=None, gt=0, le=1)
    """Minimum fraction of elements that must be within tolerance. `None` means inherit."""
    extra: Dict[str, Any] = Field(default_factory=dict)
    """Evaluator-specific parameters that do not belong in the shared schema."""


class ResolvedEvalConfig(BaseModel):
    """Resolved eval parameters with all fields populated. This is what evaluators consume."""

    warmup_runs: int = 10
    """Warmup iterations before timing."""
    iterations: int = 50
    """Timed iterations per trial."""
    num_trials: int = 3
    """Number of benchmark trials."""
    rtol: float = 1e-2
    """Relative tolerance for numerical checks."""
    atol: float = 1e-2
    """Absolute tolerance for numerical checks."""
    required_matched_ratio: Optional[float] = None
    """Minimum fraction of elements that must be within tolerance."""
    profile_baseline: bool = True
    """Whether to profile the reference implementation for baseline latency."""
    run_baseline: bool = True
    """Whether to run the reference implementation live. If False, reference outputs are
    loaded from ``workload.outputs`` (safetensors) and the live baseline is skipped."""
    extra: Dict[str, Any] = Field(default_factory=dict)
    """Evaluator-specific parameters after all config layers have been merged."""


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs. Mirrors the YAML structure exactly.

    All fields have default values to make configuration optional.
    """

    # Service-level
    timeout_seconds: int = Field(default=300, gt=0)
    """Timeout in seconds for each solution evaluation."""
    max_mem_ratio: float = Field(default=0.85, gt=0, le=1)
    """Fraction of a GPU worker's total HBM that its persistent baseline cache may occupy.
    Used by FIBServe for per-worker baseline cache admission/eviction."""
    profile_baseline: bool = True
    """Whether to profile the reference implementation for baseline latency."""
    # Per-definition defaults
    warmup_runs: int = Field(default=10, ge=0)
    """Default warmup iterations before timing for all definitions."""
    iterations: int = Field(default=50, gt=0)
    """Default timed iterations per trial for all definitions."""
    num_trials: int = Field(default=3, gt=0)
    """Default number of benchmark trials for all definitions."""
    rtol: float = Field(default=1e-2, gt=0)
    """Default relative tolerance for numerical checks."""
    atol: float = Field(default=1e-2, gt=0)
    """Default absolute tolerance for numerical checks."""
    required_matched_ratio: Optional[float] = Field(default=None, gt=0, le=1)
    """Default minimum fraction of elements that must be within tolerance."""
    # Deprecated: use op_type_config/definition_config extra instead
    sampling_validation_trials: int = Field(default=100, gt=0)
    """Deprecated default for Sampling evaluator validation rounds."""
    sampling_tvd_threshold: float = Field(default=0.2, ge=0, le=1)
    """Deprecated default for Sampling evaluator TVD threshold."""

    # Per op_type / per definition overrides
    op_type_config: Dict[str, EvalConfig] = Field(default_factory=dict)
    """Per-op-type eval overrides keyed by `definition.op_type`."""
    definition_config: Dict[str, EvalConfig] = Field(default_factory=dict)
    """Per-definition eval overrides keyed by `definition.name`."""

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> BenchmarkConfig:
        """Load config from a YAML file, with optional field overrides."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.update(overrides)
        return cls.model_validate(data)

    @classmethod
    def default(cls, **overrides: Any) -> BenchmarkConfig:
        """Load the bundled eval_config.yaml if it exists, otherwise use defaults."""
        yaml_path = Path(__file__).parent / "eval_config.yaml"
        if yaml_path.exists():
            return cls.from_yaml(str(yaml_path), **overrides)
        return cls(**overrides)

    def resolve_eval_config(
        self,
        definition: Any,
        *,
        profile_baseline: Optional[bool] = None,
        run_baseline: Optional[bool] = None,
    ) -> ResolvedEvalConfig:
        """Merge: per-def defaults -> op_type_config -> definition_config.

        Optional ``profile_baseline`` / ``run_baseline`` overrides take precedence over
        the global config; ``None`` means "use the global config value". When
        ``run_baseline`` resolves to ``False``, ``profile_baseline`` is forced to
        ``False`` since there is no live baseline to profile.
        """
        merged = {
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
            "num_trials": self.num_trials,
            "rtol": self.rtol,
            "atol": self.atol,
            "required_matched_ratio": self.required_matched_ratio,
            "profile_baseline": self.profile_baseline,
            "run_baseline": True,
            "extra": {
                "sampling_validation_trials": self.sampling_validation_trials,
                "sampling_tvd_threshold": self.sampling_tvd_threshold,
            },
        }

        layers = [
            self.op_type_config.get(definition.op_type),
            self.definition_config.get(definition.name),
        ]

        for layer in layers:
            if layer is None:
                continue
            updates = {
                k: v for k, v in layer.model_dump(exclude={"extra"}).items() if v is not None
            }
            merged.update(updates)
            if layer.extra:
                merged["extra"].update(layer.extra)

        if profile_baseline is not None:
            merged["profile_baseline"] = profile_baseline
        if run_baseline is not None:
            merged["run_baseline"] = run_baseline
        if not merged["run_baseline"]:
            merged["profile_baseline"] = False

        return ResolvedEvalConfig(**merged)
