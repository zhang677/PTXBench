"""Tests for the optional Workload.outputs field."""

from flashinfer_bench.data import RandomInput, SafetensorsInput, Workload


def test_workload_outputs_default_none():
    wl = Workload(axes={"M": 4}, inputs={"A": RandomInput()}, uuid="w1")
    assert wl.outputs is None


def test_workload_outputs_roundtrip():
    wl = Workload(
        axes={"M": 4},
        inputs={"A": SafetensorsInput(path="data.safetensors", tensor_key="A")},
        uuid="w_with_outputs",
        outputs={"Y": SafetensorsInput(path="ref.safetensors", tensor_key="Y")},
    )
    dumped = wl.model_dump(mode="json")
    assert dumped["outputs"]["Y"]["path"] == "ref.safetensors"
    assert dumped["outputs"]["Y"]["tensor_key"] == "Y"

    restored = Workload.model_validate(dumped)
    assert restored.outputs is not None
    assert restored.outputs["Y"].path == "ref.safetensors"
    assert restored.outputs["Y"].tensor_key == "Y"
