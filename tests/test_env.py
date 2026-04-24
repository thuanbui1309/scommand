"""Environment import + basic CUDA sanity tests."""
from __future__ import annotations

import pytest


def test_torch_import() -> None:
    import torch

    assert torch.__version__.startswith("2.")


def test_cuda_available() -> None:
    import torch

    assert torch.cuda.is_available(), "RTX 4090 expected; CUDA must be available"


def test_spikingjelly_import() -> None:
    from spikingjelly.activation_based import neuron, surrogate  # noqa: F401


def test_tonic_import() -> None:
    import tonic  # noqa: F401


def test_mamba_ssm_optional() -> None:
    """mamba-ssm is optional — falls back to pure-PyTorch SSM if unavailable."""
    pytest.importorskip("mamba_ssm", reason="Optional; Track C fallback path documented.")


def test_syops_counter_import() -> None:
    pytest.importorskip("syops", reason="Optional; stub utility until Phase 01.")


def test_seed_utility() -> None:
    from scommander.utils.seed import set_seed

    set_seed(42)


def test_json_logger(tmp_path) -> None:
    from scommander.utils.logging import JsonLineLogger, RunMetadata

    md = RunMetadata(run_id="t0", seed=0, dataset="shd", variant="baseline")
    with JsonLineLogger(tmp_path / "log.jsonl", md) as log:
        log.log({"epoch": 0, "loss": 1.0})
    content = (tmp_path / "log.jsonl").read_text().strip()
    assert '"epoch": 0' in content
    assert '"variant": "baseline"' in content
