"""Environment import + basic CUDA sanity tests."""
from __future__ import annotations

import pytest


def test_torch_import() -> None:
    import torch

    assert torch.__version__.startswith("2.")


def test_cuda_available() -> None:
    import torch

    assert torch.cuda.is_available(), "RTX 5090 (Blackwell sm_120) expected; CUDA must be available"


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


def test_seed_determinism_three_rerun() -> None:
    """Phase 00 gate: identical seed produces bit-identical first-batch loss across 3 reruns."""
    import torch

    from scommander.utils.seed import set_seed

    def sample_tensor() -> torch.Tensor:
        set_seed(1337)
        x = torch.randn(4, 16)
        w = torch.randn(16, 8)
        return (x @ w).sum()

    outputs = [sample_tensor().item() for _ in range(3)]
    assert outputs[0] == outputs[1] == outputs[2], f"seed non-deterministic: {outputs}"


def test_base_config_load_omegaconf() -> None:
    """Phase 00 gate: base.yaml loads via OmegaConf; amp default is false."""
    from pathlib import Path

    from omegaconf import OmegaConf

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "base.yaml"
    assert cfg_path.exists(), f"base.yaml missing at {cfg_path}"

    cfg = OmegaConf.load(cfg_path)
    amp = OmegaConf.select(cfg, "training.amp", default=None)
    assert amp is False, f"training.amp must default to false (locked precision policy); got {amp!r}"


def test_json_logger(tmp_path) -> None:
    from scommander.utils.logging import JsonLineLogger, RunMetadata

    md = RunMetadata(run_id="t0", seed=0, dataset="shd", variant="baseline")
    with JsonLineLogger(tmp_path / "log.jsonl", md) as log:
        log.log({"epoch": 0, "loss": 1.0})
    content = (tmp_path / "log.jsonl").read_text().strip()
    assert '"epoch": 0' in content
    assert '"variant": "baseline"' in content
