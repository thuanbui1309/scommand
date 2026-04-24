"""Environment verification: prints versions, runs LIF + Mamba forward sanity.

Run: python scripts/verify_env.py
Exits 0 on success; non-zero with actionable error otherwise.
"""
from __future__ import annotations

import sys


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_torch() -> bool:
    try:
        import torch

        print(f"torch {torch.__version__} | CUDA={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            _fail("CUDA not available on this machine.")
            return False
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        _ok("torch + CUDA")
        return True
    except Exception as e:
        _fail(f"torch check failed: {e}")
        return False


def check_cupy() -> bool:
    try:
        import cupy as cp

        x = cp.array([1.0, 2.0, 3.0])
        y = (x * 2).sum().get()
        assert float(y) == 12.0
        print(f"cupy {cp.__version__}")
        _ok("cupy")
        return True
    except Exception as e:
        _warn(f"cupy not functional: {e} | SpikingJelly will fallback to torch backend (~3x slower)")
        return False


def check_spikingjelly() -> bool:
    try:
        import torch
        from spikingjelly.activation_based import neuron, surrogate

        lif = neuron.LIFNode(tau=2.0, v_reset=0.5, v_threshold=1.0, surrogate_function=surrogate.ATan())
        x = torch.randn(4, 16).cuda()
        lif = lif.cuda()
        _ = lif(x)
        _ok("spikingjelly LIF forward")
        return True
    except Exception as e:
        _fail(f"spikingjelly check failed: {e}")
        return False


def check_tonic() -> bool:
    try:
        import tonic

        print(f"tonic {tonic.__version__}")
        _ok("tonic")
        return True
    except Exception as e:
        _fail(f"tonic check failed: {e}")
        return False


def check_mamba_ssm() -> bool:
    """Day-1 mamba CUDA kernel probe. Fallback path noted if failure."""
    try:
        import torch
        from mamba_ssm import Mamba

        mamba = Mamba(d_model=64).cuda()
        x = torch.randn(1, 10, 64).cuda()
        _ = mamba(x)
        _ok("mamba-ssm forward (CUDA kernel compiled)")
        return True
    except Exception as e:
        _warn(f"mamba-ssm unavailable: {e}")
        _warn("Track C must fall back to pure-PyTorch SSM. Record in docs/env-risk.md.")
        return False


def check_syops_counter() -> bool:
    try:
        import syops

        print(f"syops-counter {getattr(syops, '__version__', 'unknown')}")
        _ok("syops-counter")
        return True
    except Exception as e:
        _warn(f"syops-counter check failed: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("SCommander environment verification")
    print("=" * 60)

    checks = [
        ("torch", check_torch()),
        ("cupy", check_cupy()),
        ("spikingjelly", check_spikingjelly()),
        ("tonic", check_tonic()),
        ("mamba-ssm", check_mamba_ssm()),
        ("syops-counter", check_syops_counter()),
    ]

    print("=" * 60)
    failures = [name for name, ok in checks if not ok and name in {"torch", "spikingjelly", "tonic"}]
    if failures:
        _fail(f"Critical checks failed: {failures}")
        return 1
    _ok("Environment ready (warnings above are informational only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
