"""Dataset shape sanity tests. Skipped if datasets not downloaded yet."""
from __future__ import annotations

import pytest


def test_shd_shape_skip_if_missing() -> None:
    pytest.skip("Dataset loaders implemented in Phase 01.")


def test_ssc_shape_skip_if_missing() -> None:
    pytest.skip("Dataset loaders implemented in Phase 01.")


def test_gsc_shape_skip_if_missing() -> None:
    pytest.skip("Dataset loaders implemented in Phase 01.")
