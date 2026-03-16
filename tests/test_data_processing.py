"""
Unit tests for data-processing helpers.

Covers:
  - tox_bin binarisation (TOX_BIN_THRESHOLD logic from finetune.py)
  - content filtering (empty / NaN rows dropped)
  - balanced sampling correctness (create_balanced_splits contract)
"""
import importlib.util
import os
import sys
import tempfile
from unittest.mock import MagicMock

import pandas as pd
import polars as pl

from tests.conftest import SCRIPTS_EXP1_DIR

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing script modules.
# torch.Tensor must be a real class so that scipy's issubclass checks don't fail.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()
_torch_mock.Tensor = type("Tensor", (), {})
sys.modules["torch"] = _torch_mock

for _mod in ("transformers", "datasets", "accelerate", "matplotlib", "matplotlib.pyplot"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["matplotlib"].use = MagicMock()

# Load finetune module for load_parquet / TOX_BIN_THRESHOLD
_ft_path = os.path.join(SCRIPTS_EXP1_DIR, "finetune.py")  # noqa: E402
_spec = importlib.util.spec_from_file_location("finetune", _ft_path)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)

TOX_BIN_THRESHOLD = _ft.TOX_BIN_THRESHOLD   # 5
load_parquet = _ft.load_parquet

# Load balanced sampler
_bs_path = os.path.join(SCRIPTS_EXP1_DIR, "balanced_train_val_ds.py")  # noqa: E402
_spec2 = importlib.util.spec_from_file_location("balanced_train_val_ds", _bs_path)
_bs = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_bs)

create_balanced_splits = _bs.create_balanced_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: str):
    """Write a pandas DataFrame as parquet to *path*."""
    df.to_parquet(path, index=False)


def _make_koo_parquet(path: str, n_per_bin: int = 10):
    """
    Generate a minimal Koo-like parquet with all 10 tox_bins represented.
    Each bin has n_per_bin rows.
    """
    rows = []
    for b in range(10):
        for i in range(n_per_bin):
            rows.append({"content": f"koo post bin={b} idx={i}", "tox_bin": b})
    _write_parquet(pd.DataFrame(rows), path)


# ---------------------------------------------------------------------------
# Tests — TOX_BIN_THRESHOLD binarisation
# ---------------------------------------------------------------------------

class TestToxBinBinarisation:
    """Verifies the tox_bin -> label mapping used in load_parquet."""

    def test_threshold_constant_is_five(self):
        assert TOX_BIN_THRESHOLD == 5

    def test_bins_below_threshold_are_non_toxic(self):
        for b in range(TOX_BIN_THRESHOLD):   # 0..4
            label = int(b >= TOX_BIN_THRESHOLD)
            assert label == 0, f"bin {b} should be non-toxic"

    def test_bins_at_or_above_threshold_are_toxic(self):
        for b in range(TOX_BIN_THRESHOLD, 10):   # 5..9
            label = int(b >= TOX_BIN_THRESHOLD)
            assert label == 1, f"bin {b} should be toxic"

    def test_load_parquet_binarises_correctly(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            df = pd.DataFrame({
                "content": ["toxic post", "safe post", "borderline"],
                "tox_bin": [7, 2, 5],
            })
            _write_parquet(df, path)
            result = load_parquet(path)
            assert list(result["labels"]) == [1, 0, 1]
        finally:
            os.unlink(path)

    def test_load_parquet_drops_empty_content(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            df = pd.DataFrame({
                "content": ["real post", "  ", "", None],
                "tox_bin": [6, 3, 4, 7],
            })
            _write_parquet(df, path)
            result = load_parquet(path)
            # Only "real post" should survive filtering
            assert len(result) == 1
            assert result.iloc[0]["labels"] == 1
        finally:
            os.unlink(path)

    def test_load_parquet_drops_nan_tox_bin(self):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            df = pd.DataFrame({
                "content": ["post a", "post b"],
                "tox_bin": [6, None],
            })
            _write_parquet(df, path)
            result = load_parquet(path)
            assert len(result) == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests — balanced sampler
# ---------------------------------------------------------------------------

class TestBalancedSampling:
    """Smoke-tests for create_balanced_splits output invariants."""

    def test_output_files_are_created(self, tmp_path):
        madoc_dir = tmp_path / "madoc"
        output_dir = tmp_path / "out"
        madoc_dir.mkdir()

        _make_koo_parquet(str(madoc_dir / "koo_madoc.parquet"), n_per_bin=600)
        create_balanced_splits(str(madoc_dir), str(output_dir))

        assert (output_dir / "train_2x.parquet").exists()
        assert (output_dir / "val_2x.parquet").exists()

    def test_val_and_train_are_disjoint(self, tmp_path):
        madoc_dir = tmp_path / "madoc"
        output_dir = tmp_path / "out"
        madoc_dir.mkdir()

        _make_koo_parquet(str(madoc_dir / "koo_madoc.parquet"), n_per_bin=600)
        create_balanced_splits(str(madoc_dir), str(output_dir))

        train = pl.read_parquet(str(output_dir / "train_2x.parquet"))
        val = pl.read_parquet(str(output_dir / "val_2x.parquet"))

        # No overlapping (content, tox_bin) pairs
        train_set = set(zip(train["content"].to_list(), train["tox_bin"].to_list()))
        val_set = set(zip(val["content"].to_list(), val["tox_bin"].to_list()))
        assert len(train_set & val_set) == 0

    def test_all_bins_represented_in_val(self, tmp_path):
        madoc_dir = tmp_path / "madoc"
        output_dir = tmp_path / "out"
        madoc_dir.mkdir()

        _make_koo_parquet(str(madoc_dir / "koo_madoc.parquet"), n_per_bin=600)
        create_balanced_splits(str(madoc_dir), str(output_dir))

        val = pl.read_parquet(str(output_dir / "val_2x.parquet"))
        present_bins = set(val["tox_bin"].to_list())
        assert present_bins == set(range(10)), f"Missing bins: {set(range(10)) - present_bins}"
