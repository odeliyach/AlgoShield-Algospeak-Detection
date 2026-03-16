"""
Unit tests for metrics computation functions from scripts/exp1/finetune.py.

Validates compute_metrics and extract_metrics without requiring torch or
any GPU-dependent dependencies.
"""
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import numpy as np
from sklearn.metrics import classification_report

from tests.conftest import SCRIPTS_EXP1_DIR

# ---------------------------------------------------------------------------
# Mock ALL heavy dependencies before importing the finetune module.
# torch.Tensor must be a real class so that scipy's issubclass checks don't fail.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()
_torch_mock.Tensor = type("Tensor", (), {})
sys.modules["torch"] = _torch_mock

for _mod in ("transformers", "datasets", "accelerate", "matplotlib", "matplotlib.pyplot"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["matplotlib"].use = MagicMock()

_script_path = os.path.join(SCRIPTS_EXP1_DIR, "finetune.py")  # noqa: E402
_spec = importlib.util.spec_from_file_location("finetune", _script_path)
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)

compute_metrics = _ft.compute_metrics
extract_metrics = _ft.extract_metrics


# ---------------------------------------------------------------------------
# Tests — compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _make_eval_pred(self, y_true, y_pred_class):
        """Build (logits, labels) where logits encode the desired class."""
        labels = np.array(y_true)
        logits = np.zeros((len(y_pred_class), 2))
        for i, c in enumerate(y_pred_class):
            if c == 1:
                logits[i] = [0.0, 1.0]
            else:
                logits[i] = [1.0, 0.0]
        return logits, labels

    def test_perfect_predictions(self):
        y = [0, 1, 1, 0, 1]
        logits, labels = self._make_eval_pred(y, y)
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["recall"] == 1.0
        assert result["precision"] == 1.0

    def test_all_wrong_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        logits, labels = self._make_eval_pred(y_true, y_pred)
        result = compute_metrics((logits, labels))
        assert result["accuracy"] == 0.0
        assert result["f1"] == 0.0
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0

    def test_partial_predictions(self):
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]  # 1 FN, 1 FP
        logits, labels = self._make_eval_pred(y_true, y_pred)
        result = compute_metrics((logits, labels))
        assert abs(result["accuracy"] - 4 / 6) < 1e-6
        assert abs(result["recall"] - 2 / 3) < 1e-6

    def test_returns_required_keys(self):
        y = [0, 1]
        logits, labels = self._make_eval_pred(y, y)
        result = compute_metrics((logits, labels))
        for key in ("accuracy", "f1", "recall", "precision", "f1_macro"):
            assert key in result, f"Missing key: {key}"

    def test_all_toxic_predicted_as_non_toxic(self):
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        logits, labels = self._make_eval_pred(y_true, y_pred)
        result = compute_metrics((logits, labels))
        assert result["recall"] == 0.0
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0


# ---------------------------------------------------------------------------
# Tests — extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:

    def _make_report(self, y_true, y_pred):
        return classification_report(
            y_true, y_pred,
            target_names=["non-toxic", "toxic"],
            output_dict=True,
        )

    def test_extract_keys_present(self):
        report = self._make_report([0, 1, 1, 0], [0, 1, 1, 1])
        result = extract_metrics(report)
        for key in ("f1", "precision", "recall", "accuracy", "f1_macro"):
            assert key in result

    def test_extract_perfect_metrics(self):
        y = [0, 1, 1, 0, 1]
        report = self._make_report(y, y)
        result = extract_metrics(report)
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["accuracy"] == 1.0

    def test_extract_partial_metrics(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]  # 1 FN
        report = self._make_report(y_true, y_pred)
        result = extract_metrics(report)
        assert abs(result["recall"] - 0.5) < 1e-6
