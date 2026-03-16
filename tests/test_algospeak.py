"""
Unit tests for algospeak pattern detection logic.

Tests the regex patterns from scripts/exp1/qualitative_analysis.py
without requiring torch or transformers.
"""
import importlib.util
import os
import re
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing the script module.
# torch.Tensor must be a real class so that scipy's issubclass checks don't fail.
# polars is NOT mocked here because: (a) qualitative_analysis.py does not import
# polars at module level, and (b) mocking polars corrupts sklearn's type checks
# in other tests running in the same process.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()
_torch_mock.Tensor = type("Tensor", (), {})
sys.modules["torch"] = _torch_mock

for _mod in ("transformers",):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

_script_path = os.path.join(  # noqa: E402
    os.path.dirname(__file__), "..", "scripts", "exp1", "qualitative_analysis.py"
)
_spec = importlib.util.spec_from_file_location("qualitative_analysis", _script_path)
_qa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qa)

find_algospeak = _qa.find_algospeak
PATTERNS = _qa.PATTERNS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindAlgospeak:

    def test_clean_text_returns_empty(self):
        assert find_algospeak("The weather is nice today") == []

    def test_leet_n_word_detected(self):
        assert "leet:n-word" in find_algospeak("n1gger is a slur")

    def test_leet_n_word_variant(self):
        # n[i1] g/9 × 1-2  optional e/3  r+ — "n1gg3r" is a canonical leet encoding
        assert "leet:n-word" in find_algospeak("n1gg3r")

    def test_leet_wh0re_detected(self):
        assert "leet:wh*re" in find_algospeak("You are such a wh0re.")

    def test_leet_kill_detected(self):
        assert "leet:kill" in find_algospeak("I will k!ll you")

    def test_kys_abbrev_detected(self):
        assert "abbrev:kill-yourself" in find_algospeak("just kys lol")

    def test_case_insensitive(self):
        assert "abbrev:kill-yourself" in find_algospeak("KYS already")

    def test_multiple_patterns_detected(self):
        text = "kys you wh0re"
        hits = find_algospeak(text)
        assert "abbrev:kill-yourself" in hits
        assert "leet:wh*re" in hits

    def test_partial_word_not_matched_for_kill(self):
        # "skill" should not trigger "leet:kill" (word boundary check)
        assert "leet:kill" not in find_algospeak("She has a unique skill")

    def test_empty_string(self):
        assert find_algospeak("") == []

    def test_whitespace_only(self):
        assert find_algospeak("   ") == []

    def test_all_patterns_compile(self):
        """Ensure every regex pattern compiles without error."""
        for pattern, _label in PATTERNS:
            compiled = re.compile(pattern)
            assert compiled is not None
