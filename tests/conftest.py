"""
Shared pytest configuration and path helpers.

All test modules can import SCRIPTS_EXP1_DIR to locate project scripts
without repeating the relative-path construction.
"""
import os

# Absolute path to the repository root (parent of this tests/ directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Absolute path to the exp1 scripts directory
SCRIPTS_EXP1_DIR = os.path.join(REPO_ROOT, "scripts", "exp1")
