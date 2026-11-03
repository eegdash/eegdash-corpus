"""Shared fixtures: locate the reve_eeg checkout and make its `src/` importable."""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:  # tests import the scripts as modules
    sys.path.insert(0, str(SCRIPTS))


def reve_eeg_dir() -> Path | None:
    for cand in (os.environ.get("REVE_EEG"), "~/GitHub/reve_eeg", "~/Documents/GitHub/reve_eeg"):
        if cand and (Path(cand).expanduser() / "src" / "train.py").exists():
            return Path(cand).expanduser()
    return None


@pytest.fixture(scope="session")
def reve_src():
    d = reve_eeg_dir()
    if d is None:
        pytest.skip("reve_eeg checkout not found (set REVE_EEG)")
    src = str(d / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return d / "src"


@pytest.fixture(scope="session")
def scripts_on_path():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return SCRIPTS
