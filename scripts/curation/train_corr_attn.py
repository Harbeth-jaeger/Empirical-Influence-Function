#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.curation.train_corr_attn import train


if __name__ == "__main__":
    train()

