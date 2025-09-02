#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || true
hurdle-forecast --train data/train/train.csv --test_dir data/test --out_dir outputs
