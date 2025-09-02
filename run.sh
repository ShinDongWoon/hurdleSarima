#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python dependency.py
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || true
python train.py --train data/train/train.csv --model_out models/model.pkl
python predict.py --model models/model.pkl --test_dir data/test --out_dir outputs
