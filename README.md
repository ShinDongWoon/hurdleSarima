# hurdle-forecast (Mac MPS-ready)

A lightweight, competition-friendly **hurdle** forecaster for zero‑inflated, weekly seasonal demand.

**Structure**
```
hurdle_forecast_pkg/
├─ pyproject.toml
├─ requirements.txt  (optional; torch is optional)
├─ src/hurdle_forecast/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ mps_utils.py
│  ├─ data.py
│  ├─ classifier.py
│  ├─ intensity.py
│  ├─ combine.py
│  ├─ metrics.py
│  ├─ pipeline.py
│  └─ cli.py
├─ data/
│  ├─ train/train.csv            # put your training file here
│  └─ test/TEST_*.csv            # put all test files here
└─ outputs/
```

**Quickstart**
```bash
# (optional) create and activate venv
python -m venv .venv && source .venv/bin/activate

# install dependencies
python dependency.py

# (optional) install package in editable mode
pip install -e .

# (optional) install torch for MPS logistic classifier
pip install torch --index-url https://download.pytorch.org/whl/cpu

# run (beta-smoothing classifier default; SARIMA intensity)
hurdle-forecast \
  --train data/train/train.csv \
  --test_dir data/test \
  --out_dir outputs \
  --sample_submission path/to/sample_submission.csv  # optional
```

**Colab example**

The following cells show a minimal end-to-end run inside Google Colab:

```python
# 1. clone repository and move into it
!git clone https://github.com/<your-user>/hurdle-forecast.git
%cd hurdle-forecast

# 2. install dependencies
!python dependency.py

# 3. train model
!python train.py --train data/train/train.csv --model_out models/model.pkl

# 4. predict
!python predict.py --model models/model.pkl --test_dir data/test --out_dir outputs
```

**Key parameters**

- `--train`: path to the training CSV (e.g. `data/train/train.csv`).
- `--model_out`: where to store the trained model (default `models/model.pkl`).
- `--test_dir`: directory with `TEST_*.csv` files (e.g. `data/test`).
- `--out_dir`: directory to write predictions (e.g. `outputs`).
- `--sample_submission`: optional CSV skeleton for competition submissions.

**Expectations / Columns**
- Train csv must include: `영업일자`, `매출수량`, and either separate `영업장명`/`메뉴명` columns or a combined `영업장명_메뉴명` column which will be split.
- Test csv must include: `영업일자` and either separate `영업장명`/`메뉴명` columns or `영업장명_메뉴명` (and any other columns are preserved).
- Dates are kept **as strings** in the final submission; internal processing uses a parsed datetime index with no leakage.
- Each `TEST_*.csv` is predicted **independently** (no cross-file leakage) and a matching output file is written to `outputs/`.
- If a `--sample_submission` is given, we **only fill the values** in that skeleton and never touch column names or date string formats.

**Design Notes**
- Classifier A: **Beta-smoothed frequency** per (series × day-of-week) over the recent window (default 8 weeks). Fallbacks: series-all-time, then global.
- Optional Classifier B: **Global logistic regression** (calendar-only, with series prior) implemented in PyTorch. It runs on **MPS** if available.
- Intensity model: **SARIMAX (m=7)** on `log1p(y)` with zeros masked as missing. Grid over (p,d,q) and (P,D,Q) in {0,1}. If too short or unstable: **ETS(A,N,A)** or **SeasonalNaive** fallback.
- Final prediction: `ŷ_t = P_t × μ_t`, clipped to `>= 0` and optional p99 capping. No integer rounding (wSMAPE-friendly).

**Performance / No-Leakage**
- Models fit **only** on train data prior to each test file’s start date.
- Feature computations are cached per series & cutoff to avoid redundant work.
- Minimal overhead: vectorized calendar features; small, safe grids; robust fallbacks.
