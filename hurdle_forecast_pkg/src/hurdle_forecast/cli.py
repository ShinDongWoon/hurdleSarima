import argparse
from .config import Config
from .pipeline import run_forecast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", dest="train_csv", required=True, help="Path to train.csv")
    ap.add_argument("--test_dir", required=True, help="Directory containing TEST_*.csv files")
    ap.add_argument("--out_dir", required=True, help="Directory to write predictions")
    ap.add_argument("--sample_submission", default=None, help="Optional sample submission to fill")
    ap.add_argument("--classifier", choices=["beta", "logit"], default="beta")
    ap.add_argument("--sarima_grid", choices=["full", "small"], default="full")
    ap.add_argument("--fallback", choices=["ets", "snaive"], default="ets")
    ap.add_argument("--cap", type=float, default=0.99, help="Quantile cap (set <=0 or >1 to disable)")
    args = ap.parse_args()

    cfg = Config(
        train_csv=args.train_csv,
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        sample_submission=args.sample_submission,
        classifier_kind=args.classifier,
        sarima_grid=args.sarima_grid,
        fallback=args.fallback,
        cap_quantile=(args.cap if 0.0 < args.cap < 1.0 else None),
    )
    run_forecast(cfg)

if __name__ == "__main__":
    main()
