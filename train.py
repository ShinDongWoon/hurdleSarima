#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hurdle_forecast.config import Config
from hurdle_forecast.model import HurdleForecastModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Train hurdle forecast model and serialize to pickle.")
    ap.add_argument("--train", dest="train_csv", required=True, help="Path to train.csv")
    ap.add_argument("--model_out", default="models/model.pkl", help="Output path for pickled model")
    ap.add_argument("--classifier", choices=["beta", "logit"], default="beta")
    ap.add_argument("--sarima_grid", choices=["full", "small"], default="full")
    ap.add_argument("--fallback", choices=["ets", "snaive"], default="ets")
    ap.add_argument("--cap", type=float, default=0.99, help="Quantile cap (set <=0 or >1 to disable)")
    args = ap.parse_args()

    cfg = Config(
        train_csv=args.train_csv,
        test_dir="",
        out_dir="",
        sample_submission=None,
        classifier_kind=args.classifier,
        sarima_grid=args.sarima_grid,
        fallback=args.fallback,
        cap_quantile=(args.cap if 0.0 < args.cap < 1.0 else None),
    )

    model = HurdleForecastModel.from_config(cfg)
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
