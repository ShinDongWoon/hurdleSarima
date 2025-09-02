#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from hurdle_forecast.model import HurdleForecastModel


def main() -> None:
    ap = argparse.ArgumentParser(description="Load trained model and run predictions on test data.")
    ap.add_argument("--model", required=True, help="Path to pickled model")
    ap.add_argument("--test_dir", required=True, help="Directory containing TEST_*.csv files")
    ap.add_argument("--out_dir", required=True, help="Directory to write predictions")
    ap.add_argument("--sample_submission", default=None, help="Optional sample submission to fill")
    args = ap.parse_args()

    model = HurdleForecastModel.load(args.model)
    model.predict_dir(args.test_dir, args.out_dir, sample_submission=args.sample_submission)


if __name__ == "__main__":
    main()
