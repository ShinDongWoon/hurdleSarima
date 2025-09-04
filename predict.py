#!/usr/bin/env python
import argparse
import logging
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
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading model from %s", args.model)
    model = HurdleForecastModel.load(args.model)
    logger.info("Running predictions for %s", args.test_dir)
    model.predict_dir(args.test_dir, args.out_dir, sample_submission=args.sample_submission)
    logger.info("Predictions written to %s", args.out_dir)


if __name__ == "__main__":
    main()
