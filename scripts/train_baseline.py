#!/usr/bin/env python3
"""Fit baseline (and optionally CatBoost) from consumption Excel and save artifacts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.preprocessing import load_consumption_record, load_cooler_config
from model.predictor import WaterDemandPredictor


def main():
    default_cons = ROOT / "data" / "Cooler_Water_Consumption_Record.csv"
    if not default_cons.exists():
        default_cons = ROOT / "data" / "Cooler_Water_Consumption_Record.xlsx"
    p = argparse.ArgumentParser()
    p.add_argument("--consumption", type=Path, default=default_cons)
    p.add_argument("--coolers", type=Path, default=ROOT / "data" / "cooler_config.csv")
    p.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    p.add_argument("--train-catboost", action="store_true", help="Attempt CatBoost (needs enough rows)")
    p.add_argument("--min-rows", type=int, default=None, help="Override CatBoost min rows (tests only)")
    args = p.parse_args()

    meta = load_cooler_config(args.coolers)
    df = load_consumption_record(args.consumption)
    print(f"Loaded {len(df)} consumption rows, {len(meta)} coolers")

    pred = WaterDemandPredictor(model_dir=args.model_dir)
    pred.load_cooler_meta(meta)
    info = pred.fit_from_consumption(
        df,
        train_catboost=args.train_catboost,
        catboost_min_rows=args.min_rows,
    )
    print(json.dumps(info, indent=2))
    print("Model version:", pred.version)


if __name__ == "__main__":
    main()