#!/usr/bin/env python3
"""Build-time conversion script: converts institute Excel tables into Parquet binary format.

Reduces cold file read latency from ~400-800ms down to < 20 ms, eliminating runtime
openpyxl dependency during Space restarts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from model.preprocessing import convert_excel_to_parquet, ensure_binary_data


def main():
    parser = argparse.ArgumentParser(description="Convert Excel tables to Parquet binary format.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Path to data directory containing Excel files (default: data/)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for Excel files in {data_dir}...")
    t0 = time.perf_counter()
    results = ensure_binary_data(data_dir)
    total_time = (time.perf_counter() - t0) * 1000

    print(f"Conversion completed in {total_time:.2f} ms")
    for src_name, pq_path in results.items():
        src_path = data_dir / src_name
        src_size = src_path.stat().st_size if src_path.exists() else 0
        pq_size = pq_path.stat().st_size if pq_path.exists() else 0
        df = pd.read_parquet(pq_path)
        print(f" - {src_name} ({src_size} bytes) -> {pq_path.name} ({pq_size} bytes, {len(df)} rows)")

    print("All tables successfully converted to Parquet.")


if __name__ == "__main__":
    main()
