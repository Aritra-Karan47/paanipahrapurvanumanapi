#!/usr/bin/env python3
"""Comprehensive latency benchmark comparing Excel (.xlsx) vs Parquet (.parquet) loading.

Measures cold load, warm load across multiple statistical iterations (p50, p95, p99,
mean, std dev, min, max), and compares raw reader vs full preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from model.preprocessing import load_temperature_table, load_timetable

DATA_DIR = ROOT / "data"


def run_benchmark_series(
    name: str,
    func: Callable[[], Any],
    iterations: int = 50,
) -> Dict[str, Any]:
    """Execute a function across cold start and N warm iterations, collecting metrics."""
    # Force garbage collection prior to timing
    gc.collect()

    # Cold run
    t0 = time.perf_counter()
    res = func()
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm runs
    warm_times: List[float] = []
    for _ in range(iterations):
        gc.disable()
        t0 = time.perf_counter()
        _ = func()
        elapsed = (time.perf_counter() - t0) * 1000
        gc.enable()
        warm_times.append(elapsed)

    warm_times.sort()
    p50 = statistics.median(warm_times)
    p95 = warm_times[int(len(warm_times) * 0.95)]
    p99 = warm_times[min(int(len(warm_times) * 0.99), len(warm_times) - 1)]
    mean = statistics.mean(warm_times)
    stdev = statistics.stdev(warm_times) if len(warm_times) > 1 else 0.0

    return {
        "name": name,
        "cold_ms": cold_ms,
        "mean_ms": mean,
        "median_p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min(warm_times),
        "max_ms": max(warm_times),
        "stdev_ms": stdev,
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive Excel vs Parquet latency benchmark.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of warm iterations (default: 50)")
    args = parser.parse_args()

    tt_xlsx = DATA_DIR / "School_Timetable_With_Recess.xlsx"
    tt_pq = DATA_DIR / "School_Timetable_With_Recess.parquet"
    temp_xlsx = DATA_DIR / "Water_Temperature_Control_Table.xlsx"
    temp_pq = DATA_DIR / "Water_Temperature_Control_Table.parquet"

    print("=" * 80)
    print(f"COMPREHENSIVE LATENCY BENCHMARK: EXCEL (.xlsx) vs PARQUET (.parquet)")
    print(f"Warm iterations: {args.iterations} | System: Python {sys.version.split()[0]}")
    print("=" * 80)

    # 1. Low-level raw read benchmarks
    benchmarks = [
        # Timetable
        ("Raw pd.read_excel (Timetable)", lambda: pd.read_excel(tt_xlsx, sheet_name="Timetable")),
        ("Raw pd.read_parquet (Timetable)", lambda: pd.read_parquet(tt_pq)),
        # Temperature table
        ("Raw pd.read_excel (Temp Table)", lambda: pd.read_excel(temp_xlsx, sheet_name="Rules")),
        ("Raw pd.read_parquet (Temp Table)", lambda: pd.read_parquet(temp_pq)),
        # Full Pipeline: load_timetable
        ("Full load_timetable (Excel)", lambda: load_timetable(tt_xlsx, prefer_binary=False)),
        ("Full load_timetable (Parquet)", lambda: load_timetable(tt_pq, prefer_binary=True)),
        # Full Pipeline: load_temperature_table
        ("Full load_temp_table (Excel)", lambda: load_temperature_table(temp_xlsx, prefer_binary=False)),
        ("Full load_temp_table (Parquet)", lambda: load_temperature_table(temp_pq, prefer_binary=True)),
    ]

    results = []
    for name, fn in benchmarks:
        print(f"Benchmarking: {name} ...", end="", flush=True)
        stats = run_benchmark_series(name, fn, iterations=args.iterations)
        results.append(stats)
        print(f" done. (Warm Mean: {stats['mean_ms']:.2f} ms, p95: {stats['p95_ms']:.2f} ms)")

    print("\n" + "=" * 105)
    header = (
        f"{'Operation':<35} | {'Cold (ms)':<9} | {'Mean (ms)':<9} | {'p50 (ms)':<9} | "
        f"{'p95 (ms)':<9} | {'p99 (ms)':<9} | {'Min/Max (ms)':<14}"
    )
    print(header)
    print("-" * 105)

    for r in results:
        min_max = f"{r['min_ms']:.2f} / {r['max_ms']:.2f}"
        line = (
            f"{r['name']:<35} | {r['cold_ms']:<9.2f} | {r['mean_ms']:<9.2f} | {r['median_p50_ms']:<9.2f} | "
            f"{r['p95_ms']:<9.2f} | {r['p99_ms']:<9.2f} | {min_max:<14}"
        )
        print(line)

    print("=" * 105)

    # Comparisons and speedups
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS & SLA VERIFICATION (< 20 ms requirement):")
    print("=" * 80)

    comparisons = [
        ("Raw Timetable Read", results[0], results[1]),
        ("Raw Temp Table Read", results[2], results[3]),
        ("Full Timetable Pipeline", results[4], results[5]),
        ("Full Temp Table Pipeline", results[6], results[7]),
    ]

    all_passed = True
    for label, xl_stat, pq_stat in comparisons:
        mean_speedup = xl_stat["mean_ms"] / max(pq_stat["mean_ms"], 0.001)
        p95_speedup = xl_stat["p95_ms"] / max(pq_stat["p95_ms"], 0.001)
        sla_met = pq_stat["p95_ms"] < 20.0 and pq_stat["mean_ms"] < 20.0
        if not sla_met:
            all_passed = False
        status_str = "PASS (< 20 ms)" if sla_met else "FAIL (>= 20 ms)"

        print(f"\n{label.upper()}:")
        print(f"  - Excel Warm Mean:   {xl_stat['mean_ms']:.2f} ms (p95: {xl_stat['p95_ms']:.2f} ms)")
        print(f"  - Parquet Warm Mean: {pq_stat['mean_ms']:.2f} ms (p95: {pq_stat['p95_ms']:.2f} ms)")
        print(f"  - Speedup Factor:    {mean_speedup:.1f}x faster on mean ({p95_speedup:.1f}x on p95)")
        print(f"  - SLA Status:        {status_str}")

    print("\n" + "=" * 80)
    file_stats = [
        ("School Timetable", tt_xlsx, tt_pq),
        ("Water Temperature Control", temp_xlsx, temp_pq),
    ]
    print("STORAGE FOOTPRINT REDUCTION:")
    for name, xl, pq in file_stats:
        xl_sz = xl.stat().st_size
        pq_sz = pq.stat().st_size
        reduction = (1 - pq_sz / xl_sz) * 100
        print(f"  - {name}: {xl_sz:,} bytes (Excel) -> {pq_sz:,} bytes (Parquet) [{reduction:.1f}% reduction]")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
