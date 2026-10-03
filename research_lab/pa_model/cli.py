from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import PAConfig
from .data import build_plate_appearances, download_statcast, save_pa_dataset
from .pipeline import run_benchmark


def _parse_years(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def _config(args: argparse.Namespace) -> PAConfig:
    return PAConfig.with_years(_parse_years(args.train_years), _parse_years(args.validation_years), _parse_years(args.test_years), bootstrap_replicates=args.bootstrap_replicates)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pa-model", description="Chronological MLB plate-appearance benchmark")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--start", required=True); download.add_argument("--end", required=True); download.add_argument("--cache-dir", required=True)
    build = sub.add_parser("build-pa")
    build.add_argument("--input-dir", required=True); build.add_argument("--output", required=True)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--pa-file", required=True); benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--train-years", default="2023"); benchmark.add_argument("--validation-years", default="2024"); benchmark.add_argument("--test-years", default="2025"); benchmark.add_argument("--bootstrap-replicates", type=int, default=1000)
    run_all = sub.add_parser("run-all")
    run_all.add_argument("--start", default="2023-03-20"); run_all.add_argument("--end", default="2025-11-05"); run_all.add_argument("--work-dir", required=True)
    run_all.add_argument("--train-years", default="2023"); run_all.add_argument("--validation-years", default="2024"); run_all.add_argument("--test-years", default="2025"); run_all.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "download":
        print(json.dumps({"files": [str(file) for file in download_statcast(args.start, args.end, args.cache_dir)]}, indent=2)); return
    if args.command == "build-pa":
        pa, report = build_plate_appearances(sorted(Path(args.input_dir).glob("statcast_*.csv.gz")))
        save_pa_dataset(pa, args.output, report); print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.command == "benchmark":
        result = run_benchmark(pd.read_csv(args.pa_file, low_memory=False, parse_dates=["game_date", "date_key"]), args.output_dir, _config(args))
        print(json.dumps(result["promotion"], indent=2, sort_keys=True)); return
    config, work = _config(args), Path(args.work_dir)
    files = download_statcast(args.start, args.end, work / "raw", config)
    pa, report = build_plate_appearances(files)
    save_pa_dataset(pa, work / "plate_appearances.csv.gz", report)
    result = run_benchmark(pa, work / "artifacts", config)
    print(json.dumps(result["promotion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
