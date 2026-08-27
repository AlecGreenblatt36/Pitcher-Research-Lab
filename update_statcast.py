"""Command-line Statcast sync for any MLB pitcher.

Examples:
    python update_statcast.py <mlbam_id>
    python update_statcast.py <mlbam_id> --full
"""
from __future__ import annotations

import argparse
import json

from pitcher_sync import sync_pitcher_statcast


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Statcast data for an MLB pitcher.")
    parser.add_argument("pitcher_id", type=int, help="MLBAM pitcher ID")
    parser.add_argument("--full", action="store_true", help="Rebuild the pitcher's full Statcast history.")
    args = parser.parse_args()
    result = sync_pitcher_statcast(args.pitcher_id, force_full=args.full)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
