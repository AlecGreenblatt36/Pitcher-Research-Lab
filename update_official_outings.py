"""Command-line official outing sync for any cached MLB pitcher."""
from __future__ import annotations

import argparse
import json

from official_sync import sync_official_outings


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync official MLB boxscore lines for a pitcher.")
    parser.add_argument("pitcher_id", type=int, help="MLBAM pitcher ID")
    parser.add_argument("--season", type=int, help="Specific season to sync. Defaults to the latest cached season.")
    args = parser.parse_args()
    result = sync_official_outings(args.pitcher_id, args.season)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
