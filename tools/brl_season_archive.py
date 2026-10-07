"""Season archive for the page: every regular-season game with the model's replay numbers, the market and the result.

The page uses it to open any past date and game (the box score and plays come from MLB's public feed in the
browser). Numbers are model outputs from a frozen replay with prior-date inputs only, labeled as a replay on the
page; the live record never counts them. Output: brl_live/archive/season-<year>.json (no player data beyond the
starters' names, which are public).

Inputs (local, not in the repository): the replay per-game files on the ledger branch (git ref), the market history
files, game conditions, the reconstructed games table, team-model features and the public Chadwick name register.

Usage: python tools/brl_season_archive.py --season 2026 --replay-tag v2-1000-2026 --scratch <dir> [--ref origin/brl-live-data]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brl_live import headline as hl  # noqa: E402

MAX_RUNS = 13          # histogram bins 0..12 and 13 or more
NAMES = {'ATH': 'Athletics', 'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks', 'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox',
         'CHC': 'Chicago Cubs', 'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies', 'CWS': 'Chicago White Sox',
         'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals', 'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
         'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets', 'NYY': 'New York Yankees',
         'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners', 'SF': 'San Francisco Giants',
         'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays', 'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'}


def jl(raw: bytes) -> list:
    return [json.loads(l) for l in gzip.decompress(raw).decode().splitlines() if l.strip()]


def implied(o):
    o = float(o)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def vigfree(a, b):
    x, y = implied(a), implied(b)
    return x / (x + y)


def replay_rows(tag: str, ref: str) -> pd.DataFrame:
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', ref, 'research'], capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in names if f.startswith(f'research/replay-{tag}-') and f.endswith('.jsonl.gz')]
    if not files:
        raise SystemExit(f'no replay files for {tag} on {ref}')
    rows = []
    for f in sorted(files):
        rows += jl(subprocess.run(['git', '-C', str(ROOT), 'show', f'{ref}:{f}'], capture_output=True, check=True).stdout)
    return pd.DataFrame(rows).drop_duplicates('game_pk')


def hist(counts, n) -> list:
    c = list(counts[:MAX_RUNS]) + [sum(counts[MAX_RUNS:])]
    c += [0] * (MAX_RUNS + 1 - len(c))
    return [int(round(1000 * x / n)) for x in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--replay-tag', required=True)
    ap.add_argument('--scratch', required=True)
    ap.add_argument('--ref', default='origin/brl-live-data')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    S = Path(a.scratch)
    r = replay_rows(a.replay_tag, a.ref)
    games = pd.read_json(S / 'harness/games_all.json', convert_dates=False)[['game_pk', 'home_starter', 'away_starter']]
    cond = pd.DataFrame(jl((S / 'harness/conditions.jsonl.gz').read_bytes()))[['game_pk', 'start', 'venue']]
    tf = pd.read_csv(S / f'harness/teamfeat_{a.season}.csv')
    names = pd.read_csv(S / 'data_root/pa_model_reference/chadwick_mlbam_names.csv.gz')
    names = names.dropna(subset=['key_mlbam']).assign(key_mlbam=lambda x: x['key_mlbam'].astype(int))
    name_of = {int(k): f'{f} {l}'.strip() for k, f, l in zip(names['key_mlbam'], names['name_first'].fillna(''), names['name_last'].fillna(''))}
    if a.season == 2026:
        mk = pd.DataFrame(jl((S / 'harness/market_2026b.jsonl.gz').read_bytes()))
        mk['p_mk'] = mk['p_home']; mk['p_mk_open'] = mk.get('p_home_open')
        mk['total'] = mk['over_under']
        mk['p_over'] = [vigfree(o, u) if pd.notna(o) and pd.notna(u) else None for o, u in zip(mk['over_odds'], mk['under_odds'])]
        mk = mk.rename(columns={'home': 'home_name', 'away': 'away_name'})
    else:
        mk = pd.DataFrame(jl((S / 'harness/market_sbr_b.jsonl.gz').read_bytes()))
        mk = mk[mk['season'] == a.season]
        mk['p_mk'] = mk['p_close_dk'].fillna(mk['p_close_avg']); mk['p_mk_open'] = mk['p_open_dk'].fillna(mk['p_open_avg'])
        mk['total'] = [t[0] if isinstance(t, list) else None for t in mk['dk_total_close']]
        mk['p_over'] = [vigfree(t[1], t[2]) if isinstance(t, list) else None for t in mk['dk_total_close']]
        mk = mk.rename(columns={'home': 'home_name', 'away': 'away_name'})
    mk = mk[['game_pk', 'p_mk', 'p_mk_open', 'total', 'p_over', 'home_name', 'away_name']]
    d = r.merge(games, on='game_pk', how='left').merge(cond, on='game_pk', how='left').merge(tf, on='game_pk', how='left').merge(mk, on='game_pk', how='left')
    version = next(v for v in hl.PARAMS['versions'] if v['name'] == 'taught-v2')
    out = []
    for g in d.sort_values(['date', 'start', 'game_pk']).itertuples():
        p_sim = (g.home_wins + 0.5 * g.ties + 0.5) / (g.n + 1.0)
        rec = {'pk': int(g.game_pk), 'd': g.date, 't': g.start if isinstance(g.start, str) else None, 'v': g.venue if isinstance(g.venue, str) else None,
               'a': g.away, 'h': g.home, 'an': g.away_name if isinstance(g.away_name, str) else None, 'hn': g.home_name if isinstance(g.home_name, str) else None,
               'as': name_of.get(int(g.away_starter)) if pd.notna(g.away_starter) else None, 'hs': name_of.get(int(g.home_starter)) if pd.notna(g.home_starter) else None,
               'fa': int(g.away_runs), 'fh': int(g.home_runs), 's': round(p_sim, 4), 'ra': hist(g.away_hist, g.n), 'rh': hist(g.home_hist, g.n)}
        if pd.notna(g.p_team):
            tm = {'p_home': float(g.p_team), 'starter_adjust': {'home': {'factor': math.exp(g.sp_home)}, 'away': {'factor': math.exp(g.sp_away)}},
                  'ratings': {'home': {'offense': math.exp(g.off_home), 'defense': math.exp(g.def_home)}, 'away': {'offense': math.exp(g.off_away), 'defense': math.exp(g.def_away)}}}
            rec['tm'] = round(float(g.p_team), 4)
            p_ours, _ = hl.ours(p_sim, tm, version)
        else:
            p_ours = p_sim
        rec['o'] = round(p_ours, 4)
        if pd.notna(g.p_mk):
            rec['m'] = round(float(g.p_mk), 4)
            if pd.notna(g.p_mk_open):
                rec['mo'] = round(float(g.p_mk_open), 4)
            rec['hl'] = round(hl.headline(p_ours, float(g.p_mk), version)[0], 4)
        else:
            rec['hl'] = rec['o']
        if pd.notna(g.total):
            rec['ou'] = float(g.total)
            if g.p_over is not None and not (isinstance(g.p_over, float) and math.isnan(g.p_over)):
                rec['po'] = round(float(g.p_over), 4)
        out.append(rec)
    # Team names where the market file has none (late 2025): the name the same club carries elsewhere in the season.
    known = {}
    for rec in out:
        for side in ('a', 'h'):
            if rec.get(side + 'n'):
                known.setdefault(rec[side], rec[side + 'n'])
    for rec in out:
        for side in ('a', 'h'):
            if not rec.get(side + 'n'):
                rec[side + 'n'] = known.get(rec[side]) or NAMES.get(rec[side])
    doc = {'schema': 'brl.season-archive.v1', 'season': a.season, 'replay': a.replay_tag, 'recipe': version['name'], 'games': out,
           'note': ('Model replay: every game re-run after the season with only information from before that game '
                    '(lineups and starters as they played, prior-date history). Not live forecasts; the Record counts only live ones.')}
    dest = Path(a.out) if a.out else ROOT / 'brl_live' / 'archive' / f'season-{a.season}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, separators=(',', ':')))
    print(dest, len(out), dest.stat().st_size)


if __name__ == '__main__':
    main()
