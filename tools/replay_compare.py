"""Compare two full-season replays, paired by game, on what a model change is supposed to move: the mean total's squared
error, the mean total itself, the cross-season reversal of park residuals (TOTALS-06), P(over the closing line) and the win
Brier. Data-branch aggregates only (the per-game replay rows and the public market files).
Usage: python tools/replay_compare.py <reference tag prefix> <candidate tag prefix> [out.json]
  tags are completed with -<season> for 2025 and 2026, e.g. v2-prod3-1000 and v5-park-1000."""
import gzip, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = str(Path(__file__).resolve().parents[1])
REF, NEW = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None


def files(prefix):
    return sorted(f for f in subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'], capture_output=True, text=True).stdout.split()
                  if f.startswith(prefix) and f.endswith('.jsonl.gz'))


def rows(f):
    return [json.loads(l) for l in gzip.decompress(subprocess.run(['git', '-C', REPO, 'show', 'origin/brl-live-data:' + f], capture_output=True).stdout).decode().splitlines() if l.strip()]


def replay(tag):
    d = pd.DataFrame([r for f in files(f'research/replay-{tag}-') for r in rows(f)]).drop_duplicates('game_pk')
    d['p_sim'] = (d['home_wins'] + 0.5 * d['ties'] + 0.5) / (d['n'] + 1.0)
    d['y'] = (d['home_runs'] > d['away_runs']).astype(float)
    d['mean_total'] = [np.dot(np.arange(len(h)), h) / sum(h) + np.dot(np.arange(len(a)), a) / sum(a) for h, a in zip(d['home_hist'], d['away_hist'])]
    d['actual'] = d['home_runs'] + d['away_runs']; d['resid'] = d['actual'] - d['mean_total']
    d['total_pmf'] = [np.convolve(np.asarray(h, float) / sum(h), np.asarray(a, float) / sum(a)) for h, a in zip(d['home_hist'], d['away_hist'])]
    return d


def lines(season):
    if season == 2026:
        mk = pd.DataFrame(rows(files('research/market-2026-')[-1])); mk = mk[mk['over_under'].notna()]
        return mk[['game_pk', 'over_under']]
    mk = pd.DataFrame(rows(files('research/market-sbr-')[-1])); mk = mk[(mk['season'] == season) & mk['dk_total_close'].notna()]
    mk['over_under'] = [t[0] for t in mk['dk_total_close']]
    return mk[['game_pk', 'over_under']]


def p_over(pmf, line):
    k = np.arange(len(pmf)); over = float(pmf[k > line].sum()); under = float(pmf[k < line].sum())
    return over / (over + under) if over + under > 0 else 0.5


def paired(dates, diff, reps=4000, seed=7):
    u = np.unique(dates); idx = {d_: np.flatnonzero(dates == d_) for d_ in u}; rng = np.random.default_rng(seed)
    vals = [float(np.mean(np.concatenate([diff[idx[d_]] for d_ in rng.choice(u, len(u))]))) for _ in range(reps)]
    return round(float(diff.mean()), 5), [round(float(v), 5) for v in np.percentile(vals, [2.5, 97.5])]


out = {'reference': REF, 'candidate': NEW}
home_resid = {}
pooled = {'sq': [], 'win': [], 'over': [], 'dates': {'sq': [], 'win': [], 'over': []}}
for season in (2025, 2026):
    a = replay(f'{REF}-{season}'); b = replay(f'{NEW}-{season}')
    m = a.merge(b, on='game_pk', suffixes=('_ref', '_new'))
    m = m[m['date_ref'].notna()]
    act = m['actual_ref'].to_numpy(float); dates = m['date_ref'].to_numpy()
    sq_ref = (m['mean_total_ref'] - act) ** 2; sq_new = (m['mean_total_new'] - act) ** 2
    win_ref = (m['p_sim_ref'] - m['y_ref']) ** 2; win_new = (m['p_sim_new'] - m['y_ref']) ** 2
    res = {'games': int(len(m)), 'mean_total': {'actual': round(float(act.mean()), 3), 'reference': round(float(m['mean_total_ref'].mean()), 3), 'candidate': round(float(m['mean_total_new'].mean()), 3)},
           'sq_error': {'reference': round(float(sq_ref.mean()), 4), 'candidate': round(float(sq_new.mean()), 4), 'candidate_minus_reference': paired(dates, (sq_new - sq_ref).to_numpy(float))},
           'win_brier': {'reference': round(float(win_ref.mean()), 5), 'candidate': round(float(win_new.mean()), 5), 'candidate_minus_reference': paired(dates, (win_new - win_ref).to_numpy(float))}}
    pooled['sq'].append((sq_new - sq_ref).to_numpy(float)); pooled['win'].append((win_new - win_ref).to_numpy(float)); pooled['dates']['sq'].append(dates); pooled['dates']['win'].append(dates)
    mk = m.merge(lines(season), on='game_pk', how='inner')
    if len(mk):
        L = mk['over_under'].to_numpy(float); nb = mk['actual_ref'].to_numpy(float) != L
        yo = (mk['actual_ref'].to_numpy(float) > L)[nb].astype(float)
        po_ref = np.array([p_over(p, l) for p, l in zip(mk['total_pmf_ref'], L)])[nb]; po_new = np.array([p_over(p, l) for p, l in zip(mk['total_pmf_new'], L)])[nb]
        d_over = (po_new - yo) ** 2 - (po_ref - yo) ** 2
        res['p_over_brier'] = {'games': int(nb.sum()), 'reference': round(float(np.mean((po_ref - yo) ** 2)), 5), 'candidate': round(float(np.mean((po_new - yo) ** 2)), 5),
                               'candidate_minus_reference': paired(mk['date_ref'].to_numpy()[nb], d_over)}
        pooled['over'].append(d_over); pooled['dates']['over'].append(mk['date_ref'].to_numpy()[nb])
    for tag, col in (('reference', 'resid_ref'), ('candidate', 'resid_new')):
        home_resid.setdefault(tag, {})[season] = m.groupby('home_ref')[col].mean()
    res['park_sd'] = {'reference': round(float(home_resid['reference'][season].std()), 3), 'candidate': round(float(home_resid['candidate'][season].std()), 3)}
    out[str(season)] = res
for tag in ('reference', 'candidate'):
    h25, h26 = home_resid[tag][2025], home_resid[tag][2026]; c = h25.index.intersection(h26.index)
    out.setdefault('park_residual_cross_season_corr', {})[tag] = round(float(np.corrcoef(h25[c], h26[c])[0, 1]), 3)
out['pooled'] = {'sq_error_candidate_minus_reference': paired(np.concatenate(pooled['dates']['sq']), np.concatenate(pooled['sq'])),
                 'win_brier_candidate_minus_reference': paired(np.concatenate(pooled['dates']['win']), np.concatenate(pooled['win']))}
if pooled['over']:
    out['pooled']['p_over_brier_candidate_minus_reference'] = paired(np.concatenate(pooled['dates']['over']), np.concatenate(pooled['over']))
print(json.dumps(out, indent=1))
if OUT:
    Path(OUT).write_text(json.dumps(out, indent=1))
