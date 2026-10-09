"""TOTALS-06: park offsets for the simulator's mean total, estimated on one season of the PROD-03 replays (v2-prod3-1000) and applied to
the other. Reads only data-branch aggregates (per-game replay rows and the public market files). Park = the home team's code (the park
proxy the replay rows carry). Empirical-Bayes shrinkage: offset_p = m_p * tau2 / (tau2 + s2 / n_p), with m_p the park's mean residual
(actual minus the simulator's mean), s2 the residual variance and tau2 the between-park variance net of noise. Paired squared error with
a date bootstrap, for the simulator's own mean and for the taught expected total (TOTALS-03's fixed weights), then the diagnostic that
decided the reading: whether park residuals persist or reverse across seasons, and whether they are team effects.
Usage: python tools/totals_park.py [out.json]"""
import gzip, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = str(Path(__file__).resolve().parents[1])
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def files(prefix):
    return sorted(f for f in subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'], capture_output=True, text=True).stdout.split()
                  if f.startswith(prefix) and f.endswith('.jsonl.gz'))


def rows(f):
    return [json.loads(l) for l in gzip.decompress(subprocess.run(['git', '-C', REPO, 'show', 'origin/brl-live-data:' + f], capture_output=True).stdout).decode().splitlines() if l.strip()]


def replay(tag):
    d = pd.DataFrame([r for f in files(f'research/replay-{tag}-') for r in rows(f)]).drop_duplicates('game_pk')
    d['mean_total'] = [np.dot(np.arange(len(h)), h) / sum(h) + np.dot(np.arange(len(a)), a) / sum(a) for h, a in zip(d['home_hist'], d['away_hist'])]
    d['actual'] = d['home_runs'] + d['away_runs']; d['resid'] = d['actual'] - d['mean_total']; d['park'] = d['home']
    return d


def lines(season):
    if season == 2026:
        mk = pd.DataFrame(rows(files('research/market-2026-')[-1])); mk = mk[mk['over_under'].notna()]
        return mk[['game_pk', 'over_under']]
    mk = pd.DataFrame(rows(files('research/market-sbr-')[-1])); mk = mk[(mk['season'] == season) & mk['dk_total_close'].notna()]
    mk['over_under'] = [t[0] for t in mk['dk_total_close']]
    return mk[['game_pk', 'over_under']]


def offsets(train):
    s2 = float(train['resid'].var()); g = train.groupby('park')['resid'].agg(['mean', 'size'])
    noise = float((s2 / g['size']).mean()); tau2 = max(0.0, float(g['mean'].var()) - noise)
    off = g['mean'] * tau2 / (tau2 + s2 / g['size'])
    return off, {'resid_var': round(s2, 2), 'sd_park_means': round(float(g['mean'].std()), 3), 'noise_sd': round(noise ** 0.5, 3), 'tau': round(tau2 ** 0.5, 3),
                 'largest': {k: round(float(v), 2) for k, v in off.sort_values().iloc[[0, 1, -2, -1]].items()}}


def paired(test, a, b, reps=3000, seed=7):
    diff = a - b; dates = test['date'].to_numpy(); u = np.unique(dates); idx = {d_: np.flatnonzero(dates == d_) for d_ in u}
    rng = np.random.default_rng(seed); vals = [float(np.mean(np.concatenate([diff[idx[d_]] for d_ in rng.choice(u, len(u))]))) for _ in range(reps)]
    return round(float(diff.mean()), 4), [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


D = {s: replay(f'v2-prod3-1000-{s}').merge(lines(s), on='game_pk', how='left') for s in (2025, 2026)}
out = {}
for train_s, test_s in ((2025, 2026), (2026, 2025)):
    off, info = offsets(D[train_s]); t = D[test_s].copy(); t['off'] = t['park'].map(off).fillna(0.0)
    act = t['actual'].to_numpy(float); sim = t['mean_total'].to_numpy(float); adj = sim + t['off'].to_numpy(float)
    res = {'trained_on': train_s, 'scored_on': test_s, 'offsets': info, 'games': int(len(t)), 'sim_sq_error': round(float(np.mean((sim - act) ** 2)), 4),
           'sim_adjusted_sq_error': round(float(np.mean((adj - act) ** 2)), 4), 'sim_improvement': paired(t, (sim - act) ** 2, (adj - act) ** 2)}
    gt = t.groupby('park')['resid'].mean(); common = gt.index.intersection(off.index)
    res['test_park_mean_on_offset'] = {'slope': round(float(np.polyfit(off[common].to_numpy(), gt[common].to_numpy(), 1)[0]), 3), 'corr': round(float(np.corrcoef(off[common], gt[common])[0, 1]), 3), 'parks': int(len(common))}
    m = t[t['over_under'].notna()]
    if len(m):
        a2 = m['actual'].to_numpy(float); s_ = m['mean_total'].to_numpy(float); o_ = m['off'].to_numpy(float); L = m['over_under'].to_numpy(float)
        taught = 0.32 * s_ + 0.78 * L - 0.543; taught_adj = 0.32 * (s_ + o_) + 0.78 * L - 0.543
        res['taught'] = {'games': int(len(m)), 'sq_error': round(float(np.mean((taught - a2) ** 2)), 4), 'sq_error_adjusted': round(float(np.mean((taught_adj - a2) ** 2)), 4),
                         'improvement': paired(m, (taught - a2) ** 2, (taught_adj - a2) ** 2), 'line_sq_error': round(float(np.mean((L - a2) ** 2)), 4)}
        gl = m.assign(lr=a2 - L).groupby('park')['lr'].agg(['mean', 'size']); s2l = float((a2 - L).var())
        res['line_park'] = {'sd_park_means': round(float(gl['mean'].std()), 3), 'noise_sd': round(float((s2l / gl['size']).mean()) ** 0.5, 3)}
    out[f'{train_s}->{test_s}'] = res
diag = {}
for season in (2025, 2026):
    d = D[season]; h = d.groupby('home')['resid'].mean(); a = d.groupby('away')['resid'].mean(); c = h.index.intersection(a.index)
    diag[str(season)] = {'corr_home_vs_away_residual_by_team': round(float(np.corrcoef(h[c], a[c])[0, 1]), 3), 'teams': int(len(c)), 'sd_home': round(float(h.std()), 3), 'sd_away': round(float(a.std()), 3),
                         'monthly_residual': {k: round(float(v), 3) for k, v in d.groupby(d['date'].str[5:7])['resid'].mean().items()}}
h25, h26 = D[2025].groupby('home')['resid'].mean(), D[2026].groupby('home')['resid'].mean(); a25, a26 = D[2025].groupby('away')['resid'].mean(), D[2026].groupby('away')['resid'].mean()
c = h25.index.intersection(h26.index)
diag['across_seasons'] = {'home_2025_vs_home_2026': round(float(np.corrcoef(h25[c], h26[c])[0, 1]), 3), 'away_2025_vs_away_2026': round(float(np.corrcoef(a25[c], a26[c])[0, 1]), 3),
                         'team_total_2025_vs_2026': round(float(np.corrcoef((h25[c] + a25[c]) / 2, (h26[c] + a26[c]) / 2)[0, 1]), 3)}
out['diagnostic'] = diag
print(json.dumps(out, indent=1))
if OUT:
    Path(OUT).write_text(json.dumps(out, indent=1))
