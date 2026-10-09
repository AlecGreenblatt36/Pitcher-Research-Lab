"""TOTALS-04: where the totals distribution goes wrong. PROD-03 replays (v2-prod3-1000, 2025 and 2026, model outputs on the data
branch): combined and team-level PIT coverage and CRPS, variance ratios, first five innings against the rest, the inning profile,
home-away dependence, extras, subsets by roof, day or night, temperature and wind (public game conditions on the data branch),
park bias, and a negative-binomial challenger on the simulator's own team mean (fitted on one season, scored on the other).
Reads only aggregates. Usage: python tools/totals_diag.py [out.json] (after git fetch origin brl-live-data)."""
import gzip, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.special import gammaln
REPO = str(Path(__file__).resolve().parents[1])
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def replay(tag):
    files = [f for f in subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'], capture_output=True, text=True).stdout.split()
             if f.startswith(f'research/replay-{tag}-') and f.endswith('.jsonl.gz')]
    rows = []
    for f in sorted(files):
        rows += [json.loads(l) for l in gzip.decompress(subprocess.run(['git', '-C', REPO, 'show', 'origin/brl-live-data:' + f], capture_output=True).stdout).decode().splitlines() if l.strip()]
    return pd.DataFrame(rows).drop_duplicates('game_pk')


def pit_disc(pmf, x):
    x = int(x); c = np.cumsum(pmf); lo = c[x - 1] if x >= 1 else 0.0; hi = c[x] if x < len(c) else 1.0
    return 0.5 * (lo + hi)


def crps(pmf, x):
    x = int(x); c = np.cumsum(pmf); k = np.arange(len(pmf)); return float(np.sum((c - (k >= x)) ** 2))


def logscore(pmf, x):
    x = int(x)
    return float(-np.log(max(pmf[x] if x < len(pmf) else 1e-9, 1e-9)))


def dist_stats(pmfs, xs):
    u = np.array([pit_disc(p, x) for p, x in zip(pmfs, xs)])
    mean = np.array([np.dot(np.arange(len(p)), p) for p in pmfs]); var = np.array([np.dot(np.arange(len(p)) ** 2, p) - np.dot(np.arange(len(p)), p) ** 2 for p in pmfs])
    return {'n': int(len(xs)), 'bias_mean_minus_actual': round(float((mean - xs).mean()), 3), 'sq_error': round(float(((mean - xs) ** 2).mean()), 3),
            'corr': round(float(np.corrcoef(mean, xs)[0, 1]), 4),
            'pred_var_mean': round(float(var.mean()), 3), 'resid_var': round(float(((xs - mean) ** 2).mean()), 3), 'actual_var': round(float(xs.var()), 3),
            'middle_half_coverage': round(float(np.mean((u >= 0.25) & (u <= 0.75))), 3), 'below_10': round(float(np.mean(u < 0.1)), 3), 'above_90': round(float(np.mean(u > 0.9)), 3),
            'crps': round(float(np.mean([crps(p, x) for p, x in zip(pmfs, xs)])), 4), 'log_score': round(float(np.mean([logscore(p, x) for p, x in zip(pmfs, xs)])), 4)}


def nb_pmf(mu, r, kmax=30):
    k = np.arange(kmax + 1); p = r / (r + mu)
    lp = gammaln(k + r) - gammaln(r) - gammaln(k + 1) + r * np.log(p) + k * np.log(1 - p)
    out = np.exp(lp); out[-1] += max(0.0, 1 - out.sum()); return out


def pois_pmf(mu, kmax=30):
    k = np.arange(kmax + 1); out = stats.poisson.pmf(k, mu); out[-1] += max(0.0, 1 - out.sum()); return out


def fit_nb(mu_sim, x):
    # actual ~ NB(mean = a + b * mu_sim, dispersion r), maximum likelihood
    def nll(th):
        a, b, lr = th; mu = np.maximum(a + b * mu_sim, 0.05); r = np.exp(lr); p = r / (r + mu)
        return -np.sum(gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(p) + x * np.log(1 - p))
    res = optimize.minimize(nll, [0.0, 1.0, np.log(5.0)], method='Nelder-Mead', options={'maxiter': 4000, 'xatol': 1e-6, 'fatol': 1e-6})
    return res.x


out = {}
D = {}
for season, tag in ((2025, 'v2-prod3-1000-2025'), (2026, 'v2-prod3-1000-2026')):
    d = replay(tag)
    d['home_pmf'] = [np.asarray(h, float) / sum(h) for h in d['home_hist']]; d['away_pmf'] = [np.asarray(h, float) / sum(h) for h in d['away_hist']]
    d['total_pmf'] = [np.convolve(h, a) for h, a in zip(d['home_pmf'], d['away_pmf'])]
    d['total'] = d['home_runs'] + d['away_runs']
    f5 = lambda sc: next((c for i, c in sc if i == 6), None)      # entries are the score at the start of inning i, so the first five innings end at entry 6
    d['home_f5'] = [f5(r['home']) for r in [i['actual_first_scores'] for i in d['innings']]]; d['away_f5'] = [f5(r['away']) for r in [i['actual_first_scores'] for i in d['innings']]]
    d['home_f5_sim'] = [sum(i['sim_mean']['home'][:5]) for i in d['innings']]; d['away_f5_sim'] = [sum(i['sim_mean']['away'][:5]) for i in d['innings']]
    d['home_rest_sim'] = [sum(i['sim_mean']['home'][5:]) for i in d['innings']]; d['away_rest_sim'] = [sum(i['sim_mean']['away'][5:]) for i in d['innings']]
    d['extras'] = [len(i['actual_first_scores']['away']) > 9 for i in d['innings']]
    D[season] = d
    o = {'games': int(len(d))}
    o['combined'] = dist_stats(list(d['total_pmf']), d['total'].to_numpy(float))
    o['home'] = dist_stats(list(d['home_pmf']), d['home_runs'].to_numpy(float)); o['away'] = dist_stats(list(d['away_pmf']), d['away_runs'].to_numpy(float))
    # the dependence check: actual covariance of home and away runs against the simulator's assumed zero
    hm = np.array([np.dot(np.arange(len(p)), p) for p in d['home_pmf']]); am = np.array([np.dot(np.arange(len(p)), p) for p in d['away_pmf']])
    rh = d['home_runs'].to_numpy(float) - hm; ra = d['away_runs'].to_numpy(float) - am
    o['residual_corr_home_away'] = round(float(np.corrcoef(rh, ra)[0, 1]), 4)
    o['residual_corr_interval'] = [round(float(v), 4) for v in np.percentile([np.corrcoef(rh[i], ra[i])[0, 1] for i in (np.random.default_rng(3).integers(0, len(rh), len(rh)) for _ in range(2000))], [2.5, 97.5])]
    # first five against the rest (means only; the replay keeps inning means, not inning distributions)
    ok = d['home_f5'].notna() & d['away_f5'].notna()
    f5a = (d.loc[ok, 'home_f5'] + d.loc[ok, 'away_f5']).to_numpy(float); f5s = (d.loc[ok, 'home_f5_sim'] + d.loc[ok, 'away_f5_sim']).to_numpy(float)
    ra_ = d.loc[ok, 'total'].to_numpy(float) - f5a; rs_ = (d.loc[ok, 'home_rest_sim'] + d.loc[ok, 'away_rest_sim']).to_numpy(float)
    o['first_five'] = {'n': int(ok.sum()), 'bias': round(float((f5s - f5a).mean()), 3), 'sq_error': round(float(((f5s - f5a) ** 2).mean()), 3), 'corr': round(float(np.corrcoef(f5s, f5a)[0, 1]), 4),
                       'actual_mean': round(float(f5a.mean()), 3), 'sim_mean': round(float(f5s.mean()), 3)}
    o['after_five'] = {'n': int(ok.sum()), 'bias': round(float((rs_ - ra_).mean()), 3), 'sq_error': round(float(((rs_ - ra_) ** 2).mean()), 3), 'corr': round(float(np.corrcoef(rs_, ra_)[0, 1]), 4),
                       'actual_mean': round(float(ra_.mean()), 3), 'sim_mean': round(float(rs_.mean()), 3)}
    # the inning profile: actual runs per inning (differences of the running score) against the simulator's inning means, both teams
    prof = {'actual': [], 'sim': []}
    for side in ('home', 'away'):
        act_i = np.full((len(d), 9), np.nan); sim_i = np.array([i['sim_mean'][side][:9] for i in d['innings']], float)
        for g_, inn in enumerate([i['actual_first_scores'][side] for i in d['innings']]):
            sc = dict(inn)
            for k_ in range(1, 10):
                if k_ in sc and (k_ + 1) in sc:
                    act_i[g_, k_ - 1] = sc[k_ + 1] - sc[k_]
        prof['actual'].append(np.nanmean(act_i, 0)); prof['sim'].append(sim_i.mean(0))
    o['inning_profile'] = {'actual_home': [round(float(v), 3) for v in prof['actual'][0]], 'sim_home': [round(float(v), 3) for v in prof['sim'][0]],
                           'actual_away': [round(float(v), 3) for v in prof['actual'][1]], 'sim_away': [round(float(v), 3) for v in prof['sim'][1]],
                           'note': 'the ninth inning for the home team and the last innings of games that ended early are observed only when played'}
    # the correlation a mean can reach: compare with the market line where known (later script); extras
    o['extras'] = {'share': round(float(np.mean(d['extras'])), 4), 'n': int(sum(d['extras'])),
                   'combined_in_extras': dist_stats([p for p, e in zip(d['total_pmf'], d['extras']) if e], d.loc[d['extras'], 'total'].to_numpy(float)),
                   'combined_nine_innings': dist_stats([p for p, e in zip(d['total_pmf'], d['extras']) if not e], d.loc[~np.array(d['extras']), 'total'].to_numpy(float))}
    out[season] = o

# subsets by conditions
cfile = sorted(f for f in subprocess.run(['git', '-C', REPO, 'ls-tree', '-r', '--name-only', 'origin/brl-live-data', 'research'], capture_output=True, text=True).stdout.split()
               if f.startswith('research/conditions-') and f.endswith('.jsonl.gz'))[-1]
cond = pd.DataFrame([json.loads(l) for l in gzip.decompress(subprocess.run(['git', '-C', REPO, 'show', 'origin/brl-live-data:' + cfile], capture_output=True).stdout).decode().splitlines() if l.strip()])
for season in (2025, 2026):
    d = D[season].merge(cond[['game_pk', 'day_night', 'roof', 'temp_f', 'wind_mph', 'wind_dir', 'venue']], on='game_pk', how='left')
    sub = {}
    def add(name, mask):
        mask = np.asarray(mask, bool)
        if mask.sum() >= 80:
            sub[name] = dist_stats([p for p, m in zip(d['total_pmf'], mask) if m], d.loc[mask, 'total'].to_numpy(float))
    add('day', d['day_night'] == 'day'); add('night', d['day_night'] == 'night')
    for r in ('Open', 'Dome', 'Retractable Roof', 'Roof Closed'):
        add('roof_' + r.replace(' ', '_').lower(), d['roof'] == r)
    t = d['temp_f'].astype(float)
    add('temp_under_60', t < 60); add('temp_60_75', (t >= 60) & (t < 75)); add('temp_75_plus', t >= 75)
    w = d['wind_mph'].astype(float)
    add('wind_under_6', w < 6); add('wind_6_12', (w >= 6) & (w < 12)); add('wind_12_plus', w >= 12)
    add('wind_out', d['wind_dir'].astype(str).str.startswith('Out')); add('wind_in', d['wind_dir'].astype(str).str.startswith('In'))
    out[season]['subsets'] = sub
    # by park: bias of the mean total
    g = d.groupby('home').apply(lambda q: pd.Series({'n': len(q), 'bias': float(np.mean([np.dot(np.arange(len(p)), p) for p in q['total_pmf']]) - q['total'].mean())}))
    out[season]['park_bias'] = {'sd_across_parks': round(float(g['bias'].std()), 3), 'worst': {k: round(float(v), 2) for k, v in g['bias'].sort_values().iloc[[0, 1, -2, -1]].items()}}

# negative-binomial challenger at the team level: fitted on one season's (sim mean, actual), scored on the other
for fit_s, score_s in ((2025, 2026), (2026, 2025)):
    df, ds = D[fit_s], D[score_s]
    mu_f = np.r_[[np.dot(np.arange(len(p)), p) for p in df['home_pmf']], [np.dot(np.arange(len(p)), p) for p in df['away_pmf']]]
    x_f = np.r_[df['home_runs'].to_numpy(float), df['away_runs'].to_numpy(float)]
    a, b, lr = fit_nb(mu_f, x_f); r = float(np.exp(lr))
    mu_s = np.r_[[np.dot(np.arange(len(p)), p) for p in ds['home_pmf']], [np.dot(np.arange(len(p)), p) for p in ds['away_pmf']]]
    x_s = np.r_[ds['home_runs'].to_numpy(float), ds['away_runs'].to_numpy(float)].astype(int)
    sim_pmfs = list(ds['home_pmf']) + list(ds['away_pmf'])
    nb_pmfs = [nb_pmf(max(a + b * m, 0.05), r) for m in mu_s]; po_pmfs = [pois_pmf(max(a + b * m, 0.05)) for m in mu_s]
    nb_raw = [nb_pmf(max(m, 0.05), r) for m in mu_s]
    res = {'fit': {'a': round(float(a), 4), 'b': round(float(b), 4), 'dispersion_r': round(r, 3)}}
    for nm, pm in (('simulator_histogram', sim_pmfs), ('nb_on_sim_mean_recentered', nb_pmfs), ('nb_on_sim_mean_raw', nb_raw), ('poisson_on_sim_mean_recentered', po_pmfs)):
        res[nm] = dist_stats(pm, x_s.astype(float))
    # combined totals under independence for each
    tot = ds['total'].to_numpy(float); nh = len(ds)
    for nm, pm in (('simulator_histogram', sim_pmfs), ('nb_on_sim_mean_recentered', nb_pmfs)):
        tp = [np.convolve(pm[i], pm[nh + i]) for i in range(nh)]
        res[nm + '_combined'] = dist_stats(tp, tot)
    out[f'nb_challenger_fit_{fit_s}_score_{score_s}'] = res
if OUT:
    json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
