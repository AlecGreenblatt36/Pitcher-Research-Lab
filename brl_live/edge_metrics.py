"""Pregame skill diagnostics. Observation/scoring only; never changes the engine.

All comparators are frozen inside the publicly timestamped box forecast. Count
forecasts use fair ensemble CRPS; fixed comparator PMFs use ordinary CRPS. A
reliever not used is a zero, not a case to discard. In-game forecasts are excluded.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re

METRICS = (
    ('pitcher_K', 'Pitcher strikeouts', 'CRPS'),
    ('pitcher_BB', 'Pitcher walks', 'CRPS'),
    ('pitcher_PC', 'Pitcher pitches', 'CRPS'),
    ('pitcher_IP', 'Pitcher innings', 'CRPS'),
    ('reliever_appears', 'Relievers used', 'Brier'),
    ('team_runs', 'Team runs', 'CRPS'),
    ('high_total', 'High or low scoring', 'Brier'),
)
SCHEMA = 'brl.skill-baselines.v1'
HIGH_TOTAL = 9  # Fixed before scoring: 9+ combined runs vs 8 or fewer.


def utc(value):
    v = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError('Timestamp must include timezone')
    return v.astimezone(timezone.utc)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def histogram(d):
    n, rows = d['n'], d['counts']
    if type(n) is not int or n < 2 or not rows:
        raise ValueError('At least two complete independent worlds required')
    if len({x for x, _ in rows}) != len(rows):
        raise ValueError('Duplicate histogram support')
    if any(type(x) is not int or x < 0 or type(c) is not int or c <= 0 for x, c in rows):
        raise ValueError('Invalid integer histogram')
    if sum(c for _, c in rows) != n:
        raise ValueError('Incomplete histogram')
    return n, sorted(rows)


def ensemble_crps(d, observed):
    n, rows = histogram(d)
    cumulative = weighted = pairs = 0
    for x, count in rows:
        pairs += count * (x * cumulative - weighted)
        cumulative += count
        weighted += x * count
    return sum(c * abs(x - observed) for x, c in rows) / n - pairs / (n * (n - 1))


def discrete_crps(pmf, observed):
    """Exact score of the fitted comparator distribution, not a Monte Carlo sample."""
    if not pmf or len({x for x, _ in pmf}) != len(pmf):
        raise ValueError('Invalid baseline PMF support')
    if any(type(x) is not int or x < 0 or not math.isfinite(p) or p <= 0 for x, p in pmf):
        raise ValueError('Invalid baseline PMF')
    if abs(sum(p for _, p in pmf) - 1.0) > 1e-10:
        raise ValueError('Baseline PMF does not sum to one')
    cumulative = weighted = pairs = 0.0
    for x, p in sorted(pmf):
        pairs += p * (x * cumulative - weighted)
        cumulative += p
        weighted += x * p
    return sum(p * abs(x - observed) for x, p in pmf) - pairs


def occurrence_score(p, observed, n=None):
    if not math.isfinite(p) or not 0 <= p <= 1 or observed not in (0, 1):
        raise ValueError('Invalid event forecast/label')
    return (p - observed) ** 2 - (p * (1 - p) / (n - 1) if n else 0.0)


def nb_crps(mean, alpha, observed):
    """Discrete NB CRPS via CDF sum; returns a bound on the omitted tail.

No empirical-sample correction is applied to an analytic baseline. The bound
is numerical truncation error, NOT uncertainty about its fitted parameters.
    """
    import numpy as np
    from scipy.stats import nbinom
    if not (math.isfinite(mean) and math.isfinite(alpha) and mean > 0 and alpha > 0):
        raise ValueError('Invalid frozen NB parameters')
    if type(observed) is not int or observed < 0:
        raise ValueError('Team runs must be a nonnegative integer')
    r, p = 1 / alpha, 1 / (1 + alpha * mean)
    cap = max(observed, int(nbinom.ppf(1 - 1e-12, r, p)))
    if cap > 100000:
        raise ValueError('NB numerical tail limit exceeded')
    values = np.arange(cap + 1)
    score = float(np.sum((nbinom.cdf(values, r, p) - (values >= observed)) ** 2))
    t = cap + 1
    sf = float(nbinom.sf(t, r, p))
    residual_mean = max(0.0, mean * float(nbinom.sf(t - 1, r + 1, p)) - t * sf)
    return score, sf * residual_mean


def nb_high_probability(away_mean, home_mean, alpha):
    import numpy as np
    from scipy.stats import nbinom
    r = 1 / alpha
    a = nbinom.pmf(np.arange(HIGH_TOTAL), r, 1 / (1 + alpha * away_mean))
    h = nbinom.pmf(np.arange(HIGH_TOTAL), r, 1 / (1 + alpha * home_mean))
    return float(1 - np.convolve(a, h)[:HIGH_TOTAL].sum())


def _empirical(values):
    c = Counter(values)
    return [[int(x), n / len(values)] for x, n in sorted(c.items())]


def _relief_pmf(team_cases, metric, n_current):
    # Each prior team box gets equal weight. Randomly assign its relief lines
    # without replacement to the current pregame pool, leaving the rest unused.
    # If it used more arms than the current pool, uniformly select that many
    # historical lines. This defines a transparent marginal comparator, not a
    # medical readiness model or a competing coherent-game engine.
    out = defaultdict(float)
    for case in team_cases:
        pen = case[1:]
        if any(r.get(metric) is None for r in pen):
            continue
        m = len(pen)
        out[0] += 1 - min(m, n_current) / n_current
        for row in pen:
            out[int(row[metric])] += 1 / max(m, n_current)
    mass = sum(out.values())
    if mass == 0:
        return None
    return [[x, p / mass] for x, p in sorted(out.items()) if p > 0]


def freeze_skill_baselines(box, prior_boxes, team_baseline, as_of):
    """prior_boxes: [{date, available_at, box: reconciled prior final box}].

Caller must supply a fully audited prior-source set. Future/target records are
rejected instead of quietly filtered. Counts come from a role-only resampling
baseline; team runs and high/low use the existing fair team's NB parameters.
    """
    from zoneinfo import ZoneInfo
    day = box['date']
    # Built on the game's date, or earlier for an early call the day before (it then rests on less history, never more);
    # an origin after the game's date is refused.
    if utc(as_of).astimezone(ZoneInfo('America/New_York')).date().isoformat() > day:
        raise ValueError('Baseline origin after the forecast date')
    cases, seen = [], set()
    for row in prior_boxes:
        pk = row['box']['game_pk']
        if pk == box['game_pk'] or row['date'] >= day or utc(row['available_at']) > utc(as_of):
            raise ValueError('Future or target-game baseline input')
        if pk in seen:
            raise ValueError('Duplicate prior game in baseline')
        seen.add(pk)
        for side in ('away', 'home'):
            lines = row['box']['pitching'][side]
            if not lines:
                raise ValueError('Prior complete box has no starting pitcher')
            cases.append(lines)
    if not cases:
        raise ValueError('No earlier complete pitching boxes for baseline')
    if (team_baseline['game_pk'] != box['game_pk'] or team_baseline['prior_through'] >= day
            or team_baseline['dispersion_fit']['training_through'] >= day):
        raise ValueError('Invalid fair team baseline identity/cutoff')
    output = {
        'schema': SCHEMA, 'lane': 'pregame', 'as_of': as_of,
        'history_through': max(row['date'] for row in prior_boxes),
        'source_available_through': max((row['available_at'] for row in prior_boxes), key=utc),
        'n_prior_games': len(seen), 'n_prior_team_boxes': len(cases),
        'history_identity_sha256': canonical_hash(prior_boxes),
        'recipe': 'Equal prior team-box role resampling; exchangeable current bullpen; analytic existing team NB',
        'high_scoring_threshold': HIGH_TOTAL, 'pitching': {},
        'team_runs': {s: {'mean': team_baseline[s + '_mean_runs'],
                         'alpha': team_baseline['nb_alpha']} for s in ('away', 'home')},
        'team_history_through': team_baseline['prior_through'],
        'team_fit_through': team_baseline['dispersion_fit']['training_through'],
        'team_prior_games': team_baseline['prior_games'],
    }
    output['high_total_probability'] = nb_high_probability(
        team_baseline['away_mean_runs'], team_baseline['home_mean_runs'], team_baseline['nb_alpha'])
    for side in ('away', 'home'):
        players = box['teams'][side]['pitching']
        starter = str(box['starters'][side]['player_id'])
        pen_size = sum(str(p['player_id']) != starter for p in players)
        index = {}
        for player in players:
            is_starter = str(player['player_id']) == starter
            pmfs = {}
            for metric in ('K', 'BB', 'PC', 'outs'):
                vals = [team[0][metric] for team in cases if team[0].get(metric) is not None]
                pmfs[metric] = (_empirical(vals) if vals else None) if is_starter else _relief_pmf(cases, metric, pen_size)
            q = 1.0 if is_starter else sum(min(len(team) - 1, pen_size) / pen_size for team in cases) / len(cases)
            index[str(player['player_id'])] = {'starter': is_starter, 'pmf': pmfs,
                                              'appearance_probability': q}
        output['pitching'][side] = index
    return output


def prior_boxes_from_cache(cache, index, origin):
    """Raw official inputs never leave encrypted source storage."""
    from .history_refresh import Source, previous_day
    from .boxscore import parse_actual_box
    rows = {}
    for date, ref in sorted(index['days'].items()):
        if date > previous_day(origin).isoformat():
            raise ValueError('Same-day/future cached game')
        day = cache.load_day(ref['id'])
        for item in day['sources']:
            source = Source.from_dict(item)
            source.check(origin)
            if '/feed/live' not in source.url or '?' in source.url:
                continue
            feed = json.loads(source.body)
            if feed['gameData']['status']['abstractGameState'] != 'Final':
                continue
            if feed['gameData']['datetime']['officialDate'] != date:
                raise ValueError('Prior game date mismatch')
            # Unresolved or future final labels must not enter a fitted baseline.
            plays = feed['liveData']['plays']['allPlays']
            if not plays or any(not p['about']['isComplete'] or
                    utc(p['about']['endTime']) > utc(source.finished_at) for p in plays):
                raise ValueError('Unresolved prior game label')
            box = parse_actual_box(feed, feed['gamePk'], source.finished_at)
            row = {'date': date, 'available_at': source.finished_at, 'box': box}
            old = rows.get(feed['gamePk'])
            if old and old != row:
                raise ValueError('Conflicting prior final boxes')
            rows[feed['gamePk']] = row
    return list(rows.values())


def _published(box, pub, actual):
    from app.common import content_hash
    if (not pub or not re.fullmatch(r'[0-9a-f]{40}', str(pub.get('commit', '')))
            or pub.get('box_sha256') != content_hash(box)):
        raise ValueError('Missing or changed published box')
    if not utc(box['forecast_origin']) <= utc(box['saved_at']) <= utc(pub['published_at']) < utc(actual['first_pitch_observed_at']):
        raise ValueError('Box not published before first pitch')
    if box['team_ids'] != actual['team_ids']:
        raise ValueError('Actual team identity mismatch')


def _eligible(box, pub, actual):
    _published(box, pub, actual)
    b = box.get('skill_baselines')
    if not b or b.get('schema') != SCHEMA or b.get('lane') != 'pregame':
        raise ValueError('No frozen pregame skill baseline; never backfill after outcome')
    if not utc(b['source_available_through']) <= utc(b['as_of']) <= utc(box['forecast_origin']):
        raise ValueError('Late baseline dependency')
    if max(b['history_through'], b['team_history_through'], b['team_fit_through']) >= box['date']:
        raise ValueError('Future baseline data')
    if b['high_scoring_threshold'] != HIGH_TOTAL:
        raise ValueError('Scoring threshold changed')


def score_one(box, actual):
    base = box['skill_baselines']; rows = []; missing = []; unmatched = []
    def add(key, side, entity, name, model_score, baseline_score, **extra):
        rows.append({'metric': key, 'side': side, 'entity_id': entity, 'name': name,
                     'model_score': model_score, 'baseline_score': baseline_score,
                     'delta': model_score - baseline_score, **extra})
    for side in ('away', 'home'):
        actual_index = {str(r['player_id']): r for r in actual['pitching'][side]}
        pool = {str(p['player_id']) for p in box['teams'][side]['pitching']}
        unmatched.extend({'side': side, 'player_id': pid} for pid in actual_index if pid not in pool)
        for player in box['teams'][side]['pitching']:
            pid = str(player['player_id']); a = actual_index.get(pid); b = base['pitching'][side].get(pid)
            if b is None:
                raise ValueError('Frozen baseline has a different pitcher pool')
            for metric in ('K', 'BB', 'PC', 'outs'):
                y = 0 if a is None else a.get(metric)
                if y is None or b['pmf'].get(metric) is None:
                    missing.append({'metric': metric, 'side': side, 'player_id': pid}); continue
                if type(y) is not int or y < 0:
                    raise ValueError('Invalid final count')
                scale = 3 if metric == 'outs' else 1
                add('pitcher_IP' if metric == 'outs' else 'pitcher_' + metric, side, pid, player['name'],
                    ensemble_crps(player['distributions'][metric], y) / scale,
                    discrete_crps(b['pmf'][metric], y) / scale,
                    actual=y / scale, score_family='CRPS', role='starter' if b['starter'] else 'reliever')
            if not b['starter']:
                add('reliever_appears', side, pid, player['name'],
                    occurrence_score(player['appearance_probability'], int(a is not None), box['n_simulations']),
                    occurrence_score(b['appearance_probability'], int(a is not None)),
                    actual=int(a is not None), score_family='Brier')
        y = actual['score'][side]
        if type(y) is not int or y < 0:
            raise ValueError('Invalid actual team runs')
        score, bound = nb_crps(**base['team_runs'][side], observed=y)
        add('team_runs', side, str(box['team_ids'][side]), side,
            ensemble_crps(box['team_run_distributions'][side], y), score,
            actual=y, score_family='CRPS', baseline_numerical_error_bound=bound)
    n, dist = histogram(box['total_run_distribution'])
    p = sum(c for x, c in dist if x >= HIGH_TOTAL) / n
    y = int(sum(actual['score'].values()) >= HIGH_TOTAL)
    add('high_total', 'both', str(box['game_pk']), '9 or more total runs',
        occurrence_score(p, y, n), occurrence_score(base['high_total_probability'], y),
        actual=y, score_family='Brier')
    return {'rows': rows, 'missing_counts': missing, 'unprojected_actual_pitchers': unmatched}


def aggregate_rows(versions):
    grouped = defaultdict(lambda: defaultdict(list))
    for version in versions:
        for row in version['rows']:
            grouped[row['metric']][version['game_pk']].append(row)
    output = []
    for key, label, family in METRICS:
        groups = grouped[key]
        per_game = [{'game_pk': pk, 'model': sum(r['model_score'] for r in rows) / len(rows),
                     'baseline': sum(r['baseline_score'] for r in rows) / len(rows)}
                    for pk, rows in sorted(groups.items())]
        n = len(per_game)
        model = sum(r['model'] for r in per_game) / n if n else None
        baseline = sum(r['baseline'] for r in per_game) / n if n else None
        output.append({'metric': key, 'label': label, 'score_family': family, 'n_games': n,
                       'n_cases': sum(len(rows) for rows in groups.values()), 'model_score': model,
                       'baseline_score': baseline, 'delta': model - baseline if n else None,
                       'per_game': per_game, 'ci95': None,
                       'inference': 'Descriptive live tracking; no acceptance claim'})
    return output


def skill_box_outcome(box, pub, actual):
    """One saved box against its game's final box: (why it is not a publication, why it is not eligible, the score
    before its identity fields); each reason None when that check passes. score_one's own errors are raised, as before."""
    try:
        _published(box, pub, actual)
    except (ValueError, KeyError, TypeError) as exc:
        return str(exc), None, None
    try:
        _eligible(box, pub, actual)
    except (ValueError, KeyError, TypeError) as exc:
        return None, str(exc), None
    return None, None, score_one(box, actual)


def score_skill_boxes(boxes, publications, actuals, cache=None):
    # Select the last eligible publication before inspecting skill fields. Missing
    # new fields in a later published box cannot silently revive an older score.
    # cache: ledger['box_cache'], the outcomes of boxes archived after their game, worked out while the full box
    # still verified against its publication (brl_live/box_runner.py archive_old_boxes).
    excluded = {}; scored = []; newest = {}; by_id = {}
    for ident, box in sorted(boxes.items(), key=lambda pair:(pair[1]['saved_at'],pair[0])):
        actual = actuals.get(str(box['game_pk']))
        if actual is None: continue
        hit = (cache or {}).get(ident) if box.get('archived') else None
        if hit is not None and 'skill' in hit:
            not_published, not_eligible, raw = hit['skill'].get('published'), hit['skill'].get('eligible'), hit['skill'].get('result')
        else:
            not_published, not_eligible, raw = skill_box_outcome(box, publications.get(ident), actual)
        if not_published is not None:
            excluded[ident] = not_published; continue
        newest[box['game_pk']] = ident
        if not_eligible is not None:
            excluded[ident] = not_eligible; continue
        result = dict(raw)
        result.update(forecast_id=ident, game_pk=box['game_pk'], saved_at=box['saved_at'])
        scored.append(result); by_id[ident] = result
    latest = [by_id[ident] for ident in newest.values() if ident in by_id]
    return {'schema': 'brl.skill-scores.v1', 'lane': 'pregame', 'n_games': len(latest),
            'aggregates': aggregate_rows(latest), 'versions': scored, 'excluded': excluded,
            'high_scoring_threshold': HIGH_TOTAL,
            'unprojected_pitcher_appearances': sum(len(v['unprojected_actual_pitchers']) for v in latest),
            'weighting': 'Within each metric average cases within game, then average games',
            'numerical_uncertainty': 'Fair finite-world model scores; analytic/exact fitted baselines; no model-uncertainty interval',
            'policy': 'Latest eligible frozen skill-box version; no same-game future inputs or in-game pooling'}
