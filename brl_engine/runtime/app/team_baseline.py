"""Team baseline for the forecast record: the decayed negative-binomial team model."""
from __future__ import annotations
import gzip, json
from pathlib import Path

from .engine_bridge import TEAM_RESULTS
from brl_live.team_model import TeamModel


ALIASES = {'game_pk': ('game_pk', 'gamePk', 'pk'), 'date': ('date', 'game_date', 'officialDate', 'official_date'),
           'away_id': ('away_id', 'away_team_id', 'awayId'), 'home_id': ('home_id', 'home_team_id', 'homeId'),
           'away_runs': ('away_runs', 'away_score', 'awayRuns', 'away'), 'home_runs': ('home_runs', 'home_score', 'homeRuns', 'home')}


def _rows_in(doc):
    """Find the list of game rows in a results document of unknown shape."""
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for key in ('rows', 'results', 'games', 'data', 'items'):
            if isinstance(doc.get(key), list):
                return doc[key]
        for v in doc.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
            if isinstance(v, dict):
                inner = _rows_in(v)
                if inner:
                    return inner
    return []


def _field(r: dict, name: str):
    for alias in ALIASES[name]:
        if alias in r:
            v = r[alias]
            if isinstance(v, dict):
                v = v.get('id', v.get('runs', v.get('score')))
            return v
    for container in ('teams', 'result'):
        if isinstance(r.get(container), dict):
            side = name.split('_')[0]
            t = r[container].get(side)
            if isinstance(t, dict):
                if name.endswith('_id'):
                    return (t.get('team') or {}).get('id', t.get('id'))
                if name.endswith('_runs'):
                    return t.get('runs', t.get('score'))
    raise KeyError(name)


def _schedule_rows(doc) -> list:
    """Rows from an official schedule document ({dates: [{date, games: [...]}]}): regular-season finals only."""
    rows = []
    for day in (doc.get('dates') or []) if isinstance(doc, dict) else []:
        for g in day.get('games') or []:
            try:
                if g.get('gameType', 'R') != 'R' or ((g.get('status') or {}).get('abstractGameState') or 'Final') != 'Final':
                    continue
                t = g['teams']
                if t['away'].get('score') is None or t['home'].get('score') is None:
                    continue
                rows.append({'game_pk': int(g['gamePk']), 'date': str(g.get('officialDate') or day.get('date'))[:10],
                             'away_id': int(t['away']['team']['id']), 'home_id': int(t['home']['team']['id']),
                             'away_runs': int(t['away']['score']), 'home_runs': int(t['home']['score'])})
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_results(root=None):
    root = Path(root or TEAM_RESULTS)
    rows, meta = [], {'files': [], 'row_keys': None, 'skipped': 0}
    for path in sorted(root.glob('team_results_*.json.gz')):
        doc = json.loads(gzip.decompress(path.read_bytes()))
        if isinstance(doc, dict) and isinstance(doc.get('dates'), list):
            found = _schedule_rows(doc)
            rows.extend(found); meta['files'].append(path.name); meta['row_keys'] = ['schedule document']
            continue
        items = _rows_in(doc)
        if items and meta['row_keys'] is None and isinstance(items[0], dict):
            meta['row_keys'] = sorted(items[0].keys())[:30]
        for r in items:
            if not isinstance(r, dict):
                continue
            try:
                rows.append({'game_pk': int(_field(r, 'game_pk')), 'date': str(_field(r, 'date'))[:10], 'away_id': int(_field(r, 'away_id')),
                             'home_id': int(_field(r, 'home_id')), 'away_runs': int(_field(r, 'away_runs')), 'home_runs': int(_field(r, 'home_runs'))})
            except (KeyError, TypeError, ValueError):
                meta['skipped'] += 1
        meta['files'].append(path.name)
    return rows, meta


def fit_dispersion(rows):
    """Kept for the public layer's call shape; the team model fits its own dispersion per date."""
    return {'model': TeamModel(rows)}


def predict(game: dict, rows, fit) -> dict:
    """Never blocks a forecast: without usable team results the baseline is absent (None)."""
    model = fit['model'] if isinstance(fit, dict) and 'model' in fit else TeamModel(rows)
    try:
        out = model.probability(game['date'], game['home']['team_id'], game['away']['team_id'])
    except Exception as exc:
        return {'probability': None, 'error': type(exc).__name__ + ': ' + str(exc)[:120], 'rows': len(model.rows)}
    prior_dates = [r['date'] for r in model.rows if r['date'] < str(game['date'])[:10]]
    through = max(prior_dates) if prior_dates else None
    # Field names the public edge-metrics baselines expect from the runtime baseline.
    return {'probability': out['p_home'], **out, 'game_pk': int(game['game_pk']), 'prior_through': through,
            'dispersion_fit': {'training_through': through, 'alpha': out['alpha'], 'window': 'last three seasons before the game date'},
            'away_mean_runs': out['mu_away'], 'home_mean_runs': out['mu_home'], 'nb_alpha': out['alpha']}
