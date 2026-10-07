"""Team baseline for the forecast record: the decayed negative-binomial team model."""
from __future__ import annotations
import gzip, json
from pathlib import Path

from .engine_bridge import TEAM_RESULTS
from brl_live.team_model import TeamModel


def load_results(root=None):
    root = Path(root or TEAM_RESULTS)
    rows, meta = [], {'files': []}
    for path in sorted(root.glob('team_results_*.json.gz')):
        doc = json.loads(gzip.decompress(path.read_bytes()))
        items = doc if isinstance(doc, list) else next((v for v in doc.values() if isinstance(v, list)), [])
        for r in items:
            try:
                rows.append({'game_pk': int(r['game_pk']), 'date': str(r['date'])[:10], 'away_id': int(r['away_id']), 'home_id': int(r['home_id']),
                             'away_runs': int(r['away_runs']), 'home_runs': int(r['home_runs'])})
            except (KeyError, TypeError, ValueError):
                continue
        meta['files'].append(path.name)
    return rows, meta


def fit_dispersion(rows):
    """Kept for the public layer's call shape; the team model fits its own dispersion per date."""
    return {'model': TeamModel(rows)}


def predict(game: dict, rows, fit) -> dict:
    model = fit['model'] if isinstance(fit, dict) and 'model' in fit else TeamModel(rows)
    out = model.probability(game['date'], game['home']['team_id'], game['away']['team_id'])
    return {'probability': out['p_home'], **out}
