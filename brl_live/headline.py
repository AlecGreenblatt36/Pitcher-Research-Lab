"""The headline win chance and our independent model (brl_live/headline_params.json).

Our model is independent of the betting market at forecast time: a fixed combination of what our
forecast knows before first pitch (the simulator's and the team model's log-odds, the starters'
adjustments and the team ratings). Its weights were taught by the market: fitted to the market's
closing log-odds on past regular seasons, a far more precise target than one game's result, and then
checked against real results on games the fit never saw (DISTILL in LEDGER.md).

The headline is the best number we can give before first pitch: our model combined with the market's
pregame line when one has been captured (weights fitted on past results, checked the same way), and
our model alone when not. The market is used only for the headline, never inside our model, so the
record can still show how our own view compares with the market's.

Versions carry an effective time. A game is always scored with the version in force at its first
pitch, so a change never rewrites a published number after the fact.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

PARAMS_PATH = Path(__file__).with_name('headline_params.json')
EPS = 1e-4


def _logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, float(p)))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _ts(value) -> datetime:
    v = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if v.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return v.astimezone(timezone.utc)


def load_params(path: Path = PARAMS_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    doc['versions'] = sorted(doc['versions'], key=lambda v: _ts(v['effective_from']))
    return doc


PARAMS = load_params()


def version_at(when=None, params: dict | None = None) -> dict:
    """The version in force at a moment (now when not given)."""
    params = params or PARAMS
    t = _ts(when) if when is not None else datetime.now(timezone.utc)
    chosen = params['versions'][0]
    for v in params['versions']:
        if _ts(v['effective_from']) <= t:
            chosen = v
    return chosen


def features(p_sim: float, team_model: dict | None) -> dict | None:
    """Our forecast's pregame inputs on the scale the weights were fitted on; None when the team model is missing."""
    tm = team_model or {}
    if tm.get('p_home') is None:
        return None
    out = {'x_sim': _logit(p_sim), 'x_team': _logit(tm['p_home'])}
    sp = tm.get('starter_adjust') or {}
    if sp.get('home') and sp.get('away'):
        out['sp_home'] = math.log(float(sp['home']['factor'])); out['sp_away'] = math.log(float(sp['away']['factor']))
    rt = tm.get('ratings') or {}
    if rt.get('home') and rt.get('away'):
        for side in ('home', 'away'):
            out['off_' + side] = math.log(float(rt[side]['offense'])); out['def_' + side] = math.log(float(rt[side]['defense']))
    return out


def ours(p_sim: float, team_model: dict | None, version: dict) -> tuple[float, str]:
    """Our independent win chance for the home team and the recipe used."""
    if p_sim is None:
        raise ValueError('simulator probability required')
    x = features(p_sim, team_model)
    if x is None:
        return float(p_sim), 'simulator only (no team model)'
    model = version.get('ours') or {}
    cols = model.get('features') or []
    if version.get('kind') == 'taught' and cols and all(c in x for c in cols):
        z = float(model.get('intercept', 0.0)) + sum(float(w) * x[c] for c, w in zip(cols, model['coef']))
        return _sigmoid(z), version['name']
    return _sigmoid(0.5 * x['x_sim'] + 0.5 * x['x_team']), 'equal blend of simulator and team model'


def headline(p_ours: float, p_market: float | None, version: dict) -> tuple[float, str]:
    """The best pregame number: our model with the market's pregame line when there is one."""
    h = version.get('headline') or {}
    if p_market is None or version.get('kind') != 'taught' or not h:
        return float(p_ours), 'our model'
    z = float(h.get('intercept', 0.0)) + float(h['market']) * _logit(p_market) + float(h['ours']) * _logit(p_ours)
    return _sigmoid(z), 'our model with the market line'
