"""Phone-first public page. Only model outputs, saved forecasts and final box summaries.

The page is a static template (brl_live/page/template.html) with the public ledger
inlined as window.BRL and the two typefaces embedded, so it needs no network to render.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from app.common import canonical
from cloud.contracts import check_public

from .record import build_record

PAGE_DIR = Path(__file__).resolve().parent / 'page'
FONTS = (
    ('Barlow Condensed', 500, 'barlow-condensed-latin-500-normal.woff2'),
    ('Barlow Condensed', 600, 'barlow-condensed-latin-600-normal.woff2'),
    ('Barlow Condensed', 700, 'barlow-condensed-latin-700-normal.woff2'),
    ('Atkinson Hyperlegible', 400, 'atkinson-hyperlegible-latin-400-normal.woff2'),
    ('Atkinson Hyperlegible', 700, 'atkinson-hyperlegible-latin-700-normal.woff2'),
)
PUBLIC_KEYS = ('date', 'forecasts', 'publications', 'actuals', 'status', 'box_scores',
               'box_publications', 'actual_boxes', 'player_scores', 'skill_scores', 'live', 'market', 'context')
BOX_DAYS = 2  # full simulated games stay on the page for the slate date and the day before


def _recent_dates(date: str, days: int) -> set:
    from datetime import date as _date, timedelta
    try:
        d = _date.fromisoformat(str(date))
    except ValueError:
        return set()
    return {(d - timedelta(days=i)).isoformat() for i in range(days)}


PAGE_BOX_DROP = ('skill_baselines',)   # scoring inputs the page never reads (kept in the ledger)


def trim_boxes(public: dict) -> dict:
    """Keep full box scores (five complete simulated games each) only for recent dates, and of
    those only the box of each game's latest forecast version, which is the one the page opens.

    Every box stays in the private ledger and is still scored on the record page; the page
    itself carries only the games a reader can open, so it does not grow with the season or
    with the lineup changes that make new versions.
    """
    keep = _recent_dates(public.get('date'), BOX_DAYS)
    boxes = public.get('box_scores') or {}
    latest = {}
    for ident, f in (public.get('forecasts') or {}).items():
        pk = str(f.get('game_pk'))
        rank = (int(f.get('version') or 0), str(f.get('saved_at') or ''))
        if pk not in latest or rank > latest[pk][1]:
            latest[pk] = (ident, rank)
    recent = {}
    for k, v in boxes.items():
        if str(v.get('date')) not in keep:
            continue
        pk = str(v.get('game_pk'))
        if pk in latest and latest[pk][0] != k:
            continue
        recent[k] = {name: value for name, value in v.items() if name not in PAGE_BOX_DROP}
    public['box_scores'] = recent
    public['live'] = {k: v for k, v in (public.get('live') or {}).items() if str(v.get('date')) in keep}
    public['box_publications'] = {k: v for k, v in (public.get('box_publications') or {}).items() if k in recent}
    pks = {str(v.get('game_pk')) for v in recent.values()}
    public['actual_boxes'] = {k: v for k, v in (public.get('actual_boxes') or {}).items() if str(k) in pks}
    return public


def font_css() -> str:
    parts = []
    for family, weight, name in FONTS:
        raw = (PAGE_DIR / 'fonts' / name).read_bytes()
        parts.append('@font-face{font-family:"%s";font-style:normal;font-weight:%d;font-display:swap;'
                     'src:url(data:font/woff2;base64,%s) format("woff2");}' % (family, weight, base64.b64encode(raw).decode()))
    return '\n'.join(parts)


def public_payload(ledger, scores, *, replay=False) -> dict:
    public = {k: ledger.get(k, {}) for k in PUBLIC_KEYS}
    public['scores'] = scores
    public['record'] = build_record(ledger)
    public['view_scope'] = 'historical_replay' if replay else 'live'
    from datetime import datetime, timezone
    public['generated_at'] = datetime.now(timezone.utc).isoformat()
    return trim_boxes(public)


def render_html(public: dict) -> str:
    data = canonical(public).decode().replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (PAGE_DIR / 'template.html').read_text(encoding='utf-8')
    if '/*FONTS*/' not in template or '/*DATA*/' not in template:
        raise ValueError('Page template placeholders missing')
    return template.replace('/*FONTS*/', font_css(), 1).replace('/*DATA*/', 'window.BRL=' + data + ';', 1)


ARCHIVE_PLAY_KEYS = ('inning', 'half', 'batter_id', 'batter_name', 'pitcher_id', 'pitcher_name', 'box_outcome', 'description',
                     'outs_before', 'outs_after', 'runs_scored', 'away_score', 'home_score', 'scoring_players', 'rbi', 'estimated_pitches', 'contact')


def lean_box(box: dict) -> dict:
    """A box for the day archive: projected sample only, player means and chances, no distributions or pitch lists."""
    out = {k: box[k] for k in ('schema', 'game_pk', 'date', 'saved_at', 'forecast_origin', 'team_ids', 'starters', 'lineup_status',
                               'history_through', 'forecast_id', 'n_simulations', 'sample_roles', 'sample_indices', 'adjustments',
                               'team_model', 'team_run_distributions', 'total_run_distribution', 'line_score', 'win_table') if k in box}
    out['teams'] = {}
    for side in ('away', 'home'):
        out['teams'][side] = {}
        for kind in ('batting', 'pitching'):
            rows = []
            for r in (box.get('teams', {}).get(side, {}).get(kind) or []):
                rows.append({k: v for k, v in r.items() if k != 'distributions'})
            out['teams'][side][kind] = rows
    samples = box.get('samples') or []
    proj = samples[0] if samples else None
    if proj is not None:
        lean = {k: proj[k] for k in ('seed', 'score', 'innings', 'batting', 'pitching', 'world_index', 'typical') if k in proj}
        lean['plays'] = [{k: pl.get(k) for k in ARCHIVE_PLAY_KEYS} for pl in proj.get('plays') or []]
        out['samples'] = [lean]
        out['sample_roles'] = {'projected': out.get('sample_roles', {}).get('projected')}
        out['sample_indices'] = out.get('sample_indices', [None])[:1]
    out['archived'] = True
    return out


def day_archives(ledger: dict) -> dict:
    """date -> compact public payload for that day (forecasts, actuals, market, lean boxes)."""
    days = {}
    for ident, box in (ledger.get('box_scores') or {}).items():
        days.setdefault(str(box.get('date')), {'date': str(box.get('date')), 'forecasts': {}, 'publications': {}, 'actuals': {}, 'status': {},
                                              'box_scores': {}, 'box_publications': {}, 'actual_boxes': {}, 'live': {}, 'market': {}, 'context': {}})
    forecasts = ledger.get('forecasts') or {}
    for d, day in days.items():
        pks = set()
        for ident, f in forecasts.items():
            if str(f.get('date')) == d:
                day['forecasts'][ident] = f; pks.add(str(f['game_pk']))
                if ident in (ledger.get('publications') or {}):
                    day['publications'][ident] = ledger['publications'][ident]
                if ident in (ledger.get('box_scores') or {}):
                    day['box_scores'][ident] = lean_box(ledger['box_scores'][ident])
                    if ident in (ledger.get('box_publications') or {}):
                        day['box_publications'][ident] = ledger['box_publications'][ident]
        for pk in pks:
            for key in ('actuals', 'status', 'actual_boxes', 'market', 'context'):
                if pk in (ledger.get(key) or {}):
                    day[key][pk] = ledger[key][pk]
    return days


def render_page(ledger, scores, destination, setup_message=None, *, replay=False):
    if replay:
        if ledger.get('publications') or ledger.get('box_publications') or scores.get('n_games'):
            raise ValueError('Historical UI replay cannot carry live publication/scoring evidence')
    else:
        for f in ledger['forecasts'].values():
            check_public(f)
    public = public_payload(ledger, scores, replay=replay)
    d = Path(destination)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'index.html').write_text(render_html(public), encoding='utf-8')
    (d / 'predictions.json').write_bytes(canonical(public))
    (d / '.nojekyll').write_text('')
    days = d / 'days'
    days.mkdir(exist_ok=True)
    archives = day_archives(ledger)
    for date, payload in archives.items():
        (days / (date + '.json')).write_bytes(canonical(payload))
    public_index = {'schema': 'brl.days.v1', 'dates': sorted(archives)}
    (days / 'index.json').write_bytes(canonical(public_index))
    return d / 'index.html'
