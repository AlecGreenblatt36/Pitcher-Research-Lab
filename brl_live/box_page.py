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
               'box_publications', 'actual_boxes', 'player_scores', 'skill_scores')
BOX_DAYS = 2  # full simulated games stay on the page for the slate date and the day before


def _recent_dates(date: str, days: int) -> set:
    from datetime import date as _date, timedelta
    try:
        d = _date.fromisoformat(str(date))
    except ValueError:
        return set()
    return {(d - timedelta(days=i)).isoformat() for i in range(days)}


def trim_boxes(public: dict) -> dict:
    """Keep full box scores (five complete simulated games each) only for recent dates.

    Every box stays in the private ledger and is still scored on the record page; the page
    itself carries only the games a reader can open, so it does not grow with the season.
    """
    keep = _recent_dates(public.get('date'), BOX_DAYS)
    boxes = public.get('box_scores') or {}
    recent = {k: v for k, v in boxes.items() if str(v.get('date')) in keep}
    public['box_scores'] = recent
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
    return trim_boxes(public)


def render_html(public: dict) -> str:
    data = canonical(public).decode().replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (PAGE_DIR / 'template.html').read_text(encoding='utf-8')
    if '/*FONTS*/' not in template or '/*DATA*/' not in template:
        raise ValueError('Page template placeholders missing')
    return template.replace('/*FONTS*/', font_css(), 1).replace('/*DATA*/', 'window.BRL=' + data + ';', 1)


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
    return d / 'index.html'
