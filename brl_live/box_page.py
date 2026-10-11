"""Phone-first public page. Only model outputs, saved forecasts and final box summaries.

The page is a static template (brl_live/page/template.html) with the public ledger
inlined as window.BRL and the two typefaces embedded, so it needs no network to render.
"""
from __future__ import annotations

import re

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


def _recent_dates(date: str, days: int, ahead: int = 0) -> set:
    """The slate date and the days before it; with ahead, only the days after it."""
    from datetime import date as _date, timedelta
    try:
        d = _date.fromisoformat(str(date))
    except ValueError:
        return set()
    if ahead:
        return {(d + timedelta(days=i)).isoformat() for i in range(1, ahead + 1)}
    return {(d - timedelta(days=i)).isoformat() for i in range(days)}


EARLY_DAYS = 1  # tomorrow's early calls carry their projected game only (the full set comes with the game-day version)


PAGE_BOX_DROP = ('skill_baselines',)   # scoring inputs the page never reads (kept in the ledger)


def trim_boxes(public: dict) -> dict:
    """Keep full box scores (five complete simulated games each) only for recent dates, and of
    those only the box of each game's latest forecast version, which is the one the page opens.

    Every box stays in the private ledger and is still scored on the record page; the page
    itself carries only the games a reader can open, so it does not grow with the season or
    with the lineup changes that make new versions.
    """
    keep = _recent_dates(public.get('date'), BOX_DAYS)
    ahead = _recent_dates(public.get('date'), 0, ahead=EARLY_DAYS)
    boxes = public.get('box_scores') or {}
    latest = {}
    for ident, f in (public.get('forecasts') or {}).items():
        pk = str(f.get('game_pk'))
        rank = (int(f.get('version') or 0), str(f.get('saved_at') or ''))
        if pk not in latest or rank > latest[pk][1]:
            latest[pk] = (ident, rank)
    recent = {}
    for k, v in boxes.items():
        d = str(v.get('date'))
        if d not in keep and d not in ahead:
            continue
        pk = str(v.get('game_pk'))
        if pk in latest and latest[pk][0] != k:
            continue
        if d in ahead:
            early = lean_box(v, home_pick=((public.get('record') or {}).get('blend') or {}).get(k))
            early.pop('archived', None); early['early'] = True
            recent[k] = early
        else:
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


def replay_summary() -> dict:
    """Season replay scores for the Record page, from the static season files (games with a closing line)."""
    out = {}
    for src in sorted(ARCHIVE_DIR.glob('season-*.json')):
        try:
            doc = json.loads(src.read_text())
        except (OSError, ValueError):
            continue
        games = [g for g in doc.get('games') or [] if g.get('m') is not None and g.get('fh') is not None and g.get('fa') is not None and g['fh'] != g['fa']]
        if not games:
            continue
        def score(key):
            vals = [(g[key], 1.0 if g['fh'] > g['fa'] else 0.0) for g in games if g.get(key) is not None]
            b = sum((p - y) ** 2 for p, y in vals) / len(vals)
            right = sum(1 for p, y in vals if (p >= 0.5) == (y == 1.0)) / len(vals)
            return {'brier': round(b, 5), 'better_than_coin_pct': round((0.25 - b) / 0.25 * 100, 2), 'picks_right_pct': round(right * 100, 1)}
        out[str(doc.get('season'))] = {'games': len(games), 'replay': doc.get('replay'), 'headline': score('hl'), 'ours': score('o'),
                                       'simulator': score('s'), 'team': score('tm'), 'market': score('m'), 'market_open': score('mo')}
    return out


def public_payload(ledger, scores, *, replay=False) -> dict:
    public = {k: ledger.get(k, {}) for k in PUBLIC_KEYS}
    public['scores'] = scores
    public['record'] = build_record(ledger)
    public['record']['replay'] = replay_summary()
    try:
        from .series import series_outlook
        public['series'] = series_outlook(ledger, public['record'])
    except Exception as exc:  # the page works without it
        public['series'] = {'schema': 'brl.series.v1', 'series': [], 'error': type(exc).__name__ + ': ' + str(exc)[:160]}
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


def _winner(sample: dict) -> str:
    score = sample.get('score') or {}
    return 'home' if (score.get('home') or 0) > (score.get('away') or 0) else 'away'


def lean_box(box: dict, home_pick: float | None = None) -> dict:
    """A box for the day archive: projected sample only, player means and chances, no distributions or pitch lists.

    The saved roles were chosen with the simulator's favorite. With home_pick (the headline's home win chance) given and
    the headline picking the other team, the kept game is the saved 'upset' world, the most typical game our pick wins,
    so the page never shows the pick losing its own projected game."""
    out = {k: box[k] for k in ('schema', 'game_pk', 'date', 'saved_at', 'forecast_origin', 'team_ids', 'starters', 'lineup_status',
                               'history_through', 'forecast_id', 'n_simulations', 'sample_roles', 'sample_indices', 'adjustments',
                               'team_model', 'team_run_distributions', 'total_run_distribution', 'line_score', 'win_table', 'matchups') if k in box}
    out['teams'] = {}
    for side in ('away', 'home'):
        out['teams'][side] = {}
        for kind in ('batting', 'pitching'):
            rows = []
            for r in (box.get('teams', {}).get(side, {}).get(kind) or []):
                rows.append({k: v for k, v in r.items() if k != 'distributions'})
            out['teams'][side][kind] = rows
    samples = box.get('samples') or []
    roles, indices = box.get('sample_roles') or {}, list(box.get('sample_indices') or [])
    choice = 0
    if home_pick is not None and samples:
        fav = 'home' if home_pick >= 0.5 else 'away'
        up = roles.get('upset')
        j = indices.index(up) if up is not None and up in indices else -1
        if _winner(samples[0]) != fav and 0 <= j < len(samples) and _winner(samples[j]) == fav:
            choice = j
    proj = samples[choice] if samples else None
    if proj is not None:
        lean = {k: proj[k] for k in ('seed', 'score', 'innings', 'batting', 'pitching', 'world_index', 'typical') if k in proj}
        lean['plays'] = [{k: pl.get(k) for k in ARCHIVE_PLAY_KEYS} for pl in proj.get('plays') or []]
        out['samples'] = [lean]
        if choice:
            out['sample_roles'] = {'projected': indices[choice]}
            out['sample_indices'] = [indices[choice]]
            out['projected_follows_pick'] = True
        else:
            out['sample_roles'] = {'projected': out.get('sample_roles', {}).get('projected')}
            out['sample_indices'] = out.get('sample_indices', [None])[:1]
    out['archived'] = True
    return out


def day_archives(ledger: dict, blend: dict | None = None) -> dict:
    """date -> compact public payload for that day (forecasts, actuals, market, lean boxes). blend: the record's headline
    home win chance by forecast, so each day's projected game is the most typical game our pick wins (lean_box)."""
    blend = blend or {}
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
                    day['box_scores'][ident] = lean_box(ledger['box_scores'][ident], home_pick=blend.get(ident))
                    if ident in (ledger.get('box_publications') or {}):
                        day['box_publications'][ident] = ledger['box_publications'][ident]
        for pk in pks:
            for key in ('actuals', 'status', 'actual_boxes', 'market', 'context'):
                if pk in (ledger.get(key) or {}):
                    day[key][pk] = ledger[key][pk]
    return days


ARCHIVE_DIR = Path(__file__).resolve().parent / 'archive'
ARCHIVE_NAME = re.compile(r'(season-\d{4}|win-expectancy)\.json')


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
    archives = day_archives(ledger, (public.get('record') or {}).get('blend'))
    for date, payload in archives.items():
        (days / (date + '.json')).write_bytes(canonical(payload))
    public_index = {'schema': 'brl.days.v1', 'dates': sorted(archives)}
    (days / 'index.json').write_bytes(canonical(public_index))
    # Static files for past dates: the season replays (model outputs only) and the league win expectancy table.
    for src in sorted(ARCHIVE_DIR.glob('*.json')):
        if ARCHIVE_NAME.fullmatch(src.name):
            (days / src.name).write_bytes(src.read_bytes())
    return d / 'index.html'
