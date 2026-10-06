"""Prediction-only rendering with explicit accepted history coverage."""
from html import escape
from pathlib import Path
from cloud.page import render_page as original_render

OLD = ('The locked public-data history in this runtime ends September 27, 2026. '
       'It is captured in the encrypted package, filtered strictly before the target game, '
       'and is not silently labeled current. New lineups and roster inputs are captured '
       'on each scheduled check. An automatic Statcast-history refresh is not implemented in this checkpoint.')

def render_page(ledger, scores, destination, setup_message=None):
    path = original_render(ledger, scores, destination, setup_message)
    coverages = sorted({f['history_through'] for f in ledger.get('forecasts', {}).values()})
    used = ', '.join(escape(day) for day in coverages) or 'No forecast saved'
    replacement = ('History used in the saved forecasts runs through: ' + used + '. '
        'The daily refresh checks completed games through the previous Eastern-calendar day and '
        'reconciles plate appearances against official final feeds before accepting a date. '
        'Source snapshots and history revisions stay encrypted; original forecast inputs are never '
        'rewritten. A coverage gap blocks new forecasts. The initial seed history retains its '
        'previously disclosed, unverified historical publication vintages.')
    text = Path(path).read_text(encoding='utf-8')
    if OLD not in text:
        raise ValueError('Expected private-runtime page template changed')
    Path(path).write_text(text.replace(OLD, replacement), encoding='utf-8')
    return path
