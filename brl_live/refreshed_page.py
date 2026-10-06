"""Prediction-only rendering with explicit accepted history coverage."""
from html import escape
from pathlib import Path
import json
from cloud.page import render_page as original_render

OLD = ('The locked public-data history in this runtime ends September 27, 2026. '
       'It is captured in the encrypted package, filtered strictly before the target game, '
       'and is not silently labeled current. New lineups and roster inputs are captured '
       'on each scheduled check. An automatic Statcast-history refresh is not implemented in this checkpoint.')

def render_page(ledger, scores, destination, setup_message=None):
    # Past results without a forecast belong in the record, not as unnamed
    # "Waiting for forecast" cards on today's slate. This is display-only:
    # preserve all ledger versions/results in the public JSON and scorebook.
    view = dict(ledger)
    if ledger.get('date'):
        view['status'] = {pk: status for pk, status in ledger.get('status', {}).items()
                          if status.get('date') == ledger['date']}
    path = original_render(view, scores, destination, setup_message)
    data_path = Path(destination) / 'predictions.json'
    public_data = json.loads(data_path.read_text(encoding='utf-8'))
    public_data['status'] = ledger.get('status', {})
    from app.common import canonical
    data_path.write_bytes(canonical(public_data))
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
