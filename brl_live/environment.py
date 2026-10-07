"""Run environment at simulation time: weather, wind, roof, day or night and venue (ENV-01, ENV-02 in LEDGER.md).

The plate-appearance model knows the park only through the home team's code and knows nothing about weather.
This table multiplies the seven outcome probabilities of every plate appearance in a game by exp(effect),
then renormalizes (provider_adjust.EnvironmentAdjust). Effects were fitted on the model's own out-of-sample
predictions (Poisson per outcome class with the model probability as offset): temperature, wind blowing out or
in (Wrigley Field separately), roof closed, day game, and a shrunk residual per venue. They are centered so the
average plate appearance of the fit is unchanged, which keeps the context offsets' level.

Conditions come from the official feed at forecast time (gameData.weather, venue, dayNight). When the weather is
not posted yet, the venue's usual temperature for the month and its usual share of games with the roof closed
stand in, with no wind.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TABLE_PATH = Path(__file__).with_name('environment.json')
WIND = re.compile(r'^\s*(\d+)\s*mph\s*,?\s*(.*)$', re.I)
CLOSED = ('Dome', 'Roof Closed')


def load_table(path: Path = TABLE_PATH) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != 'brl.environment.v1' or len(doc['labels']) != 7:
        raise ValueError('environment table shape')
    return doc


def conditions_from_feed(gd: dict) -> dict:
    """Venue, weather and day or night from the official feed's gameData."""
    w = gd.get('weather') or {}
    dt = gd.get('datetime') or {}
    return conditions(venue=(gd.get('venue') or {}).get('name'), temp=w.get('temp'), wind=w.get('wind'), condition=w.get('condition'),
                      day_night=dt.get('dayNight'), date=dt.get('officialDate'))


def conditions(*, venue=None, temp=None, wind=None, condition=None, day_night=None, date=None, wind_mph=None, wind_dir=None) -> dict:
    """One shape for the live feed and the replay's conditions file."""
    try:
        temp_f = int(str(temp).strip()) if temp not in (None, '') else None
    except ValueError:
        temp_f = None
    if wind is not None and wind_mph is None:
        m = WIND.match(str(wind))
        if m:
            wind_mph, wind_dir = int(m.group(1)), (m.group(2).strip() or None)
    month = int(str(date)[5:7]) if date and len(str(date)) >= 7 else None
    return {'venue': venue, 'temp_f': temp_f, 'wind_mph': wind_mph, 'wind_dir': wind_dir, 'condition': condition or None,
            'day_night': day_night, 'month': month}


def features(cond: dict, table: dict) -> tuple[dict, list]:
    """The table's features for a game, and which values were filled from the venue's climate."""
    filled = []
    climate = ((table.get('climate') or {}).get(cond.get('venue') or '') or {}).get(str(cond.get('month'))) or {}
    if cond.get('condition'):
        closed = 1.0 if cond['condition'] in CLOSED else 0.0
    else:
        closed = float(climate.get('closed_share') or 0.0); filled.append('roof')
    temp = cond.get('temp_f')
    if temp is None:
        temp = climate.get('temp_f') if climate.get('temp_f') is not None else 72.0; filled.append('temperature')
    mph, direction = cond.get('wind_mph'), str(cond.get('wind_dir') or '')
    if mph is None:
        mph = 0.0; filled.append('wind')
    open_air = 1.0 - closed
    out_ = open_air * mph / 10.0 if direction.startswith('Out') else 0.0
    in_ = open_air * mph / 10.0 if direction.startswith('In') else 0.0
    wrigley = cond.get('venue') == 'Wrigley Field'
    x = {'temp10': open_air * (float(temp) - 72.0) / 10.0, 'wind_out': out_, 'wind_in': in_,
         'wind_out_wrigley': out_ if wrigley else 0.0, 'wind_in_wrigley': in_ if wrigley else 0.0,
         'closed': closed, 'day': 1.0 if cond.get('day_night') == 'day' else 0.0}
    return x, filled


def log_multipliers(cond: dict, table: dict) -> np.ndarray:
    """Seven log-multipliers in the model's label order (BIP_OUT, K, BB_HBP, 1B, 2B_3B, HR, OTHER_REACH)."""
    x, _ = features(cond, table)
    z = np.zeros(7)
    for f in table['features']:
        z += float(x.get(f, 0.0)) * np.asarray(table['coef'][f], float)
    venue = (table.get('venue') or {}).get(cond.get('venue') or '')
    if venue is not None:
        z += np.asarray(venue, float)
    return z - np.asarray(table['center'], float)


def describe(cond: dict, table: dict) -> str:
    x, filled = features(cond, table)
    parts = [cond.get('venue') or 'unknown venue']
    if 'temperature' not in filled and not x['closed']:
        parts.append(f"{cond['temp_f']}F")
    if x['closed'] >= 1.0:
        parts.append('roof closed')
    if cond.get('wind_mph') and cond.get('wind_dir') and not x['closed']:
        parts.append(f"wind {cond['wind_mph']} mph {cond['wind_dir']}")
    if cond.get('day_night'):
        parts.append(cond['day_night'])
    if filled:
        parts.append('usual ' + ' and '.join(filled) + ' for the venue and month')
    return 'run environment (' + table.get('name', 'environment') + '): ' + ', '.join(parts)
