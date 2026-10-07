"""Run environment at simulation time (ENV-02): conditions from the feed, climate fallback, multipliers, provider wrapper."""
import numpy as np

from brl_live import environment as env
from brl_live.provider_adjust import EnvironmentAdjust, SIM_LABELS


class Flat:
    name = 'flat'; validation_status = 'test'

    def probabilities(self, ctx):
        return {k: 1 / 7 for k in SIM_LABELS}


def test_conditions_from_feed_and_features():
    table = env.load_table()
    gd = {'weather': {'condition': 'Clear', 'temp': '92', 'wind': '15 mph, Out To CF'}, 'venue': {'name': 'Wrigley Field'},
          'datetime': {'dayNight': 'day', 'officialDate': '2026-07-04'}}
    c = env.conditions_from_feed(gd)
    assert c == {'venue': 'Wrigley Field', 'temp_f': 92, 'wind_mph': 15, 'wind_dir': 'Out To CF', 'condition': 'Clear', 'day_night': 'day', 'month': 7}
    x, filled = env.features(c, table)
    assert filled == [] and abs(x['temp10'] - 2.0) < 1e-12 and x['wind_out'] == 1.5 and x['wind_out_wrigley'] == 1.5 and x['closed'] == 0 and x['day'] == 1
    z = env.log_multipliers(c, table)
    hr = table['labels'].index('HR')
    assert z.shape == (7,) and z[hr] > 0.3                    # a hot day with the wind blowing out at Wrigley: many more home runs
    # roof closed: weather does not count
    c2 = env.conditions(venue='Chase Field', temp='70', wind='0 mph, None', condition='Roof Closed', day_night='night', date='2026-07-04')
    x2, _ = env.features(c2, table)
    assert x2['closed'] == 1 and x2['temp10'] == 0 and x2['wind_out'] == 0
    assert 'roof closed' in env.describe(c2, table)


def test_climate_fallback_when_weather_is_not_posted():
    table = env.load_table()
    c = env.conditions_from_feed({'venue': {'name': 'Coors Field'}, 'datetime': {'dayNight': 'night', 'officialDate': '2026-08-10'}})
    x, filled = env.features(c, table)
    clim = table['climate']['Coors Field']['8']
    assert filled == ['roof', 'temperature', 'wind'] and abs(x['temp10'] - (clim['temp_f'] - 72) / 10) < 1e-9 and x['wind_out'] == 0
    assert 'usual roof and temperature and wind' in env.describe(c, table)
    # an unknown venue and month: neutral temperature, no venue effect
    x, filled = env.features(env.conditions(venue='Nowhere Park'), table)
    assert x['temp10'] == 0 and x['closed'] == 0


def test_average_conditions_leave_the_model_nearly_unchanged():
    table = env.load_table()
    z = env.log_multipliers(env.conditions(venue='Nowhere Park', temp='72', wind='0 mph, Calm', condition='Clear', day_night='night', date='2026-06-01'), table)
    assert np.max(np.abs(z)) < 0.05


def test_environment_adjust_multiplies_and_renormalizes():
    z = np.zeros(7); z[5] = np.log(2.0)
    p = EnvironmentAdjust(Flat(), z).probabilities(None)
    assert abs(sum(p.values()) - 1) < 1e-12 and abs(p['home_run'] / p['single'] - 2.0) < 1e-12
    a = EnvironmentAdjust(Flat()); assert abs(a.probabilities(None)['home_run'] - 1 / 7) < 1e-12
    a.set_environment(z); assert abs(a.probabilities(None)['home_run'] - 2 / 8) < 1e-12


def test_adjusted_provider_adds_the_environment_only_when_enabled():
    from brl_live.boxscore import adjusted_provider, ADJUST
    c = env.conditions(venue='Coors Field', temp='85', wind='10 mph, Out To CF', condition='Clear', day_night='night', date='2026-07-01')
    on = dict(ADJUST, environment=True, context_offsets=False)
    prov, hook, label = adjusted_provider(Flat(), None, '2026-07-01', on, c)
    assert isinstance(prov, EnvironmentAdjust) and any('run environment' in l and 'Coors Field' in l for l in label)
    prov, hook, label = adjusted_provider(Flat(), None, '2026-07-01', dict(on, environment=False), c)
    assert not isinstance(prov, EnvironmentAdjust) and not any('run environment' in l for l in label)
