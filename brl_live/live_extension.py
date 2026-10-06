"""Use accepted prior-day history in the live worker; leave locked models intact.

Imports the pinned decrypted runtime. No plaintext model, history, player IDs,
or source snapshots are embedded in this public source module.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from app.common import ROOT, canonical, content_hash, digest, timestamp
from app.safety import Blocked
from app import engine_bridge as bridge
from app.portable import PortableLockedProvider, PortableStarterPolicy, pinned_document
from app.team_baseline import load_results, fit_dispersion, predict
from research_lab.game_sim.engine import GameSimulator
from research_lab.game_sim.models import SimulationConfig
from research_lab.game_sim.fixtures import load_names
from research_lab.game_sim.starter_hazard import tendencies_at
from research_lab.game_sim.locked_pa_provider import EXPECTED_MODEL_SHA256
from cloud import runner as original
from cloud.contracts import (live_inputs, config_for, draw_seeds, utcnow,
                             assert_finished_before_start, summarize, score_versions)
from .refreshed_page import render_page
from cloud.security import key_bytes
from .history_refresh import HistoryCache, Fetcher, previous_day


def make_history_engine(parameters: dict, history_path: Path):
    """Same coefficients, preprocessing and policies; explicit history dependency."""
    _, births = load_names(bridge.NAMES)
    if digest(bridge.MODEL) != EXPECTED_MODEL_SHA256:
        raise Blocked('Locked PA artifact changed')
    if parameters['date'] < '2026-01-01':
        raise Blocked('Live lock used 2025 calibration; cannot forecast earlier dates')
    provider = PortableLockedProvider(bridge.MODEL, history_path, parameters['date'],
                                     parameters['date'], parameters['park'], birthdates=births)
    history = pd.read_csv(history_path, low_memory=False)
    manager_doc = pinned_document('manager')
    if manager_doc['source_sha256'] != digest(bridge.HAZARD):
        raise Blocked('Original manager artifact changed')
    pexp, texp = tendencies_at(history, parameters['date'], float(manager_doc['league_mean_bf']))
    matchup = bridge.decode_matchup(parameters['matchup'])
    manager = PortableStarterPolicy(manager_doc, pexp, texp,
        {int(matchup.away.starter.player_id): matchup.away.team_id,
         int(matchup.home.starter.player_id): matchup.home.team_id})
    return GameSimulator(provider, manager_policy=manager,
                         config=SimulationConfig(**parameters.get('config', {}))), matchup


class RefreshedSimulator:
    def __init__(self, history_info: dict):
        self.info = history_info
        self.path = Path(history_info['history_path'])
        if digest(self.path) != history_info['history_sha256']:
            raise Blocked('Assembled history hash mismatch')
        self.history = pd.read_csv(self.path, low_memory=False)
        self.history['date_key'] = self.history['date_key'].astype(str).str[:10]
        rows, _ = load_results()
        by_game = {row['game_pk']: row for row in rows}
        for row in history_info['new_team_results']:
            # Preserve the inherited regular-season-only baseline recipe.
            if row['game_type'] != 'R':
                continue
            value = {k: v for k, v in row.items() if k != 'game_type'}
            old = by_game.get(value['game_pk'])
            if old is not None and old != value:
                raise Blocked('Team result revision conflicts with locked seed data')
            by_game[value['game_pk']] = value
        self.rows = list(by_game.values())
        self.fit = fit_dispersion(self.rows)  # still 2023-2024 only

    def prepare(self, feed, receipt):
        origin = timestamp(receipt['finished_at'])
        if timestamp(self.info['captured_before']) > origin:
            raise Blocked('History was assembled after input capture')
        if self.info['coverage_through'] != previous_day(origin).isoformat():
            raise Blocked('History coverage is not through yesterday')
        game, matchup, notes, statuses, fingerprint = live_inputs(feed, receipt, self.history)
        notes['source_vintage'] = 'Frozen seed plus encrypted, captured-before-forecast daily updates'
        notes['coverage_through'] = self.info['coverage_through']
        return game, matchup, notes, statuses, content_hash({
            'lineup_fingerprint': fingerprint, 'history_sha256': self.info['history_sha256']})

    def run(self, game, matchup):
        parameters = {'date': game['date'], 'park': game['home']['abbr'], 'game': game,
                      'matchup': asdict(matchup), 'config': config_for(game['game_type'])}
        sim, decoded = make_history_engine(parameters, self.path)
        results = [sim.simulate(decoded, int(seed), record_events=False)
                   for seed in draw_seeds(game['game_pk'])]
        return results, predict(game, self.rows, self.fit)


class RefreshRunner(original.Runner):
    def process(self, pk, item):
        url = f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        feed, receipt = self.net.json(url)
        gd = feed['gameData']
        if feed.get('gamePk') != pk:
            raise Blocked('Wrong game returned by official source')
        if gd['status']['abstractGameState'] == 'Final':
            result = original.parse_final(feed, pk, receipt['finished_at'])
            old = self.store.ledger['actuals'].get(str(pk))
            if old and all(old[k] == result[k] for k in ('away', 'home', 'team_ids', 'first_pitch_observed_at')):
                return
            self.store.private('results', content_hash(result), {'result': result, 'source': feed, 'receipt': receipt})
            self.store.ledger['actuals'][str(pk)] = {k: result[k] for k in
                ('game_pk', 'away', 'home', 'team_ids', 'first_pitch_observed_at', 'fetched_at')}
            self.store.ledger['status'][str(pk)] = {'state': 'final', 'date': gd['datetime']['officialDate'],
                                                 'checked_at': self.clock().isoformat()}
            self.store.persist(); return
        if gd['status']['abstractGameState'] != 'Preview':
            self.store.ledger['status'][str(pk)] = {'state': gd['status']['detailedState'],
                'date': gd['datetime']['officialDate'], 'checked_at': self.clock().isoformat()}
            self.store.persist(); return
        game, matchup, notes, statuses, fingerprint = self.sim.prepare(feed, receipt)
        previous = original.fingerprint_from_forecasts(self.store, pk)
        if any(f['snapshot_hash'] == fingerprint for _, f in previous):
            return
        self.store.private('inputs', content_hash({'receipt': receipt, 'game': game}),
            {'source': feed, 'receipt': receipt, 'derived_game': game, 'lineup_status': statuses,
             'history_sha256': self.sim.info['history_sha256'],
             'history_index_sha256': self.sim.info['index_sha256'],
             'history_coverage_through': self.sim.info['coverage_through']})
        results, baseline = self.sim.run(game, matchup)
        guard, guard_receipt = self.net.json(url)
        finished = self.clock().isoformat()
        assert_finished_before_start(game, guard, finished)
        if timestamp(guard_receipt['finished_at']) > timestamp(finished):
            raise Blocked('Guard response clock invalid')
        forecast = summarize(game, statuses, results, receipt['finished_at'], finished,
                             baseline, notes, fingerprint, len(previous) + 1, self.run_id)
        ident = self.store.publish(forecast)
        self.store.ledger['status'][str(pk)] = {'state': 'forecast_saved', 'date': game['date'],
                                              'forecast_id': ident, 'checked_at': finished}
        self.store.persist()


def main(public_dir: str) -> dict:
    key = key_bytes(os.environ.get('BRL_PA_PACKAGE_KEY', ''))
    store = original.GitStore(os.environ['GITHUB_REPOSITORY'], os.environ['GH_TOKEN'], key)
    before = len(store.ledger['forecasts'])
    cache = HistoryCache(store, key)
    index = cache.refresh(Fetcher())
    info = cache.assemble(index, bridge.HISTORY, ROOT / 'private_work/history.csv.gz', utcnow())
    simulator = RefreshedSimulator(info)
    runner = RefreshRunner(original.Network(), store, simulator,
                           run_id=os.environ.get('GITHUB_RUN_ID', 'local'))
    try:
        runner.iteration()
    finally:
        ledger = store.ledger
        scores = score_versions(ledger['forecasts'], ledger['publications'], ledger['actuals'])
        render_page(ledger, scores, public_dir)
    for name in ('index.html', 'predictions.json', '.nojekyll'):
        store.put('public/' + name, (Path(public_dir) / name).read_bytes())
    return {'status': 'iteration_finished', 'forecasts': len(ledger['forecasts']),
            'live_forecasts_created': len(ledger['forecasts']) - before,
            'scored_games': scores['n_games'], 'raw_data_published': False,
            'history_coverage_through': info['coverage_through'],
            'history_last_game_date': str(simulator.history['date_key'].max()),
            'history_added_PA': info['added_PA'], 'history_index_sha256': info['index_sha256']}
