"""Box-score publishing layer around the unchanged live PA/game forecast engine."""
from __future__ import annotations
import json,os
from dataclasses import asdict
from pathlib import Path
from datetime import timedelta
from app.common import ROOT,content_hash,canonical,timestamp
from app.safety import Blocked
from app import engine_bridge as bridge
from app.team_baseline import predict
from cloud import runner as original
from cloud.contracts import (config_for,draw_seeds,utcnow,summarize,score_versions,assert_finished_before_start)
from cloud.security import key_bytes
from .live_extension import RefreshedSimulator,make_history_engine
from .history_refresh import HistoryCache,Fetcher
from .verified_store import VerifiedGitStore
from .boxscore import run_box_worlds,parse_actual_box,score_player_boxes,bookkeeping_history_from_cache,validate_box_payload
from .box_page import render_page
from .edge_metrics import (freeze_skill_baselines, prior_boxes_from_cache, score_skill_boxes)


class BoxSimulator(RefreshedSimulator):
    def __init__(self,history_info,annotations,skill_history=None,skill_origin=None):
        super().__init__(history_info)
        self.annotations=annotations
        self.box_output=None
        self.skill_history=skill_history
        self.skill_origin=skill_origin
    def run(self,game,matchup):
        parameters={'date':game['date'],'park':game['home']['abbr'],'game':game,
                    'matchup':asdict(matchup),'config':config_for(game['game_type'])}
        engine,decoded=make_history_engine(parameters,self.path)
        results,box=run_box_worlds(engine,decoded,self.annotations,game['date'],draw_seeds(game['game_pk']),full_history=self.history)
        baseline=predict(game,self.rows,self.fit)
        metadata={**box,'game_pk':game['game_pk'],'date':game['date'],
                  'starters':{s:{'player_id':getattr(decoded,s).starter.player_id} for s in ('away','home')}}
        if self.skill_history is not None:
            box['skill_baselines']=freeze_skill_baselines(metadata,self.skill_history,baseline,self.skill_origin)
        self.box_output=box
        return results,baseline


def _ensure_fields(ledger):
    for k in ('box_scores','box_publications','actual_boxes','player_scores','skill_scores'):ledger.setdefault(k,{})

class BoxRunner(original.Runner):
    def process(self,pk,item):
        _ensure_fields(self.store.ledger)
        url=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        feed,receipt=self.net.json(url);gd=feed['gameData']
        if feed.get('gamePk')!=pk:raise Blocked('Wrong official game identity')
        if gd['status']['abstractGameState']=='Final':
            result=original.parse_final(feed,pk,receipt['finished_at'])
            self.store.private('results',content_hash(result),{'result':result,'source':feed,'receipt':receipt})
            self.store.ledger['actuals'][str(pk)]={k:result[k] for k in ('game_pk','away','home','team_ids','first_pitch_observed_at','fetched_at')}
            # This parser and its sources are never supplied to the same game's
            # prediction methods. Final comparisons use player IDs, not names.
            actual=parse_actual_box(feed,pk,receipt['finished_at'])
            self.store.ledger['actual_boxes'][str(pk)]=actual
            self.store.ledger['status'][str(pk)]={'state':'final','date':gd['datetime']['officialDate'],'checked_at':self.clock().isoformat()}
            self.store.persist();return
        if gd['status']['abstractGameState']!='Preview':
            self.store.ledger['status'][str(pk)]={'state':gd['status']['detailedState'],'date':gd['datetime']['officialDate'],'checked_at':self.clock().isoformat()}
            self.store.persist();return
        game,matchup,notes,statuses,fingerprint=self.sim.prepare(feed,receipt)
        previous=original.fingerprint_from_forecasts(self.store,pk)
        if any(f['snapshot_hash']==fingerprint and ident in self.store.ledger['box_scores'] and 'skill_baselines' in self.store.ledger['box_scores'][ident] for ident,f in previous):return
        self.store.private('inputs',content_hash({'receipt':receipt,'game':game}),
            {'source':feed,'receipt':receipt,'derived_game':game,'lineup_status':statuses,
             'history_sha256':self.sim.info['history_sha256'],'history_index_sha256':self.sim.info['index_sha256'],
             'history_coverage_through':self.sim.info['coverage_through']})
        results,baseline=self.sim.run(game,matchup)
        guard,guard_receipt=self.net.json(url);finished=self.clock().isoformat()
        assert_finished_before_start(game,guard,finished)
        if timestamp(guard_receipt['finished_at'])>timestamp(finished):raise Blocked('Invalid guard clock')
        forecast=summarize(game,statuses,results,receipt['finished_at'],finished,baseline,notes,fingerprint,len(previous)+1,self.run_id)
        box={**self.sim.box_output,'schema':'brl.box-forecast.v1','game_pk':pk,'date':game['date'],
             'saved_at':finished,'forecast_origin':receipt['finished_at'],
             'team_ids':{s:game[s]['team_id'] for s in ('away','home')},
             'starters':{s:{k:game[s]['starter'][k] for k in ('player_id','name')} for s in ('away','home')},
             'lineup_status':statuses,'history_through':notes['history_through_used']}
        # Publishing the distributions now, not backdating them to an earlier
        # win-only forecast. Old immutable summaries remain untouched.
        if box.get('skill_baselines') and timestamp(box['skill_baselines']['as_of'])>timestamp(box['forecast_origin']):
            raise Blocked('Baseline built after the forecast origin')
        validate_box_payload(box)
        ident=self.store.publish(forecast);box['forecast_id']=ident
        box_id=content_hash(box)
        reply=self.store.put('box_forecasts/'+box_id+'.json',canonical(box),immutable=True)
        if reply is None:raise Blocked('New player forecast has no fresh publication receipt')
        commit=reply['commit']['sha']
        pub={'commit':commit,'published_at':self.clock().isoformat(),'box_sha256':box_id}
        self.store.ledger['box_scores'][ident]=box
        self.store.ledger['box_publications'][ident]=pub
        self.store.ledger['status'][str(pk)]={'state':'forecast_saved','date':game['date'],'forecast_id':ident,'checked_at':finished}
        self.store.persist()


def main(public_dir):
    key=key_bytes(os.environ.get('BRL_PA_PACKAGE_KEY',''))
    store=VerifiedGitStore(os.environ['GITHUB_REPOSITORY'],os.environ['GH_TOKEN'],key)
    _ensure_fields(store.ledger);before=len(store.ledger['forecasts']);old_boxes=len(store.ledger['box_scores'])
    cache=HistoryCache(store,key);index=cache.refresh(Fetcher())
    info=cache.assemble(index,bridge.HISTORY,ROOT/'private_work/history.csv.gz',utcnow())
    annotations=bookkeeping_history_from_cache(cache,index,utcnow())
    skill_origin=utcnow()
    skill_history=prior_boxes_from_cache(cache,index,skill_origin)
    sim=BoxSimulator(info,annotations,skill_history,skill_origin.isoformat())
    runner=BoxRunner(original.Network(),store,sim,run_id=os.environ.get('GITHUB_RUN_ID','local'))
    try:runner.iteration()
    finally:
        ledger=store.ledger
        scores=score_versions(ledger['forecasts'],ledger['publications'],ledger['actuals'])
        ledger['player_scores']=score_player_boxes(ledger['box_scores'],ledger['box_publications'],ledger['actual_boxes'])
        ledger['skill_scores']=score_skill_boxes(ledger['box_scores'],ledger['box_publications'],ledger['actual_boxes'])
        store.persist();render_page(ledger,scores,public_dir)
    for name in ('index.html','predictions.json','.nojekyll'):store.put('public/'+name,(Path(public_dir)/name).read_bytes())
    return {'status':'iteration_finished','forecasts':len(ledger['forecasts']),
        'live_forecasts_created':len(ledger['forecasts'])-before,'box_forecasts_created':len(ledger['box_scores'])-old_boxes,
        'box_forecasts':len(ledger['box_scores']),'player_scored_games':ledger['player_scores']['n_games'],
        'scored_games':scores['n_games'],'skill_scored_games':ledger['skill_scores']['n_games'],
        'skill_baseline_prior_games':len(skill_history),'raw_data_published':False,
        'history_coverage_through':info['coverage_through'],'history_added_PA':info['added_PA'],
        'pitch_bookkeeping_prior_PA':len(annotations),'engine_rules_changed':False,
        'model_parameters_changed':False,'storage_read_audit':store.read_audit}
