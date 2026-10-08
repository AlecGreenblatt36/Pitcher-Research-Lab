"""Box-score publishing layer around the unchanged live PA/game forecast engine."""
from __future__ import annotations
import json,os,re,traceback
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
from .history_refresh import HistoryCache,Fetcher,previous_day
from .verified_store import VerifiedGitStore
from .boxscore import run_box_worlds,parse_actual_box,score_player_boxes,bookkeeping_history_from_cache,validate_box_payload,ADJUST,adjusted_provider
from .box_page import render_page
from .live_feed import live_matchup,appearances,matchup_parameters
from .live_update import run_live_update
from .market import capture as capture_market
from .team_model import TeamModel, StarterAdjust
from zoneinfo import ZoneInfo
from .edge_metrics import (freeze_skill_baselines, prior_boxes_from_cache, score_skill_boxes)


class BoxSimulator(RefreshedSimulator):
    def __init__(self,history_info,annotations,skill_history=None,skill_origin=None):
        super().__init__(history_info)
        self.annotations=annotations
        self.box_output=None
        self.skill_history=skill_history
        self.skill_origin=skill_origin
        self._appearances=None
        try:self.team_model=TeamModel(self.rows)
        except Exception:self.team_model=TeamModel([])
        # Starting-pitcher adjustment of the team model (TEAM-02); without it the team model is the v1 recipe.
        try:self.starter_adjust=StarterAdjust(self.history,self.rows);self.starter_adjust_error=None
        except Exception as exc:self.starter_adjust=None;self.starter_adjust_error=type(exc).__name__+': '+str(exc)[:160]
    def appearances(self):
        if self._appearances is None:self._appearances=appearances(self.history)
        return self._appearances
    def update_live(self,feed,date,updated_at):
        """Continue an in-progress game from its observed state; a snapshot, not a forecast version."""
        matchup,start,state,game=live_matchup(feed,self.history,date,self.appearances())
        engine,_=make_history_engine(matchup_parameters(game,matchup,config_for(game['game_type'])),self.path,getattr(self,'physics_table',None))
        try:
            from .environment import conditions_from_feed
            environment=conditions_from_feed(feed.get('gameData') or {})
        except Exception:
            environment=None
        return run_live_update(engine,matchup,start,state,game,self.history,updated_at=updated_at,environment=environment,
                               teams=((game.get('away') or {}).get('abbr'),(game.get('home') or {}).get('abbr')))
    def run(self,game,matchup):
        parameters={'date':game['date'],'park':game['home']['abbr'],'game':game,
                    'matchup':asdict(matchup),'config':config_for(game['game_type'])}
        engine,decoded=make_history_engine(parameters,self.path,getattr(self,'physics_table',None))
        teams=(game['away'].get('abbr'),game['home'].get('abbr'))
        results,box=run_box_worlds(engine,decoded,self.annotations,game['date'],draw_seeds(game['game_pk']),full_history=self.history,
                                   environment=getattr(self,'environment',None),teams=teams)
        # Hitter-against-pitcher grid from the same adjusted model (brl_live/matchups.py); never blocks the forecast.
        try:
            from .matchups import matchup_grid
            grid_provider,_,_=adjusted_provider(engine.provider,self.history,game['date'],ADJUST,getattr(self,'environment',None),teams)
            box['matchups']=matchup_grid(grid_provider,decoded,box)
        except Exception as exc:
            box['matchups']={'error':type(exc).__name__+': '+str(exc)[:160]}
        baseline=predict(game,self.rows,self.fit)
        # The team half of the headline blend: our decayed negative-binomial team model, the one
        # measured on the 2026 replay next to the simulator. The runtime's own baseline is kept
        # in the forecast record unchanged.
        starters=None;sp_error=self.starter_adjust_error
        if self.starter_adjust is not None:
            try:
                scale=float((ADJUST.get('postseason_exp_scale') or {}).get(game['game_type'],1.0))
                starters=self.starter_adjust.factors(game['date'],game['home']['team_id'],game['away']['team_id'],
                                                     decoded.home.starter.player_id,decoded.away.starter.player_id,share_scale=scale)
            except Exception as exc:sp_error=type(exc).__name__+': '+str(exc)[:160]
        try:
            box['team_model']=self.team_model.probability(game['date'],game['home']['team_id'],game['away']['team_id'],starters=starters)
            if sp_error:box['team_model']['starter_adjust_error']=sp_error
        except Exception as exc:box['team_model']={'error':type(exc).__name__+': '+str(exc)[:160],'rows_seen':len(self.team_model.rows)}
        metadata={**box,'game_pk':game['game_pk'],'date':game['date'],
                  'starters':{s:{'player_id':getattr(decoded,s).starter.player_id} for s in ('away','home')}}
        if self.skill_history is not None:
            box['skill_baselines']=freeze_skill_baselines(metadata,self.skill_history,baseline,self.skill_origin)
        self.box_output=box
        return results,baseline


def _ensure_fields(ledger):
    for k in ('box_scores','box_publications','actual_boxes','player_scores','skill_scores','live','context'):ledger.setdefault(k,{})

def game_context(item,gd,now):
    """Public game context for the page: schedule facts plus the official feed's venue and weather (no player data)."""
    out=dict((item or {}).get('context') or {})
    venue=(gd.get('venue') or {}).get('name')
    if venue:out['venue']=venue
    w=gd.get('weather') or {}
    weather={k:w.get(k) for k in ('condition','temp','wind') if w.get(k) not in (None,'')}
    if weather:out['weather']=weather
    dt=gd.get('datetime') or {}
    if dt.get('officialDate'):out['date']=dt['officialDate']
    out['updated_at']=now
    return out

LIVE_STATES=('In Progress','Manager challenge','Umpire review','Delayed')

class BoxRunner(original.Runner):
    def live_update(self,pk,feed,date):
        # Never lets an in-game problem block pregame forecasts: errors are recorded, not raised.
        now=self.clock().isoformat()
        try:
            snapshot=self.sim.update_live(feed,date,now)
        except Exception as exc:
            frames=traceback.extract_tb(exc.__traceback__)
            where={'file':Path(frames[-1].filename).name,'function':frames[-1].name,'line':frames[-1].lineno} if frames else {}
            snapshot={'schema':'brl.live-update.v1','game_pk':pk,'date':date,'updated_at':now,
                'error':type(exc).__name__+': '+str(exc)[:200],'error_location':where}
        try:
            # The real game so far, for the page: the plays with their pitches, and the lines (display only).
            from .real_game import plays_from_feed,box_from_feed
            snapshot['plays']=plays_from_feed(feed);snapshot['box']=box_from_feed(feed)
        except Exception as exc:
            snapshot['plays_error']=type(exc).__name__+': '+str(exc)[:200]
        self.store.ledger.setdefault('live',{})[str(pk)]=snapshot
    def process(self,pk,item):
        _ensure_fields(self.store.ledger)
        url=f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'
        feed,receipt=self.net.json(url);gd=feed['gameData']
        if feed.get('gamePk')!=pk:raise Blocked('Wrong official game identity')
        self.store.ledger['context'][str(pk)]=game_context(item,gd,self.clock().isoformat())
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
            if gd['status']['abstractGameState']=='Live' and str(gd['status'].get('detailedState','')).startswith(LIVE_STATES):
                self.live_update(pk,feed,gd['datetime']['officialDate'])
            self.store.persist();return
        game,matchup,notes,statuses,fingerprint=self.sim.prepare(feed,receipt)
        # The game's run environment as the official feed shows it now (ENV-02); not part of the fingerprint,
        # so a weather update alone never makes a new version. The box records what was used.
        try:
            from .environment import conditions_from_feed
            self.sim.environment=conditions_from_feed(gd)
        except Exception:
            self.sim.environment=None
        if bridge.MODEL_NAME!='locked-pa-2026-v1':
            # A different PA model is a different forecast: the snapshot hash carries it, so a model
            # switch produces a new saved version for every pending game and earlier versions stand.
            fingerprint=content_hash({'inputs':fingerprint,'pa_model':bridge.MODEL_NAME,'model_sha256':bridge.MODEL_SHA256})
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


def _market(runner,ledger):
    # Reference line only; failures are recorded, never raised.
    now=runner.clock()
    try:
        day=now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        from .market_books import fetch_html
        ledger['market_receipt']=capture_market(runner.net.json,day,ledger,now.isoformat(),books_fetch=fetch_html)
    except Exception as exc:
        ledger['market_receipt']={'error':type(exc).__name__+': '+str(exc)[:200],'at':now.isoformat()}

def _prune_live(ledger):
    """Drop live snapshots of games that are final; the record scores forecasts, not snapshots."""
    for pk in list(ledger.get('live',{})):
        if pk in ledger.get('actuals',{}):ledger['live'].pop(pk,None)

def main(public_dir):
    key=key_bytes(os.environ.get('BRL_PA_PACKAGE_KEY',''))
    store=VerifiedGitStore(os.environ['GITHUB_REPOSITORY'],os.environ['GH_TOKEN'],key)
    _ensure_fields(store.ledger);_prune_live(store.ledger);before=len(store.ledger['forecasts']);old_boxes=len(store.ledger['box_scores'])
    cache=HistoryCache(store,key);refresh_note=None
    try:index=cache.refresh(Fetcher())
    except Blocked as exc:
        # Yesterday cannot be accepted yet (its last game is still being played past midnight
        # Eastern, or the source has not published the day): carry on with the last accepted
        # history, which may run through the day before yesterday but never older.
        from datetime import timedelta
        index=cache.index();refresh_note=str(exc)
        previous=previous_day(utcnow());accepted=index.get('coverage_through')
        if accepted not in (previous.isoformat(),(previous-timedelta(days=1)).isoformat()):raise
    info=cache.assemble(index,bridge.HISTORY,ROOT/'private_work/history.csv.gz',utcnow())
    info['yesterday_incomplete']=bool(refresh_note);info['refresh_note']=refresh_note
    annotations=bookkeeping_history_from_cache(cache,index,utcnow())
    skill_origin=utcnow()
    skill_history=prior_boxes_from_cache(cache,index,skill_origin)
    sim=BoxSimulator(info,annotations,skill_history,skill_origin.isoformat())
    physics_receipt=None
    if bridge.MODEL_NAME!='locked-pa-2026-v1':
        # A selected model with physics features needs the per-PA physics table through yesterday.
        from .physics_inputs import assemble_physics_table
        sim.physics_table,physics_receipt=assemble_physics_table(cache,index,utcnow())
        manifest_path=ROOT/'brl_engine'/'models'/(bridge.MODEL_NAME+'.json')
        manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if 'f_def' in (manifest.get('physics_features') or []):
            # The model reads each fielding team's defense: out rate on fieldable balls in play above the
            # league over the prior window, from the same history the forecasts use, through yesterday.
            from research_lab.pa_model.physics import DefenseState
            from zoneinfo import ZoneInfo
            today=utcnow().astimezone(ZoneInfo('America/New_York')).date().isoformat()
            sim.defense_state=DefenseState.build(sim.history,today,manifest.get('physics_params') or {})
            physics_receipt['team_defense']={t:round(sim.defense_state.value(t),4) for t in sorted(sim.defense_state.entries)}
    runner=BoxRunner(original.Network(),store,sim,run_id=os.environ.get('GITHUB_RUN_ID','local'))
    _market(runner,store.ledger)
    try:runner.iteration()
    finally:
        ledger=store.ledger;_prune_live(ledger)
        scores=score_versions(ledger['forecasts'],ledger['publications'],ledger['actuals'])
        ledger['player_scores']=score_player_boxes(ledger['box_scores'],ledger['box_publications'],ledger['actual_boxes'])
        ledger['skill_scores']=score_skill_boxes(ledger['box_scores'],ledger['box_publications'],ledger['actual_boxes'])
        store.persist();render_page(ledger,scores,public_dir)
    for name in ('index.html','predictions.json','.nojekyll'):store.put('public/'+name,(Path(public_dir)/name).read_bytes())
    # Day archives go to the ledger branch; the static season files live in the repository (brl_live/archive).
    for p in sorted((Path(public_dir)/'days').glob('*.json')):
        if re.fullmatch(r'(\d{4}-\d{2}-\d{2}|index)\.json',p.name):store.put('public/days/'+p.name,p.read_bytes())
    return {'status':'iteration_finished','forecasts':len(ledger['forecasts']),
        'live_forecasts_created':len(ledger['forecasts'])-before,'box_forecasts_created':len(ledger['box_scores'])-old_boxes,
        'box_forecasts':len(ledger['box_scores']),'player_scored_games':ledger['player_scores']['n_games'],
        'scored_games':scores['n_games'],'skill_scored_games':ledger['skill_scores']['n_games'],
        'skill_baseline_prior_games':len(skill_history),'raw_data_published':False,
        'history_coverage_through':info['coverage_through'],'history_added_PA':info['added_PA'],'history_refresh_note':refresh_note,
        'history_day_notes':{day:entry.get('notes') for day,entry in sorted(index.get('days',{}).items())[-3:] if entry.get('notes')},
        'pa_model':{'name':bridge.MODEL_NAME,'sha256':bridge.MODEL_SHA256,'physics':physics_receipt},
        'pitch_bookkeeping_prior_PA':len(annotations),'pitch_bookkeeping_season_games':int(getattr(annotations,'attrs',{}).get('season_games',0)),'engine_rules_changed':False,
        'model_parameters_changed':False,'simulation_adjustments':'context offsets (brl_live/context_offsets.json)',
        'live_updates':len(ledger['live']),'live_update_errors':sum(1 for v in ledger['live'].values() if v.get('error')),
        'market_receipt':ledger.get('market_receipt'),'team_results_rows':len(getattr(sim,'rows',[]) or []),
        'team_results_meta':getattr(sim,'results_meta',None),
        'storage_read_audit':store.read_audit}
