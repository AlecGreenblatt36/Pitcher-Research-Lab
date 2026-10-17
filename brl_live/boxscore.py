"""Observation-only box scoring of the locked simulator's exact game paths.

No engine random stream, PA probability, runner transition or manager decision is
changed. BB/HBP and pitch counts are *bookkeeping estimates* sampled from earlier
terminal PAs on an independent keyed stream. Pitch sequences are not generated.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import asdict
from typing import Any
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from app.safety import Blocked
from research_lab.game_sim.engine import GameSimulator
from .world_selection import world_features, select_worlds, SELECTION_NOTE
from .pitch_bridge import PitchBridge, pitch_list, contact_of, MIN_BATTER_CONTACT
from .win_table import WinTable

BAT = ('PA','AB','H','2B','3B','HR','R','RBI','BB','HBP','K','SF','SB','CS')
STEALS = ('stolen_base','caught_stealing')
# Running plays between plate appearances other than steals (research_lab.game_sim.running_events, TRANS-02).
RUNNING = ('wild_pitch','passed_ball','balk','defensive_indifference','pickoff','pickoff_error','error','other_advance','runner_out')
PIT = ('outs','PC','H','R','BB','HBP','K','HR','BF')
SIDE = ('away','home')

class BookkeepingFit:
    """Pitch counts and the walk/hit-by-pitch split for box scores.

Pitch counts: empirical pools of terminal pitch numbers from official prior-date
playEvents (the seed corpus has no real pitch numbers). A pitcher's own pool is
used only when it holds at least MIN_POOL compatible plate appearances; smaller
pools fall back to the league pool for the outcome and batter hand, then outcome.

Hit by pitch: drawn separately from a shrunk rate, never from raw pools. The
share of walk-plus-HBP events that are HBP is the league share for the batter
hand, times the pitcher's shrunk ratio, times the batter's shrunk ratio, each
pulled toward one with K_HBP pseudo-events. The rates come from `full_history`
(the assembled PA history, whose terminal_event column is reliable), or from the
annotation rows when no full history is given. A pitcher who has never hit a
batter still can; no rate is ever zero.
    """
    MIN_POOL=25
    K_HBP=150.0
    def __init__(self, history: pd.DataFrame, date: str, full_history: pd.DataFrame|None=None):
        self.cutoff=date
        h=history.loc[history.date_key.astype(str).str[:10] < date].copy()
        if h.empty: raise Blocked('No prior plate appearances for box bookkeeping')
        if h.date_key.astype(str).str[:10].max() >= date: raise Blocked('Future bookkeeping input')
        self.max_input_date=str(h.date_key.astype(str).str[:10].max())
        h['label']=h.outcome.astype(str).str.lower()
        mapping={'k':'strikeout','1b':'single','2b_3b':'double_triple','hr':'home_run','walk_hbp':'bb_hbp','out':'bip_out'}
        h['label']=h.label.replace(mapping)
        h['pitcher']=h.pitcher.astype(int).astype(str)
        h['hbp']=(h.terminal_event=='hit_by_pitch').astype(int)
        h['pitch_number']=pd.to_numeric(h.pitch_number,errors='coerce')
        h=h.loc[h.pitch_number.notna() & (h.pitch_number>=0) & (h.pitch_number%1==0)]
        # Only genuinely compatible observed counts enter the categorical pools.
        valid=(h.label!='strikeout')|(h.pitch_number>=3)
        valid &= (h.label!='bb_hbp')|(h.hbp==1)|(h.terminal_event=='intent_walk')|(h.pitch_number>=4)
        pools_source=h.loc[valid]
        self.pools={}; self.fallback_counts=Counter()
        for keys,tag in [(('pitcher','label','stand'),'poh'),(('pitcher','label'),'po'),
                         (('label','stand'),'oh'),(('label',),'o')]:
            for key,rows in pools_source.groupby(list(keys),sort=False):
                if not isinstance(key,tuple):key=(key,)
                if tag in ('poh','po') and len(rows)<self.MIN_POOL:continue
                self.pools[(tag,*key)]=rows['pitch_number'].to_numpy(dtype=np.int32)
        self._fit_hbp(h if full_history is None else self._prepare_full(full_history,date))
        records=[]
        if 'pitches' in h.columns:
            has_batter='batter' in h.columns;has_contact='contact' in h.columns
            for r in h.loc[h.pitches.notna()].itertuples():
                if isinstance(r.pitches,list) and r.pitches:
                    batter=getattr(r,'batter',None) if has_batter else None
                    contact=getattr(r,'contact',None) if has_contact else None
                    records.append({'pitcher':int(r.pitcher),'stand':str(r.stand),'outcome':str(r.outcome).lower(),
                                    'terminal_event':str(r.terminal_event),'pitches':r.pitches,
                                    'batter':None if batter is None or (isinstance(batter,float) and np.isnan(batter)) else int(batter),
                                    'contact':contact if isinstance(contact,dict) else None})
        self.bridge=PitchBridge(records)
    @staticmethod
    def _prepare_full(full: pd.DataFrame, date: str) -> pd.DataFrame:
        f=full.loc[full.date_key.astype(str).str[:10] < date, ['outcome','terminal_event','pitcher','stand']+(['batter'] if 'batter' in full.columns else [])].copy()
        f['label']=f.outcome.astype(str).str.lower().replace({'bb_hbp':'bb_hbp','walk_hbp':'bb_hbp'})
        f['pitcher']=f.pitcher.astype(int).astype(str)
        f['hbp']=(f.terminal_event=='hit_by_pitch').astype(int)
        return f
    def _fit_hbp(self, frame: pd.DataFrame) -> None:
        w=frame.loc[frame.label=='bb_hbp']
        if w.empty: raise Blocked('No prior walk or hit-by-pitch events for box bookkeeping')
        self.hbp_league_all=float(w.hbp.mean())
        self.hbp_league={str(k):float(v) for k,v in w.groupby('stand').hbp.mean().items()}
        self.hbp_pitcher=self._ratios(w,'pitcher')
        self.hbp_batter=self._ratios(w,'batter') if 'batter' in w.columns else {}
        self.hbp_support=int(len(w))
    def _ratios(self, w: pd.DataFrame, who: str) -> dict:
        grp=w.groupby(w[who].astype(str)).hbp.agg(['sum','count'])
        shrunk=(grp['sum']+self.K_HBP*self.hbp_league_all)/(grp['count']+self.K_HBP)
        return (shrunk/max(self.hbp_league_all,1e-6)).to_dict()
    def hbp_probability(self, pitcher_id, batter_id, hand) -> float:
        base=self.hbp_league.get(str(hand),self.hbp_league_all)
        p=base*self.hbp_pitcher.get(str(pitcher_id),1.0)*self.hbp_batter.get(str(batter_id),1.0)
        return float(min(0.5,max(0.005,p)))
    def draw(self,event,rng):
        """(hbp, pitch count, pitch list or None) for one simulated plate appearance."""
        pid=str(event['pitcher_id']);label=event['outcome'];hand=event.get('batter_hand','R')
        hbp=int(label=='bb_hbp' and rng.random()<self.hbp_probability(pid,event.get('batter_id'),hand))
        pitches=self.bridge.draw(pid,label,hand,bool(hbp),rng) if self.bridge.n_sequences else None
        if pitches:
            return hbp,len(pitches),pitches
        for key in [('poh',pid,label,hand),('po',pid,label),('oh',label,hand),('o',label)]:
            if key in self.pools:
                a=self.pools[key];self.fallback_counts[key[0]]+=1
                pc=int(a[int(rng.integers(len(a)))])
                if label=='bb_hbp' and not hbp:pc=max(4,pc)
                return hbp,pc,None
        raise Blocked('No compatible earlier PA count pool for '+label)
    @staticmethod
    def contact_kind(event) -> str|None:
        """Which real batted balls describe this simulated play (None when no ball was put in play)."""
        o=event['outcome'];d=event.get('description','')
        if o=='single':return 'single'
        if o=='double_triple':return 'triple' if 'tripled.' in d else 'double'
        if o=='home_run':return 'home_run'
        if o=='bip_out':return 'dp' if 'double play' in d else 'sf' if 'sacrifice fly' in d else 'out'
        if o=='other_reach':return 'fc' if 'fielder' in d else 'error'
        return None
    def contact(self,event,rng):
        """Batted-ball record (t, loc, dist, ev) for one simulated ball in play, or None."""
        kind=self.contact_kind(event)
        if kind is None or not self.bridge.n_contacts:return None
        return self.bridge.draw_contact(event.get('batter_id'),event.get('batter_hand','R'),kind,rng)

class ObservedSimulator(GameSimulator):
    """Listen to already-selected scoring runners, without consuming randomness."""
    def simulate(self,*args,**kwargs):
        self.scoring_trace=[]
        result=super().simulate(*args,**kwargs)
        result.box_scoring_trace=self.scoring_trace
        return result
    def _charge_scored_runners(self,scored_runners,lines):
        self.scoring_trace.append([(r.player_id,r.responsible_pitcher_id,r.automatic) for r in scored_runners])
        return super()._charge_scored_runners(scored_runners,lines)

def _zeros(fields):return {k:0 for k in fields}

def build_game_box(result,matchup,fit:BookkeepingFit) -> dict[str,Any]:
    if not result.events or len(result.events)!=len(result.box_scoring_trace):
        raise Blocked('Complete observed events required for box scoring')
    batting={s:{p.player_id:dict(player_id=p.player_id,name=p.name,spot=i+1,**_zeros(BAT))
                for i,p in enumerate(getattr(matchup,s).lineup)} for s in SIDE}
    pitching={s:{} for s in SIDE};hands={p.player_id:p.bats for s in SIDE for p in getattr(matchup,s).lineup}
    throws={p.player_id:p.throws for s in SIDE for p in (getattr(matchup,s).starter,*getattr(matchup,s).bullpen)}
    rng=np.random.default_rng(np.random.SeedSequence([int(result.seed),0x424F58]))
    pa_counts=Counter();innings={s:{} for s in SIDE};plays=[]
    for event,scored in zip(result.events,result.box_scoring_trace):
        e=dict(event);s=e['batting_side'];fs='home' if s=='away' else 'away';pid=str(e['pitcher_id']);bid=str(e['batter_id'])
        if e['outcome'] in STEALS:
            # A steal between plate appearances: the runner's SB or CS, the pitcher's out on a caught stealing; no PA.
            runner=batting[s].get(bid)
            if runner is not None:runner['SB' if e['outcome']=='stolen_base' else 'CS']+=1
            p=pitching[fs].setdefault(pid,dict(player_id=pid,name=e['pitcher_name'],**_zeros(PIT)))
            p['outs']+=e['outs_after']-e['outs_before']
            e.update(box_outcome=e['outcome'],rbi=0,scoring_players=[],estimated_pitches=0,pitches=[],contact=None)
            plays.append({k:e[k] for k in ('inning','half','batter_id','batter_name','pitcher_id','pitcher_name',
                'box_outcome','description','outs_before','outs_after','runs_scored','away_score','home_score',
                'bases_before','bases_after','scoring_players','rbi','estimated_pitches','pitches','contact')})
            continue
        if e['outcome'] in RUNNING:
            # A wild pitch, passed ball, balk, pickoff ... between plate appearances: runs for the runners who scored
            # (charged to their responsible pitchers), the out to the pitcher on the mound; no PA, no RBI.
            p=pitching[fs].setdefault(pid,dict(player_id=pid,name=e['pitcher_name'],**_zeros(PIT)))
            p['outs']+=e['outs_after']-e['outs_before']
            for runner,responsible,automatic in scored:
                batting[s][str(runner)]['R']+=1
                charged=pid if responsible is None else str(responsible)
                if charged not in pitching[fs]:raise Blocked('Missing responsible pitcher in box score')
                pitching[fs][charged]['R']+=1
            line=innings[s].setdefault(str(e['inning']),{'R':0,'H':0});line['R']+=e['runs_scored']
            e.update(box_outcome=e['outcome'],rbi=0,scoring_players=[str(x[0]) for x in scored],estimated_pitches=0,pitches=[],contact=None)
            plays.append({k:e[k] for k in ('inning','half','batter_id','batter_name','pitcher_id','pitcher_name',
                'box_outcome','description','outs_before','outs_after','runs_scored','away_score','home_score',
                'bases_before','bases_after','scoring_players','rbi','estimated_pitches','pitches','contact')})
            continue
        b=batting[s][bid];p=pitching[fs].setdefault(pid,dict(player_id=pid,name=e['pitcher_name'],**_zeros(PIT)))
        b['PA']+=1;p['BF']+=1;pa_counts[s]+=1
        o=e['outcome'];e['batter_hand']=('L' if throws[pid]=='R' else 'R') if hands[bid]=='S' else hands[bid];hbp,pc,pitches=fit.draw(e,rng);p['PC']+=pc
        e['estimated_pitches']=pc;e['pitches']=pitches;e['contact']=fit.contact(e,rng)
        hit=o in ('single','double_triple','home_run');sf='sacrifice fly' in e['description']
        b['AB']+=int(o!='bb_hbp' and not sf)
        b['SF']+=int(sf)
        if o=='bb_hbp':
            k='HBP' if hbp else 'BB';b[k]+=1;p[k]+=1
            e['box_outcome']='hit_by_pitch' if hbp else 'walk'
        elif hit:
            b['H']+=1;p['H']+=1
            k='HR' if o=='home_run' else ('3B' if 'tripled.' in e['description'] else '2B')
            if o!='single':b[k]+=1
            if o=='home_run':p['HR']+=1
            e['box_outcome']='triple' if k=='3B' else 'double' if o=='double_triple' else o
        elif o=='strikeout': b['K']+=1;p['K']+=1;e['box_outcome']='strikeout'
        else:e['box_outcome']=o
        p['outs']+=e['outs_after']-e['outs_before']
        # Report R for automatic runners against the active pitcher, without
        # changing the legacy manager's own run totals or its subsequent choices.
        for runner,responsible,automatic in scored:
            batting[s][str(runner)]['R']+=1
            charged=pid if responsible is None else str(responsible)
            if charged not in pitching[fs]:raise Blocked('Missing responsible pitcher in box score')
            pitching[fs][charged]['R']+=1
        # MLB's scoring rule 9.04: a run that scores on a hit, a walk or hit-by-pitch, a sacrifice fly, an out in play other
        # than a double or triple play, or a fielder's choice is batted in; one that scores on an error is not
        dsc=e['description']
        rbi=e['runs_scored'] if hit or o=='bb_hbp' or sf or (o=='bip_out' and 'double play' not in dsc and 'triple play' not in dsc) or (o=='other_reach' and "fielder's choice" in dsc) else 0
        b['RBI']+=rbi;e['rbi']=rbi;e['scoring_players']=[str(x[0]) for x in scored]
        inning=str(e['inning']);line=innings[s].setdefault(inning,{'R':0,'H':0})
        line['R']+=e['runs_scored'];line['H']+=int(hit)
        plays.append({k:e[k] for k in ('inning','half','batter_id','batter_name','pitcher_id','pitcher_name',
            'box_outcome','description','outs_before','outs_after','runs_scored','away_score','home_score',
            'bases_before','bases_after','scoring_players','rbi','estimated_pitches','pitches','contact')})
    for s in SIDE:
        opp='home' if s=='away' else 'away';rows=list(batting[s].values());ps=list(pitching[opp].values())
        score=getattr(result,s+'_score')
        if sum(x['R'] for x in rows)!=score or sum(x['R'] for x in ps)!=score:raise Blocked('Player run accounting failed')
        if sum(x['H'] for x in rows)!=sum(x['H'] for x in ps):raise Blocked('Hit accounting failed')
        if sum(x['PA'] for x in rows)!=sum(x['BF'] for x in ps):raise Blocked('PA/BF accounting failed')
        for k in ('BB','HBP','K','HR'):
            if sum(x[k] for x in rows)!=sum(x[k] for x in ps):raise Blocked(k+' accounting failed')
        for row in rows:
            if row['AB']+row['BB']+row['HBP']+row['SF']!=row['PA']:raise Blocked('PA/AB accounting failed')
        for row in ps:
            legacy=result.pitcher_lines[row['player_id']]
            for new,old in [('outs','outs_recorded'),('H','hits_allowed'),('K','strikeouts'),('HR','home_runs'),('BF','batters_faced')]:
                if row[new]!=legacy[old]:raise Blocked('Legacy pitcher line mismatch: '+new)
            if row['BB']+row['HBP']!=legacy['walks_hbp']:raise Blocked('Legacy BB/HBP mismatch')
    return {'seed':int(result.seed),'score':{s:getattr(result,s+'_score') for s in SIDE},
            'innings':innings,'batting':{s:list(batting[s].values()) for s in SIDE},
            'pitching':{s:list(pitching[s].values()) for s in SIDE},'plays':plays}

def _distribution(counter,n):
    return {'n':n,'counts':[[int(k),int(v)] for k,v in sorted(counter.items())]}

# Share of a lineup slot's plate appearances taken by the hitter who started there, by batting order spot, from every
# 2023-2025 regular-season game (the team's k-th plate appearance belongs to spot (k-1) mod 9 + 1). Scored on 2026 (PLAYER-02):
# the hit chance's mean 0.633 to 0.617 against 0.607 actual, Brier -0.00064; home runs -0.00010.
STARTER_SHARE=(0.974,0.976,0.976,0.977,0.969,0.962,0.953,0.942,0.927)

class BoxAccumulator:
    def __init__(self,matchup):
        self.matchup=matchup;self.n=0;self.hist={s:{'batting':{},'pitching':{}} for s in SIDE};self.early=[]
        self.innings={s:defaultdict(lambda:Counter()) for s in SIDE};self.score_pairs=[];self.seeds=[];self.features=[];self.win=WinTable()
        # What decides the game (KEYS-01): for each condition, the worlds where it happened and those the team then won
        self.keys=Counter()
        for s in SIDE:
            t=getattr(matchup,s)
            for i,p in enumerate(t.lineup):self.hist[s]['batting'][p.player_id]={'name':p.name,'spot':i+1,'stats':{k:Counter() for k in BAT},'hs':Counter()}
            for i,p in enumerate((t.starter,*t.bullpen)):self.hist[s]['pitching'][p.player_id]={'name':p.name,'role':p.role,'order':i,'appeared':0,'stats':{k:Counter() for k in PIT}}
    def add(self,box):
        self.n+=1;self.score_pairs.append((box['score']['away'],box['score']['home']));self.seeds.append(box['seed']);self.features.append(world_features(box))
        self.win.add(box)
        for s in SIDE:
            for kind,fields in [('batting',BAT),('pitching',PIT)]:
                index={r['player_id']:r for r in box[kind][s]}
                for pid,agg in self.hist[s][kind].items():
                    row=index.get(pid,_zeros(fields))
                    if kind=='pitching':agg['appeared']+=int(pid in index)
                    for k in fields:agg['stats'][k][row[k]]+=1
                    # hits by singles among them, for his chance of two or more total bases (PLAYER-06)
                    if kind=='batting':agg['hs'][(min(int(row['H']),5),min(int(row['H'])-int(row['2B'])-int(row['3B'])-int(row['HR']),5))]+=1
            for inning,values in box['innings'][s].items():self.innings[s][int(inning)].update(values)
        self._keys(box)
        # each world's first inning and first five innings, road and home runs (NRFI-01, FIRST5-01)
        inn={s:{int(k):int(v.get('R',0)) for k,v in box['innings'][s].items()} for s in SIDE}
        self.early.append((inn['away'].get(1,0),inn['home'].get(1,0),sum(v for k,v in inn['away'].items() if k<=5),sum(v for k,v in inn['home'].items() if k<=5)))
    def _keys(self,box):
        """KEYS-01: count, per team, the worlds where each condition held and where it held and the team won: scoring first,
        leading after five innings, its starter getting 18 or more outs, and each of its hitters homering or getting two or
        more hits."""
        win='home' if box['score']['home']>box['score']['away'] else 'away'
        first=next(('away' if pl.get('half')=='top' else 'home' for pl in box.get('plays') or [] if int(pl.get('runs_scored') or 0)>0),None)
        r5={s:sum(int(v.get('R',0)) for k,v in box['innings'][s].items() if int(k)<=5) for s in SIDE}
        lead5='home' if r5['home']>r5['away'] else 'away' if r5['away']>r5['home'] else None
        for s in SIDE:
            w=int(win==s);conds=[('first',''),] if first==s else []
            if lead5==s:conds.append(('lead5',''))
            sp=str(getattr(self.matchup,s).starter.player_id)
            for row in box['pitching'][s]:
                if str(row['player_id'])==sp and int(row.get('outs') or 0)>=18:conds.append(('sp6',sp))
            for row in box['batting'][s]:
                if int(row.get('HR') or 0)>=1:conds.append(('hr',str(row['player_id'])))
                if int(row.get('H') or 0)>=2:conds.append(('h2',str(row['player_id'])))
            for c in conds:
                self.keys[(s,)+c+('n',)]+=1;self.keys[(s,)+c+('w',)]+=w
            self.keys[(s,'all','','w')]+=w
    def keys_summary(self):
        """KEYS-01: [{'side','kind','player_id','share','win_if','win_else'}]: how often each condition happened and the team's
        win chance when it did and when it did not, over this game's worlds."""
        out=[]
        for (s,kind,pid,what),n in sorted(self.keys.items()):
            if what!='n' or kind=='all' or not n:continue
            w=self.keys[(s,kind,pid,'w')];wa=self.keys[(s,'all','','w')]
            out.append({'side':s,'kind':kind,'player_id':pid,'share':round(n/self.n,4),'win_if':round(w/n,4),
                        'win_else':round((wa-w)/(self.n-n),4) if self.n>n else None})
        return out
    def finish(self):
        if self.n!=10000:raise Blocked('Box projections require 10,000 finished worlds')
        result={'n_simulations':self.n,'teams':{},'line_score':{},'sample_selection':SELECTION_NOTE,'samples':[]}
        for s in SIDE:
            result['teams'][s]={}
            for kind,fields in [('batting',BAT),('pitching',PIT)]:
                result['teams'][s][kind]=[]
                for pid,v in self.hist[s][kind].items():
                    row={'player_id':pid,**{k:x for k,x in v.items() if k not in ('stats','appeared','hs')},
                         'means':{k:sum(value*count for value,count in v['stats'][k].items())/self.n for k in fields},
                         'distributions':{k:_distribution(v['stats'][k],self.n) for k in fields}}
                    if kind=='batting':
                        # The simulated lineup slot bats all game; the hitter himself does not (pinch hitters, defensive
                        # changes): his chance keeps each of the slot's hits with his share of the slot's plate appearances
                        # (PLAYER-02).
                        keep=STARTER_SHARE[min(max(int(v.get('spot') or 1),1),9)-1]
                        row['hit_probability']=1-sum(c*(1-keep)**k for k,c in v['stats']['H'].items())/self.n
                        row['hr_probability']=1-sum(c*(1-keep)**k for k,c in v['stats']['HR'].items())/self.n
                        # two or more total bases of his own: not (no hit kept, or one kept hit that is a single)
                        row['tb2_probability']=1-sum(c*((1-keep)**h+(s1*keep*(1-keep)**(h-1) if h else 0)) for (h,s1),c in v['hs'].items())/self.n
                    else:
                        row['appearance_probability']=v['appeared']/self.n
                        row['means']['IP']=row['means']['outs']/3
                    result['teams'][s][kind].append(row)
            result['line_score'][s]={str(i):{k:v.get(k,0)/self.n for k in ('R','H')} for i,v in sorted(self.innings[s].items())}
        counts=Counter(self.score_pairs)
        features=self.features if len(self.features)==self.n else [
            {'innings':9,'away':a,'home':h,**{k:0 for k in ('away_hits','home_hits','away_outs','home_outs','away_K','home_K','away_PC','home_PC')}}
            for a,h in self.score_pairs]
        picks=select_worlds(features)
        selected=[picks[k] for k in ('projected','high','low','upset') if picks.get(k) is not None]
        for i in picks.get('runners_up',[]):
            if i not in selected:selected.append(i)
            if len(selected)==5:break
        result['sample_roles']={k:picks[k] for k in ('projected','high','low','upset') if picks.get(k) is not None}
        modal=self.score_pairs[selected[0]]
        # Preserve dependence across the entire world: marginal player run
        # histograms cannot be added to reconstruct team or game distributions.
        result['team_run_distributions']={side:_distribution(Counter(p[i] for p in self.score_pairs),self.n)
                                         for i,side in enumerate(SIDE)}
        result['total_run_distribution']=_distribution(Counter(a+h for a,h in self.score_pairs),self.n)
        # home minus away runs over the same worlds, for a run line that keeps the dependence between the two scores (RUNLINE-01)
        result['margin_distribution']=_distribution(Counter(h-a for a,h in self.score_pairs),self.n)
        if len(self.early)==self.n:
            result['first_inning']={'n':self.n,'no_run':sum(1 for a1,h1,_,_ in self.early if a1+h1==0),'away_scores':sum(1 for a1,_,_,_ in self.early if a1>0),
                                    'home_scores':sum(1 for _,h1,_,_ in self.early if h1>0)}
            result['f5_margin_distribution']=_distribution(Counter(h5-a5 for _,_,a5,h5 in self.early),self.n)
            result['f5_total_distribution']=_distribution(Counter(a5+h5 for _,_,a5,h5 in self.early),self.n)
        result['sample_indices']=selected
        result['typical_score_frequency']=counts[modal]/self.n
        if self.win.games:result['win_table']=self.win.finish()
        result['keys']=self.keys_summary()
        return result

# Simulation-time adjustments on the locked model (brl_live/provider_adjust.py), measured on the
# 2026 out-of-sample replay before being switched on (LEDGER.md): context offsets are on; per-world
# talent noise at c=1 scored worse than the offsets alone and stays off. Since CTX-02 (October 8) the context table is
# estimated on the model as the simulator serves it (physics table of the previous and current season), which also sets
# the league's strikeout, walk and home-run level.
# postseason_exp_scale: by game type, the factor on starters' and teams' expected batters faced that
# drives the fitted starter hazard in postseason games. Measured on 294 postseason starts of 2023-2026
# (research/postseason_usage-37606652416.json): starters face 0.866 of their own regular-season
# median in the Wild Card, Division and League Championship rounds (0.99 in the World Series), and
# 28% of those starts end before 16 batters against 8% in the regular season. A factor of 0.91
# reproduces that ratio in the simulator (0.869; share under 16 batters 0.25). Recorded in each
# box's adjustments; relievers keep their regular-season roles.
# team_offsets: the batting and the fielding team's offsets beyond the PA model (brl_live/team_offsets.py, TEAM-01,
# TEAM-05). On: both full replays improved (2026 win Brier -0.00041, 2025 -0.00052; totals closer to the market line
# and to actual park scoring), and the production model stays v2 (RETRAIN-02). Table: brl_live/team_offsets.json,
# v2 residuals through 2026-09-27, k 4,000, half-life 120 days, batting and fielding. Centered on the league since CENTER-01
# (the context offsets carry the league level; uncentered, the replays ran 0.09 to 0.16 runs per game low).
# transitions: base running after each outcome from real play-by-play (TRANS-01; research_lab.game_sim.transitions.EmpiricalKernel,
# table brl_live/transitions.json from every 2023-2026 regular-season play, tools/brl_transition_kernel.py). The hand-set kernel
# had half the real double plays and scored the runner from third on 13% of outs with fewer than two out (real 54%). Full
# replays with cross-season kernels: batters per team-game 38.67 to 37.89 (actual 37.84, 2026) and 38.63 to 37.87 (37.66, 2025);
# simulator win Brier -0.00061 (2026) and -0.00010 (2025); in-game win chance on the page -0.00024 (2026); totals better in 2026,
# mixed in 2025. On since October 8, 2026.
# steals: runner speeds from sprint speed and stolen-base attempts (brl_live/running.py, RUN-01). On since the full
# 2026 replay (RUN-02): simulator win Brier 0.24405 to 0.24378, correlation with the market's closing log-odds 0.8235 to
# 0.8354, totals closer to the market line, mean total unchanged.
# running_events: wild pitches, passed balls, balks, pickoffs and the other running plays between plate appearances
# (TRANS-02; research_lab.game_sim.running_events.RunningEvents, table brl_live/running_events.json from the transitions
# lane, tools/brl_running_events.py). Off until its replays are read.
# reliever_choice: which reliever enters at each pitching change, from a conditional logit fitted on every 2023-2026
# relief entry (BULLPEN-01; research_lab.game_sim.reliever_choice.RelieverChoice, table brl_live/reliever_choice.json from
# tools/brl_reliever_choice.py): recent use, rest, role over the year and the last three weeks, strikeout record, the
# next three hitters' hands and the game situation. Off until its replays are read; off, the hand-set scoring chooses.
# relief_exit: when a reliever comes out, from logistic hazards fitted on every real relief decision point (RELIEF-02;
# research_lab.game_sim.relief_exit.ReliefExit, table brl_live/relief_exit.json from tools/brl_relief_exit.py). Off until
# its replays are read; off, the hand-set rule decides (it pulls relievers mid-inning far more often than managers do).
# day_form_sigma: a per-team, per-world shock on reaching base (DISP-01; brl_live.provider_adjust.DayForm) so team runs
# vary from game to game as much as real ones do. 0 is off.
# postseason_exit_offset: with relief_exit, log-odds added to the exit hazard in postseason games (POST-02): postseason
# managers use 4.2 relievers a team-game against 3.3 in the regular season (POST-01); with the starter scale and the
# fitted exits the engine gives 3.8, and +0.4 gives 4.15. The World Series gets no offset, as its starters get no scale;
# like every postseason game it runs the plain hazards, without LEASH-02's and RELIEF-04's regular-season corrections.
# relief_hooks: with relief_exit, each team's tendency to pull relievers faster or slower than the fitted hazard
# (RELIEF-03; table brl_live/relief_hooks.json from tools/brl_relief_hooks.py). Off until its replays are read.
# base_state: the stack's probabilities shaped by bases and outs (RUNS-02; brl_live.provider_adjust.BaseStateAdjust,
# table brl_live/base_state_offsets.json from tools/brl_base_state.py). Off until its replays are read.
# leash: in regular-season games each starter's expected batters faced is moved for short rest (openers), a relief
# outing before the start, the first two starts back from a layoff of 20 days or more, March and September (LEASH-01;
# research_lab.game_sim.starter_leash, table brl_live/leash.json from tools/brl_leash.py). Off until its replays are read.
# PROD-03 (LEDGER.md) switched reliever_choice, relief_exit, relief_hooks and leash on together on October 9, 2026: the
# package passed its do-no-harm replays in 2025 and 2026 (win Brier better in both, totals within bounds), and goes live
# under Alec's standing rule for proven improvements. The headline version taught-v5 was fitted on those replays.
ADJUST={'context_offsets':True,'talent_noise_c':0.0,'player_prior_pa':180.0,
        'postseason_exp_scale':{'F':0.91,'D':0.91,'L':0.91,'W':1.0},
        'environment':True,'team_offsets':True,'steals':True,'transitions':True,'running_events':False,
        'reliever_choice':True,'leash':True,'base_state':False,'relief_exit':True,'relief_hooks':True,
        'postseason_exit_offset':{'F':0.4,'D':0.4,'L':0.4,'W':0.0},'day_form_sigma':0.0,
        # DISP-03 (off until its replays are read): {'sigma':..,'center':..,'worlds':..}: extra worlds with the centered
        # day-form shock for the market numbers only (the total's over and under, the run line); win chances stay plain
        'market_day_form':None,
        # POST-03 (on from October 16, 2026, 11:50 p.m. Eastern, under Alec's standing approval of model improvements): the
        # starter's postseason factor by his tier of strikeouts minus walks this season (postseason starts 2023-2026: the
        # bottom third kept 0.772 of its regular length, the middle 0.846, the top 0.900; x0.91 treated everyone alike)
        'postseason_starter_tiers':{'cuts':[0.1518,0.1876],'scale':[0.81,0.89,0.94]},
        # LEASH-04 (on from October 16, 2026, 6:20 p.m. Eastern; gates held on 2025 and on 7 of 2026's 8 shards, the eighth
        # stalled): 'chain' draws each plate appearance's pitches from the matchup's own count chain (needs a model with
        # chain features) and the starter's exit reads LEASH-03's correction with pitch terms (regular season)
        'pitch_counts':'chain',
        # BULLPEN-07 (off until its replays are read): a shock on each reliever's utility drawn once per simulated game,
        # the day's unseen availability; spreads the bullpen's chances to pitch (BULLPEN-03's miss at the top)
        'reliever_game_sigma':0.0,
        # BULLPEN-09 (off until its replays are read): the choice's utilities over a temperature; 1.0 = the fitted choice as is
        'reliever_choice_temperature':1.0,
        # LEASH-02 (on from October 11, 2026; its gates held on both seasons): the inning correction on the starter-exit
        # hazard's log-odds, regular season only (brl_live/hazard_correction.json); the postseason keeps its measured usage
        # on the plain hazard
        'hazard_correction':True,
        # RELIEF-04 (on from October 11, 2026; its gates held on both seasons): the entry-inning correction on the
        # relief-exit hazard, regular season only (brl_live/relief_correction.json); with it the team hooks come from
        # brl_live/relief_hooks_corrected.json, measured against the corrected hazard. The postseason keeps the plain
        # hazard, its hooks and its measured exit offset
        'relief_correction':True}

TRANSITIONS_PATH=Path(__file__).resolve().parent/'transitions.json'
RUNNING_EVENTS_PATH=Path(__file__).resolve().parent/'running_events.json'
RELIEVER_CHOICE_PATH=Path(__file__).resolve().parent/'reliever_choice.json'
RELIEF_EXIT_PATH=Path(__file__).resolve().parent/'relief_exit.json'
RELIEF_HOOKS_PATH=Path(__file__).resolve().parent/'relief_hooks.json'
RELIEF_CORRECTION_PATH=Path(__file__).resolve().parent/'relief_correction.json'
RELIEF_HOOKS_CORRECTED_PATH=Path(__file__).resolve().parent/'relief_hooks_corrected.json'
LEASH_PATH=Path(__file__).resolve().parent/'leash.json'
_KERNEL={}

def leash_for(settings):
    """The starter-leash table (brl_live/leash.json, LEASH-01) when switched on, else None."""
    if not settings.get('leash'):return None
    if 'l' not in _KERNEL:
        from research_lab.game_sim.starter_leash import Leash
        _KERNEL['l']=Leash(json.loads(LEASH_PATH.read_text()))
    return _KERNEL['l']

HAZARD_CORRECTION_PATH=Path(__file__).resolve().parent/'hazard_correction.json'
HAZARD_CORRECTION_V2_PATH=Path(__file__).resolve().parent/'hazard_correction_v2.json'

def hazard_correction_for(settings,game_type):
    """LEASH-02's correction ({'terms','coef'}) for a regular-season game when switched on, else None. With LEASH-04's pitch
    draws on (settings pitch_counts 'chain') the correction is LEASH-03's refit with pitch terms (44 terms), which every
    live simulator can read because each one draws pitches (pitch_counts_for); otherwise the live file must hold LEASH-02's
    32 terms in order."""
    if not settings.get('hazard_correction') or str(game_type) not in ('R','W'):return None   # POST-04: the World Series as the regular season
    from research_lab.game_sim.starter_hazard import CORRECTION_TERMS,CORRECTION_TERMS_V2
    if settings.get('pitch_counts')=='chain':
        if 'hc2' not in _KERNEL:
            doc=json.loads(HAZARD_CORRECTION_V2_PATH.read_text())
            if list(doc['terms'])!=list(CORRECTION_TERMS_V2) or len(doc['coef'])!=len(CORRECTION_TERMS_V2):
                raise ValueError('brl_live/hazard_correction_v2.json must hold LEASH-03\'s %d terms in order'%len(CORRECTION_TERMS_V2))
            _KERNEL['hc2']={'terms':doc['terms'],'coef':doc['coef']}
        return _KERNEL['hc2']
    if 'hc' not in _KERNEL:
        doc=json.loads(HAZARD_CORRECTION_PATH.read_text())
        if list(doc['terms'])!=list(CORRECTION_TERMS) or len(doc['coef'])!=len(CORRECTION_TERMS):
            raise ValueError('brl_live/hazard_correction.json must hold LEASH-02\'s %d terms in order; pitch terms (LEASH-03) are not wired into every simulator'%len(CORRECTION_TERMS))
        _KERNEL['hc']={'terms':doc['terms'],'coef':doc['coef']}
    return _KERNEL['hc']

PITCH_COUNTS_FALLBACK_PATH=Path(__file__).resolve().parent/'pitch_counts_by_outcome.json'

def pitch_counts_for(settings,provider):
    """LEASH-04: the matchup pitch-count draws for a provider that carries the chain when switched on, else None. Every
    live simulator (the engine, the box worlds, the market worlds) is built through this so the starter's exit reads the
    same pitch counts everywhere."""
    if settings.get('pitch_counts')!='chain':return None
    if getattr(provider,'chain',None) is None:
        raise ValueError('pitch counts from the chain need a model with chain features')
    from research_lab.game_sim.pitch_counts import PitchCounts,MatchupPitchCounts
    if 'pc_fallback' not in _KERNEL:_KERNEL['pc_fallback']=PitchCounts(json.loads(PITCH_COUNTS_FALLBACK_PATH.read_text()))
    return MatchupPitchCounts(provider,_KERNEL['pc_fallback'])

def leash_pexp(leash,history,pexp,starters,date,league):
    """pexp with each of the game's starters moved by the leash table, from his regular-season appearances before the
    game's date."""
    from research_lab.game_sim.starter_leash import AppearanceIndex
    from .live_feed import appearances
    ids=[int(x) for x in starters]
    h=history[history['date_key'].astype(str).str[:10]<str(date)[:10]]
    if 'game_type' in h.columns:h=h[h['game_type']=='R']
    app=appearances(h[h['game_pk'].isin(h.loc[h['pitcher'].isin(ids),'game_pk'].unique())])
    index=AppearanceIndex.from_frame(app[app['pitcher'].isin(ids)])
    out=dict(pexp)
    for pid in ids:out[pid]=leash.adjust(float(pexp.get(pid,league)),index.facts(pid,date))
    return out

def reliever_choice_for(settings):
    """The fitted reliever choice (brl_live/reliever_choice.json, BULLPEN-01) when switched on, else None; with
    reliever_game_sigma (BULLPEN-07) each reliever's utility carries a shock drawn once per simulated game."""
    if not settings.get('reliever_choice'):return None
    sigma=float(settings.get('reliever_game_sigma') or 0.0);temp=float(settings.get('reliever_choice_temperature') or 1.0)
    key=('c',sigma,temp)
    if key not in _KERNEL:
        from research_lab.game_sim.reliever_choice import RelieverChoice
        _KERNEL[key]=RelieverChoice(json.loads(RELIEVER_CHOICE_PATH.read_text()),game_sigma=sigma,temperature=temp)
    return _KERNEL[key]

def relief_corrected(settings,game_type='R'):
    """True when RELIEF-04's entry-inning correction applies: switched on with the fitted exits, in regular-season games and
    (POST-04) World Series games, whose starter and reliever usage measured like the regular season's (POST-01, POST-02)."""
    return bool(settings.get('relief_exit') and settings.get('relief_correction')) and str(game_type) in ('R','W')

def relief_exit_for(settings,game_type='R'):
    """The fitted reliever exits (brl_live/relief_exit.json, RELIEF-02) when switched on, else None; with RELIEF-04's
    entry-inning correction (brl_live/relief_correction.json) in regular-season games when that is switched on."""
    if not settings.get('relief_exit'):return None
    corrected=relief_corrected(settings,game_type);key='xc' if corrected else 'x'
    if key not in _KERNEL:
        from research_lab.game_sim.relief_exit import ReliefExit
        corr=None
        if corrected:
            c=json.loads(RELIEF_CORRECTION_PATH.read_text());corr={'terms':c['terms'],'coef':c['coef']}
        _KERNEL[key]=ReliefExit(json.loads(RELIEF_EXIT_PATH.read_text()),correction=corr)
    return _KERNEL[key]

def relief_hooks_for(settings,game_type='R'):
    """The team hook offsets (brl_live/relief_hooks.json, RELIEF-03) when switched on with the fitted exits, else None;
    measured against the corrected hazard (relief_hooks_corrected.json) when RELIEF-04's correction applies."""
    if not (settings.get('relief_hooks') and settings.get('relief_exit')):return None
    corrected=relief_corrected(settings,game_type);key='h2' if corrected else 'h'
    if key not in _KERNEL:_KERNEL[key]=json.loads((RELIEF_HOOKS_CORRECTED_PATH if corrected else RELIEF_HOOKS_PATH).read_text())
    return _KERNEL[key]

def manager_for(manager,settings,teams=None,game_type='R'):
    """(manager, label): the engine's manager with the fitted reliever choice and exits (and, given the game's teams as
    (away, home) abbreviations, their hook offsets; in postseason games the postseason exit offset) attached when
    switched on (a copy, so the engine's own manager is unchanged), else the manager itself and None."""
    choice=reliever_choice_for(settings);exits=relief_exit_for(settings,game_type);hooks=relief_hooks_for(settings,game_type)
    if choice is None and exits is None:return manager,None
    import copy
    m=copy.copy(manager);labels=[]
    if choice is not None:m.reliever_choice=choice;labels.append(choice.name)
    if exits is not None:
        m.relief_exit=exits;labels.append(exits.name)
        offs={'away':{'mid':0.0,'end':0.0},'home':{'mid':0.0,'end':0.0}}
        if hooks is not None and teams and all(teams):
            t=hooks.get('teams') or {}
            for side,team in zip(('away','home'),teams):
                o=t.get(str(team).upper()) or {}
                offs[side]={'mid':float(o.get('mid',0.0)),'end':float(o.get('end',0.0))}
            labels.append(str(hooks.get('name')))
        post=float((settings.get('postseason_exit_offset') or {}).get(str(game_type),0.0)) if str(game_type)!='R' else 0.0
        if post:
            for side in offs:offs[side]={k:v+post for k,v in offs[side].items()}
            labels.append('postseason reliever exits (log-odds +%.2f)'%post)
        if any(v for o in offs.values() for v in o.values()):m.relief_offsets=offs
    return m,'; '.join(labels)

def running_events_for(settings):
    """The running plays between plate appearances (brl_live/running_events.json, TRANS-02) when switched on, else None."""
    if not settings.get('running_events'):return None
    if 'r' not in _KERNEL:
        from research_lab.game_sim.running_events import RunningEvents
        _KERNEL['r']=RunningEvents(json.loads(RUNNING_EVENTS_PATH.read_text()))
    return _KERNEL['r']

def transitions_for(settings):
    """The empirical base-running kernel (brl_live/transitions.json, TRANS-01) when it is switched on, else None (the
    engine's hand-set kernel)."""
    if not settings.get('transitions'):return None
    if 'k' not in _KERNEL:
        from research_lab.game_sim.transitions import EmpiricalKernel
        _KERNEL['k']=EmpiricalKernel(json.loads(TRANSITIONS_PATH.read_text()))
    return _KERNEL['k']

def steal_model_for(settings,matchup,date):
    """The base-running model for a game when steals are on, else None. Regular-season games use the current
    season's running statistics only when the data was fetched before the game's date; postseason games always do."""
    if not settings.get('steals'):return None
    from .running import steal_model,load_running
    season=int(str(date)[:4]);through=season
    if getattr(matchup,'game_type','R')=='R':
        fetched=str((load_running().get('fetched_at') or {}).get(str(season)) or '')[:10]
        if not fetched or fetched>=str(date)[:10]:through=season-1
    return steal_model(through)

def adjusted_provider(provider,full_history,date,settings=ADJUST,environment=None,teams=None):
    """Wrap the engine's provider with the enabled adjustments. Returns (provider, world_hook, label).

    environment: the game's conditions (brl_live/environment.conditions) when the run environment is on.
    teams: (away abbreviation, home abbreviation) for the team offsets."""
    from .provider_adjust import ContextAdjust,TalentNoise,EnvironmentAdjust,TeamAdjust,history_talent_inputs
    label=[];hook=None
    if settings.get('context_offsets'):
        provider=ContextAdjust(provider);label.append('context offsets through '+str(provider.offsets.get('estimated_through')))
    if settings.get('environment') and environment is not None:
        from .environment import load_table,log_multipliers,describe
        table=load_table()
        provider=EnvironmentAdjust(provider,log_multipliers(environment,table));label.append(describe(environment,table))
    if settings.get('team_offsets') and teams and all(teams):
        from .team_offsets import load_table as load_team_table,game_log_multipliers
        tt=load_team_table()
        provider=TeamAdjust(provider,game_log_multipliers(tt,teams[0],teams[1],date))
        label.append('team offsets through '+str(tt.get('estimated_through'))+(' (centered on the league)' if tt.get('center') else ''))
    if settings.get('base_state'):
        from .provider_adjust import BaseStateAdjust
        provider=BaseStateAdjust(provider);label.append('bases and outs shape ('+str(provider.offsets.get('name'))+')')
    if settings.get('role_offsets'):
        from .role_offsets import load_table as load_role_table,log_multipliers as role_multipliers
        from .provider_adjust import RoleAdjust
        rt=load_role_table()
        provider=RoleAdjust(provider,role_multipliers(rt,date))
        label.append('starter and reliever levels through '+str(rt.get('estimated_through')))
    sg=float(settings.get('day_form_sigma') or 0.0)
    if sg>0:
        from .provider_adjust import DayForm
        provider=DayForm(provider,sg);label.append('day form (sd %.2f per team and world)'%sg)
        hook=provider.new_world
    c=float(settings.get('talent_noise_c') or 0.0)
    if c>0:
        if full_history is None:raise Blocked('Talent noise needs the assembled PA history')
        counts_for,league=history_talent_inputs(full_history,date)
        provider=TalentNoise(provider,c,float(settings.get('player_prior_pa',180.0)),league,counts_for)
        prev=hook;tn=provider.new_world
        hook=(lambda seed,_a=prev,_b=tn:(_a(seed),_b(seed))) if prev else tn
        label.append('per-world talent noise c=%g'%c)
    return provider,hook,label

def market_worlds(sim_for,matchup,provider,seeds,spec,world_hook=None):
    """DISP-03: worlds with the centered day-form shock (brl_live.provider_adjust.DayForm) on top of the game's provider, for
    the market numbers only. sim_for(provider) builds the game's simulator around a provider; each world's seed is its box
    seed mixed with a constant, so it differs from that world's plain seed (it can equal another world's), and the shock
    draws from the seed with its own salt. run_box_worlds does not call it when the provider already has a day-form shock.
    Returns the joint distributions the market numbers need."""
    from .provider_adjust import DayForm
    dform=DayForm(provider,float(spec['sigma']),float(spec.get('center',0.0)))
    sim=sim_for(dform);pairs=[]
    for seed in list(seeds)[:int(spec.get('worlds',4000))]:
        s2=(int(seed)*2654435761+0x5EED)&0x7FFFFFFF
        if world_hook:world_hook(s2)
        dform.new_world(s2^0x3C6EF372)
        r=sim.simulate(matchup,s2)
        if r.winner=='tie' or r.ended_by_plate_appearance_cap:raise Blocked('Unfinished market world')
        pairs.append((r.away_score,r.home_score))
    n=len(pairs)
    return {'n':n,'day_form':{'sigma':float(spec['sigma']),'center':float(spec.get('center',0.0))},
            'total_run_distribution':_distribution(Counter(a+h for a,h in pairs),n),
            'margin_distribution':_distribution(Counter(h-a for a,h in pairs),n),
            'team_run_distributions':{side:_distribution(Counter(p[i] for p in pairs),n) for i,side in enumerate(SIDE)}}


def run_box_worlds(engine,matchup,history,date,seeds,full_history=None,settings=ADJUST,environment=None,teams=None):
    fit=BookkeepingFit(history,date,full_history=full_history)
    provider,world_hook,adjust_label=adjusted_provider(engine.provider,full_history,date,settings,environment,teams)
    if getattr(engine.manager,'adjust_label',None):adjust_label=list(adjust_label)+[engine.manager.adjust_label]
    steals=steal_model_for(settings,matchup,date)
    if steals is not None:adjust_label=list(adjust_label)+[steals.describe()]
    kernel=transitions_for(settings)
    if kernel is not None:adjust_label=list(adjust_label)+[kernel.name]
    running=running_events_for(settings)
    if running is not None:adjust_label=list(adjust_label)+[running.name]
    manager,choice_label=manager_for(engine.manager,settings,teams,getattr(matchup,'game_type','R'))
    if choice_label:adjust_label=list(adjust_label)+[choice_label]
    pitches=pitch_counts_for(settings,provider)
    if pitches is not None:adjust_label=list(adjust_label)+[pitches.name]
    sim=ObservedSimulator(provider,config=engine.config,manager_policy=manager,steals=steals,transitions=kernel,running_events=running,pitch_counts=pitches)
    accumulator=BoxAccumulator(matchup);results=[]
    for seed in seeds:
        if world_hook:world_hook(int(seed))
        result=sim.simulate(matchup,int(seed),record_events=True)
        if result.winner=='tie' or result.ended_by_plate_appearance_cap:raise Blocked('Unfinished box world')
        accumulator.add(build_game_box(result,matchup,fit))
        result.events=[];del result.box_scoring_trace
        results.append(result)
    payload=accumulator.finish()
    payload['adjustments']=adjust_label
    if settings.get('market_day_form') and not float(settings.get('day_form_sigma') or 0.0)>0:
        # never stacked on a day-form shock already in the provider; a failure here costs the market numbers only, never
        # the forecast (the page falls back to the plain worlds)
        try:
            payload['market_worlds']=market_worlds(lambda pv:ObservedSimulator(pv,config=engine.config,manager_policy=manager,steals=steals,transitions=kernel,running_events=running,pitch_counts=pitch_counts_for(settings,pv)),
                                                   matchup,provider,seeds,settings['market_day_form'],world_hook)
        except Exception as exc:
            payload['market_worlds']={'error':type(exc).__name__+': '+str(exc)[:160]}
    for j,i in enumerate(payload['sample_indices']):
        if world_hook:world_hook(int(seeds[i]))
        sample=build_game_box(sim.simulate(matchup,int(seeds[i]),record_events=True),matchup,fit)
        if tuple(sample['score'][s] for s in SIDE)!=accumulator.score_pairs[i]:raise Blocked('Sample seed parity failed')
        sample['world_index']=i;sample['typical']=j==0
        payload['samples'].append(sample)
    payload['bookkeeping']={'method':'prior-date pitch-count pools (pitcher pools only when at least %d PA) plus a shrunk hit-by-pitch rate by batter hand, pitcher and batter'%BookkeepingFit.MIN_POOL,
        'input_max_date':fit.max_input_date,'cutoff_exclusive':date,'sampling_stream':'independent of engine',
        'pitch_counts_are_estimates':True,'pitch_sequences_generated':bool(fit.bridge.n_sequences),
        'counts_do_not_influence_removal':True,'fallback_tiers_used':dict(fit.fallback_counts),
        'hbp_support_events':fit.hbp_support,'hbp_league_share':round(fit.hbp_league_all,4),
        'pitch_sequences':{'prior_sequences':fit.bridge.n_sequences,'path_tiers_used':dict(fit.bridge.fallbacks),
                           'method':'real prior count paths by outcome (own pitcher with at least %d, else league by hand); pitch types from the pitcher\'s mix by hand and count, speeds from his distribution by type; league fallbacks'%12},
        'contact':{'prior_balls_in_play':fit.bridge.n_contacts,'tiers_used':dict(fit.bridge.contact_fallbacks),
                   'method':'batted-ball shape, fielder and distance from real prior balls in play of the same kind (own batter with at least %d, else league by batter hand); double plays from ground-ball double plays, sacrifice flies from sacrifice flies'%MIN_BATTER_CONTACT}}
    return results,payload


def fair_crps(distribution,observed):
    n=distribution['n'];pairs=distribution['counts']
    if n<2 or sum(c for _,c in pairs)!=n:raise Blocked('Incomplete count distribution')
    ordered=sorted(pairs);absolute=sum(c*abs(x-observed) for x,c in ordered)/n
    cumulative=0;weighted=0;pair_sum=0
    for x,c in ordered:
        pair_sum+=c*(x*cumulative-weighted);cumulative+=c;weighted+=x*c
    # Unbiased finite-ensemble score: exclude a draw's distance from itself.
    return absolute-pair_sum/(n*(n-1))

def corrected_brier(distribution,observed):
    n=distribution['n'];p=sum(c for x,c in distribution['counts'] if x>0)/n
    return (p-int(observed>0))**2-p*(1-p)/(n-1)


def parse_actual_box(feed,game_pk,fetched_at):
    from app.results import parse_final
    result=parse_final(feed,game_pk,fetched_at)
    if result['first_pitch_observed_at'] is None:raise Blocked('First pitch unavailable for player scoring')
    live=feed['liveData'];out={'game_pk':game_pk,'fetched_at':fetched_at,
        'first_pitch_observed_at':result['first_pitch_observed_at'],'team_ids':result['team_ids'],
        'score':{s:result[s] for s in SIDE},'batting':{},'pitching':{},'innings':{s:{} for s in SIDE}}
    for item in live['linescore']['innings']:
        for s in SIDE:
            if 'runs' in item.get(s,{}):out['innings'][s][str(item['num'])]={k:int(item[s][v]) for k,v in [('R','runs'),('H','hits')]}
    # errors for the line score's E column (display only; missing in the feed means left out, never zero)
    errs={s:((live['linescore'].get('teams') or {}).get(s) or {}).get('errors') for s in SIDE}
    if all(v is not None for v in errs.values()):out['errors']={s:int(errs[s]) for s in SIDE}
    for s in SIDE:
        team=live['boxscore']['teams'][s];bat=[];pit=[]
        for pid in team['batters']:
            p=team['players']['ID'+str(pid)];stats=p.get('stats',{}).get('batting',{})
            if not stats:continue
            mapping={'PA':'plateAppearances','AB':'atBats','H':'hits','2B':'doubles','3B':'triples','HR':'homeRuns','R':'runs','RBI':'rbi','BB':'baseOnBalls','HBP':'hitByPitch','K':'strikeOuts','SF':'sacFlies'}
            # Missing fields are unavailable, never fabricated as zero.
            row={'player_id':str(pid),'name':p['person']['fullName'],'spot':int(p.get('battingOrder','0'))//100,'order':int(p.get('battingOrder','0')),
                 **{k:int(stats[v]) if v in stats else None for k,v in mapping.items()}}
            # his average through this game, as box scores print it (display only)
            from .real_game import season_mark
            sa=season_mark(p,'batting','avg')
            if sa:row['AVG']=sa
            bat.append(row)
        for pid in team['pitchers']:
            p=team['players']['ID'+str(pid)];stats=p.get('stats',{}).get('pitching',{})
            if not stats:continue
            mapping={'PC':'numberOfPitches','H':'hits','R':'runs','ER':'earnedRuns','BB':'baseOnBalls','HBP':'hitBatsmen','K':'strikeOuts','HR':'homeRuns','BF':'battersFaced'}
            ip=str(stats['inningsPitched']).split('.')
            if len(ip)!=2 or ip[1] not in ('0','1','2'):raise Blocked('Invalid baseball innings notation')
            row={'player_id':str(pid),'name':p['person']['fullName'],'outs':3*int(ip[0])+int(ip[1]),
                 **{k:int(stats[v]) if v in stats else None for k,v in mapping.items()}}
            from .real_game import season_mark
            se=season_mark(p,'pitching','era')
            if se:row['ERA']=se
            pit.append(row)
        out['batting'][s]=bat;out['pitching'][s]=pit
        for k,official in [('R','runs'),('H','hits')]:
            if any(row[k] is None for row in bat+pit):raise Blocked('Incomplete actual totals')
            if sum(row[k] for row in bat)!=int(team['teamStats']['batting'][official]):raise Blocked('Actual batting sum mismatch')
            if sum(v[k] for v in out['innings'][s].values())!=sum(row[k] for row in bat):raise Blocked('Actual inning sum mismatch')
    for s in SIDE:
        opp='home' if s=='away' else 'away'
        for k in ('R','H','BB','K','HR'):
            if all(r[k] is not None for r in out['pitching'][s]+out['batting'][opp]):
                if sum(r[k] for r in out['pitching'][s])!=sum(r[k] for r in out['batting'][opp]):raise Blocked('Actual opposite-side totals mismatch')
    # The real plays, for the page (every plate appearance, its pitches and where the ball went); display only,
    # so a reading problem is recorded rather than allowed to stop the final from being scored.
    from .real_game import plays_from_feed
    try:out['plays']=plays_from_feed(feed)
    except Exception as exc:out['plays']=[];out['plays_error']=type(exc).__name__+': '+str(exc)[:200]
    return out

def player_box_version(ident,box,pub,actual):
    """One saved box against its game's final box: (version, None), or (None, reason) when it cannot count."""
    from app.common import timestamp,content_hash
    import re
    if (not pub or not re.fullmatch(r'[0-9a-f]{40}',str(pub.get('commit','')))
            or pub.get('box_sha256')!=content_hash(box)
            or not timestamp(box['saved_at'])<=timestamp(pub['published_at'])<timestamp(actual['first_pitch_observed_at'])):
        return None,'Player distributions were not publicly saved before first pitch'
    if box['team_ids']!=actual['team_ids']:
        return None,'Player actual team mismatch'
    rows=[]
    for s in SIDE:
        for kind,metrics in [('batting',('H','HR','K')),('pitching',('K','outs'))]:
            actual_index={r['player_id']:r for r in actual[kind][s]}
            for p in box['teams'][s][kind]:
                a=actual_index.get(p['player_id'])
                for metric in metrics:
                    # In a FINAL, complete MLB participant listing, absent
                    # projected participants have zero realized opportunity.
                    obs=0 if a is None else a[metric]
                    if obs is None:continue
                    d=p['distributions'][metric]
                    rows.append({'side':s,'kind':kind,'player_id':p['player_id'],'name':p['name'],
                                 'metric':'IP' if metric=='outs' else metric,
                                 'expected':p['means'][metric]/(3 if metric=='outs' else 1),
                                 'actual':obs/(3 if metric=='outs' else 1),
                                 'crps':fair_crps(d,obs)/(3 if metric=='outs' else 1),
                                 'event_brier':corrected_brier(d,obs) if kind=='batting' and metric in ('H','HR') else None})
    return {'forecast_id':ident,'game_pk':box['game_pk'],'saved_at':box['saved_at'],'rows':rows},None

def score_player_boxes(boxes,publications,actuals,cache=None):
    """cache: ledger['box_cache'], the scores of boxes archived after their game (brl_live/box_runner.py
    archive_old_boxes), worked out while the full box still verified against its publication."""
    per_version=[];excluded={};latest={}
    for ident,box in boxes.items():
        actual=actuals.get(str(box['game_pk']))
        if actual is None:continue
        hit=(cache or {}).get(ident) if box.get('archived') else None
        if hit is not None and 'player' in hit:
            version,reason=hit['player'].get('version'),hit['player'].get('excluded')
        else:
            version,reason=player_box_version(ident,box,publications.get(ident),actual)
        if reason is not None:
            excluded[ident]=reason;continue
        per_version.append(version)
        if box['game_pk'] not in latest or latest[box['game_pk']]['saved_at']<box['saved_at']:latest[box['game_pk']]=version
    groups=defaultdict(list)
    for game in latest.values():
        for row in game['rows']:groups[(row['kind'],row['metric'])].append(row)
    aggregate=[{'kind':kind,'metric':metric,'n_player_games':len(rows),'n_games':len(latest),
                'crps':sum(r['crps'] for r in rows)/len(rows),
                'event_brier':sum(r['event_brier'] for r in rows)/len(rows) if rows[0]['event_brier'] is not None else None}
               for (kind,metric),rows in groups.items()]
    return {'n_games':len(latest),'aggregates':aggregate,'versions':per_version,'excluded':excluded,
            'policy':'Latest publicly saved pregame box per game; unconditional zero for nonappearance',
            'uncertainty':'Players in a game are dependent; no independent-player significance claim'}


def bookkeeping_history_from_cache(cache,index,origin):
    """Read actual pitch events, not the PA corpus's normalized pitch_number.

The seed corpus sets pitch_number=1 for all terminal PAs and therefore cannot
support pitch-count estimation. Official prior-date playEvents supply counts.
    """
    from brl_live.history_refresh import Source,previous_day
    from research_lab.pa_model.outcomes import map_event
    from app.common import timestamp
    rows=[];seen=set()
    for date,ref in sorted(index['days'].items()):
        if date>previous_day(origin).isoformat():raise Blocked('Future bookkeeping date')
        day=cache.load_day(ref['id'])
        for value in day['sources']:
            source=Source.from_dict(value);source.check(origin)
            if '/feed/live' not in source.url or '?' in source.url:continue
            import json
            feed=json.loads(source.body)
            if feed['gameData']['status']['abstractGameState']!='Final':continue
            for play in feed['liveData']['plays']['allPlays']:
                outcome=map_event(play.get('result',{}).get('eventType',''))
                if outcome is None:continue
                if not play['about']['isComplete'] or timestamp(play['about']['endTime'])>timestamp(source.finished_at):raise Blocked('Unresolved bookkeeping label')
                key=(int(feed['gamePk']),int(play['about']['atBatIndex']))
                if key in seen:continue
                seen.add(key)
                m=play['matchup'];count=sum(e.get('isPitch') is True for e in play.get('playEvents',[]))
                batter=(m.get('batter') or {}).get('id')
                rows.append({'date_key':feed['gameData']['datetime']['officialDate'],
                    'outcome':outcome,'terminal_event':play['result']['eventType'],
                    'pitcher':int(m['pitcher']['id']),'batter':None if batter is None else int(batter),'stand':m['batSide']['code'],
                    'pitch_number':count,'available_at':source.finished_at,'pitches':pitch_list(play),'contact':contact_of(play)})
    # The sealed season backfill (when one exists) supplies every earlier game of the year;
    # the day cache above wins on overlap, and nothing dated today or later is used.
    season_games=0
    if cache is not None and getattr(cache,'store',None) is not None and getattr(cache,'key',None) is not None:
        from brl_live.bookkeeping_season import load_season,season_rows
        from zoneinfo import ZoneInfo
        today=origin.astimezone(ZoneInfo('America/New_York')).date()
        for year in sorted({today.year-1,today.year}):
            doc=load_season(cache.store,cache.key,year)
            if doc is None:continue
            season_games+=len(doc['games'])
            rows.extend(season_rows(doc,today.isoformat(),seen))
    if not rows:raise Blocked('No prior official pitch-count observations')
    frame=pd.DataFrame(rows);frame.attrs['season_games']=season_games
    return frame


def validate_box_payload(box):
    from app.common import timestamp
    import re
    if box.get('schema')!='brl.box-forecast.v1' or box.get('n_simulations')!=10000:raise Blocked('Invalid box schema/count')
    if len(box.get('samples',[]))!=5 or len({s['seed'] for s in box['samples']})!=5:raise Blocked('Five distinct sample worlds required')
    if not timestamp(box['forecast_origin'])<=timestamp(box['saved_at']):raise Blocked('Box timing invalid')
    if box['bookkeeping']['input_max_date']>=box['date']:raise Blocked('Future box bookkeeping input')
    for side in SIDE:
        if len(box['teams'][side]['batting'])!=9:raise Blocked('Nine projected lineup spots required')
        for kind,fields in [('batting',BAT),('pitching',PIT)]:
            rows=box['teams'][side][kind]
            if len({r['player_id'] for r in rows})!=len(rows):raise Blocked('Duplicate box player')
            for row in rows:
                for metric in fields:
                    dist=row['distributions'][metric];pairs=dist['counts']
                    if dist['n']!=10000 or sum(c for _,c in pairs)!=10000:raise Blocked('Incomplete player distribution')
                    if len(set(x for x,_ in pairs))!=len(pairs) or any(type(x)is not int or type(c)is not int or x<0 or c<=0 for x,c in pairs):raise Blocked('Invalid histogram bin')
                    mean=sum(x*c for x,c in pairs)/10000
                    if not math.isfinite(row['means'][metric]) or abs(mean-row['means'][metric])>1e-10:raise Blocked('Box means differ from saved distribution')
    return box
