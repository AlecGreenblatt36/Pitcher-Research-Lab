"""Do past ESPN scoreboards still carry the pregame moneylines? Counts only, for a few dates.

For each date: events, events with odds, events whose moneylines brl_live.market can read, the
providers named, and whether open and close lines both appear. Writes diagnostics/market_probe.json.
"""
from __future__ import annotations
import base64, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brl_live.market import ESPN_SCOREBOARD, parse_scoreboard  # noqa: E402


def get(url):
    with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0 (BRL probe)', 'Accept': 'application/json'}), timeout=30) as r:
        return json.loads(r.read())


def main():
    dates = (os.environ.get('BRL_PROBE_DATES') or '20260405,20260620,20260920,20250815,20241003,20231005').split(',')
    out = {'schema': 'brl.market-probe.v1', 'at': datetime.now(timezone.utc).isoformat(), 'dates': {}}
    for d in dates:
        try:
            doc = get(ESPN_SCOREBOARD.format(date=d.strip()))
            events = doc.get('events') or []
            with_odds = [e for e in events for c in (e.get('competitions') or []) if c.get('odds')]
            lines = parse_scoreboard(doc)
            providers = sorted({str((o.get('provider') or {}).get('name')) for e in events for c in (e.get('competitions') or []) for o in (c.get('odds') or [])})
            sample = None
            for e in events:
                for c in e.get('competitions') or []:
                    for o in c.get('odds') or []:
                        ml = o.get('moneyline') or {}
                        if isinstance(ml, dict) and ml:
                            sample = {k: (sorted(v.keys()) if isinstance(v, dict) else type(v).__name__) for k, v in ml.items()}
                            home = ml.get('home') or {}
                            if isinstance(home, dict):
                                sample['home_detail'] = {k: (sorted(v.keys()) if isinstance(v, dict) else v) for k, v in home.items()}
                            break
                    if sample: break
                if sample: break
            states = sorted({((c.get('status') or {}).get('type') or {}).get('state') for e in events for c in (e.get('competitions') or [])})
            out['dates'][d] = {'events': len(events), 'with_odds': len(with_odds), 'readable_moneylines': len(lines), 'providers': providers, 'states': states,
                               'moneyline_shape': sample, 'p_home_examples': [l['p_home'] for l in lines[:3]]}
            # the per-event summary keeps a pick center with the lines for finished games
            summaries = []
            for e in events[:3]:
                try:
                    sm = get('https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event=' + str(e.get('id')))
                    pc = sm.get('pickcenter') or []
                    first = pc[0] if pc else {}
                    summaries.append({'pickcenter': len(pc), 'odds_key': 'odds' in sm, 'providers': [str((x.get('provider') or {}).get('name')) for x in pc][:4],
                                      'keys': sorted(first.keys())[:20], 'home_ml': (first.get('homeTeamOdds') or {}).get('moneyLine'),
                                      'away_ml': (first.get('awayTeamOdds') or {}).get('moneyLine'), 'details': first.get('details'),
                                      'home_odds_keys': sorted((first.get('homeTeamOdds') or {}).keys())[:20]})
                except Exception as exc:
                    summaries.append({'error': type(exc).__name__ + ': ' + str(exc)[:100]})
            out['dates'][d]['summaries'] = summaries
        except Exception as exc:
            out['dates'][d] = {'error': type(exc).__name__ + ': ' + str(exc)[:150]}
    text = json.dumps(out, indent=1)
    print(text)
    repo, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GH_TOKEN')
    if repo and token:
        url = f'https://api.github.com/repos/{repo}/contents/diagnostics/market_probe.json'
        headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'BRL-probe/1.0', 'Content-Type': 'application/json'}
        sha = None
        try:
            with urlopen(Request(url + '?ref=brl-live-data', headers=headers), timeout=30) as r:
                sha = json.loads(r.read()).get('sha')
        except HTTPError:
            pass
        payload = {'message': 'BRL: market probe', 'content': base64.b64encode(text.encode()).decode(), 'branch': 'brl-live-data'}
        if sha:
            payload['sha'] = sha
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers=headers, data=json.dumps(payload).encode(), method='PUT'), timeout=30) as r:
                    r.read(); break
            except HTTPError as exc:
                if exc.code != 409 or attempt == 3:
                    raise
                time.sleep(3)


if __name__ == '__main__':
    main()
