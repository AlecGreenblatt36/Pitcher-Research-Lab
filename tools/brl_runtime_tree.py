"""List the files inside the encrypted runtime (names, sizes, hashes only; never contents).

Runs in the cloud with the repository secret; writes diagnostics/runtime_tree.json to the
ledger branch so the layout can be read back through git. No file contents, no key material.
"""
from __future__ import annotations
import base64,hashlib,json,os,sys
from pathlib import Path
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from brl_live.bootstrap import api,fetch,restore,TAG,ASSET

def main():
    repo=os.environ['GITHUB_REPOSITORY'];token=os.environ['GH_TOKEN'];key=os.environ.get('BRL_PA_PACKAGE_KEY','')
    out={'runtime_tag':TAG,'files':[],'error':None}
    try:
        if not key:raise ValueError('secret missing')
        config=json.loads(Path('brl_live/runtime.json').read_text())
        runtime=restore(fetch(f'https://github.com/{repo}/releases/download/{TAG}/{ASSET}'),key,config,os.environ['RUNNER_TEMP']+'/brl-private-tree')
        for p in sorted(runtime.rglob('*')):
            if p.is_file():
                rel=str(p.relative_to(runtime))
                h=hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size<200_000_000 else None
                out['files'].append({'path':rel,'size':p.stat().st_size,'sha256':h})
                if rel.endswith('.py') and ('app/' in rel or 'cloud/' in rel):
                    # public API surface only: def/class lines, no bodies
                    lines=[l.strip() for l in p.read_text(errors='replace').splitlines() if l.lstrip().startswith(('def ','class ','from ','import '))]
                    out['files'][-1]['signatures']=lines[:120]
    except Exception as exc:
        out['error']=type(exc).__name__+': '+str(exc)[:200]
    body=json.dumps(out,indent=1).encode()
    url=f'https://api.github.com/repos/{repo}/contents/diagnostics/runtime_tree.json'
    sha=None
    try:sha=api(url+'?ref=brl-live-data',token).get('sha')
    except HTTPError as exc:
        if exc.code!=404:raise
    payload={'message':'BRL: runtime tree diagnostic','content':base64.b64encode(body).decode(),'branch':'brl-live-data'}
    if sha:payload['sha']=sha
    api(url,token,'PUT',payload)
    print(json.dumps({'files':len(out['files']),'error':out['error']}))
if __name__=='__main__':main()
