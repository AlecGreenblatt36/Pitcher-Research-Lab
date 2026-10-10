"""The report lane writes a day's plans in one commit (Git Data API) and rebases when another writer moved the branch."""
import importlib.util
import io
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location('brl_report_put_many_test', ROOT / 'tools' / 'brl_report.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeGit:
    def __init__(self, moves=1):
        self.head = 'c0'; self.trees = {'c0': {'old.json': 'x'}}; self.commits = {'c0': 'c0tree'}; self.tree_of = {'c0tree': {'old.json': 'x'}}
        self.blobs = {}; self.moves = moves; self.n = 0; self.patches = 0

    def __call__(self, url, token, method='GET', payload=None):
        tail = url.split('/git/')[1]
        if tail == 'blobs':
            sha = 'b%d' % len(self.blobs); self.blobs[sha] = payload['content']; return {'sha': sha}
        if tail.startswith('ref/heads/'):
            return {'object': {'sha': self.head}}
        if tail.startswith('commits/'):
            return {'tree': {'sha': self.commits[tail.split('/')[1]]}}
        if tail == 'trees':
            self.n += 1; t = 't%d' % self.n
            files = dict(self.tree_of[payload['base_tree']])
            files.update({e['path']: self.blobs[e['sha']] for e in payload['tree']})
            self.tree_of[t] = files; return {'sha': t}
        if tail == 'commits':
            self.n += 1; c = 'c%d' % self.n; self.commits[c] = payload['tree']; self.parent = payload['parents']; return {'sha': c}
        if tail.startswith('refs/heads/'):
            self.patches += 1
            if self.moves:
                # someone else committed first: the head moved, the update is not a fast forward
                self.moves -= 1; self.n += 1; other = 'c%d' % self.n
                self.commits[other] = 'other_tree'; self.tree_of['other_tree'] = dict(self.tree_of['c0tree'], **{'live.json': 'y'}); self.head = other
                raise HTTPError(url, 422, 'Update is not a fast forward', {}, io.BytesIO(b'{}'))
            self.head = payload['sha']; return {}
        raise AssertionError(url)


def test_put_many_is_one_commit_and_rebases_on_a_moved_head(monkeypatch):
    B = _report()
    fake = FakeGit(moves=2)
    monkeypatch.setattr(B.D, 'api', fake)
    monkeypatch.setattr(B.time, 'sleep', lambda s: None)
    files = {'public/reports/2026-10-10/1.json': '{"a":1}', 'public/reports/2026-10-10/index.json': '{"games":{}}'}
    commit = B.put_many('o/r', 't', files, 'brl-live-data', 'BRL report 2026-10-10')
    assert fake.head == commit and fake.patches == 3
    final = fake.tree_of[fake.commits[commit]]
    assert final['live.json'] == 'y' and final['old.json'] == 'x'          # the other writer's file is kept
    assert final['public/reports/2026-10-10/1.json'] == '{"a":1}' and final['public/reports/2026-10-10/index.json'] == '{"games":{}}'
    assert len(fake.blobs) == 2                                              # blobs made once, reused on each try
    assert B.put_many('o/r', 't', {}, 'b', 'nothing') is None


def test_put_many_falls_back_to_one_file_at_a_time_when_the_branch_keeps_moving(monkeypatch):
    B = _report()
    fake = FakeGit(moves=100)
    monkeypatch.setattr(B.D, 'api', fake)
    monkeypatch.setattr(B.time, 'sleep', lambda s: None)
    wrote = []
    monkeypatch.setattr(B.D, 'put_text', lambda repo, token, path, text, branch, message: wrote.append(path))
    files = {'public/reports/d/index.json': '{}', 'public/reports/d/1.json': '{"a":1}', 'public/reports/d/2.json': '{"b":2}'}
    assert B.put_many('o/r', 't', files, 'b', 'm', fast_tries=3) is None
    assert fake.patches == 3
    assert wrote == ['public/reports/d/1.json', 'public/reports/d/2.json', 'public/reports/d/index.json']   # the index last
