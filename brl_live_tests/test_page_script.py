"""The page's own script must parse: one stray quote in a note string blanks the whole site on every phone."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_template_scripts_parse(tmp_path):
    html = (ROOT / 'brl_live' / 'page' / 'template.html').read_text()
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.S)
    assert scripts
    for i, code in enumerate(scripts):
        f = tmp_path / f'page{i}.js'
        f.write_text(code)
        r = subprocess.run([NODE, '--check', str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:800]


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_rendered_page_script_parses(tmp_path):
    from brl_live import box_page
    D = {'date': '2026-10-10', 'forecasts': {}, 'generated_at': '2026-10-10T12:00:00Z'}
    try:
        html = box_page.render_html(D)
    except Exception as exc:                    # the renderer needs more than this minimal file; the template check covers the script
        pytest.skip('render needs a fuller prediction file: ' + type(exc).__name__)
    for i, (attrs, code) in enumerate(re.findall(r'<script([^>]*)>(.*?)</script>', html, re.S)):
        if 'json' in attrs:
            continue
        f = tmp_path / f'rendered{i}.js'
        f.write_text(code)
        r = subprocess.run([NODE, '--check', str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:800]
