"""The page's script is one scope: a second function with the same name silently replaces the first everywhere (a new
runLine for the run line market once nearly replaced the running game's runLine). Every top-level name appears once."""
import re
from collections import Counter
from pathlib import Path

TEMPLATE = Path(__file__).parents[1] / 'brl_live/page/template.html'


def test_no_two_top_level_functions_share_a_name():
    s = TEMPLATE.read_text(encoding='utf-8')
    names = re.findall(r'\n  function ([A-Za-z0-9_$]+)\(', s)
    dup = sorted(n for n, c in Counter(names).items() if c > 1)
    assert not dup, dup
    assert len(names) > 300


def test_no_top_level_var_is_declared_twice():
    s = TEMPLATE.read_text(encoding='utf-8')
    names = re.findall(r'\n  var ([A-Za-z0-9_$]+) = ', s)
    dup = sorted(n for n, c in Counter(names).items() if c > 1)
    assert not dup, dup
