"""Legacy entry point kept for live_extension.main; the box page is the page."""
from __future__ import annotations


def render_page(ledger, scores, destination, setup_message=None):
    from .box_page import render_page as box_render
    return box_render(ledger, scores, destination, setup_message)
