"""Focused tests for documentation markdown HTML rendering."""

from __future__ import annotations

from ui_reports import documentation_body_html


def test_documentation_body_html_renders_tables() -> None:
    """Pipe tables should become real HTML tables, not one long paragraph."""
    md = (
        "## Modules\n\n"
        "| Module | Bug |\n"
        "|--------|-----|\n"
        "| shop.auth | identity check |\n"
        "| shop.billing | off-by-one |\n"
    )
    html = documentation_body_html(md)
    assert "<table>" in html
    assert "<th>" in html
    assert "shop.auth" in html
    assert "| Module | Bug |" not in html


def test_documentation_body_html_renders_horizontal_rules() -> None:
    """Markdown thematic breaks should render as <hr>."""
    html = documentation_body_html("Intro\n\n---\n\nNext")
    assert "<hr />" in html


def test_documentation_body_html_strips_object_object_and_mojibake() -> None:
    """Display path should clean model junk before rendering."""
    html = documentation_body_html(
        "Title â Demo\n\n,[object Object],\n\nDone."
    )
    assert "[object Object]" not in html
    assert "object Object" not in html
    assert "â" not in html
    assert "Title - Demo" in html
    assert "Done." in html


def test_documentation_body_html_strips_inline_code_sample_junk() -> None:
    """Real model output with ,[object Object], lines must never reach HTML."""
    sample = (
        "from requests_html import HTMLSession\n"
        ",[object Object],\n"
        ",[object Object],\n"
        "print(r.html.links)\n"
        "from requests_html import AsyncHTMLSession,[object Object],\n"
    )
    html = documentation_body_html(sample)
    assert "[object Object]" not in html
    assert "object Object" not in html
    assert "&#91;object Object&#93;" not in html
    assert "HTMLSession" in html
    assert "print(r.html.links)" in html

