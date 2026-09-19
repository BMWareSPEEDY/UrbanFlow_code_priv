#!/usr/bin/env python3
"""Render the corrected UrbanFLOW Research Paper V2 (HTML) to a PDF.

Reads the canonical V2 HTML in misc/old_artifacts/, rewrites the relative
image src attributes to absolute file:// URLs pointing into docs/figures/,
then prints to A4 via headless Chrome (Playwright) with page-number footers.
"""
import base64
import mimetypes
import pathlib
import sys

from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).resolve().parent
ROOT = BASE.parent
HTML = BASE / "old_artifacts" / "UrbanFLOW_Research_Paper_v2.html"
PDF = ROOT / "docs" / "UrbanFLOW_Research_Paper_v2.pdf"
FIGURES = ROOT / "docs" / "figures"

IMAGES = {
    "fig1_architecture.png",
    "fig_dashboard.png",
    "UrbanFLOW_paper_fig_scatter.png",
    "UrbanFLOW_paper_fig_speedup.png",
    "UrbanFLOW_paper_fig_zeroinfl.png",
}


def rewrite_image_srcs(text: str) -> str:
    for name in IMAGES:
        target = FIGURES / name
        if not target.exists():
            print(f"WARNING: missing figure {name} (expected at {target})")
            continue
        mime = mimetypes.guess_type(name)[0] or "image/png"
        data = base64.b64encode(target.read_bytes()).decode("ascii")
        data_uri = f"data:{mime};base64,{data}"
        text = text.replace(f'src="{name}"', f'src="{data_uri}"')
        print(f"embedded {name} ({len(data) / 1024:.0f} KiB b64)")
    return text


def main() -> None:
    if not HTML.exists():
        print(f"FATAL: source HTML not found: {HTML}", file=sys.stderr)
        return 1

    text = rewrite_image_srcs(HTML.read_text(encoding="utf-8"))
    url = HTML.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(base_url=url)
        page.set_content(text, wait_until="networkidle", timeout=120_000)
        katex = page.evaluate("document.querySelectorAll('.katex').length")
        print(f"katex elements rendered: {katex}")
        if katex == 0:
            print("WARNING: no KaTeX elements found; math may not have rendered")
        imgs = page.evaluate(
            "Array.from(document.images).map(i => i.naturalWidth || 0)"
        )
        print(f"figure natural widths (0 = failed load): {imgs}")
        page.pdf(
            path=str(PDF),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            margin={"top": "1in", "bottom": "1in", "left": "1in", "right": "1in"},
            header_template="<div></div>",
            footer_template=(
                "<div style='width:100%; text-align:center; "
                "font-family:\"Times New Roman\",Times,serif; font-size:9.5pt;'>"
                "Page <span class='pageNumber'></span> of "
                "<span class='totalPages'></span></div>"
            ),
        )
        browser.close()
    print(f"PDF written: {PDF} ({PDF.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())