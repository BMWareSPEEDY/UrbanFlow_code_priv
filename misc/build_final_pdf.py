#!/usr/bin/env python3
"""Render doc/UrbanFLOW_Technical_Research_Paper_Final.md to HTML + PDF.

Uses the `markdown` package for conversion and Playwright (system Chrome)
for headless A4 PDF printing with page-number footers.
"""
import pathlib
import sys

import markdown
from playwright.sync_api import sync_playwright

BASE = pathlib.Path(__file__).resolve().parent
MD = BASE / "doc" / "UrbanFLOW_Technical_Research_Paper_Final.md"
HTML = BASE / "doc" / "UrbanFLOW_Technical_Research_Paper_Final.html"
PDF = BASE / "doc" / "UrbanFLOW_Technical_Research_Paper_Final.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 10.5pt; line-height: 1.4; color: #111;
  max-width: 100%;
}
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 10pt 0; }
h2 {
  font-size: 13.5pt; margin: 18pt 0 6pt 0; padding-bottom: 2pt;
  border-bottom: 0.8pt solid #999; break-after: avoid;
}
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 10pt 0 3pt 0; break-after: avoid; }
p { margin: 0 0 6pt 0; text-align: justify; }
a { color: #0645ad; text-decoration: none; word-break: break-all; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 8.8pt; background: #f4f4f4; padding: 0 2px; border-radius: 2px;
}
pre {
  background: #f7f7f7; border: 0.6pt solid #ddd; border-radius: 3px;
  padding: 7pt 9pt; overflow-x: auto; break-inside: avoid;
  font-size: 8.6pt; line-height: 1.3;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 8pt 0;
  font-size: 8.8pt; line-height: 1.25; break-inside: avoid;
}
th, td { border: 0.5pt solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eef1f4; font-weight: bold; }
blockquote {
  margin: 8pt 0; padding: 4pt 10pt; border-left: 3pt solid #ccc;
  background: #fafafa; color: #333;
}
ul, ol { margin: 0 0 6pt 0; padding-left: 1.4em; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 0.8pt solid #ccc; margin: 14pt 0; }
img { max-width: 100%; }
"""


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc", "nl2br"],
    )
    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>UrbanFLOW — Comprehensive Technical Research Paper (Final)</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )
    HTML.write_text(html, encoding="utf-8")
    print(f"HTML written: {HTML}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.goto(HTML.as_uri(), wait_until="networkidle", timeout=120_000)
        page.pdf(
            path=str(PDF),
            format="A4",
            print_background=True,
            display_header_footer=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
            header_template="<div></div>",
            footer_template=(
                "<div style='width:100%; text-align:center; "
                "font-family:\"Times New Roman\",Times,serif; font-size:8.5pt; color:#555;'>"
                "UrbanFLOW Technical Research Paper (Final) — Page "
                "<span class='pageNumber'></span> of "
                "<span class='totalPages'></span></div>"
            ),
        )
        browser.close()
    print(f"PDF written: {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
