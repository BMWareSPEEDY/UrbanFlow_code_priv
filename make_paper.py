# Python script to build full paper
import os, subprocess, pypdf

# Import the HTML parts
from generate_html_content import build_complete_html

def run():
    html_content = build_complete_html()
    with open("UrbanFLOW_Paper_Print.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Wrote UrbanFLOW_Paper_Print.html")

    msedge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    pdf_path = os.path.abspath("UrbanFLOW_Research_Paper.pdf")
    html_path = os.path.abspath("UrbanFLOW_Paper_Print.html")

    cmd = [
        msedge_path,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        f"file:///{html_path.replace(os.sep, '/')}"
    ]
    print("Executing Edge headless PDF compilation...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"SUCCESS: Generated {pdf_path}")
        print(f"File size: {size_kb:.1f} KB | Total Pages: {num_pages}")
    else:
        print("Compilation failed!")

if __name__ == '__main__':
    run()
