"""
Builds every HTML deliverable from the current run and publishes it under docs/.

  unit screen and shift chart   ml/reports/unit_screen.html
  quality dashboard             analytics/reports/dashboard.html
  ML overview                   ml/reports/ml_overview.html
  ML technical report           ml/reports/ml_technical.html
  MLOps monitoring              ml/reports/monitoring_report.html

Screenshots of the screen and the dashboard come from headless Chrome when it is installed.

Usage: python -m ml.reports.build
"""
import shutil
import subprocess
from pathlib import Path

from analytics.reports import generate_dashboard
from ml.reports import generate_ml_overview, generate_ml_technical, generate_monitoring_report, generate_unit_screen

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CHROME = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
          Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
          Path("/usr/bin/google-chrome"), Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]


def screenshot(html, png, w=1400, h=1000):
    chrome = next((c for c in CHROME if c.exists()), None)
    if chrome is None:
        print("  chrome not found; screenshot skipped:", png.name)
        return
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                    f"--screenshot={png}", html.resolve().as_uri()], check=False, capture_output=True, timeout=120)
    print("  screenshot", png.relative_to(ROOT))


def main():
    for mod in (generate_unit_screen, generate_dashboard, generate_ml_overview, generate_ml_technical, generate_monitoring_report):
        mod.main()
    (DOCS / "reports").mkdir(parents=True, exist_ok=True)
    (DOCS / "screenshots").mkdir(parents=True, exist_ok=True)
    pages = {"unit_screen.html": ROOT / "ml/reports/unit_screen.html",
             "dashboard.html": ROOT / "analytics/reports/dashboard.html",
             "ml_overview.html": ROOT / "ml/reports/ml_overview.html",
             "ml_technical.html": ROOT / "ml/reports/ml_technical.html",
             "monitoring_report.html": ROOT / "ml/reports/monitoring_report.html"}
    for name, src in pages.items():
        shutil.copy(src, DOCS / "reports" / name)
    screenshot(DOCS / "reports/unit_screen.html", DOCS / "screenshots/unit_screen.png", 1400, 1100)
    screenshot(DOCS / "reports/dashboard.html", DOCS / "screenshots/dashboard.png", 1100, 1400)
    print("published to", DOCS)


if __name__ == "__main__":
    main()
