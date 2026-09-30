"""Shared palette, chart helpers and the HTML shell for the reports and dashboard."""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BRAND_BLUE = "#3D5166"
ACCENT = "#6B8FA8"
LIGHT_BLUE = "#A8C0D1"
AMBER = "#D4881E"
RED = "#CC0000"
GREEN = "#1A7A3A"
GREY = "#AAAAAA"
DARK_GREY = "#555555"
TEXT = "#222222"

CHART_W, CHART_H, CHART_DPI = 8.2, 3.8, 130
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#DDDDDD", "axes.grid": False,
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": CHART_DPI,
})


def chart_style(ax):
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#DDDDDD")


def fig(h=None, w=None, ncols=1, nrows=1, **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=(w or CHART_W, h or CHART_H), **kw)
    for a in (ax.ravel() if hasattr(ax, "ravel") else [ax]):
        chart_style(a)
    return f, ax


def img(f, alt=""):
    buf = io.BytesIO()
    f.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(f)
    return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def table(df, fmt=None, cls="data"):
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            s = f(v) if f and v == v else ("" if v != v else str(v))
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            cells += f'<td class="{"num" if num else ""}">{s}</td>'
        body += f"<tr>{cells}</tr>"
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


CSS = f"""
:root {{ --brand: {BRAND_BLUE}; --accent: {ACCENT}; --text: {TEXT}; --muted: {DARK_GREY}; --rule: #DDDDDD; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #F4F5F7; color: var(--text); font-family: "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 15px; line-height: 1.55; }}
.page {{ max-width: 980px; margin: 0 auto; background: white; padding: 40px 56px 64px; }}
header.doc {{ border-bottom: 3px solid var(--brand); padding-bottom: 14px; margin-bottom: 28px; }}
header.doc .kicker {{ color: var(--accent); font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; font-weight: 600; }}
header.doc h1 {{ margin: 4px 0 6px; font-size: 28px; color: var(--brand); }}
header.doc .meta {{ color: var(--muted); font-size: 13px; }}
h2 {{ color: var(--brand); font-size: 21px; margin: 36px 0 8px; border-bottom: 1px solid var(--rule); padding-bottom: 4px; }}
h3 {{ color: var(--brand); font-size: 16px; margin: 22px 0 6px; }}
p {{ margin: 8px 0 12px; }}
img {{ max-width: 100%; display: block; margin: 10px 0 4px; }}
.caption {{ color: var(--muted); font-size: 12.5px; margin-bottom: 18px; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 16px; }}
table.data th {{ background: var(--brand); color: white; text-align: left; padding: 6px 8px; font-weight: 600; }}
table.data td {{ border-bottom: 1px solid #EEE; padding: 5px 8px; }}
table.data td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0 22px; }}
.kpi {{ border: 1px solid var(--rule); border-top: 3px solid var(--accent); padding: 10px 12px; }}
.kpi .v {{ font-size: 24px; font-weight: 700; color: var(--brand); }}
.kpi .l {{ font-size: 12px; color: var(--muted); }}
.note {{ background: #F7F9FB; border-left: 3px solid var(--accent); padding: 10px 14px; margin: 14px 0; font-size: 14px; }}
nav.toc {{ font-size: 13.5px; margin: 0 0 24px; }}
nav.toc a {{ color: var(--accent); text-decoration: none; margin-right: 14px; }}
@media (max-width: 700px) {{ .page {{ padding: 20px 16px 40px; }} }}
"""


def shell(title, kicker, meta, body, toc=None):
    nav = ""
    if toc:
        nav = '<nav class="toc">' + "".join(f'<a href="#{a}">{t}</a>' for a, t in toc) + "</nav>"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head>
<body><div class="page">
<header class="doc"><div class="kicker">{kicker}</div><h1>{title}</h1><div class="meta">{meta}</div></header>
{nav}{body}
</div></body></html>"""
