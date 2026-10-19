# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright"]
# ///
"""Draws the model card's charts: speed.png, quality.png, size.png, size-vs-quality.png.

Run:  uv run cards/dsx-2bit/assets/make_charts.py      (uv installs playwright; any Chromium works)
Each chart is an SVG drawn here, rendered by headless Chromium at 2x (1200 px wide) on a white card.
Chromium: PW_CHROMIUM=/path/to/chrome, else the first one found in ~/.cache/ms-playwright, else
`uv run --with playwright playwright install chromium` once. Fonts: Pathway Extreme from Google Fonts,
Iosevka for numbers (CHART_NUM_FONT=/path/to/font.woff2 to use a specific file); both fall back to system fonts.
All numbers are the ones below; edit them here and rerun.
"""
import glob, os, pathlib
from playwright.sync_api import sync_playwright

OUT = pathlib.Path(__file__).resolve().parent
W = 600  # CSS px; PNGs are 2x

# ---------------------------------------------------------------- data
SPEED = {  # tok/s, exact routing, greedy
    "2x RTX 3090 + 128 GB DDR4": {"prose": 37.2, "agent": 30.0, "p3k": 334, "p8k": 508},
    "RTX 5090 + 128 GB DDR5": {"prose": 12.0, "agent": 15.2, "p3k": 93, "p8k": 172},
}
QUALITY = [  # Lucebox 1017 eval set: 8 x 1024 tokens vs FP8 top-256 log-probs
    # name, KL, top-1 %, perplexity, highlight
    ("FP8 original", 0.0, 100.0, 8.56, False),
    ("Lucebox ROCMFP, 2.5 bpw", 0.253, 78.7, 9.78, False),
    ("dsx 2-bit, 2.03 bpw", 0.397, 74.1, 10.81, True),
]
SIZE = [  # GB to download, without the 203 GB Engram tables unless marked
    ("FP8 original", 510, "includes Engram", False),
    ("Q2_K GGUF (mrfakename)", 265, "", False),
    ("EXL3 3.5 bpw", 258, "", False),
    ("EXL3 2.9 bpw", 211, "", False),
    ("dsx 2-bit", 148, "", True),
    ("VQ (Baekpica)", 115, "approx.", False),
]
SIZE_QUALITY = [  # sizes all include the Engram tables; perplexity on the Lucebox eval
    # name, GB, GB range or None, perplexity, highlight
    ("FP8 original", 510, None, 8.56, False),
    ("Lucebox ROCMFP 2.5 bpw", 395, (383, 405), 9.78, False),
    ("dsx 2-bit", 351, None, 10.81, True),
]

# ---------------------------------------------------------------- style
BLUE, BLUE2, GREY, GREY2, INK, MUTED, GRID = "#4d6bfe", "#a9b7ff", "#9aa3ad", "#5e6872", "#14181f", "#5d6670", "#e4e8ec"
FONT = os.environ.get("CHART_NUM_FONT") if os.path.exists(os.environ.get("CHART_NUM_FONT", "")) else None
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Pathway+Extreme:opsz,wdth,wght@8..12,75..100,400..600&display=swap');
{f"@font-face{{font-family:Num;src:url(file://{FONT})}}" if FONT else ""}
html,body{{margin:0;background:transparent}}
.card{{width:{W}px;box-sizing:border-box;background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:20px 22px 16px}}
h1{{margin:0;font:600 21px/1.15 'Pathway Extreme',system-ui,sans-serif;font-variation-settings:'wdth' 85;color:{INK}}}
.sub{{margin-top:4px;font:400 12.5px/1.35 system-ui,-apple-system,'Segoe UI',sans-serif;color:{MUTED}}}
svg{{display:block;margin-top:10px;overflow:visible}}
svg text{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:{INK};font-size:12px}}
svg .n{{font-family:Num,Iosevka,'Iosevka Term',ui-monospace,Menlo,monospace}}
svg .m{{fill:{MUTED}}}
svg .v{{paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}}
.key{{display:flex;gap:18px;margin-top:10px;font:400 12.5px system-ui,sans-serif;color:{INK}}}
.key i{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.note{{margin-top:8px;font:400 11.5px/1.35 system-ui,sans-serif;color:{MUTED}}}
"""

def page(title, sub, svg, key="", note="", width=W):
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style><div class=card style='width:{width}px'><h1>{title}</h1><div class=sub>{sub}</div>{key}{svg}{f'<div class=note>{note}</div>' if note else ''}</div>"

def t(x, y, s, cls="", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{s}</text>'

def key(items):
    return "<div class=key>" + "".join(f"<span><i style='background:{c}'></i>{n}</span>" for n, c in items) + "</div>"

# ---------------------------------------------------------------- charts
def speed():
    machines = list(SPEED)
    cols = [BLUE, GREY]
    panels = [("Decode", [("prose", "prose"), ("agent", "agent, 3K prompt")], 45, 10),
              ("Prefill", [("p3k", "3K prompt"), ("p8k", "8K prompt")], 650, 100)]
    pw, h, top, bot = 262, 230, 26, 196
    s = []
    for pi, (name, groups, mx, step) in enumerate(panels):
        ox = pi * (pw + 32) + 34
        s.append(t(ox - 30, 12, name + " (tok/s)", extra='font-weight="600"'))
        y = lambda v: bot - v / mx * (bot - top)
        for v in range(0, mx + 1 - step // 2, step):
            s.append(f'<line x1="{ox}" x2="{ox+pw-34}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID if v else "#b8c0c8"}"/>')
            s.append(t(ox - 6, y(v) + 4, v, "n m", "end", 'font-size="10.5"'))
        gw = (pw - 34) / len(groups)
        for gi, (k, label) in enumerate(groups):
            gx = ox + gi * gw + (gw - 2 * 40 - 6) / 2
            for mi, mname in enumerate(machines):
                v = SPEED[mname][k]; x = gx + mi * 46
                s.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="40" height="{bot-y(v):.1f}" rx="2" fill="{cols[mi]}"/>')
                s.append(t(x + 20, y(v) - 5, f"{v:.1f}" if mx < 100 else f"{v:g}", "n v", "middle", 'font-size="11.5"'))
            s.append(t(ox + gi * gw + gw / 2, bot + 17, label, "", "middle"))
    svg = f'<svg width="{W-44}" height="{h}">{"".join(s)}</svg>'
    return page("dsx speed: DeepSeek-V4.1-Flash-1017 2-bit", "Tokens per second, exact routing, greedy decoding.", svg, key(list(zip(machines, cols))))

def quality():
    lw, pw, gap, rh, top = 168, 112, 16, 30, 24
    panels = [("Perplexity ↓", 3, 12, "{:.2f}"), ("Top-1 match ↑", 2, 100, "{:.1f}%"), ("KL divergence ↓", 1, 0.5, "{:.3f}")]
    s = []
    for pi, (name, idx, mx, fmt) in enumerate(panels):
        ox = lw + pi * (pw + gap)
        s.append(t(ox, 12, name, extra='font-weight="600"'))
        for ri, row in enumerate(QUALITY):
            v, hl = row[idx], row[4]
            y = top + ri * rh
            c = BLUE if hl else (GREY2 if ri == 0 else GREY)
            bw = v / mx * (pw - 44)
            if bw > 0: s.append(f'<rect x="{ox}" y="{y+4}" width="{bw:.1f}" height="18" rx="2" fill="{c}"/>')
            s.append(t(ox + bw + 5, y + 17, fmt.format(v), "n v", extra=f'font-size="11.5" {"font-weight=\"700\"" if hl else ""}'))
    for ri, row in enumerate(QUALITY):
        s.append(t(0, top + ri * rh + 17, row[0], "", extra=f'{"font-weight=\"700\"" if row[4] else ""}'))
    svg = f'<svg width="{W-44}" height="{top + rh*len(QUALITY)}">{"".join(s)}</svg>'
    return page("Quality against the FP8 original", "Lucebox 1017 eval set: 8 × 1,024 tokens, compared with FP8's top-256 log-probs.", svg)

def size():
    lw, rh, top, mx = 168, 28, 4, 550
    bx = lambda v: lw + v / mx * (W - 44 - lw - 165)
    s = []
    for g in range(0, mx + 1, 100):
        s.append(f'<line x1="{bx(g):.1f}" x2="{bx(g):.1f}" y1="{top}" y2="{top + rh*len(SIZE)}" stroke="{GRID}"/>')
        s.append(t(bx(g), top + rh * len(SIZE) + 14, g, "n m", "middle", 'font-size="10.5"'))
    for ri, (name, gb, tag, hl) in enumerate(SIZE):
        y = top + ri * rh
        c = BLUE if hl else GREY
        pat = ' fill="url(#hatch)"' if tag == "includes Engram" else f' fill="{c}"'
        s.append(t(0, y + 16, name, "", extra='font-weight="700"' if hl else ""))
        s.append(f'<rect x="{lw}" y="{y+4}" width="{bx(gb)-lw:.1f}" height="17" rx="2"{pat}/>')
        s.append(t(bx(gb) + 6, y + 16, ("~" if tag == "approx." else "") + f"{gb} GB" + (" · includes Engram" if tag == "includes Engram" else ""), "n", extra=f'font-size="11.5" {"font-weight=\"700\"" if hl else ""}'))
    defs = f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="{GREY}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" stroke-width="2.5"/></pattern></defs>'
    svg = f'<svg width="{W-44}" height="{top + rh*len(SIZE) + 20}">{defs}{"".join(s)}</svg>'
    return page("Download size of DeepSeek-V4.1-Flash-1017 builds", "Gigabytes, without the 203 GB Engram tables unless marked.", svg)

def size_quality():
    X0, X1, Y0, Y1 = 46, W - 44 - 20, 226, 14
    xl, xh, yl, yh = 300, 550, 8, 12
    x = lambda v: X0 + (v - xl) / (xh - xl) * (X1 - X0)
    y = lambda v: Y0 - (v - yl) / (yh - yl) * (Y0 - Y1)
    s = []
    for g in range(xl, xh + 1, 50):
        s.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{Y1}" y2="{Y0}" stroke="{GRID}"/>')
        s.append(t(x(g), Y0 + 15, g, "n m", "middle", 'font-size="10.5"'))
    for g in range(yl, yh + 1):
        s.append(f'<line x1="{X0}" x2="{X1}" y1="{y(g):.1f}" y2="{y(g):.1f}" stroke="{GRID}"/>')
        s.append(t(X0 - 7, y(g) + 4, g, "n m", "end", 'font-size="10.5"'))
    s.append(t((X0 + X1) / 2, Y0 + 34, "Size with Engram tables (GB)", "m", "middle"))
    s.append(t(X0 - 34, (Y0 + Y1) / 2, "Perplexity ↓", "m", "middle", f'transform="rotate(-90 {X0-34} {(Y0+Y1)/2})"'))
    place = {"FP8 original": (-10, 4, "end"), "Lucebox ROCMFP 2.5 bpw": (0, -14, "middle"), "dsx 2-bit": (12, 4, "start")}
    for name, gb, rng, ppl, hl in SIZE_QUALITY:
        c = BLUE if hl else (BLUE2 if name.startswith("dsx") else GREY2)
        if rng: s.append(f'<line x1="{x(rng[0]):.1f}" x2="{x(rng[1]):.1f}" y1="{y(ppl):.1f}" y2="{y(ppl):.1f}" stroke="{GREY2}" stroke-width="2"/>')
        s.append(f'<circle cx="{x(gb):.1f}" cy="{y(ppl):.1f}" r="{7 if hl else 6}" fill="{c}"/>')
        dx, dy, a = place[name]
        size_txt = f"~{gb} ({rng[0]}–{rng[1]})" if rng else f"{gb}"
        s.append(t(x(gb) + dx, y(ppl) + dy, f'{name} <tspan class="n m">{size_txt} GB, {ppl:.2f}</tspan>', "", a, 'font-weight="700"' if hl else ""))
    svg = f'<svg width="{W-44}" height="{Y0 + 42}">{"".join(s)}</svg>'
    return page("Size against perplexity", "Lucebox 1017 eval set. Sizes include the 203 GB Engram tables.", svg)

BRAND_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Pathway+Extreme:ital,opsz,wdth,wght@0,8..12,75..100,300..600;1,8..12,75..100,300..600&display=swap');
{f"@font-face{{font-family:Num;src:url(file://{FONT})}}" if FONT else ""}
html,body{{margin:0}}
.card{{width:1600px;height:900px;box-sizing:border-box;position:relative;overflow:hidden;padding:64px 76px;color:#f0f6f7;
  font:400 22px/1.4 Num,Iosevka,ui-monospace,Menlo,monospace;
  background:radial-gradient(60% 70% at 88% 0%,rgba(125,240,212,.16),transparent 65%),radial-gradient(50% 60% at 0% 100%,rgba(154,168,255,.10),transparent 70%),#0d1318}}
.eyebrow{{font-weight:700;font-size:20px;letter-spacing:.16em;color:#7df0d4;text-transform:uppercase}}
h1{{margin:18px 0 0;font:300 66px/1.02 'Pathway Extreme',sans-serif;font-variation-settings:'wdth' 75,'opsz' 12;letter-spacing:-.01em}}
h1 em{{color:#7df0d4;font-style:italic}}
.hero{{margin-top:54px;display:grid;grid-template-columns:repeat(3,1fr);gap:28px}}
.tile{{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:26px 30px}}
.big{{font:500 88px/1 'Pathway Extreme',sans-serif;font-variation-settings:'wdth' 75;color:#7df0d4;white-space:nowrap}}
.big small{{font-size:40px;margin-left:10px;color:#bff7e9}}
.lab{{margin-top:12px;color:#a9b8bd;font-size:21px}}
.cmp{{margin-top:44px}}
.cmp h2{{margin:0 0 18px;font:600 26px/1 Num,Iosevka,ui-monospace,monospace}}
.cmp h2 span{{color:#a9b8bd;font-weight:400}}
.row{{display:grid;grid-template-columns:260px 1fr 110px;align-items:center;gap:22px;height:46px}}
.nm{{font-weight:700;font-size:24px}} .pc{{font-weight:700;font-size:30px;text-align:right}}
.track{{height:26px;border-radius:6px;background:rgba(255,255,255,.06)}} .track b{{display:block;height:100%;border-radius:6px}}
.foot{{position:absolute;left:76px;right:76px;bottom:40px;display:flex;justify-content:space-between;color:#8a9a9f;font-size:20px}}
"""

POST = {  # the X post: a few big numbers, one comparison
    "speed": "30+", "size": "148", "prefill": "500",  # prefill: 508 tok/s on an 8K prompt
    # top-1 match with FP8 on the Lucebox 1017 eval set (QUALITY above), rounded
    "match": [("FP8 original", 100, "#c9d6da"), ("Lucebox 2.5-bit", 79, "#9aa8ff"), ("dsx 2-bit", 74, "#7df0d4")],
}

def post():
    """16:9 social card in Pocket Model's site style: dark ground, mint accent, Pathway Extreme headings."""
    bars = "".join(
        f"<div class=row><span class=nm style='color:{c}'>{n}</span><div class=track><b style='width:{v}%;background:{c}'></b></div>"
        f"<span class=pc style='color:{c}'>{v}%</span></div>" for n, v, c in POST["match"])
    css = BRAND_CSS
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · 2-bit quant + engine</div>
<h1>DeepSeek-V4.1-Flash-1017 (552B MoE)<br><em>on 2x RTX 3090 + 128 GB RAM</em></h1>
<div class=hero>
  <div class=tile><div class=big>{POST['speed']}<small>tok/s</small></div><div class=lab>generation speed</div></div>
  <div class=tile><div class=big>{POST['size']}<small>GB</small></div><div class=lab>model download</div></div>
  <div class=tile><div class=big>{POST['prefill']}<small>tok/s</small></div><div class=lab>prompt processing</div></div>
</div>
<div class=cmp><h2>Same next token as FP8 <span>top-1 match</span></h2>{bars}</div>
<div class=foot><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div>
</div>"""

EVAL_POST = {  # Pocket Model dogfood v26b: dsx 2-bit vs the hosted model, temperature 1.0, think low, graded in code
    "rows": [("Tool calls", 20, 16, 14), ("Olympiad math", 3, 3, 3), ("Hard multiple choice", 3, 1, 1), ("All", 26, 20, 18)],
    "dsx": ("dsx 2-bit, at home", "#7df0d4"), "hosted": ("Full model, hosted", "#5b7cff"),
}

def post_eval():
    """16:9 social card: dsx 2-bit vs the hosted model on a 26-row Pocket Model eval."""
    dn, dc = EVAL_POST["dsx"]; hn, hc = EVAL_POST["hosted"]
    groups = ""
    for name, n, d, h in EVAL_POST["rows"]:
        bar = lambda k, c: (f"<div class=vb><span style='color:{c}'>{k}/{n}</span>"
                            f"<b style='height:{k / n * 240:.0f}px;background:{c}'></b></div>")
        groups += (f"<div class='grp{' all' if name == 'All' else ''}'><div class=pair>{bar(d, dc)}{bar(h, hc)}</div>"
                   f"<div class=gn>{name}</div><div class=gr>{n} rows</div></div>")
    css = BRAND_CSS + """
h1{font-size:54px;margin-top:14px}
.card{padding-top:52px}
.body{margin-top:30px;display:grid;grid-template-columns:1fr 430px;gap:40px}
.chart{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:22px 30px 16px}
.chart h2{margin:0;font:600 25px/1 Num,Iosevka,ui-monospace,monospace}
.chart .k{display:flex;gap:26px;margin-top:14px;font-size:20px}
.chart .k i{display:inline-block;width:16px;height:16px;border-radius:4px;margin-right:9px;vertical-align:-2px}
.groups{margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;align-items:end;height:310px;border-bottom:2px solid rgba(228,241,244,.3)}
.grp{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.grp.all{background:rgba(255,255,255,.04);border-radius:12px 12px 0 0}
.pair{display:flex;gap:10px;align-items:flex-end}
.vb{width:62px;display:flex;flex-direction:column;align-items:center}
.vb span{font-weight:700;font-size:23px;margin-bottom:8px;white-space:nowrap}
.vb b{display:block;width:100%;border-radius:5px 5px 0 0}
.gn,.gr{position:relative;top:0}
.grp .gn{order:3;margin-top:0}
.labels{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;text-align:center;margin-top:12px}
.labels b{display:block;font-size:20px} .labels span{color:#a9b8bd;font-size:17px}
.tiles{display:flex;flex-direction:column;gap:16px}
.tiles .tile{padding:18px 26px}
.tiles .big{font-size:66px}
.tiles .lab{margin-top:8px;font-size:20px}
.foot{flex-direction:column;gap:6px;bottom:28px;font-size:18px}
.foot .r{display:flex;justify-content:space-between}
.foot .note{color:#a9b8bd}
"""
    names = [r[0] for r in EVAL_POST["rows"]]; ns = [r[1] for r in EVAL_POST["rows"]]
    groups = groups.replace("<div class=gn>", "<div style='display:none'>").replace("<div class=gr>", "<div style='display:none'>")
    labels = "".join(f"<div><b>{a}</b><span>{n} rows</span></div>" for a, n in zip(names, ns))
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · Pocket Model eval · 26 hard rows</div>
<h1>2-bit DeepSeek-V4.1-Flash-1017 on two RTX 3090s<br><em>matches the hosted full model</em></h1>
<div class=body>
 <div class=chart><h2>Rows passed</h2>
  <div class=k><span><i style='background:{dc}'></i>{dn}</span><span><i style='background:{hc}'></i>{hn}</span></div>
  <div class=groups>{groups}</div><div class=labels>{labels}</div></div>
 <div class=tiles>
  <div class=tile><div class=big>2×<small>tokens</small></div><div class=lab>thinking on math and multiple choice</div></div>
  <div class=tile><div class=big>29<small>tok/s</small></div><div class=lab>2x RTX 3090 + 128 GB RAM</div></div>
  <div class=tile><div class=big>$0<small>per token</small></div><div class=lab>self-hosted, 148 GB</div></div>
 </div>
</div>
<div class=foot><div class=note>Both at temperature 1.0, think low, graded in code. Within noise (p = 0.69): the hosted model scored 20/26 here at temperature 0.</div>
<div class=r><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div></div>
</div>"""

OVERTHINK = {  # reasoning tokens per run, temperature 1.0 unless noted; every run correct unless noted
    "builds": [("hosted", "Full model, hosted", "#5b7cff"), ("v2", "v2 2-bit", "#ff6b7a"), ("v3a", "v3a 2-bit", "#7df0d4")],
    "problems": [
        ("Problem A", "olympiad math, answer 480/49", {
            "hosted": [17426, 11069, 10465, 6371, 6799, 5665, 8426, 13485],
            "v2": [10126, 39479, 12159, 31487], "v3a": [10469, 10263, 15311]}, None),
        ("Problem B", "olympiad math, answer 500", {
            "hosted": [12523, 7507, 11651, 9523, 8879, 13870, 9741, 12147],
            "v2": [26462, 22645], "v3a": [21489, 20180]}, "v2"),  # v2 at temperature 0: looped past 55K, no answer
    ],
    "xmax": 60000,
}

def post_overthink():
    """16:9 social card: reasoning tokens per run, v2 vs v3a vs the hosted full model, on two hard math problems."""
    W, L, R, rowh = 1448, 250, 56, 49
    xmax = OVERTHINK["xmax"]; builds = OVERTHINK["builds"]
    x = lambda v: L + min(v, xmax) / xmax * (W - L - R)
    panels = ""
    for name, sub, runs, loop in OVERTHINK["problems"]:
        H = 40 + rowh * len(builds) + (56 if name == OVERTHINK["problems"][-1][0] else 30)
        top = 40
        s = [f'<text x="0" y="22" class="pt">{name}</text><text x="{len(name) * 15 + 18}" y="22" class="ps">{sub}</text>']
        lo, hi = min(runs["hosted"]), max(runs["hosted"])
        s.append(f'<rect x="{x(lo):.1f}" y="{top}" width="{x(hi) - x(lo):.1f}" height="{rowh * len(builds)}" rx="6" fill="rgba(91,124,255,.16)"/>')
        for t in range(0, xmax + 1, 10000):
            s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + rowh * len(builds)}" stroke="rgba(228,241,244,.10)"/>')
            s.append(f'<text x="{x(t):.1f}" y="{top + rowh * len(builds) + 24}" class="ax" text-anchor="middle">{"0" if t == 0 else f"{t // 1000}K"}</text>')
        for i, (k, label, col) in enumerate(builds):
            cy = top + rowh * i + rowh / 2
            s.append(f'<text x="0" y="{cy + 7}" class="rl" style="fill:{col}">{label}</text>')
            for j, v in enumerate(sorted(runs[k])):
                s.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="12.5" fill="{col}" stroke="#0d1318" stroke-width="2.5"/>')
            if loop == k:
                ax = x(xmax)
                s.append(f'<path d="M {ax - 22} {cy - 11} L {ax} {cy} L {ax - 22} {cy + 11} Z" fill="{col}"/>')
                s.append(f'<text x="{ax - 32}" y="{cy + 7}" class="lp" text-anchor="end">temperature 0: looped past 55K, no answer</text>')
        if name == OVERTHINK["problems"][-1][0]:
            s.append(f'<text x="{W - R}" y="{top + rowh * len(builds) + 52}" class="ax" text-anchor="end">reasoning tokens per run</text>')
        panels += f'<div class=panel><svg width="{W}" height="{H}">{"".join(s)}</svg></div>'
    css = BRAND_CSS + """
.card{padding-top:52px}
h1{font-size:56px;margin-top:14px}
.panels{margin-top:30px;display:flex;flex-direction:column;gap:18px}
.panel{background:rgba(255,255,255,.045);border:1px solid rgba(228,241,244,.14);border-radius:18px;padding:18px 26px 8px}
svg{display:block;overflow:visible}
svg text{font-family:Num,Iosevka,ui-monospace,Menlo,monospace;fill:#f0f6f7}
.pt{font-size:26px;font-weight:700} .ps{font-size:21px;fill:#a9b8bd} .rl{font-size:22px;font-weight:700}
.ax{font-size:17px;fill:#8a9a9f} .lp{font-size:19px;fill:#ffb3bc}
.foot{flex-direction:column;gap:6px;bottom:28px;font-size:18px}
.foot .r{display:flex;justify-content:space-between} .foot .note{color:#a9b8bd}
"""
    return f"""<!doctype html><meta charset=utf-8><style>{css}</style><div class=card>
<div class=eyebrow>dsx · 2-bit DeepSeek-V4.1-Flash for 2x RTX 3090 · now with image input</div>
<h1>v3a stops the runaway reasoning<br><em>same size, speed and KL as v2</em></h1>
<div class=panels>{panels}</div>
<div class=foot><div class=note>Each dot is one run at temperature 1.0, all correct. Blue band: the full model's range. v3a still thinks about 2× as long as the full model on problem B.</div>
<div class=r><span>github.com/lxe/dsx</span><span>huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-2bit</span></div></div>
</div>"""

# ---------------------------------------------------------------- render
def chromium():
    if os.environ.get("PW_CHROMIUM"): return os.environ["PW_CHROMIUM"]
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    return found[-1] if found else None

if __name__ == "__main__":
    charts = {"speed.png": speed(), "quality.png": quality(), "size.png": size(), "size-vs-quality.png": size_quality(), "post.png": post(), "post-eval.png": post_eval(), "post-overthink.png": post_overthink()}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium(), args=["--allow-file-access-from-files"])
        pg = b.new_page(viewport={"width": 840, "height": 900}, device_scale_factor=2)
        for name, html in charts.items():
            tmp = OUT / f".{name}.html"
            tmp.write_text(html)
            pg.set_viewport_size({"width": 1600, "height": 900} if name.startswith("post") else {"width": 840, "height": 900})
            pg.goto(tmp.as_uri(), wait_until="networkidle")
            pg.evaluate("document.fonts.ready")
            pg.locator(".card").screenshot(path=str(OUT / name), omit_background=True)
            tmp.unlink()
            print("wrote", OUT / name)
        b.close()
