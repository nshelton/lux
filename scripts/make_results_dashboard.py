#!/usr/bin/env python3
"""Build a results dashboard (results.html at the repo root) over eval_capture +
eval_hemisphere outputs: a checkpoint x scene capture-accuracy table, a checkpoint x
obliquity-bin hemisphere table, and an embedded figure gallery (summary.png per
scene + hemisphere_overview.png), with a link to the per-pixel viewer.

    python scripts/make_results_dashboard.py
    ~/.venvs/lux/bin/python -m http.server   # from repo root -> :8000/results.html

Served from the REPO ROOT (so both captures/ figures and evals/ figures resolve).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OBL = [0, 15, 30, 45, 60, 75]


def _cap_metrics(root: Path, ckpt: str) -> dict:
    """{scene: {...}} for a checkpoint's capture evals."""
    out = {}
    for mj in sorted(root.glob(f"captures/*/eval_{ckpt}/metrics.json")):
        m = json.loads(mj.read_text())
        scene = mj.parent.parent.name
        row = m.get("row", {})
        out[scene] = {"u": m.get("bin_acc"), "cov": m.get("coverage"), "du": m.get("med_du"),
                      "uv": row.get("uv_bin_acc"), "v": row.get("v_bin_acc"),
                      "fig": f"captures/{scene}/eval_{ckpt}/summary.png"}
    return out


def _hemi_bins(root: Path, ckpt: str) -> dict:
    """{obliquity_bin: mean bin_acc} from per_sample.csv, + the overview fig path."""
    csvp = root / f"evals/hemisphere/results_{ckpt}/per_sample.csv"
    if not csvp.exists():
        return {}
    rows = list(csv.DictReader(csvp.open()))
    if not rows:
        return {}
    theta = np.array([max(float(r["theta_cam_deg"]), float(r["theta_proj_deg"])) for r in rows])
    ba = np.array([float(r["bin_acc"]) for r in rows])
    out = {}
    for lo, hi in zip(OBL, OBL[1:]):
        sel = (theta >= lo) & (theta < hi)
        out[f"{lo}-{hi}"] = float(ba[sel].mean()) if sel.any() else None
    out["_fig"] = f"evals/hemisphere/results_{ckpt}/hemisphere_overview.png"
    return out


def _cell(v):
    if v is None:
        return '<td class="na">–</td>'
    g = int(40 + 150 * max(0.0, min(1.0, v)))     # green ramp
    r = int(150 * (1 - max(0.0, min(1.0, v))))
    return f'<td style="background:rgb({r},{g},40)">{v * 100:.1f}</td>'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="*", default=None, help="checkpoint stems; default = discover")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    if args.ckpts:
        ckpts = [Path(c).stem for c in args.ckpts]
    else:
        ckpts = sorted({p.parent.name[len("eval_"):] for p in root.glob("captures/*/eval_*/metrics.json")})
    scenes = sorted({mj.parent.parent.name for mj in root.glob("captures/*/eval_*/metrics.json")})

    cap = {c: _cap_metrics(root, c) for c in ckpts}
    hemi = {c: _hemi_bins(root, c) for c in ckpts}

    # capture table: rows=ckpt, cols=scene, cell = u-bin %
    cap_tbl = ["<tr><th>checkpoint \\ scene (u-bin %)</th>" + "".join(f"<th>{s}</th>" for s in scenes) + "</tr>"]
    for c in ckpts:
        cells = "".join(_cell(cap[c].get(s, {}).get("u")) for s in scenes)
        cap_tbl.append(f"<tr><th class='l'>{c}</th>{cells}</tr>")

    # hemisphere table: rows=ckpt, cols=obliquity bins
    obl_cols = [f"{lo}-{hi}" for lo, hi in zip(OBL, OBL[1:])]
    hemi_tbl = ["<tr><th>checkpoint \\ obliquity (bin %)</th>" + "".join(f"<th>{b}°</th>" for b in obl_cols) + "</tr>"]
    for c in ckpts:
        cells = "".join(_cell(hemi[c].get(b)) for b in obl_cols)
        hemi_tbl.append(f"<tr><th class='l'>{c}</th>{cells}</tr>")

    # figure gallery data (per ckpt: scene figs + hemi fig)
    gallery = {c: {"scenes": {s: cap[c][s]["fig"] for s in cap[c]}, "hemi": hemi[c].get("_fig")} for c in ckpts}

    html = (_HTML.replace("__CAPTBL__", "\n".join(cap_tbl))
                 .replace("__HEMITBL__", "\n".join(hemi_tbl))
                 .replace("__GALLERY__", json.dumps(gallery))
                 .replace("__CKPTS__", json.dumps(ckpts)))
    (root / "results.html").write_text(html)
    print(f"dashboard -> {root / 'results.html'}   ({len(ckpts)} ckpts, {len(scenes)} scenes)")
    print(f"open:  ~/.venvs/lux/bin/python -m http.server   (from {root.resolve()})  ->  :8000/results.html")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>lux eval results</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;padding:16px}
 h2{margin:22px 0 8px;font-size:16px;color:#9cf}
 table{border-collapse:collapse;margin-bottom:8px}
 th,td{border:1px solid #333;padding:5px 10px;text-align:center;font-variant-numeric:tabular-nums}
 th{background:#1c1c1c} th.l{text-align:left;font-family:ui-monospace,monospace;font-weight:500}
 td.na{color:#555}
 select{background:#222;color:#ddd;border:1px solid #444;padding:5px 9px;border-radius:4px;font-size:14px}
 .grid{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
 .card{background:#181818;border:1px solid #333;border-radius:6px;padding:6px}
 .card div{font-family:ui-monospace,monospace;color:#9cf;margin-bottom:4px}
 .card img{display:block;max-width:560px;width:100%;cursor:zoom-in}
 img.big{max-width:96vw}
 a{color:#6cf}
</style></head><body>
<h1 style="font-size:18px">lux eval results <a href="captures/view.html" style="font-size:13px;margin-left:14px">→ per-pixel viewer</a></h1>
<h2>Capture u-bin accuracy (full coverage)</h2>
<table>__CAPTBL__</table>
<h2>Hemisphere bin accuracy by obliquity (synthetic bench)</h2>
<table>__HEMITBL__</table>
<h2>Figures <select id="ck"></select></h2>
<div id="gal" class="grid"></div>
<script>
const G=__GALLERY__, CK=__CKPTS__;
const sel=document.getElementById('ck'), gal=document.getElementById('gal');
CK.forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);});
function show(c){ gal.innerHTML=''; const g=G[c]; if(!g)return;
  Object.keys(g.scenes).sort().forEach(s=>{ gal.appendChild(card(s, g.scenes[s])); });
  if(g.hemi) gal.appendChild(card('hemisphere', g.hemi));
}
function card(label,src){ const d=document.createElement('div'); d.className='card';
  const t=document.createElement('div'); t.textContent=label; const im=document.createElement('img');
  im.src=src+'?'+Date.now(); im.onclick=()=>im.classList.toggle('big');
  im.onerror=()=>{t.textContent=label+'  (no figure — run eval_capture with --maps? summary.png missing)';im.remove();};
  d.appendChild(t); d.appendChild(im); return d;
}
sel.onchange=()=>show(sel.value);
const def=CK.find(c=>c.includes('newaug'))||CK[0]; if(def){sel.value=def;show(def);}
</script></body></html>"""


if __name__ == "__main__":
    main()
