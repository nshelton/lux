#!/usr/bin/env python3
"""Generate a self-contained interactive viewer for eval_capture results.

Finds every captures/<scene>/eval_<ckpt>/ for a checkpoint, packs each scene's
pred/ref/conf .npy into a compact uint16 binary, and writes ONE view_<ckpt>.html
that switches between scenes and data layers via dropdowns.

    python scripts/make_capture_viewer.py --ckpt proj_net_conv_newaug_ep06
    ~/.venvs/lux/bin/python -m http.server --directory captures   # -> :8000/view_<ckpt>.html

Layers (base image): x = pred col, y = pred row, confidence (turbo), and the signed
du/dv error clamped ±2px (coolwarm — small errors saturate). Hover for a crosshair +
live scanline plots (du across the row, dv down the column; residual or absolute
pred/ref curves) and a per-pixel pred/ref/du/dv/conf readout.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

SCALE = 32           # coord -> uint16 fixed point (1/32 px)
SENT = 0xFFFF


def _pack_coord(a: np.ndarray) -> bytes:
    q = np.round(np.where(np.isfinite(a), a, -1.0) * SCALE)
    return np.where(np.isfinite(a) & (a >= 0), np.clip(q, 0, SENT - 1), SENT).astype("<u2").tobytes()


def _pack(d: Path) -> tuple[int, int, int, int]:
    """Pack one eval dir's npy into d/viewer_data.bin; return (H, W, pw, ph)."""
    pred = np.load(d / "pred.npy")
    ref_col = np.load(d / "ref_col.npy")
    ref_row = (np.load(d / "ref_row.npy") if (d / "ref_row.npy").exists()
               else np.full(ref_col.shape, np.nan, np.float32))
    conf = np.load(d / "conf.npy")
    meta = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    pw, ph = meta.get("proj_wh", [1920, 1080])
    H, W = ref_col.shape
    blob = bytearray(struct.pack("<6I", 0x4C555856, H, W, int(pw), int(ph), SCALE))
    for a in (pred[..., 0], pred[..., 1], ref_col, ref_row):
        blob += _pack_coord(np.ascontiguousarray(a))
    blob += np.clip(np.round(np.nan_to_num(conf) * 255), 0, 255).astype(np.uint8).tobytes()
    (d / "viewer_data.bin").write_bytes(blob)
    return H, W, int(pw), int(ph)


def _lut(name: str) -> list:
    import matplotlib
    cmap = matplotlib.colormaps[name]
    return (np.asarray(cmap(np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="restrict to these checkpoint stems; default = every eval_*/ found")
    ap.add_argument("--captures-root", default="captures")
    ap.add_argument("--force", action="store_true", help="repack even if viewer_data.bin is fresh")
    args = ap.parse_args()
    root = Path(args.captures_root)
    want = {Path(c).stem for c in args.ckpts} if args.ckpts else None
    # discover every captures/<scene>/eval_<ckpt>/ that has the required arrays
    manifest: dict[str, dict[str, str]] = {}
    for f in sorted(root.glob("*/eval_*/pred.npy")):
        d = f.parent
        if not ((d / "ref_col.npy").exists() and (d / "conf.npy").exists()):
            continue
        ckpt = d.name[len("eval_"):]
        scene = d.parent.name
        if want and ckpt not in want:
            continue
        bin_ = d / "viewer_data.bin"
        if args.force or not bin_.exists() or bin_.stat().st_mtime < f.stat().st_mtime:
            try:
                _pack(d)
                print(f"  packed {ckpt} / {scene}")
            except Exception as e:  # noqa: BLE001 - skip incompatible old dirs
                print(f"  skip   {ckpt} / {scene}  ({e})")
                continue
        manifest.setdefault(ckpt, {})[scene] = f"{scene}/{d.name}/viewer_data.bin"
    if not manifest:
        raise SystemExit(f"no usable eval_*/ dirs under {root}")
    html = (_HTML
            .replace("__MANIFEST__", json.dumps(manifest))
            .replace("__TURBO__", json.dumps(_lut("turbo")))
            .replace("__DIVERGE__", json.dumps(_lut("coolwarm"))))
    out = root / "view.html"
    out.write_text(html)
    nck, nsc = len(manifest), len({s for v in manifest.values() for s in v})
    print(f"\nviewer -> {out}   ({nck} checkpoints x up to {nsc} scenes)")
    print(f"open:    ~/.venvs/lux/bin/python -m http.server --directory {root}  ->  :8000/view.html")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>capture viewer</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
 #ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 10px;background:#181818}
 select,button{background:#222;color:#ddd;border:1px solid #444;padding:4px 8px;border-radius:4px}
 button.on{background:#2a6;border-color:#2a6;color:#000;font-weight:600}
 #wrap{display:grid;grid-template-columns:auto 320px;gap:8px;padding:10px}
 canvas{background:#000}
 #img{cursor:crosshair;border:1px solid #333;image-rendering:pixelated}
 #vline,#hline{border:1px solid #333;background:#181818}
 #hline{grid-column:1/2}
 #read{font-family:ui-monospace,monospace;background:#181818;padding:5px 9px;border-radius:4px}
</style></head><body>
<div id="ctl">
 <label>ckpt <select id="ckpt"></select></label>
 <label>scene <select id="scene"></select></label>
 <label>layer <select id="layer">
   <option value="du">du error ±2px</option>
   <option value="dv">dv error ±2px</option>
   <option value="x">x = pred col</option>
   <option value="y">y = pred row</option>
   <option value="conf">confidence</option>
 </select></label>
 <label>scanline <select id="smode">
   <option value="resid">residual (pred-ref)</option>
   <option value="abs">absolute (pred & ref)</option>
 </select></label>
 <span id="read">loading…</span>
</div>
<div id="wrap">
 <canvas id="img"></canvas>
 <canvas id="vline" title="dv down the hovered column"></canvas>
 <canvas id="hline" title="du across the hovered row"></canvas>
</div>
<script>
const MANIFEST=__MANIFEST__, TURBO=__TURBO__, DIV=__DIVERGE__, SENT=0xFFFF, DARK=[20,20,20];
const DISP=Math.min(1180,Math.round(0.6*screen.width));
const img=document.getElementById('img'), vlc=document.getElementById('vline'), hlc=document.getElementById('hline'),
      read=document.getElementById('read'), selScene=document.getElementById('scene'),
      selLayer=document.getElementById('layer'), selMode=document.getElementById('smode');
let D=null, base=null, last=null;
let layer='du', smode='resid';
const ictx=img.getContext('2d');
const cv=(a,i)=>a[i]===SENT?NaN:a[i]/D.SCALE;
const cl=(v,a,b)=>v<a?a:v>b?b:v;

function renderBase(){
 if(!D) return; const {W,H,predC,predR,refC,refR,conf,PW,PH}=D;
 const id=ictx.createImageData(W,H), px=id.data;
 for(let i=0;i<W*H;i++){ let rgb;
   if(layer==='conf'){ rgb=TURBO[conf[i]]; }
   else if(layer==='x'){ const v=cv(predC,i); rgb=isNaN(v)?DARK:TURBO[cl(Math.round(v/PW*255),0,255)]; }
   else if(layer==='y'){ const v=cv(predR,i); rgb=isNaN(v)?DARK:TURBO[cl(Math.round(v/PH*255),0,255)]; }
   else { const p=layer==='du'?cv(predC,i):cv(predR,i), r=layer==='du'?cv(refC,i):cv(refR,i);
          rgb=(isNaN(p)||isNaN(r))?DARK:DIV[cl(Math.round(((p-r)/4+0.5)*255),0,255)]; }   // +-2px -> coolwarm
   const o=i*4; px[o]=rgb[0];px[o+1]=rgb[1];px[o+2]=rgb[2];px[o+3]=255;
 }
 img.width=W; img.height=H; img.style.width=DISP+'px'; img.style.height=Math.round(DISP*H/W)+'px';
 ictx.putImageData(id,0,0); base=id;
 if(last) draw(last[0],last[1]);
}
function draw(x,y){ ictx.putImageData(base,0,0);
 ictx.strokeStyle='rgba(0,255,140,.85)'; ictx.lineWidth=Math.max(1,D.W/DISP);
 ictx.beginPath(); ictx.moveTo(x,0);ictx.lineTo(x,D.H); ictx.moveTo(0,y);ictx.lineTo(D.W,y); ictx.stroke();
}
function plot(c,series,horiz){
 const w=c.width=horiz?DISP:300, h=c.height=horiz?180:Math.round(DISP*D.H/D.W), g=c.getContext('2d');
 g.fillStyle='#181818'; g.fillRect(0,0,w,h);
 let lo=Infinity,hi=-Infinity; for(const s of series)for(const v of s.data)if(isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
 if(!isFinite(lo)){g.fillStyle='#666';g.font='11px monospace';g.fillText('no valid pixels',8,18);return;}
 if(smode==='resid'){const m=Math.max(0.5,Math.abs(lo),Math.abs(hi));lo=-m;hi=m;}
 const pad=(hi-lo)||1, mapV=v=>{const t=(v-lo)/pad; return horiz?(h-8-t*(h-16)):(8+t*(h-16));};
 if(smode==='resid'){g.strokeStyle='#444';g.beginPath();const z=mapV(0); if(horiz){g.moveTo(0,z);g.lineTo(w,z);}else{g.moveTo(z,0);g.lineTo(z,h);}g.stroke();}
 for(const s of series){g.strokeStyle=s.c;g.beginPath();let st=false;
   for(let i=0;i<s.data.length;i++){const v=s.data[i];if(!isFinite(v)){st=false;continue;}
     const p=i/(s.data.length-1)*(horiz?w:h),q=mapV(v),cx=horiz?p:q,cy=horiz?q:p;
     if(!st){g.moveTo(cx,cy);st=true;}else g.lineTo(cx,cy);} g.stroke();}
 g.fillStyle='#aaa';g.font='11px monospace';
 g.fillText((horiz?'du across row':'dv down col')+'  ['+lo.toFixed(2)+','+hi.toFixed(2)+']px',8,14);
}
function hover(x,y){ last=[x,y]; draw(x,y); const {W,H,predC,predR,refC,refR,conf}=D;
 const i=y*W+x, pc=cv(predC,i),pr=cv(predR,i),rc=cv(refC,i),rr=cv(refR,i),cf=conf[i]/255;
 const f=v=>isNaN(v)?'  --':v.toFixed(2);
 read.textContent=`x${x} y${y}  pred(${f(pc)},${f(pr)})  ref(${f(rc)},${f(rr)})  du${f(pc-rc)} dv${f(pr-rr)}  conf ${cf.toFixed(2)}`;
 const duR=new Float32Array(W),pcR=new Float32Array(W),rcR=new Float32Array(W);
 for(let xx=0;xx<W;xx++){const j=y*W+xx,a=cv(predC,j),b=cv(refC,j);duR[xx]=a-b;pcR[xx]=a;rcR[xx]=b;}
 plot(hlc, smode==='resid'?[{data:duR,c:'#5cf'}]:[{data:pcR,c:'#5cf'},{data:rcR,c:'#fb5'}], true);
 const dvC=new Float32Array(H),prC=new Float32Array(H),rrC=new Float32Array(H);
 for(let yy=0;yy<H;yy++){const j=yy*W+x,a=cv(predR,j),b=cv(refR,j);dvC[yy]=a-b;prC[yy]=a;rrC[yy]=b;}
 plot(vlc, smode==='resid'?[{data:dvC,c:'#5cf'}]:[{data:prC,c:'#5cf'},{data:rrC,c:'#fb5'}], false);
}
img.addEventListener('mousemove',e=>{if(!D)return;const r=img.getBoundingClientRect();
 const x=Math.floor((e.clientX-r.left)/r.width*D.W), y=Math.floor((e.clientY-r.top)/r.height*D.H);
 if(x>=0&&x<D.W&&y>=0&&y<D.H) hover(x,y);});
selLayer.onchange=()=>{layer=selLayer.value;renderBase();};
selMode.onchange=()=>{smode=selMode.value;if(last)hover(last[0],last[1]);};
function load(url){ read.textContent='loading '+url+'…';
 fetch(url).then(r=>r.arrayBuffer()).then(buf=>{
   const hd=new Uint32Array(buf,0,6),W=hd[2],H=hd[1],n=W*H; let o=24;
   const u16=()=>{const a=new Uint16Array(buf,o,n);o+=n*2;return a;};
   D={W,H,PW:hd[3],PH:hd[4],SCALE:hd[5],predC:u16(),predR:u16(),refC:u16(),refR:u16(),conf:new Uint8Array(buf,o,n)};
   last=null; renderBase(); read.textContent=`${W}x${H}, proj ${D.PW}x${D.PH} — hover the image`;
 }).catch(e=>read.textContent='load failed: '+e+' (serve via http.server from captures/)');
}
const selCkpt=document.getElementById('ckpt');
function opt(sel,val){const o=document.createElement('option');o.value=val;o.textContent=val;sel.appendChild(o);}
function fillScenes(ck){const keep=selScene.value;selScene.innerHTML='';Object.keys(MANIFEST[ck]).sort().forEach(s=>opt(selScene,s));if([...selScene.options].some(o=>o.value===keep))selScene.value=keep;}
function loadCur(){const ck=selCkpt.value,sc=selScene.value;if(MANIFEST[ck]&&MANIFEST[ck][sc])load(MANIFEST[ck][sc]);}
Object.keys(MANIFEST).sort().forEach(ck=>opt(selCkpt,ck));
selCkpt.onchange=()=>{fillScenes(selCkpt.value);loadCur();};
selScene.onchange=loadCur;
const cks=Object.keys(MANIFEST).sort(), def=cks.find(c=>c.includes('newaug'))||cks[0];
if(def){selCkpt.value=def;fillScenes(def);loadCur();}else read.textContent='no checkpoints in manifest';
</script></body></html>"""


if __name__ == "__main__":
    main()
