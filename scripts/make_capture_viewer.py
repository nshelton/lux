#!/usr/bin/env python3
"""Generate a self-contained interactive viewer for an eval_capture result folder.

Reads pred.npy / ref_col.npy / ref_row.npy / conf.npy (+ metrics.json) from an
eval_<ckpt> dir, packs them into a compact binary, and writes view.html beside them.

    python scripts/make_capture_viewer.py --eval-dir captures/test4/eval_<ckpt>
    cd captures/test4/eval_<ckpt> && python -m http.server   # -> :8000/view.html

What it does that a static image can't: hover the image for a crosshair + live
**scanline** plots — du (pred_col - ref_col) across the hovered row, dv down the
hovered column — so you scrub to find exactly where the net and the Gray-code
reference diverge sub-pixel. Base layer toggles between the clamped signed-error
maps and confidence. Per-pixel readout of pred/ref/du/dv/conf.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

SCALE = 32           # coord -> uint16 fixed point (1/32 px ~= 0.03 px)
SENT = 0xFFFF        # invalid sentinel


def _pack_coord(a: np.ndarray) -> np.ndarray:
    q = np.round(np.where(np.isfinite(a), a, -1.0) * SCALE)
    return np.where(np.isfinite(a) & (a >= 0), np.clip(q, 0, SENT - 1), SENT).astype("<u2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="an eval_<ckpt> folder (with pred.npy etc.)")
    args = ap.parse_args()
    d = Path(args.eval_dir)
    pred = np.load(d / "pred.npy")                 # (H, W, 2) col,row
    ref_col = np.load(d / "ref_col.npy")           # (H, W)
    ref_row = (np.load(d / "ref_row.npy") if (d / "ref_row.npy").exists()
               else np.full(ref_col.shape, np.nan, np.float32))
    conf = np.load(d / "conf.npy")                 # (H, W)
    meta = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    pw, ph = meta.get("proj_wh", [1920, 1080])
    H, W = ref_col.shape

    blob = bytearray()
    blob += struct.pack("<6I", 0x4C555856, H, W, int(pw), int(ph), SCALE)   # 'LUXV' + dims
    for a in (pred[..., 0], pred[..., 1], ref_col, ref_row):
        blob += _pack_coord(np.ascontiguousarray(a)).tobytes()
    blob += np.clip(np.round(np.nan_to_num(conf) * 255), 0, 255).astype(np.uint8).tobytes()
    (d / "viewer_data.bin").write_bytes(blob)
    (d / "view.html").write_text(_HTML.replace("__NAME__", d.parent.name))
    print(f"viewer -> {d}/view.html   ({len(blob) / 1e6:.1f} MB data)")
    print(f"open:    cd {d} && ~/.venvs/lux/bin/python -m http.server  ->  http://localhost:8000/view.html")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>capture viewer — __NAME__</title>
<style>
  body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
  #wrap{display:grid;grid-template-columns:auto 320px;grid-template-rows:auto 200px;gap:8px;padding:10px}
  canvas{background:#000;image-rendering:pixelated}
  #img{cursor:crosshair;border:1px solid #333}
  #vline{border:1px solid #333}#hline{border:1px solid #333}
  #ctl{grid-column:1/3;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  button{background:#222;color:#ddd;border:1px solid #444;padding:5px 10px;border-radius:4px;cursor:pointer}
  button.on{background:#2a6;border-color:#2a6;color:#000;font-weight:600}
  #read{font-family:ui-monospace,monospace;white-space:pre;background:#181818;padding:6px 10px;border-radius:4px}
  label{user-select:none}
</style></head><body>
<div id="ctl">
  <b>__NAME__</b>
  <span>layer:</span>
  <button id="bDu" class="on">du ±2px</button>
  <button id="bDv">dv ±2px</button>
  <button id="bConf">confidence</button>
  <span style="margin-left:14px">scanline:</span>
  <button id="bResid" class="on">residual</button>
  <button id="bAbs">absolute</button>
  <span id="read">hover the image…</span>
</div>
<div id="wrap">
  <canvas id="img"></canvas>
  <canvas id="vline" title="dv down the hovered column"></canvas>
  <canvas id="hline" title="du across the hovered row"></canvas>
</div>
<script>
const SENT=0xFFFF;
let H,W,PW,PH,SCALE, predC,predR,refC,refR,conf, layer='du', smode='resid';
const img=document.getElementById('img'), vlc=document.getElementById('vline'), hlc=document.getElementById('hline');
const read=document.getElementById('read');
const DISP=Math.min(1100, 0.62*screen.width);   // displayed image width

function c(arr,i){const v=arr[i];return v===SENT?NaN:v/SCALE;}
function rdbu(t){ // t in [-1,1] -> diverging blue-white-red
  t=Math.max(-1,Math.min(1,t));
  const w=[247,247,247], b=[33,102,172], r=[178,24,43];
  const a=t<0?b:r, k=Math.abs(t);
  return [w[0]+(a[0]-w[0])*k, w[1]+(a[1]-w[1])*k, w[2]+(a[2]-w[2])*k];
}
function turbo(t){ // crude turbo-ish 0..1
  t=Math.max(0,Math.min(1,t));
  return [34+220*Math.min(1,Math.max(0,1.8-Math.abs(4*t-3))),
          Math.round(255*Math.sin(Math.PI*t)*0.9+25),
          Math.round(255*Math.max(0,1-1.6*t))+20];
}
function render(){
  const id=new ImageData(W,H), px=id.data;
  for(let i=0;i<W*H;i++){
    let col;
    if(layer==='conf'){ col=turbo(conf[i]/255); }
    else{
      const pc=layer==='du'?c(predC,i):c(predR,i), rc=layer==='du'?c(refC,i):c(refR,i);
      if(isNaN(pc)||isNaN(rc)){ col=[18,18,18]; } else col=rdbu((pc-rc)/2);
    }
    px[i*4]=col[0]; px[i*4+1]=col[1]; px[i*4+2]=col[2]; px[i*4+3]=255;
  }
  const off=document.createElement('canvas'); off.width=W; off.height=H;
  off.getContext('2d').putImageData(id,0,0);
  img.width=W; img.height=H; img.style.width=DISP+'px'; img.style.height=(DISP*H/W)+'px';
  const g=img.getContext('2d'); g.clearRect(0,0,W,H); g.drawImage(off,0,0);
  if(lastXY) crosshair(lastXY[0],lastXY[1]);
}
let lastXY=null;
function crosshair(x,y){ render0(); const g=img.getContext('2d');
  g.strokeStyle='rgba(0,255,140,.8)'; g.lineWidth=Math.max(1,W/DISP);
  g.beginPath(); g.moveTo(x,0); g.lineTo(x,H); g.moveTo(0,y); g.lineTo(W,y); g.stroke();
}
let base=null;
function render0(){ if(base){img.getContext('2d').putImageData(base,0,0);} }
function plot(cv,vals,horizontal){
  const w=cv.width=horizontal?DISP:300, h=cv.height=horizontal?180:Math.round(DISP*H/W);
  const g=cv.getContext('2d'); g.fillStyle='#181818'; g.fillRect(0,0,w,h);
  const N=vals.length; let lo=Infinity,hi=-Infinity;
  for(const s of vals) for(const v of s.data){ if(isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);} }
  if(!isFinite(lo)){g.fillStyle='#666';g.fillText('no valid pixels',8,20);return;}
  if(smode==='resid'){ const m=Math.max(0.5,Math.max(Math.abs(lo),Math.abs(hi))); lo=-m;hi=m; }
  const pad=hi-lo||1; const X=horizontal?w:h, mapV=v=>{const t=(v-lo)/pad; return horizontal?(h-8-t*(h-16)):(8+t*(h-16));};
  // zero line (resid)
  if(smode==='resid'){ g.strokeStyle='#444'; g.beginPath(); const z=mapV(0);
    if(horizontal){g.moveTo(0,z);g.lineTo(w,z);}else{g.moveTo(z,0);g.lineTo(z,h);} g.stroke(); }
  for(const s of vals){ g.strokeStyle=s.color; g.beginPath(); let started=false;
    for(let i=0;i<s.data.length;i++){ const v=s.data[i]; if(!isFinite(v)){started=false;continue;}
      const p=i/(s.data.length-1)*(horizontal?w:h), q=mapV(v);
      const cx=horizontal?p:q, cy=horizontal?q:p;
      if(!started){g.moveTo(cx,cy);started=true;}else g.lineTo(cx,cy); }
    g.stroke(); }
  g.fillStyle='#999'; g.font='11px monospace';
  g.fillText((horizontal?'du across row':'dv down col')+'  ['+lo.toFixed(2)+','+hi.toFixed(2)+']px',8,14);
}
function update(x,y){
  lastXY=[x,y]; crosshair(x,y);
  const i=y*W+x, pc=c(predC,i),pr=c(predR,i),rc=c(refC,i),rr=c(refR,i),cf=conf[i]/255;
  const du=pc-rc, dv=pr-rr;
  read.textContent=`x${x} y${y}  pred(${f(pc)},${f(pr)})  ref(${f(rc)},${f(rr)})  du ${f(du)} dv ${f(dv)}  conf ${cf.toFixed(2)}`;
  // horizontal scanline (row y): du across columns
  const duRow=new Float32Array(W), pcRow=new Float32Array(W), rcRow=new Float32Array(W);
  for(let xx=0;xx<W;xx++){const j=y*W+xx;const a=c(predC,j),b=c(refC,j);duRow[xx]=a-b;pcRow[xx]=a;rcRow[xx]=b;}
  plot(hlc, smode==='resid'?[{data:duRow,color:'#5cf'}]:[{data:pcRow,color:'#5cf'},{data:rcRow,color:'#fb5'}], true);
  // vertical scanline (col x): dv down rows
  const dvCol=new Float32Array(H), prCol=new Float32Array(H), rrCol=new Float32Array(H);
  for(let yy=0;yy<H;yy++){const j=yy*W+x;const a=c(predR,j),b=c(refR,j);dvCol[yy]=a-b;prCol[yy]=a;rrCol[yy]=b;}
  plot(vlc, smode==='resid'?[{data:dvCol,color:'#5cf'}]:[{data:prCol,color:'#5cf'},{data:rrCol,color:'#fb5'}], false);
}
function f(v){return isNaN(v)?' --':v.toFixed(2);}
img.addEventListener('mousemove',e=>{const r=img.getBoundingClientRect();
  const x=Math.floor((e.clientX-r.left)/r.width*W), y=Math.floor((e.clientY-r.top)/r.height*H);
  if(x>=0&&x<W&&y>=0&&y<H) update(x,y);});
function setLayer(l){layer=l;['Du','Dv','Conf'].forEach(k=>document.getElementById('b'+k).classList.toggle('on',k.toLowerCase()==='conf'?l==='conf':l==='d'+k[1].toLowerCase()));
  renderFull();}
function renderFull(){render(); base=img.getContext('2d').getImageData(0,0,W,H); if(lastXY)update(lastXY[0],lastXY[1]);}
document.getElementById('bDu').onclick=()=>setLayer('du');
document.getElementById('bDv').onclick=()=>setLayer('dv');
document.getElementById('bConf').onclick=()=>setLayer('conf');
document.getElementById('bResid').onclick=()=>{smode='resid';document.getElementById('bResid').classList.add('on');document.getElementById('bAbs').classList.remove('on');if(lastXY)update(...lastXY);};
document.getElementById('bAbs').onclick=()=>{smode='abs';document.getElementById('bAbs').classList.add('on');document.getElementById('bResid').classList.remove('on');if(lastXY)update(...lastXY);};
fetch('viewer_data.bin').then(r=>r.arrayBuffer()).then(buf=>{
  const hd=new Uint32Array(buf,0,6); H=hd[1];W=hd[2];PW=hd[3];PH=hd[4];SCALE=hd[5];
  let o=24; const n=W*H;
  predC=new Uint16Array(buf,o,n);o+=n*2; predR=new Uint16Array(buf,o,n);o+=n*2;
  refC=new Uint16Array(buf,o,n);o+=n*2; refR=new Uint16Array(buf,o,n);o+=n*2;
  conf=new Uint8Array(buf,o,n);
  renderFull();
  read.textContent=`loaded ${W}x${H}, proj ${PW}x${PH} — hover the image`;
}).catch(e=>{read.textContent='failed to load viewer_data.bin: '+e+'  (serve via python -m http.server)';});
</script></body></html>"""


if __name__ == "__main__":
    main()
