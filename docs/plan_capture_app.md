# Plan — Capture App & Real-World Eval

## Status
- **`scripts/capture_app.py`** rewritten as a **PySide6 GUI** (dropdowns + live
  feed + Capture button; pixel-exact frameless projector window). Syntax-clean.
  **Hardware-untested** (no rig here). `cv2` installed; still need
  `pip install PySide6`.
- **`scripts/eval_capture.py` — BUILT.** Decodes a `graycode/` capture to a
  reference projector-column map (`GrayCodeMethod.decode_columns`, no rig/calib
  needed) and scores the net's `marray/` prediction on the column (validity IoU,
  u-bin acc, median/p95 |du|, confidence sweep + overview plot). First real result
  on `captures/test0` (non-ideal): `proj_net.pt` transfers well — at conf≥0.5,
  0.68px median |du|, 94.5% bin acc, 62% coverage; conf≥0.9 → 0.60px / 99.3% / 51%.
  `proj_net_mixed.pt` is worse on real (2.6px / 85% @0.5) despite being newer.

## Design (capture_app.py)
One PySide6 operator window you run with no flags (`python scripts/capture_app.py`);
everything is in the GUI — no CLI workflow. macOS allows only one GUI event loop on
the main thread, so the operator UI *and* the projector window are both Qt (this is
why we moved off GLFW). Layout:
- **Devices**: camera dropdown + projector-monitor dropdown (auto-enumerated;
  defaults to the first non-primary monitor) + Refresh.
- **Camera**: resolution preset (ZED 1080p/720p/2.2K + native + plain HD) and a
  stereo-eye crop (Left/Right/Full). The capture camera is a **Stereolabs ZED**,
  used as a plain UVC device (its SDK is CUDA-only / no Mac) — macOS grabs it
  directly via AVFoundation. It returns a side-by-side L|R frame, so the default
  is ZED 1080p + **Left eye** → a clean 1920×1080; the crop is width-based (post-
  read, applied to both preview and saved captures) so it adapts to any mode.
  (RealSense/Structure-Core need SDKs painful on arm64-mac; Azure-Kinect &
  PrimeSense have no usable Mac path — ZED is the one that just works.)
- **Live feed**: the selected camera at ~30 fps (QTimer single-read preview).
- **Projector test**: Black / Flat gray / White buttons — project a solid field on
  the chosen monitor to verify the camera lock is steady (the drift check).
- **Capture**: scene-name field, checkable pattern-set list (marray/graycode/
  phaseshift pre-ticked), Settle + Flush + Exposure knobs, **Capture** button +
  progress bar. Capture projects each frame (`show_image`→synchronous `repaint`),
  spins `settle` while pumping events, flush-grabs, writes
  `captures/<scene>/<set>/cap_<stem>.png` (same layout the renderer produces), and
  mirrors each grab into the feed for live feedback.

Projector pixel-exactness (the "HiDPI silent killer"): frameless window sized to the
monitor's native geometry; pattern blitted top-left with smoothing OFF (nearest) and
tagged with the screen's `devicePixelRatio` so 1 pattern px → 1 physical px even on
HiDPI; logs a warning if a pattern's pixel size ≠ the projector's native size, and a
note if dpr ≠ 1. Camera auto-exposure/focus/WB disabled (best-effort, codes are
backend-dependent on macOS — verify with the Flat gray button + feed).

(`--ckpt` is reserved for a future in-GUI live-decode panel; the old GLFW `list`/
`capture`/`live` CLI modes are gone — the dropdowns and buttons replace them.)

## Next steps (in order)
1. `pip install PySide6`.
2. `python scripts/capture_app.py` → pick camera + projector from the dropdowns.
3. Confirm pixel-exactness on the rig: project Flat gray, watch the log for the
   `[warn] pattern … != projector native …` / HiDPI dpr note; pattern should be 1:1.
4. Lock the camera (Flat gray → verify the feed is constant across a few seconds;
   tune Exposure if it drifts).
5. **Capture protocol** (hand to the company): *static* scene; project `marray`
   (single-shot input) + `graycode` + `phaseshift` (multi-shot reference GT); repeat
   over flat sections at varied poses. Camera fixed, scene still between shots.
6. ~~**Build `scripts/eval_capture.py`**~~ **DONE** (Gray-code column reference + net
   `marray` prediction → IoU / bin-acc / median|du| / p95 / confidence sweep +
   overview plot). `python scripts/eval_capture.py --captures captures/<scene> --ckpt
   checkpoints/proj_net.pt`. Currently scores the **column** only (Gray code encodes
   columns); phaseshift not yet wired as a sub-pixel reference. Possible next: add a
   phaseshift-on-graycode reference for sub-pixel GT, and score the row head too.
7. **Real-data fine-tune** (sim-to-real): treat the Gray-code reference as the target,
   build a real loaf, fine-tune the net a few epochs. Expect a sim-to-real drop on
   first contact (renderer is Lambertian-only; real sensor/projector differ) — this
   step is both the fix and the selling point ("pretrain synthetic, adapt to your rig").

## Notes
- Net is **pattern-specific** — real eval must use *our* marray; off-the-shelf SL
  datasets won't work.
- Calibration from the captured correspondence: see `docs/calibration_design.md`
  (Step 2). Build `calibrate_from_correspondence.py` after eval_capture.
