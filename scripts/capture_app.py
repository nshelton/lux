#!/usr/bin/env python3
"""Projector-camera capture GUI (PySide6, pixel-exact fullscreen).

A single operator window you run with no flags: pick the camera and projector
from dropdowns, watch the live camera feed, and hit **Capture** to project a
pattern set (or several) frame-by-frame and grab each one. Output lands in the
same ``captures/<scene>/<set>/cap_<stem>.png`` layout the synthetic renderer
produces, so ``eval_capture.py`` decodes real and synthetic captures identically.

Why Qt for both windows: the projector needs a true, frameless, native-resolution
window on the projector monitor (so the M-array code is presented 1:1 and never
resampled by the compositor), and the operator needs normal widgets. macOS only
allows one GUI event loop on the main thread, so both live in one PySide6 app.

Pixel-exactness safeguards (the "HiDPI silent killer"):
  * projector window is frameless and sized to the monitor's native geometry;
  * patterns are drawn with smoothing OFF (nearest) at the top-left, 1:1;
  * the image carries the screen's devicePixelRatio so 1 pattern px -> 1 physical
    px on HiDPI displays too;
  * the log warns if a pattern's pixel size != the projector's native size.

Run it:
    pip install PySide6           # opencv-python already required by lux
    python scripts/capture_app.py

Then in the GUI:
  1. pick the camera + projector monitor from the dropdowns;
  2. hit "Flat gray" and confirm the live feed is steady (camera lock check);
  3. type a scene name, tick the pattern sets (marray + graycode + phaseshift),
     hit Capture. Keep the scene + camera dead still across the whole sequence.
  4. test the model live: project the marray pattern, then under **Process** hit
     "Net ▶ current frame" — the predicted projector-column map appears in the
     result panel beside the feed. After a Capture (or for an existing scene), a
     Process button appears per captured set: marray → net, graycode/phaseshift →
     reference column. Toggle View (column / confidence / packed) and min-conf.

Hardware-dependent bits (true fullscreen pixel-exactness, camera property
locking) can only be shaken out on the real rig; every knob is in the GUI.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402

try:
    from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QPointF
    from PySide6.QtGui import (QImage, QPixmap, QPainter, QColor, QGuiApplication)
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton,
        QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
        QDoubleSpinBox, QSpinBox, QListWidget, QListWidgetItem, QPlainTextEdit,
        QProgressBar, QGroupBox,
    )
except ImportError:
    raise SystemExit("missing dependency 'PySide6'  ->  pip install PySide6")


# --------------------------------------------------------------------------
# Pattern set discovery
# --------------------------------------------------------------------------
_IMG_EXT = (".png", ".jpg", ".jpeg")

# Camera capture-resolution presets (label -> (w, h) or None for device native).
# ZED modes are side-by-side L|R; the listed eye size is after a left/right crop.
RES_OPTIONS = [
    ("ZED 1080p  (3840x1080 → eye 1920x1080)", (3840, 1080)),
    ("ZED 720p  (2560x720 → eye 1280x720)", (2560, 720)),
    ("ZED 2.2K  (4416x1242 → eye 2208x1242)", (4416, 1242)),
    ("Default (device native)", None),
    ("1920x1080", (1920, 1080)),
    ("1280x720", (1280, 720)),
]
EYE_OPTIONS = [("Left eye", "left"), ("Right eye", "right"), ("Full frame", "full")]


def list_pattern_sets(root: str) -> list[str]:
    """Names of subfolders of ``root`` that contain at least one image."""
    rp = Path(root)
    if not rp.is_dir():
        return []
    out = []
    for d in sorted(rp.iterdir()):
        if d.is_dir() and any(p.suffix.lower() in _IMG_EXT for p in d.iterdir()):
            out.append(d.name)
    return out


def set_frames(root: str, name: str) -> list[Path]:
    """Image files in pattern set ``root/name``, in filename order."""
    d = Path(root) / name
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT)


# --------------------------------------------------------------------------
# Camera: locked exposure/gain/focus, flushed grab. Same cap serves preview
# (single read) and capture (flush stale frames, then grab).
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, cv2):
        self.cv2 = cv2
        self.cap = None
        self.index = None
        self.crop = "full"          # "full" | "left" | "right" (stereo eye select)

    def open(self, index: int, width=None, height=None, exposure=None):
        self.close()
        cv2 = self.cv2
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"camera {index} did not open")
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)              # freshest preview frame
        except Exception:
            pass
        # Lock anything that could drift between frames of a pattern sequence.
        # These property codes are backend-dependent and best-effort; verify the
        # capture is actually constant (project flat gray, watch the feed).
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)            # 0.25 == manual on many UVC backends
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass
        if exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap = cap
        self.index = index

    def set_exposure(self, exposure: float):
        if self.cap is not None:
            self.cap.set(self.cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(self.cv2.CAP_PROP_EXPOSURE, exposure)

    def set_crop(self, crop: str):
        self.crop = crop if crop in ("full", "left", "right") else "full"

    # A ZED hands back a side-by-side L|R frame (16:9 per eye → ~3.56:1 overall);
    # a normal single camera is ~16:9 (1.78) or 4:3 (1.33). Only split when the
    # frame is actually that wide, so selecting an eye never *double-crops* a
    # single-image source (which would just halve its FOV).
    _STEREO_ASPECT = 2.4

    def apply_crop(self, frame: np.ndarray) -> np.ndarray:
        if self.crop == "full":
            return frame
        h, w = frame.shape[:2]
        if w / max(h, 1) < self._STEREO_ASPECT:
            return frame                 # not side-by-side: nothing to split
        half = w // 2
        return frame[:, :half] if self.crop == "left" else frame[:, half:]

    @property
    def is_open(self) -> bool:
        return self.cap is not None

    def read_rgb_u8(self, raw: bool = False):
        """Latest frame as RGB uint8 (H,W,3), or None. For the live preview.
        ``raw`` skips the eye-crop (used to report the device's delivered size)."""
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return rgb if raw else self.apply_crop(rgb)

    def grab_rgb_float(self, flush: int = 5) -> np.ndarray:
        """Drop ``flush`` stale buffered frames, return next as RGB float [0,1]."""
        if self.cap is None:
            raise RuntimeError("camera not open")
        for _ in range(max(0, flush)):
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera grab failed")
        rgb = self.apply_crop(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
        return rgb.astype(np.float32) / 255.0

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.index = None


def probe_cameras(cv2, n: int = 5) -> list[tuple[int, int, int]]:
    """Open indices 0..n-1; return (index, w, h) for those that respond."""
    found = []
    for i in range(n):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
        cap.release()
    return found


# --------------------------------------------------------------------------
# Projector window: frameless, native-resolution, pixel-exact presenter
# --------------------------------------------------------------------------
class ProjectorWindow(QWidget):
    """Frameless top-level window placed on a chosen monitor at native geometry.

    Patterns are blitted top-left with smoothing OFF and the image tagged with
    the screen's devicePixelRatio, so one pattern pixel == one physical pixel."""

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("lux-projector")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setCursor(Qt.BlankCursor)
        self._img = None            # QImage or None
        self._solid = QColor(0, 0, 0)

    def place_on(self, screen) -> float:
        """Move/resize to cover ``screen`` exactly; return its devicePixelRatio."""
        geo = screen.geometry()
        self.setGeometry(geo)
        self.show()
        h = self.windowHandle()
        if h is not None:
            h.setScreen(screen)
        self.setGeometry(geo)        # re-assert after screen bind
        self.raise_()
        return float(screen.devicePixelRatio())

    def show_image(self, arr: np.ndarray, dpr: float = 1.0):
        img = arr
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img[..., :3])
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        qimg.setDevicePixelRatio(dpr)
        self._img = qimg
        self.repaint()               # synchronous: pattern is on screen on return

    def show_solid(self, value: float):
        v = int(np.clip(value, 0, 1) * 255 + 0.5)
        self._img = None
        self._solid = QColor(v, v, v)
        self.repaint()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), self._solid)
        if self._img is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.drawImage(QPointF(0, 0), self._img)
        p.end()


def np_rgb_to_qimage(arr_u8: np.ndarray) -> QImage:
    arr_u8 = np.ascontiguousarray(arr_u8[..., :3])
    h, w = arr_u8.shape[:2]
    return QImage(arr_u8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# --------------------------------------------------------------------------
# Operator window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cv2 = self._require_cv2()
        self.cam = Camera(self.cv2)
        self.projector = ProjectorWindow()
        self.screens: list = []
        self._busy = False           # True during a capture sequence
        self._processing = False     # True while running the net / a decoder
        self._last_qimage = None     # keep preview image alive for the label
        self._result_qimage = None   # keep result-panel image alive for the label
        self.model = None            # ProjUNet, lazy-loaded on first Process
        self.proj_wh = None          # (w, h) from the checkpoint
        self._device = "cpu"
        self._last_result = None     # ("net", pred, conf, label) | ("ref", col, label)

        self.setWindowTitle("lux — capture")
        self._build_ui()
        self._refresh_devices()

        self.preview = QTimer(self)
        self.preview.setInterval(33)
        self.preview.timeout.connect(self._preview_tick)
        self.preview.start()

        self.log("Ready. Pick a camera + projector, project flat gray to verify "
                 "the feed is steady, then Capture.")

    @staticmethod
    def _require_cv2():
        try:
            import cv2
            return cv2
        except ImportError:
            raise SystemExit("missing dependency 'cv2'  ->  pip install opencv-python")

    # -- UI construction ---------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---- left: controls ----
        left = QVBoxLayout()
        root.addLayout(left, 0)

        dev = QGroupBox("Devices")
        dev_l = QFormLayout(dev)
        self.cam_box = QComboBox()
        self.cam_box.currentIndexChanged.connect(self._reopen_camera)
        self.screen_box = QComboBox()
        self.refresh_btn = QPushButton("Refresh devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        dev_l.addRow("Camera", self.cam_box)
        dev_l.addRow("Projector", self.screen_box)
        dev_l.addRow(self.refresh_btn)
        left.addWidget(dev)

        cam = QGroupBox("Camera")
        cam_l = QFormLayout(cam)
        self.res_box = QComboBox()
        for label, wh in RES_OPTIONS:
            self.res_box.addItem(label, wh)
        self.eye_box = QComboBox()
        for label, mode in EYE_OPTIONS:
            self.eye_box.addItem(label, mode)
        self.res_box.currentIndexChanged.connect(self._reopen_camera)
        self.eye_box.currentIndexChanged.connect(self._on_eye_changed)
        cam_l.addRow("Resolution", self.res_box)
        cam_l.addRow("Stereo eye", self.eye_box)
        left.addWidget(cam)

        proj = QGroupBox("Projector test (lock check / aim)")
        proj_l = QHBoxLayout(proj)
        for label, val in (("Black", 0.0), ("Flat gray", 0.5), ("White", 1.0)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, v=val: self._project_solid(v))
            proj_l.addWidget(b)
        self.project_pat_btn = QPushButton("Project pattern")
        self.project_pat_btn.setToolTip("Project the highlighted pattern set's first "
                                        "frame (or double-click a set in the list)")
        self.project_pat_btn.clicked.connect(lambda: self._project_pattern())
        proj_l.addWidget(self.project_pat_btn)
        left.addWidget(proj)

        cap = QGroupBox("Capture")
        cap_l = QFormLayout(cap)
        self.scene_edit = QLineEdit("scene01")
        self.pattern_list = QListWidget()
        self.pattern_list.setMaximumHeight(140)
        self.pattern_list.itemDoubleClicked.connect(
            lambda it: self._project_pattern(it.text()))
        self.settle_spin = QDoubleSpinBox()
        self.settle_spin.setRange(0.0, 5.0)
        self.settle_spin.setSingleStep(0.05)
        self.settle_spin.setValue(0.25)
        self.settle_spin.setSuffix(" s")
        self.flush_spin = QSpinBox()
        self.flush_spin.setRange(0, 30)
        self.flush_spin.setValue(5)
        exp_row = QHBoxLayout()
        self.exposure_edit = QLineEdit()
        self.exposure_edit.setPlaceholderText("blank = auto-lock")
        self.exposure_apply = QPushButton("Apply")
        self.exposure_apply.clicked.connect(self._apply_exposure)
        exp_row.addWidget(self.exposure_edit)
        exp_row.addWidget(self.exposure_apply)
        exp_w = QWidget()
        exp_w.setLayout(exp_row)
        cap_l.addRow("Scene name", self.scene_edit)
        cap_l.addRow("Pattern sets", self.pattern_list)
        cap_l.addRow("Settle", self.settle_spin)
        cap_l.addRow("Flush frames", self.flush_spin)
        cap_l.addRow("Exposure", exp_w)
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setMinimumHeight(40)
        self.capture_btn.clicked.connect(self._run_capture)
        cap_l.addRow(self.capture_btn)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        cap_l.addRow(self.progress)
        left.addWidget(cap)

        proc = QGroupBox("Process — test model in the wild")
        proc_l = QVBoxLayout(proc)
        self.net_live_btn = QPushButton("Net  ▶  current frame")
        self.net_live_btn.setToolTip("Grab the current camera frame and run the net. "
                                     "Project the marray pattern first.")
        self.net_live_btn.clicked.connect(self._process_net_live)
        proc_l.addWidget(self.net_live_btn)
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View"))
        self.view_box = QComboBox()
        self.view_box.addItems(["Column (turbo)", "Confidence", "Packed RGB"])
        self.view_box.currentIndexChanged.connect(self._rerender_result)
        view_row.addWidget(self.view_box, 1)
        self.minconf_spin = QDoubleSpinBox()
        self.minconf_spin.setRange(0.0, 1.0)
        self.minconf_spin.setSingleStep(0.05)
        self.minconf_spin.setValue(0.0)
        self.minconf_spin.setPrefix("min-conf ")
        self.minconf_spin.valueChanged.connect(self._rerender_result)
        view_row.addWidget(self.minconf_spin)
        proc_l.addLayout(view_row)
        self.captured_box = QGroupBox("captured sets (current scene)")
        self.captured_layout = QVBoxLayout(self.captured_box)
        proc_l.addWidget(self.captured_box)
        left.addWidget(proc)

        left.addStretch(1)

        # ---- right: live feed + result panel + log ----
        right = QVBoxLayout()
        root.addLayout(right, 1)
        panels = QHBoxLayout()
        feed_col = QVBoxLayout()
        feed_col.addWidget(QLabel("live feed"))
        self.feed = QLabel("no camera")
        self.feed.setAlignment(Qt.AlignCenter)
        self.feed.setMinimumSize(480, 360)
        self.feed.setStyleSheet("background:#111; color:#888;")
        feed_col.addWidget(self.feed, 1)
        panels.addLayout(feed_col, 1)
        res_col = QVBoxLayout()
        self.result_caption = QLabel("result")
        self.result_caption.setStyleSheet("color:#aaa;")
        res_col.addWidget(self.result_caption)
        self.result = QLabel("no result — Process a capture or the current frame")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setMinimumSize(480, 360)
        self.result.setStyleSheet("background:#111; color:#888;")
        res_col.addWidget(self.result, 1)
        panels.addLayout(res_col, 1)
        right.addLayout(panels, 1)
        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumHeight(150)
        right.addWidget(self.status)

        # process buttons track the current scene's captures on disk
        self.scene_edit.textChanged.connect(self._refresh_process_buttons)
        self._refresh_process_buttons()

    # -- logging -----------------------------------------------------------
    def log(self, msg: str):
        self.status.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # -- device discovery --------------------------------------------------
    def _refresh_devices(self):
        # screens
        self.screen_box.blockSignals(True)
        self.screen_box.clear()
        self.screens = list(QGuiApplication.screens())
        primary = QGuiApplication.primaryScreen()
        default_idx = 0
        for i, s in enumerate(self.screens):
            g = s.geometry()
            phys_w = int(round(g.width() * s.devicePixelRatio()))
            phys_h = int(round(g.height() * s.devicePixelRatio()))
            tag = " (primary)" if s is primary else ""
            self.screen_box.addItem(
                f"[{i}] {s.name()}  {phys_w}x{phys_h}  dpr {s.devicePixelRatio():g}{tag}", i)
            if s is not primary and default_idx == 0:
                default_idx = i        # prefer a non-primary monitor = the projector
        if self.screens:
            self.screen_box.setCurrentIndex(default_idx)
        self.screen_box.blockSignals(False)

        # cameras
        self.cam_box.blockSignals(True)
        self.cam_box.clear()
        cams = probe_cameras(self.cv2, n=5)
        for idx, w, h in cams:
            self.cam_box.addItem(f"cam {idx}  ({w}x{h})", idx)
        self.cam_box.blockSignals(False)

        self.log(f"found {len(self.screens)} monitor(s), {len(cams)} camera(s)")
        if cams:
            self.cam_box.setCurrentIndex(0)
            self._reopen_camera()

    def _reopen_camera(self):
        """(Re)open the selected camera at the chosen resolution + eye crop.
        Called when the camera, resolution, or device list changes."""
        idx = self.cam_box.currentData()
        if idx is None:
            return
        wh = self.res_box.currentData()
        w, h = wh if wh else (None, None)
        try:
            self.cam.open(int(idx), width=w, height=h)
            self.cam.set_crop(self.eye_box.currentData() or "full")
            self._apply_exposure(announce=False)
            raw = self.cam.read_rgb_u8(raw=True)     # report raw -> cropped size
            if raw is not None:
                rh, rw = raw.shape[:2]
                out = self.cam.apply_crop(raw)
                oh, ow = out.shape[:2]
                if (ow, oh) != (rw, rh):
                    self.log(f"camera {idx}: {rw}x{rh} -> {ow}x{oh} "
                             f"({self.eye_box.currentText()})")
                else:
                    self.log(f"camera {idx}: {rw}x{rh}"
                             + ("  [not side-by-side: eye-crop skipped]"
                                if self.cam.crop != "full" else ""))
        except RuntimeError as e:
            self.log(f"[error] {e}")

    def _on_eye_changed(self):
        # crop is applied post-read, so no reopen needed
        self.cam.set_crop(self.eye_box.currentData() or "full")

    def _apply_exposure(self, announce: bool = True):
        txt = self.exposure_edit.text().strip()
        if not txt or not self.cam.is_open:
            return
        try:
            val = float(txt)
        except ValueError:
            self.log(f"[error] exposure '{txt}' is not a number")
            return
        self.cam.set_exposure(val)
        if announce:
            self.log(f"set exposure -> {val}")

    # -- projector placement ----------------------------------------------
    def _current_screen(self):
        i = self.screen_box.currentData()
        if i is None or i >= len(self.screens):
            return None
        return self.screens[i]

    def _ensure_projector(self):
        """Place the projector window on the selected screen; return (dpr, w, h)
        in physical pixels, or None if no screen is selected."""
        screen = self._current_screen()
        if screen is None:
            self.log("[error] no projector monitor selected")
            return None
        dpr = self.projector.place_on(screen)
        g = screen.geometry()
        phys_w = int(round(g.width() * dpr))
        phys_h = int(round(g.height() * dpr))
        return dpr, phys_w, phys_h

    def _project_solid(self, value: float):
        info = self._ensure_projector()
        if info is None:
            return
        self.projector.show_solid(value)
        name = {0.0: "black", 0.5: "flat gray", 1.0: "white"}.get(value, f"{value:g}")
        self.log(f"projecting {name} on {self.screen_box.currentText()}")

    def _project_pattern(self, name: str | None = None):
        """Project the first frame of a pattern set as a test/aim image. ``name``
        defaults to the highlighted list item, else the first set in the list."""
        if name is None:
            it = self.pattern_list.currentItem() or (
                self.pattern_list.item(0) if self.pattern_list.count() else None)
            if it is None:
                self.log("[error] no pattern sets to project")
                return
            name = it.text()
        frames = set_frames(self.args.patterns_root, name)
        if not frames:
            self.log(f"[error] no frames in pattern set '{name}'")
            return
        info = self._ensure_projector()
        if info is None:
            return
        dpr, phys_w, phys_h = info
        pat = io.load_image(str(frames[0]), gray=False)
        ph, pw = pat.shape[:2]
        self.projector.show_image(pat, dpr)
        note = "" if (pw, ph) == (phys_w, phys_h) else \
            f"  [warn: {pw}x{ph} != projector {phys_w}x{phys_h}, not pixel-exact]"
        self.log(f"projecting pattern '{name}' / {frames[0].name}{note}")

    # -- live preview ------------------------------------------------------
    def _preview_tick(self):
        if self._busy or self._processing or not self.cam.is_open:
            return
        frame = self.cam.read_rgb_u8()
        if frame is None:
            return
        self._show_feed(frame)

    def _show_feed(self, rgb_u8: np.ndarray):
        self._last_qimage = np_rgb_to_qimage(rgb_u8)
        pix = QPixmap.fromImage(self._last_qimage).scaled(
            self.feed.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.feed.setPixmap(pix)

    # -- model + processing (run net / decoders into the result panel) -----
    def _ensure_model(self) -> bool:
        """Lazy-load the checkpoint on first use (keeps startup torch-free)."""
        if self.model is not None:
            return True
        try:
            import torch
            from lux.proj_net import load_checkpoint
        except ImportError as e:
            self.log(f"[error] inference needs torch: {e}")
            return False
        ckpt = self.args.ckpt
        if not Path(ckpt).exists():
            self.log(f"[error] checkpoint not found: {ckpt}  (pass --ckpt)")
            return False
        self._device = self.args.device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"loading net {ckpt} on {self._device} …")
        QApplication.processEvents()
        try:
            self.model, self.proj_wh = load_checkpoint(ckpt, device=self._device)
        except Exception as e:
            self.log(f"[error] failed to load checkpoint: {e}")
            return False
        self.log(f"net ready (projector {self.proj_wh[0]}x{self.proj_wh[1]}); "
                 f"inference shares the GPU with any running training")
        return True

    def _projector_width(self) -> int:
        """Column span for reference decoders: the net ckpt if loaded, else the
        authored graycode pattern width, else 1920."""
        if self.proj_wh:
            return int(self.proj_wh[0])
        try:
            frames = set_frames(self.args.patterns_root, "graycode")
            if frames:
                return io.load_image(str(frames[0]), gray=True).shape[1]
        except Exception:
            pass
        return 1920

    def _projector_height(self) -> int:
        """Row span for the horizontal Gray-code decoder (see _projector_width)."""
        if self.proj_wh:
            return int(self.proj_wh[1])
        for s in ("graycode_h", "graycode"):
            try:
                frames = set_frames(self.args.patterns_root, s)
                if frames:
                    return io.load_image(str(frames[0]), gray=True).shape[0]
            except Exception:
                pass
        return 1080

    def _to_gray(self, rgb_float: np.ndarray) -> np.ndarray:
        """RGB [0,1] -> gray [0,1] matching io.load_image(gray=True) (cv2 luma)."""
        u8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        g = self.cv2.cvtColor(u8, self.cv2.COLOR_RGB2GRAY)
        return g.astype(np.float64) / 255.0

    def _captured_set_frames(self, setname: str) -> list[Path]:
        d = Path(self.args.out) / self.scene_edit.text().strip() / setname
        return sorted(d.glob("cap_*.png")) if d.is_dir() else []

    def _refresh_process_buttons(self):
        """One process button per pattern set captured for the current scene."""
        lay = self.captured_layout
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        any_set = False
        for setname, label in (("marray", "marray ▶ net"),
                               ("graycode", "graycode ▶ column"),
                               ("graycode_h", "graycode_h ▶ row"),
                               ("phaseshift", "phaseshift ▶ column")):
            if self._captured_set_frames(setname):
                b = QPushButton(f"Process: {label}")
                b.setEnabled(not self._processing)
                b.clicked.connect(lambda _=False, s=setname: self._process_set(s))
                lay.addWidget(b)
                any_set = True
        if self._captured_set_frames("graycode") and self._captured_set_frames("graycode_h"):
            b = QPushButton("Process: graycode ▶ (u,v) exact")
            b.setToolTip("Decode vertical + horizontal Gray codes into an exact "
                         "per-pixel (column, row) correspondence.")
            b.setEnabled(not self._processing)
            b.clicked.connect(lambda _=False: self._process_graycode_uv())
            lay.addWidget(b)
            any_set = True
        if not any_set:
            scene = self.scene_edit.text().strip() or "(unnamed)"
            lay.addWidget(QLabel(f"no captures for '{scene}' yet"))

    def _process_net_live(self):
        if self._processing or not self.cam.is_open:
            if not self.cam.is_open:
                self.log("[error] no camera open")
            return
        if not self._ensure_model():
            return
        self._set_processing(True)
        try:
            rgb = self.cam.grab_rgb_float(self.flush_spin.value())
            self._run_net(self._to_gray(rgb), "current frame")
        except Exception as e:
            self.log(f"[error] process failed: {e}")
        finally:
            self._set_processing(False)

    def _process_set(self, setname: str):
        if self._processing:
            return
        frames = self._captured_set_frames(setname)
        if not frames:
            self.log(f"[error] no captured frames for '{setname}'")
            return
        if setname == "marray" and not self._ensure_model():
            return
        self._set_processing(True)
        try:
            if setname == "marray":
                gray = io.load_image(str(frames[0]), gray=True)
                self._run_net(gray, f"captured {setname}/{frames[0].name}")
            else:
                self._run_reference(setname, frames)
        except Exception as e:
            self.log(f"[error] process failed: {e}")
        finally:
            self._set_processing(False)

    def _run_net(self, gray: np.ndarray, label: str):
        from lux.proj_net import predict_full
        self.log(f"running net on {label} ({gray.shape[1]}x{gray.shape[0]}) …")
        QApplication.processEvents()
        pred, conf = predict_full(self.model, gray, self.proj_wh,
                                  device=self._device, return_conf=True)
        self._last_result = ("net", pred, conf, label)
        valid = np.isfinite(pred[..., 0])
        medc = float(np.median(conf[valid])) if valid.any() else 0.0
        self.log(f"  net: {int(valid.sum()):,} valid px ({100*valid.mean():.1f}%), "
                 f"median confidence {medc:.2f}")
        self._rerender_result()

    def _run_reference(self, setname: str, frames: list[Path]):
        from lux.methods.graycode import GrayCodeMethod
        from lux.methods.phaseshift import PhaseShiftMethod
        stack = np.stack([io.load_image(str(f), gray=True) for f in frames], axis=0)
        self.log(f"decoding {setname} ({len(frames)} frames) …")
        QApplication.processEvents()
        if setname == "graycode":
            vmax, axis = self._projector_width(), "column"
            val, _ = GrayCodeMethod().decode_columns(stack, vmax)
        elif setname == "graycode_h":
            vmax, axis = self._projector_height(), "row"
            val, _ = GrayCodeMethod().decode_rows(stack, vmax)
        elif setname == "phaseshift":
            vmax, axis = self._projector_width(), "column"
            val, _ = PhaseShiftMethod(shifts=4, high_periods=16).decode_columns(stack, vmax)
        else:
            self.log(f"[error] no decoder for '{setname}'")
            return
        self._last_result = ("ref", val, f"{setname} ({axis})", vmax)
        valid = np.isfinite(val)
        self.log(f"  {setname}: {int(valid.sum()):,} valid px ({100*valid.mean():.1f}%)")
        self._rerender_result()

    def _process_graycode_uv(self):
        """Decode vertical + horizontal Gray codes into an exact (column, row) map."""
        if self._processing:
            return
        vframes = self._captured_set_frames("graycode")
        hframes = self._captured_set_frames("graycode_h")
        if not (vframes and hframes):
            self.log("[error] need both graycode and graycode_h captures")
            return
        self._set_processing(True)
        try:
            from lux.methods.graycode import GrayCodeMethod
            gc = GrayCodeMethod()
            pw, ph = self._projector_width(), self._projector_height()
            self.log(f"decoding exact (u,v): graycode {len(vframes)}f + "
                     f"graycode_h {len(hframes)}f …")
            QApplication.processEvents()
            col, _ = gc.decode_columns(
                np.stack([io.load_image(str(f), gray=True) for f in vframes], axis=0), pw)
            row, _ = gc.decode_rows(
                np.stack([io.load_image(str(f), gray=True) for f in hframes], axis=0), ph)
            both = np.isfinite(col) & np.isfinite(row)
            uv = np.stack([col, row], axis=-1)
            uv[~both] = np.nan
            self._last_result = ("refuv", uv, "graycode exact (u,v)")
            self.log(f"  exact (u,v): {int(both.sum()):,} px with both column + row")
            self._rerender_result()
        except Exception as e:
            self.log(f"[error] process failed: {e}")
        finally:
            self._set_processing(False)

    def _colorize(self, value: np.ndarray, valid: np.ndarray,
                  vmin: float, vmax: float) -> np.ndarray:
        """Turbo-colormap a scalar field to RGB uint8; invalid pixels -> black."""
        norm = np.clip((np.nan_to_num(value) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        u8 = (norm * 255).astype(np.uint8)
        cmap = getattr(self.cv2, "COLORMAP_TURBO", self.cv2.COLORMAP_JET)
        rgb = np.ascontiguousarray(self.cv2.applyColorMap(u8, cmap)[..., ::-1])
        rgb[~valid] = 0
        return rgb

    def _rerender_result(self):
        if self._last_result is None:
            return
        kind = self._last_result[0]
        if kind == "net":
            _, pred, conf, label = self._last_result
            pw, ph = self.proj_wh
            view = self.view_box.currentText()
            if view.startswith("Confidence"):
                rgb = self._colorize(conf, np.isfinite(pred[..., 0]), 0.0, 1.0)
                cap = f"net confidence — {label}"
            elif view.startswith("Packed"):
                rgb = (io.uv_conf_to_rgb(pred, conf, pw, ph) * 255).astype(np.uint8)
                cap = f"packed (R=col G=row B=conf) — {label}"
            else:
                col = pred[..., 0]
                valid = np.isfinite(col) & (conf >= self.minconf_spin.value())
                rgb = self._colorize(col, valid, 0.0, pw)
                cap = f"predicted column (net) — {label}"
        elif kind == "ref":
            _, val, label, vmax = self._last_result
            rgb = self._colorize(val, np.isfinite(val), 0.0, vmax)
            cap = f"reference {label}"
        else:  # refuv — exact (column, row) from V + H Gray codes
            _, uv, label = self._last_result
            valid = np.isfinite(uv[..., 0]) & np.isfinite(uv[..., 1])
            rgb = (io.uv_conf_to_rgb(uv, valid.astype(np.float64),
                                     self._projector_width(),
                                     self._projector_height()) * 255).astype(np.uint8)
            cap = f"{label}  (R=col, G=row)"
        self._show_result(rgb, cap)

    def _show_result(self, rgb_u8: np.ndarray, caption: str):
        self.result_caption.setText(caption)
        self._result_qimage = np_rgb_to_qimage(rgb_u8)
        pix = QPixmap.fromImage(self._result_qimage).scaled(
            self.result.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.result.setPixmap(pix)

    def _set_processing(self, busy: bool):
        self._processing = busy
        self.net_live_btn.setEnabled(not busy)
        for i in range(self.captured_layout.count()):
            w = self.captured_layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                w.setEnabled(not busy)
        if busy:
            self.preview.stop()
        else:
            self.preview.start()
        QApplication.processEvents()

    # -- capture sequence --------------------------------------------------
    def _selected_sets(self) -> list[str]:
        out = []
        for i in range(self.pattern_list.count()):
            it = self.pattern_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.text())
        return out

    def _run_capture(self):
        if self._busy:
            return
        if not self.cam.is_open:
            self.log("[error] no camera open")
            return
        sets = self._selected_sets()
        if not sets:
            self.log("[error] tick at least one pattern set")
            return
        scene = self.scene_edit.text().strip()
        if not scene:
            self.log("[error] enter a scene name")
            return
        info = self._ensure_projector()
        if info is None:
            return
        dpr, phys_w, phys_h = info
        if dpr != 1.0:
            self.log(f"[note] projector is HiDPI (dpr={dpr:g}); compensating per-image. "
                     f"Verify 1:1 by decoding a graycode capture.")

        # gather frames
        jobs: list[tuple[str, Path]] = []
        for s in sets:
            frames = set_frames(self.args.patterns_root, s)
            if not frames:
                self.log(f"[skip] no frames in {s}")
                continue
            jobs += [(s, f) for f in frames]
        if not jobs:
            self.log("[error] selected sets had no frames")
            return

        self._apply_exposure(announce=False)
        settle = self.settle_spin.value()
        flush = self.flush_spin.value()
        out_root = Path(self.args.out)

        self._set_busy(True)
        self.progress.setMaximum(len(jobs))
        self.progress.setValue(0)
        self.log(f"capturing {len(jobs)} frame(s) across {len(sets)} set(s) "
                 f"-> {out_root}/{scene}/")
        warned_size = False
        done = 0
        try:
            for i, (sname, fpath) in enumerate(jobs):
                pat = io.load_image(str(fpath), gray=False)
                ph, pw = pat.shape[:2]
                if not warned_size and (pw, ph) != (phys_w, phys_h):
                    self.log(f"[warn] pattern {pw}x{ph} != projector native "
                             f"{phys_w}x{phys_h}: NOT pixel-exact (drawn top-left, "
                             f"cropped/letterboxed). Match pattern res to the projector.")
                    warned_size = True
                self.projector.show_image(pat, dpr)
                self._spin(settle)                     # let projector + sensor settle
                frame = self.cam.grab_rgb_float(flush)
                odir = io.ensure_dir(out_root / scene / sname)
                io.save_image(str(Path(odir) / f"cap_{fpath.stem}.png"), frame)
                self._show_feed((frame * 255).astype(np.uint8))   # live feedback
                done += 1
                self.progress.setValue(done)
                if i == 0 or jobs[i - 1][0] != sname:              # entering a new set
                    self.log(f"[{sname}] -> {odir}/")
        except Exception as e:
            self.log(f"[error] capture aborted: {e}")
        finally:
            self.projector.show_solid(0.0)            # go dark
            self._set_busy(False)
            self._refresh_process_buttons()           # enable Process for new sets
            self.log(f"done: wrote {done} capture(s) to {out_root}/{scene}/ "
                     f"(Process them here, or decode with eval_capture.py)")

    def _spin(self, seconds: float):
        """Wait ``seconds`` while keeping the projector painted + UI responsive."""
        t = QElapsedTimer()
        t.start()
        ms = int(seconds * 1000)
        while t.elapsed() < ms:
            QApplication.processEvents()
            time.sleep(0.002)

    def _set_busy(self, busy: bool):
        self._busy = busy
        for w in (self.capture_btn, self.cam_box, self.screen_box,
                  self.refresh_btn, self.pattern_list, self.scene_edit,
                  self.res_box, self.eye_box):
            w.setEnabled(not busy)
        if busy:
            self.preview.stop()
        else:
            self.preview.start()

    # -- pattern list population (called once devices known) ---------------
    def populate_patterns(self):
        sets = list_pattern_sets(self.args.patterns_root)
        default_on = {"marray", "graycode", "graycode_h", "phaseshift"}
        self.pattern_list.clear()
        for name in sets:
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in default_on else Qt.Unchecked)
            self.pattern_list.addItem(it)
        if not sets:
            self.log(f"[warn] no pattern sets under {self.args.patterns_root}/")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, e):
        self.preview.stop()
        self.cam.close()
        self.projector.close()
        super().closeEvent(e)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="captures", help="output root for captures/")
    ap.add_argument("--patterns-root", default="patterns",
                    help="folder holding pattern set subfolders")
    ap.add_argument("--ckpt", default="checkpoints/proj_net.pt",
                    help="checkpoint for the in-app Process panel (net inference)")
    ap.add_argument("--device", default=None,
                    help="torch device for inference (default: auto; pass 'cpu' to "
                         "leave a running training's GPU untouched)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    win = MainWindow(args)
    win.populate_patterns()
    win.resize(1480, 820)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
