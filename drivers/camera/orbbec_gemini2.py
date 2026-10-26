"""Orbbec Gemini 2 camera driver.

OrbbecGemini2:
  open() / close(): manage the pyorbbecsdk pipeline.
  get_frame(): return aligned BGR and depth-mm frames.
  K / D: camera matrix and distortion coefficients.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver, CameraFrameError


class OrbbecGemini2(CameraDriver):
    """Orbbec Gemini 2 RGB-D camera driver."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
    ) -> None:
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None

        self._pipeline = None
        self._depth_scale_mm: float = 1.0
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._reset_frame_failures()

    # Lifecycle

    def open(self) -> None:
        """Open the camera pipeline."""
        # Import first so native load errors stay visible.
        try:
            from pyorbbecsdk import (
                Pipeline, Config,
                OBSensorType, OBFormat, OBAlignMode,
                Context,
            )
        except ImportError as e:
            raise RuntimeError(f"pyorbbecsdk is not installed: {e}") from e

        # Silence noisy native logs during SDK initialization.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            try:
                from pyorbbecsdk import OBLogSeverity
                Context().set_logger_severity(OBLogSeverity.FATAL)
            except Exception:
                pass

            try:
                self._pipeline = Pipeline()
            except Exception as e:
                raise RuntimeError(
                    f"Orbbec camera not found: {e}\n"
                    "  Check USB connection and udev permissions.\n"
                    "  Permission quick fix: sudo chmod a+rw /dev/bus/usb/*/*"
                ) from e

            cfg = Config()

            # Color stream
            plist = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            cp = None
            for fmt in (OBFormat.MJPG, OBFormat.RGB):
                try:
                    cp = plist.get_video_stream_profile(self._w, self._h, fmt, self._fps)
                    break
                except Exception:
                    pass
            if cp is None:
                cp = plist.get_default_video_stream_profile()
            cfg.enable_stream(cp)

            # Depth stream
            dplist = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                dp = dplist.get_video_stream_profile(self._w, self._h, OBFormat.Y16, self._fps)
            except Exception:
                dp = dplist.get_default_video_stream_profile()
            cfg.enable_stream(dp)

            cfg.set_align_mode(OBAlignMode.HW_MODE)
            self._pipeline.start(cfg)
            self._reset_frame_failures()

            # Intrinsics from SDK
            intr = self._pipeline.get_camera_param().rgb_intrinsic
            self._K = np.array([
                [intr.fx, 0,       intr.cx],
                [0,       intr.fy, intr.cy],
                [0,       0,       1      ],
            ], dtype=np.float64)

            # Distortion
            self._D = self._load_distortion()

        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # Frames

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._pipeline is None:
            return None, None
        try:
            from pyorbbecsdk import OBFormat
            frames = self._pipeline.wait_for_frames(500)
            if frames is None:
                self._record_frame_failure("wait_for_frames timeout")
                return None, None

            color_bgr = None
            cf = frames.get_color_frame()
            if cf is not None:
                w, h = cf.get_width(), cf.get_height()
                raw = np.asanyarray(cf.get_data(), dtype=np.uint8)
                fmt = cf.get_format()
                try:
                    if fmt == OBFormat.MJPG:
                        color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    elif fmt == OBFormat.RGB:
                        color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                    else:
                        color_bgr = raw.reshape(h, w, 3)
                except Exception:
                    pass

            depth_mm = None
            df = frames.get_depth_frame()
            if df is not None:
                dw, dh = df.get_width(), df.get_height()
                depth_raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
                depth_scale = self._depth_scale_mm
                try:
                    depth_scale = float(df.get_depth_scale())
                    self._depth_scale_mm = depth_scale
                except Exception:
                    pass
                depth_mm = np.clip(
                    np.rint(depth_raw.astype(np.float32) * depth_scale),
                    0,
                    np.iinfo(np.uint16).max,
                ).astype(np.uint16)

            if color_bgr is None or depth_mm is None:
                self._record_frame_failure("missing color or depth frame")
            else:
                self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except Exception as exc:
            self._record_frame_failure(str(exc))
            return None, None

    # Intrinsics

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("Camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            raise RuntimeError("Camera is not open")
        return self._D

    # Internals

    def _load_distortion(self) -> np.ndarray:
        """Load distortion; fall back to zeros for invalid calibration."""
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    if abs(D[0]) > 5.0:
                        print(f"[OrbbecGemini2] Invalid k1={D[0]:.2f}; using zero distortion")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        return np.zeros((1, 5), dtype=np.float64)
