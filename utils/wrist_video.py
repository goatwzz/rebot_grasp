"""Continuous wrist-camera RGB recording without blocking robot motions."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np


class RecordingCamera:
    def __init__(self, camera, output_dir: Path, fps: float) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("腕部相机录像帧率必须为正数")
        self._camera = camera
        self.path = Path(output_dir) / "recordings" / f"wrist_rgb_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture, name="wrist-video", daemon=True)
        self._writer = None
        self._recording_enabled = True
        self._latest = None
        self._sequence = 0
        self._consumed = 0
        self._capture_time = 0.0
        self._recorded_frames = 0
        self._first_recorded_at = None
        self._last_recorded_at = None
        self._missing_color_frames = 0
        self._missing_depth_frames = 0
        self._recording_error = None
        self._error = None
        self._closed = False
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (self._writer is not None and self._latest is not None) or self._error is not None,
                timeout=5.0,
            )
            error = self._error
        if not ready or error is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            raise RuntimeError(f"腕部相机录像启动失败：{error or '没有获取到有效 RGB-D 帧'}")
        print(f"[录像] 正在录制腕部相机原始 RGB：{self.path}")

    @property
    def K(self) -> np.ndarray:
        return self._camera.K

    def _capture(self) -> None:
        try:
            while not self._stop.is_set():
                color, depth = self._camera.get_frame()
                if color is None:
                    self._missing_color_frames += 1
                    continue
                color = np.ascontiguousarray(color).copy()
                if self._writer is None and self._recording_enabled:
                    height, width = color.shape[:2]
                    writer = cv2.VideoWriter(
                        str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (width, height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        raise RuntimeError("无法创建 MP4 文件或初始化 mp4v 编码器")
                    with self._condition:
                        self._writer = writer
                        self._condition.notify_all()
                try:
                    if self._writer is not None:
                        self._writer.write(color)
                        recorded_at = time.monotonic()
                        if self._first_recorded_at is None:
                            self._first_recorded_at = recorded_at
                        self._last_recorded_at = recorded_at
                        self._recorded_frames += 1
                except cv2.error as exc:
                    print(f"[录像] 写入失败，已停止录像但保留相机采集：{exc}")
                    self._writer.release()
                    self._writer = None
                    self._recording_enabled = False
                    self._recording_error = str(exc)
                if depth is None:
                    self._missing_depth_frames += 1
                    continue
                with self._condition:
                    self._latest = (color, depth.copy())
                    self._sequence += 1
                    self._capture_time = time.monotonic()
                    self._condition.notify_all()
        except Exception as exc:
            print(f"[录像] 相机采集线程停止：{exc}")
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if self._writer is not None:
                self._writer.release()
            with self._condition:
                self._condition.notify_all()

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        requested_after = time.monotonic()
        with self._condition:
            self._condition.wait_for(
                lambda: (self._sequence > self._consumed and self._capture_time >= requested_after)
                or self._error is not None or self._stop.is_set(),
                timeout=1.0,
            )
            if self._error is not None:
                raise RuntimeError(f"腕部相机采集失败：{self._error}") from self._error
            if self._sequence <= self._consumed or self._capture_time < requested_after:
                return None, None
            self._consumed = self._sequence
            color, depth = self._latest
            return color.copy(), depth.copy()

    def warm_up(self, count: int = 20) -> None:
        for _ in range(count):
            self.get_frame()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        try:
            self._camera.close()
        except Exception as exc:
            print(f"[录像] 关闭相机失败：{exc}")
        self._thread.join(timeout=2.0)
        capture_span_s = (self._last_recorded_at - self._first_recorded_at
                          if self._first_recorded_at is not None else 0.0)
        metadata = {
            "started_at": self._started_at,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "recorded_frames": self._recorded_frames,
            "recording_fps": self._fps,
            "capture_span_s": round(capture_span_s, 3),
            "video_duration_s": round(self._recorded_frames / self._fps, 3),
            "missing_color_frames": self._missing_color_frames,
            "missing_depth_frames": self._missing_depth_frames,
            "capture_error": None if self._error is None else str(self._error),
            "recording_error": self._recording_error,
            "video_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
        try:
            self.path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        except OSError as exc:
            print(f"[录像] 无法写入录像状态报告：{exc}")
        if self._thread.is_alive():
            print("[录像] 相机采集线程未能及时退出，请检查设备连接。")
        print(f"[录像] 视频已结束：{self.path}；帧数={self._recorded_frames}，"
              f"无彩色帧={self._missing_color_frames}，无深度帧={self._missing_depth_frames}，"
              f"采集错误={metadata['capture_error'] or '无'}。")
        if capture_span_s > metadata["video_duration_s"] * 1.5 and self._recorded_frames > 1:
            print(f"[录像] 采集横跨 {capture_span_s:.1f}s，视频仅 {metadata['video_duration_s']:.1f}s；"
                  "实际采集帧率低于写入帧率，播放会加速，并不单独证明相机断连。")
