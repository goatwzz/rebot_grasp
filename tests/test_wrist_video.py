"""Hardware-free checks for continuous wrist-camera recording."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from utils.wrist_video import RecordingCamera


class FakeCamera:
    def __init__(self) -> None:
        self.count = 0
        self.closed = False
        self.K = np.eye(3)

    def get_frame(self):
        time.sleep(0.01)
        self.count += 1
        return np.full((8, 12, 3), self.count % 256, np.uint8), np.full((8, 12), 500, np.uint16)

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class RecordingCameraTests(unittest.TestCase):
    def test_capture_continues_while_main_thread_is_busy(self):
        camera = FakeCamera()
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.wrist_video.cv2.VideoWriter", return_value=writer,
        ) as open_writer:
            recorder = RecordingCamera(camera, Path(directory), 30.0)
            try:
                time.sleep(0.08)
                self.assertGreaterEqual(len(writer.frames), 4)
                color, depth = recorder.get_frame()
                self.assertEqual(color.shape, (8, 12, 3))
                self.assertEqual(depth.shape, (8, 12))
                np.testing.assert_array_equal(recorder.K, camera.K)
                self.assertEqual(open_writer.call_args.args[-1], (12, 8))
                self.assertEqual(recorder.path.suffix, ".mp4")
            finally:
                recorder.close()
            self.assertTrue(camera.closed)
            self.assertTrue(writer.released)
            report = json.loads(recorder.path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["recorded_frames"], 4)
            self.assertIsNone(report["capture_error"])
            self.assertGreater(report["capture_span_s"], 0)
            self.assertAlmostEqual(report["video_duration_s"], report["recorded_frames"] / 30.0, places=2)

    def test_color_recording_continues_when_depth_is_missing(self):
        class ColorOnlyCamera(FakeCamera):
            def get_frame(self):
                color, depth = super().get_frame()
                return color, depth if self.count == 1 else None

        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as directory, patch(
            "utils.wrist_video.cv2.VideoWriter", return_value=writer,
        ):
            recorder = RecordingCamera(ColorOnlyCamera(), Path(directory), 30.0)
            try:
                time.sleep(0.06)
                self.assertGreaterEqual(len(writer.frames), 4)
                self.assertEqual(recorder.get_frame(), (None, None))
            finally:
                recorder.close()
            report = json.loads(recorder.path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertGreater(report["missing_depth_frames"], 0)

    def test_invalid_frame_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                RecordingCamera(FakeCamera(), Path(directory), 0.0)


if __name__ == "__main__":
    unittest.main()
