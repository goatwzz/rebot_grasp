"""Offline VLM contract, request, and preview checks; no API key or camera needed."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np

from utils.vlm_utils import VLMClient, VLMError, VLMSettings, draw_grounding, encode_image, parse_grounding
from scripts import vlm_target_demo


def reply(**overrides):
    value = {"found": True, "target_name": "左边的香蕉", "bbox_xyxy": [10, 20, 50, 60],
             "coordinate_space": "pixel", "reason": "位于图像左侧，颜色和指令匹配",
             "placement": {"required": False, "found": False, "target_name": "", "bbox_xyxy": None,
                           "relation": None, "reason": "明确要求不放置", "requested_name": "", "selection_mode": "disabled"}}
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def parse(content, **kwargs):
    settings = dict(instruction="把左边的香蕉放进盒子", image_size=(200, 160),
                    request_image_size=(100, 80), coordinate_space="pixel")
    settings.update(kwargs)
    return parse_grounding(content, **settings)


class GroundingTests(unittest.TestCase):
    def test_resize_coordinates_map_to_original(self):
        result = parse(reply())
        self.assertEqual(result.bbox_xyxy, (20, 40, 100, 120))
        self.assertEqual(result.target_name, "左边的香蕉")

    def test_normalized_coordinates_and_border(self):
        result = parse(reply(bbox_xyxy=[0, 100, 1000, 1000], coordinate_space="normalized_1000"),
                       coordinate_space="normalized_1000")
        self.assertEqual(result.bbox_xyxy, (0, 16, 199, 159))

    def test_widescreen_normalized_box_maps_each_axis_independently(self):
        result = parse(reply(bbox_xyxy=[200, 600, 400, 950], coordinate_space="normalized_1000"),
                       coordinate_space="normalized_1000", image_size=(1280, 720), request_image_size=(1280, 720))
        self.assertEqual(result.bbox_xyxy, (256, 432, 512, 684))
        # The same raw Y=950 is invalid as a pixel coordinate in a 720-pixel-high image.
        with self.assertRaises(VLMError) as error:
            parse(reply(bbox_xyxy=[200, 600, 400, 950]),
                  image_size=(1280, 720), request_image_size=(1280, 720))
        self.assertIn("bbox_xyxy=[200, 600, 400, 950]", str(error.exception))
        self.assertIn("1280x720", str(error.exception))
        self.assertIn("pixel", str(error.exception))

    def test_no_unique_target_is_a_valid_result_without_bbox(self):
        result = parse(reply(found=False, bbox_xyxy=None, reason="两个杯子无法区分"))
        self.assertFalse(result.found)
        self.assertIsNone(result.bbox_xyxy)
        with self.assertRaises(VLMError):
            parse(reply(found=False))

    def test_fenced_json_is_supported(self):
        self.assertTrue(parse("```json\n" + reply() + "\n```").found)

    def test_invalid_or_ambiguous_geometry_is_rejected(self):
        cases = [
            reply(bbox_xyxy=[-1, 20, 50, 60]), reply(bbox_xyxy=[10, 20, 101, 60]),
            reply(bbox_xyxy=[50, 20, 10, 60]), reply(bbox_xyxy=[10, 20, 10, 60]),
            reply(bbox_xyxy=[True, 20, 50, 60]), reply(bbox_xyxy=[float("nan"), 20, 50, 60]),
            reply(bbox_xyxy=[10, 20, 50]), reply(coordinate_space="normalized_1000"),
            reply(found="true"), reply(target_name=""), "explanation\n" + reply(),
            reply() + "\n" + reply(), "[]", "", "not JSON",
        ]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(VLMError):
                parse(content)

    def test_drawing_only_modifies_a_copy_of_the_snapshot(self):
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        annotated = draw_grounding(image, parse(reply()))
        np.testing.assert_array_equal(annotated[80, 60], [0, 0, 255])
        np.testing.assert_array_equal(annotated[80, 20], [0, 255, 0])
        self.assertFalse(image.any())

    def test_jpeg_encoding_keeps_aspect_ratio(self):
        url, size = encode_image(np.zeros((200, 400, 3), dtype=np.uint8), 100)
        self.assertEqual(size, (100, 50))
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


class SettingsAndRequestTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"VLM_API_KEY": "test-secret", "VLM_MODEL": "test-vision-model"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cfg = {"vlm": {"base_url": "https://example.invalid/v1"}}

    def test_missing_configuration_lists_what_to_fill_without_secrets(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(VLMError) as error:
            VLMSettings.from_config(self.cfg)
        self.assertIn("VLM_MODEL", str(error.exception))
        self.assertIn("VLM_API_KEY", str(error.exception))

    def test_invalid_settings(self):
        for values in ({"timeout_s": 0}, {"coordinate_space": "guess"}, {"base_url": "file:///tmp/key"},
                       {"base_url": "https://user:password@example.invalid/v1"}, {"max_tokens": 0}):
            with self.subTest(values=values), self.assertRaises(VLMError):
                VLMSettings.from_config({"vlm": {**self.cfg["vlm"], **values}})

    def test_local_key_file_is_used_without_environment_and_env_can_override(self):
        with tempfile.TemporaryDirectory() as temp:
            key_path = Path(temp) / "key.local"
            key_path.write_text("local-test-secret\n")
            config = {"vlm": {**self.cfg["vlm"], "api_key_file": str(key_path)}}
            with patch.dict(os.environ, {"VLM_MODEL": "test-vision-model"}, clear=True):
                settings = VLMSettings.from_config(config)
                client = VLMClient(settings)
                with patch("utils.vlm_utils.urlopen") as opening:
                    opening.return_value.__enter__.return_value.read.return_value = json.dumps({
                        "choices": [{"message": {"content": reply()}}],
                    }).encode()
                    client.ground("抓香蕉", np.zeros((80, 100, 3), dtype=np.uint8))
                    self.assertEqual(opening.call_args.args[0].get_header("Authorization"), "Bearer local-test-secret")
                self.assertNotIn("local-test-secret", repr(settings))
                with patch.dict(os.environ, {"VLM_API_KEY": "env-test-secret"}):
                    with patch("utils.vlm_utils.urlopen") as opening:
                        opening.return_value.__enter__.return_value.read.return_value = json.dumps({
                            "choices": [{"message": {"content": reply()}}],
                        }).encode()
                        client.ground("抓香蕉", np.zeros((80, 100, 3), dtype=np.uint8))
                        self.assertEqual(opening.call_args.args[0].get_header("Authorization"), "Bearer env-test-secret")

    def test_actual_wire_payload_and_response(self):
        settings = VLMSettings.from_config(self.cfg)
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": reply()}}]}
        with patch("utils.vlm_utils.urlopen") as opening:
            opening.return_value.__enter__.return_value.read.return_value = json.dumps(envelope).encode()
            result = VLMClient(settings).ground("抓左边香蕉", np.zeros((80, 100, 3), dtype=np.uint8))
        request = opening.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "test-vision-model")
        self.assertEqual(body["messages"][1]["content"][1]["text"], "抓左边香蕉")
        self.assertTrue(body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(opening.call_args.kwargs["timeout"], 45.)
        self.assertEqual(result.bbox_xyxy, (10, 20, 50, 60))

    def test_http_errors_do_not_expose_credentials_or_server_body(self):
        errors = [HTTPError("https://example.invalid", 401, "test-secret", {}, None),
                  URLError("test-secret"), TimeoutError("test-secret")]
        for failure in errors:
            with self.subTest(failure=type(failure).__name__), patch("utils.vlm_utils.urlopen", side_effect=failure):
                with self.assertRaises(VLMError) as error:
                    VLMClient(VLMSettings.from_config(self.cfg)).ground("抓香蕉", np.zeros((80, 100, 3), dtype=np.uint8))
                self.assertNotIn("test-secret", str(error.exception))

    def test_402_reports_account_balance_and_service_code_without_leaking_key(self):
        for payload in (
            {"code": 30001, "message": "Sorry, your account balance is insufficient"},
            {"error": {"code": 30001, "message": "account balance is insufficient test-secret sk-extra-token"}},
        ):
            response = HTTPError("https://example.invalid", 402, "Payment Required", {},
                                 io.BytesIO(json.dumps(payload).encode()))
            with patch("utils.vlm_utils.urlopen", side_effect=response), self.assertRaises(VLMError) as error:
                VLMClient(VLMSettings.from_config(self.cfg)).ground("抓香蕉", np.zeros((80, 100, 3), dtype=np.uint8))
            text = str(error.exception)
            self.assertIn("HTTP 402", text)
            self.assertIn("余额", text)
            self.assertIn("30001", text)
            self.assertIn("balance is insufficient", text)
            self.assertNotIn("test-secret", text)
            self.assertNotIn("sk-extra-token", text)

    def test_truncated_or_invalid_envelope_is_rejected(self):
        for envelope in ({"choices": []}, {"choices": [{"finish_reason": "length", "message": {"content": reply()}}]}):
            with patch("utils.vlm_utils.urlopen") as opening:
                opening.return_value.__enter__.return_value.read.return_value = json.dumps(envelope).encode()
                with self.assertRaises(VLMError):
                    VLMClient(VLMSettings.from_config(self.cfg)).ground("抓香蕉", np.zeros((80, 100, 3), dtype=np.uint8))


class PreviewTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Requires a POSIX pseudo-terminal")
    def test_terminal_backspace_deletes_complete_chinese_characters(self):
        import pty
        import select
        import subprocess
        import termios
        import time

        try:
            import readline  # noqa: F401
        except ImportError:
            self.skipTest("Python readline module is unavailable")
        master, slave = pty.openpty()
        process = None
        try:
            attrs = termios.tcgetattr(slave)
            # Reproduce terminals that erase bytes unless readline is loaded.
            attrs[0] &= ~getattr(termios, "IUTF8", 0x4000)
            attrs[6][termios.VERASE] = b"\x7f"
            termios.tcsetattr(slave, termios.TCSANOW, attrs)
            code = (
                "import queue\n"
                "from scripts.vlm_target_demo import start_terminal_input\n"
                "commands = queue.Queue()\n"
                "start_terminal_input(commands)\n"
                "print('\\nINPUT_RESULT=' + ascii(commands.get(timeout=10)), flush=True)\n"
            )
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", code], cwd=Path(__file__).resolve().parents[1],
                stdin=slave, stdout=slave, stderr=slave,
            )
            captured = bytearray()
            deadline = time.monotonic() + 10
            marker = "并回车：".encode("utf-8")
            while marker not in captured and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    captured.extend(os.read(master, 8192))
            self.assertIn(marker, captured, captured.decode(errors="replace"))
            # Delete all five Chinese characters, then enter the quit command.
            os.write(master, "我想玩积木".encode("utf-8") + b"\x7f" * 5 + b"q\n")
            while b"INPUT_RESULT=" not in captured and time.monotonic() < deadline:
                if select.select([master], [], [], 0.1)[0]:
                    captured.extend(os.read(master, 8192))
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 0, captured.decode(errors="replace"))
            self.assertIn(b"INPUT_RESULT='/quit'", captured)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
            os.close(slave)

    def test_terminal_reader_reads_once_and_signals_empty_input_or_eof(self):
        class ImmediateThread:
            def __init__(self, *, target, **kwargs):
                self.target = target
            def start(self):
                self.target()
        for answer, expected in (("抓香蕉", "抓香蕉"), ("", ""), (EOFError(), None)):
            with self.subTest(answer=answer):
                commands = queue.Queue()
                options = {"side_effect": answer} if isinstance(answer, Exception) else {"return_value": answer}
                with patch.object(vlm_target_demo.threading, "Thread", ImmediateThread), patch("builtins.input", **options) as read:
                    vlm_target_demo.start_terminal_input(commands)
                read.assert_called_once()
                self.assertEqual(commands.get_nowait(), expected)
                self.assertTrue(commands.empty())

    def test_two_instructions_prompt_after_results_and_replace_old_box(self):
        pending = []
        events = []
        answers = iter(["抓左边的香蕉", "抓右边的杯子", "q"])
        class ScriptedThread:
            def __init__(self, *, target, name, **kwargs):
                self.target, self.name = target, name
            def start(self):
                if self.name == "vlm-terminal-input":
                    self.target()
                else:
                    pending.append(self.target)
        def read(prompt):
            events.append("prompt")
            return next(answers)
        def ground(command, snapshot):
            events.append("inference")
            return parse(reply(target_name="香蕉" if "香蕉" in command else "杯子",
                               bbox_xyxy=[10, 20, 50, 60] if "香蕉" in command else [60, 20, 90, 60]),
                         instruction=command)
        frame = 0
        def get_frame():
            nonlocal frame
            frame += 1
            if frame in (3, 5):
                pending.pop(0)()
            if frame > 6:
                self.fail("The preview did not exit after the terminal q command")
            return np.full((160, 200, 3), frame * 10, dtype=np.uint8), None
        camera = SimpleNamespace(open=Mock(), close=Mock(), warm_up=Mock(), K=np.eye(3), get_frame=get_frame)
        client = SimpleNamespace(ground=Mock(side_effect=ground))
        args = SimpleNamespace(instruction=None, warmup=0, output_dir=Path("/unused"))
        shown = []
        with patch.object(vlm_target_demo, "make_camera", return_value=camera), \
             patch.object(vlm_target_demo, "configure_camera", side_effect=lambda cfg, args: cfg), \
             patch.object(vlm_target_demo.sys.stdin, "isatty", return_value=True), \
             patch.object(vlm_target_demo.threading, "Thread", ScriptedThread), \
             patch("builtins.input", side_effect=read), \
             patch.object(vlm_target_demo, "save_result", return_value=Path("/unused")) as save, \
             patch.object(vlm_target_demo, "report_result", side_effect=lambda *args: events.append("result")), \
             patch.object(cv2, "namedWindow"), patch.object(cv2, "destroyAllWindows"), \
             patch.object(cv2, "getWindowProperty", return_value=1), \
             patch.object(cv2, "imshow", side_effect=lambda title, image: shown.append(image.copy())), \
             patch.object(cv2, "waitKey", return_value=-1), redirect_stdout(io.StringIO()):
            self.assertEqual(vlm_target_demo.run_camera(args, {}, client), 0)
        self.assertEqual(events, ["prompt", "inference", "result", "prompt", "inference", "result", "prompt"])
        self.assertEqual(client.ground.call_count, 2)
        self.assertEqual(save.call_count, 2)
        # First query uses frame 2; the second uses a new frame 4, without requiring R.
        self.assertTrue(np.all(client.ground.call_args_list[0].args[1] == 20))
        self.assertTrue(np.all(client.ground.call_args_list[1].args[1] == 40))
        np.testing.assert_array_equal(shown[2][80, 60], [0, 0, 255])
        np.testing.assert_array_equal(shown[3][80, 60], [40, 40, 40])  # old box cleared while processing
        np.testing.assert_array_equal(shown[4][80, 60], [40, 40, 40])  # old center stays cleared
        np.testing.assert_array_equal(shown[4][80, 150], [0, 0, 255])  # new target center
        camera.close.assert_called_once()

    def test_terminal_controls_are_not_natural_language_instructions(self):
        aliases = {"r": "/live", " R ": "/live", "g": "/retry", "q": "/quit", "/quit": "/quit"}
        for text, expected in aliases.items():
            self.assertEqual(vlm_target_demo.normalize_terminal_command(text), expected)
        self.assertEqual(vlm_target_demo.normalize_terminal_command("抓左边的香蕉"), "抓左边的香蕉")

    def test_terminal_r_and_q_do_not_send_api_requests(self):
        for command in ("r", "q"):
            with self.subTest(command=command):
                camera = SimpleNamespace(open=Mock(), close=Mock(), warm_up=Mock(), K=np.eye(3),
                                         get_frame=Mock(return_value=(np.zeros((160, 200, 3), dtype=np.uint8), None)))
                client = SimpleNamespace(ground=Mock())
                args = SimpleNamespace(instruction=command, warmup=0)
                with patch.object(vlm_target_demo, "make_camera", return_value=camera), \
                     patch.object(vlm_target_demo, "configure_camera", side_effect=lambda cfg, args: cfg), \
                     patch.object(vlm_target_demo.sys.stdin, "isatty", return_value=False), \
                     patch.object(cv2, "namedWindow"), patch.object(cv2, "destroyAllWindows"), \
                     patch.object(cv2, "getWindowProperty", return_value=1), patch.object(cv2, "imshow"), \
                     patch.object(cv2, "waitKey", return_value=ord("q")), redirect_stdout(io.StringIO()):
                    self.assertEqual(vlm_target_demo.run_camera(args, {}, client), 0)
                client.ground.assert_not_called()
                camera.close.assert_called_once()

    def test_headless_image_mode_saves_original_box_and_json_without_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = directory / "scene.png"
            cv2.imwrite(str(image), np.zeros((160, 200, 3), dtype=np.uint8))
            args = SimpleNamespace(image=image, instruction="抓左边香蕉", output_dir=directory / "results", no_window=True)
            client = SimpleNamespace(ground=Mock(return_value=parse(reply())))
            with patch.object(vlm_target_demo, "make_camera") as camera, redirect_stdout(io.StringIO()):
                self.assertEqual(vlm_target_demo.run_image(args, client), 0)
            camera.assert_not_called()
            output = next(args.output_dir.iterdir())
            data = json.loads((output / "result.json").read_text())
            self.assertEqual(data["bbox_xyxy"], [20, 40, 100, 120])
            self.assertIsNotNone(cv2.imread(str(output / "annotated.png")))
            self.assertIsNotNone(cv2.imread(str(output / "image.png")))

    def test_missing_config_exits_before_camera_or_network(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text("vlm:\n  base_url: https://example.invalid/v1\n")
            with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["demo", "--config", str(config), "--check-config"]):
                with patch.object(vlm_target_demo, "make_camera") as camera, patch("utils.vlm_utils.urlopen") as network:
                    with redirect_stderr(io.StringIO()) as error:
                        self.assertEqual(vlm_target_demo.main(), 2)
            self.assertIn("VLM_API_KEY", error.getvalue())
            camera.assert_not_called()
            network.assert_not_called()

    def test_camera_result_stays_on_original_snapshot_and_r_discards_old_result(self):
        # Delay the worker until the next frame to exercise the image/result pairing.
        for discard in (False, True):
            with self.subTest(discard=discard), tempfile.TemporaryDirectory() as temp:
                pending = []
                class DelayedThread:
                    def __init__(self, *, target, **kwargs):
                        self.target = target
                    def start(self):
                        pending.append(self.target)
                first = np.full((160, 200, 3), 20, dtype=np.uint8)
                second = np.full((160, 200, 3), 90, dtype=np.uint8)
                frames = 0
                def get_frame():
                    nonlocal frames
                    frames += 1
                    if frames == 2:
                        pending.pop(0)()
                    return (first if frames == 1 else second), np.full((160, 200), 500, dtype=np.uint16)
                camera = SimpleNamespace(open=Mock(), close=Mock(), warm_up=Mock(), K=np.eye(3), get_frame=get_frame)
                args = SimpleNamespace(instruction="抓左边香蕉", output_dir=Path(temp), warmup=0)
                client = SimpleNamespace(ground=Mock(return_value=parse(reply())))
                shown = []
                with patch.object(vlm_target_demo, "make_camera", return_value=camera), \
                     patch.object(vlm_target_demo, "configure_camera", side_effect=lambda cfg, args: cfg), \
                     patch.object(vlm_target_demo.sys.stdin, "isatty", return_value=False), \
                     patch.object(vlm_target_demo.threading, "Thread", DelayedThread), \
                     patch.object(cv2, "namedWindow"), patch.object(cv2, "destroyAllWindows"), \
                     patch.object(cv2, "getWindowProperty", return_value=1), \
                     patch.object(cv2, "imshow", side_effect=lambda title, image: shown.append(image.copy())), \
                     patch.object(cv2, "waitKey", side_effect=[ord("r") if discard else -1, ord("q")]), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(vlm_target_demo.run_camera(args, {}, client), 0)
                camera.close.assert_called_once()
                client.ground.assert_called_once()
                np.testing.assert_array_equal(client.ground.call_args.args[1], first)
                np.testing.assert_array_equal(shown[-1][-1, -1], [90, 90, 90] if discard else [20, 20, 20])
                results = list(Path(temp).glob("*/result.json"))
                self.assertEqual(len(results), 0 if discard else 1)


if __name__ == "__main__":
    unittest.main()
