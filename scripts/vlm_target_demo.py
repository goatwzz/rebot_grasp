"""Stage 1: natural-language + image grounding, with no robot connection.

Camera: python scripts/vlm_target_demo.py
Image:  python scripts/vlm_target_demo.py --image scene.jpg --instruction "抓左边的香蕉" --no-window
Config: python scripts/vlm_target_demo.py --check-config

Enter Chinese instructions in the terminal; terminal g/r/q also control the preview.
Window: G=repeat, R=live/discard pending result, Q/Esc=exit. Requests use a worker.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Any

# input() otherwise falls back to terminal byte-based erasing on Linux. Loading
# readline enables UTF-8 character editing, including for the input worker.
try:
    import readline  # noqa: F401
except ImportError:
    pass  # Optional standard-library module (unavailable on some platforms).

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from drivers.camera import make_camera  # noqa: E402
from utils.camera_utils import configure_camera, load_config  # noqa: E402
from utils.vlm_utils import TargetGrounding, VLMClient, VLMError, VLMSettings, draw_grounding  # noqa: E402
from utils.terminal_utils import TerminalStateGuard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文字＋图像理解预览（不连接或移动机械臂）")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/default.yaml")
    parser.add_argument("--instruction", help="中文自然语言指令；省略时在终端输入")
    parser.add_argument("--image", type=Path, help="使用已有图片，不打开相机")
    parser.add_argument("--no-window", action="store_true", help="图片模式下只保存结果，不打开 GUI")
    parser.add_argument("--check-config", action="store_true", help="只检查服务配置，不请求 API 或打开相机")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/vlm_preview")
    parser.add_argument("--camera-type", choices=("realsense_d435i", "realsense_d405", "orbbec_gemini2"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()
    if not args.check_config:
        if args.no_window and not args.image:
            parser.error("--no-window 仅用于 --image 图片模式")
        if args.image and not (args.instruction and args.instruction.strip()):
            parser.error("--image 需要同时提供 --instruction")
        if not args.image and not args.instruction and not sys.stdin.isatty():
            parser.error("当前终端不能交互输入，请提供 --instruction")
    return args


def save_result(
    output_dir: Path, image: np.ndarray, result: TargetGrounding,
    *, depth_mm: np.ndarray | None = None, K: np.ndarray | None = None,
) -> Path:
    directory = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    for name, data in (("image.png", image), ("annotated.png", draw_grounding(image, result))):
        if not cv2.imwrite(str(directory / name), data):
            raise OSError(f"无法保存 {directory / name}")
    (directory / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if depth_mm is not None and K is not None:
        np.savez_compressed(directory / "rgbd.npz", depth_mm=depth_mm, camera_matrix=K)
    return directory


def report_result(result: TargetGrounding, directory: Path) -> None:
    print("\n[VLM] 指令：", result.instruction)
    if result.found:
        print("[VLM] 抓取目标：", result.target_name)
        print("[VLM] 抓取框 [x1,y1,x2,y2]：", result.bbox_xyxy)
    else:
        print("[VLM] 画面中没有可定位的抓取目标")
    print("[VLM] 说明：", result.reason)
    placement = result.placement
    if placement is not None and placement.required:
        if placement.found:
            relation = {"in": "放入", "on": "放到上面", "near": "放到旁边"}[placement.relation]
            print(f"[VLM] 放置目标：{placement.target_name}（{relation}）")
            mode = {"explicit": "按指定地点选择", "auto": "自动选择容器/放置区域", "fallback": "使用替代放置目标"}[placement.selection_mode]
            print(f"[VLM] 放置选择：{mode}" + (f"；原要求={placement.requested_name}" if placement.requested_name else ""))
            print("[VLM] 放置框 [x1,y1,x2,y2]：", placement.bbox_xyxy)
            print("[VLM] 放置目标说明：", placement.reason)
            if result.found:
                print(f"[任务识别] 抓取 {result.target_name} → {relation} {placement.target_name}")
        else:
            print(f"[VLM] 放置目标未确认：{placement.target_name or '未知目的地'}；{placement.reason}")
        print("[任务识别] 两个目标均已确认。" if result.targets_complete else "[任务识别] 目标尚未全部确认，仅保留识别预览。")
    elif placement is not None and placement.selection_mode == "grasp_only":
        print("[VLM] 错误的容器摆放姿态：", placement.reason)
        if result.found:
            print(f"[任务识别] 仅抓取 {result.target_name}，本轮不执行放置。")
    if placement is not None and placement.rejected_container is not None:
        rejected = placement.rejected_container
        print(f"[VLM] 不可用容器：{rejected.target_name}；框={rejected.bbox_xyxy}；{rejected.reason}")
    print("[VLM] 图像、标注和 JSON 已保存：", directory, flush=True)


class TargetPreviewProcessor:
    """Snapshot processing hooks shared by the VLM and VLM/SAM/GraspNet demos."""

    def __init__(self, client: VLMClient) -> None:
        self.client = client

    def process(self, command, color, depth_mm, K):
        return self.client.ground(command, color)

    def draw(self, image, result):
        return draw_grounding(image, result)

    def save(self, output_dir, image, result, *, depth_mm=None, K=None):
        return save_result(output_dir, image, result, depth_mm=depth_mm, K=K)

    def report(self, result, directory):
        report_result(result, directory)

    def status(self, result):
        if not result.found:
            return "NO VISIBLE PICK TARGET"
        if result.placement is not None and result.placement.required:
            return "PICK + PLACE TARGETS FOUND" if result.placement.found else "PLACE TARGET NOT CONFIRMED"
        if result.placement is not None and result.placement.selection_mode == "grasp_only":
            return "INVALID CONTAINER POSE | GRASP ONLY"
        return "FOUND"

    def intro(self):
        return "仅进行文字＋图像目标识别，不连接机械臂。"

    def window_title(self):
        return "VLM Target - Stage 1"

    def compose_display(self, main_image):
        """Allow richer processors to replace the main image with a dashboard."""
        return main_image

    def display_size(self):
        """Requested visible window size, or None to let OpenCV choose."""
        return None

    def bind_window(self, window):
        """Attach optional mouse interaction after the OpenCV window exists."""

    def handle_preview_key(self, key):
        """Handle optional preview-only keys and report whether one was used."""
        return False

    def show_result(self, image, result):
        """Update optional auxiliary windows, on the main GUI thread."""

    def reset_preview(self, image_shape, status):
        """Clear auxiliary results when a snapshot is replaced or discarded."""

    def poll_preview(self):
        """Keep optional native visualizers responsive between results."""

    def close_preview(self):
        """Release auxiliary windows before shutting down the preview."""


def run_image(args: argparse.Namespace, client: VLMClient, *, processor=None, depth_mm=None, K=None) -> int:
    processor = processor or TargetPreviewProcessor(client)
    window = processor.window_title() + " - Q/Esc to exit"
    image = cv2.imread(str(args.image))
    if image is None:
        raise VLMError(f"无法读取图片：{args.image}")
    try:
        result = processor.process(args.instruction, image, depth_mm, K)
        if not args.no_window:
            processor.show_result(image, result)
        directory = processor.save(args.output_dir, image, result, depth_mm=depth_mm, K=K)
        processor.report(result, directory)
        if not args.no_window:
            display = processor.draw(image, result)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if processor.display_size() is not None:
                cv2.resizeWindow(window, *processor.display_size())
            processor.bind_window(window)
            while True:
                processor.poll_preview()
                cv2.imshow(window, processor.compose_display(display))
                key = cv2.waitKey(30) & 0xFF
                processor.handle_preview_key(key)
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        processor.close_preview()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


def normalize_terminal_command(command: str) -> str:
    command = command.strip()
    aliases = {"q": "/quit", "quit": "/quit", "exit": "/quit", "/exit": "/quit",
               "r": "/live", "/r": "/live", "g": "/retry", "/g": "/retry"}
    return aliases.get(command.casefold(), command)


def start_terminal_input(commands: queue.Queue, prompt: str | None = None) -> None:
    """Read one command; the UI requests the next prompt after processing finishes."""
    def reader() -> None:
        try:
            command = normalize_terminal_command(input(
                prompt or "\n请输入下一条中文指令（r=预览，g=重试，q=退出）并回车："
            ))
        except (EOFError, OSError):
            commands.put(None)
            return
        # Empty input still acknowledges that this prompt has finished.
        commands.put(command)
    threading.Thread(target=reader, daemon=True, name="vlm-terminal-input").start()


def run_camera(args: argparse.Namespace, cfg: dict[str, Any], client: VLMClient, *, processor=None) -> int:
    processor = processor or TargetPreviewProcessor(client)
    terminal_state = TerminalStateGuard()
    commands: queue.Queue = queue.Queue()
    completions: queue.Queue = queue.Queue()
    if args.instruction:
        commands.put(args.instruction)
    window = processor.window_title()
    camera = make_camera(configure_camera(cfg, args))
    generation = 0
    busy = False
    last_instruction = ""
    terminal_enabled = sys.stdin.isatty()
    input_pending = False
    frozen: np.ndarray | None = None
    status = "LIVE | Type instruction in terminal"

    def submit(command: str, color: np.ndarray, depth: np.ndarray | None, K: np.ndarray) -> None:
        nonlocal generation, busy, last_instruction, frozen, status
        generation += 1
        request_id = generation
        snapshot = color.copy()
        snapshot_depth = depth.copy() if depth is not None else None
        snapshot_K = K.copy()
        frozen = snapshot.copy()
        processor.reset_preview(color.shape[:2], "PROCESSING NEW SNAPSHOT")
        busy = True
        last_instruction = command
        status = "SNAPSHOT | VLM processing... | R=discard Q=exit"
        print(f"\n[VLM] 正在分析本轮快照，指令：{command}", flush=True)

        def worker() -> None:
            try:
                result = processor.process(command, snapshot, snapshot_depth, snapshot_K)
                error = None
            except (VLMError, ValueError, RuntimeError) as exc:
                result, error = None, str(exc)
            except Exception:
                result, error = None, "VLM 处理异常，请检查服务配置后重试"
            completions.put((request_id, snapshot, snapshot_depth, snapshot_K, result, error))

        threading.Thread(target=worker, daemon=True, name="vlm-grounding").start()

    try:
        camera.open()
        camera.warm_up(args.warmup)
        K = camera.K.astype(np.float64)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if processor.display_size() is not None:
            cv2.resizeWindow(window, *processor.display_size())
        processor.bind_window(window)
        print("\n[说明] " + processor.intro())
        print("[窗口按键] G=用上一条指令识别新快照，R=实时预览/丢弃待返回结果，Q/Esc=退出。")
        if processor.display_size() is not None:
            print("[总览交互] 双击任意格放大；1~4=放大对应格，0=返回四宫格；3D 格左拖旋转、右拖平移、滚轮缩放。")
        print("[终端命令] r=恢复预览，g=重试上一条指令，q 或 /quit=退出，输入后回车。")
        print("[说明] RGB 图使用英文框标签；红框＋×表示不可用容器。目标名称和具体原因见终端。", flush=True)
        while True:
            color, depth = camera.get_frame()
            processor.poll_preview()
            if color is None:
                continue
            try:
                request_id, snapshot, snap_depth, snap_K, result, error = completions.get_nowait()
            except queue.Empty:
                pass
            else:
                busy = False
                if request_id == generation:
                    if error:
                        print(f"\n[VLM] {error}", flush=True)
                        status = "ERROR | See terminal | G=retry R=live Q=exit"
                        processor.reset_preview(color.shape[:2], "ERROR | See terminal")
                    else:
                        frozen = processor.draw(snapshot, result)
                        processor.show_result(snapshot, result)
                        try:
                            directory = processor.save(args.output_dir, snapshot, result, depth_mm=snap_depth, K=snap_K)
                            processor.report(result, directory)
                        except OSError as exc:
                            print(f"[保存失败] {exc}", flush=True)
                        status = processor.status(result)
                        status += " | G=new snapshot R=live Q=exit"
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = ""
            else:
                input_pending = False
                if command is None:
                    terminal_enabled = False
                    command = ""
                    print("[终端] 输入已关闭，可使用窗口 G/R/Q 按键。", flush=True)
            if command:
                command = normalize_terminal_command(command)
            if command == "/quit":
                break
            if command == "/live":
                generation += 1
                frozen = None
                processor.reset_preview(color.shape[:2], "LIVE | Result discarded")
                status = "LIVE | Pending result discarded" if busy else "LIVE | Type instruction or press G"
                command = None
            elif command == "/retry":
                command = last_instruction or None
                if not command:
                    print("[VLM] 请先输入一条中文指令。", flush=True)
            if command:
                if busy:
                    print("[VLM] 上一次请求尚未结束，请完成后重新输入指令。", flush=True)
                else:
                    submit(command, color, depth, K)
            display = frozen.copy() if frozen is not None else color.copy()
            if getattr(processor, "show_status_overlay", True) and status:
                cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 220, 80), 2)
            cv2.imshow(window, processor.compose_display(display))
            key = cv2.waitKey(1) & 0xFF
            processor.handle_preview_key(key)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                generation += 1
                frozen = None
                processor.reset_preview(color.shape[:2], "LIVE | Result discarded")
                status = "LIVE | Pending result discarded" if busy else "LIVE | Type instruction or press G"
            elif key in (ord("g"), ord("G"), ord(" ")):
                if busy:
                    print("[VLM] 上一次请求尚未结束。", flush=True)
                elif last_instruction:
                    submit(last_instruction, color, depth, K)
                else:
                    print("[VLM] 请先在终端输入中文指令并回车。", flush=True)
            if terminal_enabled and not busy and not input_pending:
                input_pending = True
                terminal_state.restore()
                start_terminal_input(commands)
    finally:
        try:
            camera.close()
        finally:
            try:
                processor.close_preview()
                cv2.destroyAllWindows()
            finally:
                # Q/Esc can end the GUI while its daemon reader is inside
                # readline, with ECHO and ICANON temporarily disabled.
                terminal_state.restore()
    return 0


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
        settings = VLMSettings.from_config(cfg)
        if args.check_config:
            key_source = f"环境变量 {settings.api_key_env}" if os.environ.get(settings.api_key_env, "").strip() else "本地密钥文件"
            print(f"[配置就绪] base_url={settings.base_url}\nmodel={settings.model}\n密钥来源={key_source}（已配置，不显示内容）")
            print("尚未请求 API；模型是否可用需要实际识别验证。")
            return 0
        client = VLMClient(settings)
        return run_image(args, client) if args.image else run_camera(args, cfg, client)
    except (VLMError, OSError, ValueError, cv2.error) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[相机错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n退出预览。")
        raise SystemExit(130)
