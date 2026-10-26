"""Real PTY checks for pending readline cleanup and inherited no-echo state."""

import importlib.util
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest

from utils.terminal_utils import TerminalStateGuard

ROOT = Path(__file__).resolve().parents[1]


class TerminalStateTests(unittest.TestCase):
    def test_non_terminal_stream_is_left_alone(self):
        guard = TerminalStateGuard(io.StringIO())
        guard.restore()
        self.assertIsNone(guard._fd)

    @unittest.skipUnless(os.name == "posix" and importlib.util.find_spec("readline"), "Requires POSIX readline")
    def test_pending_readline_is_restored_on_normal_and_error_exit(self):
        import pty
        import termios

        for fail in (False, True):
            with self.subTest(fail=fail):
                master, slave = pty.openpty()
                process = None
                try:
                    code = (
                        "import readline,threading,time,termios,sys\n"
                        "from utils.terminal_utils import TerminalStateGuard\n"
                        "guard=TerminalStateGuard()\n"
                        "threading.Thread(target=lambda: input('INPUT_PENDING:'),daemon=True).start()\n"
                        "deadline=time.monotonic()+5\n"
                        "while termios.tcgetattr(sys.stdin.fileno())[3]&termios.ECHO:\n"
                        "    if time.monotonic()>deadline: raise RuntimeError('readline did not start')\n"
                        "    time.sleep(.005)\n"
                        "print('\\nACTIVE_NO_ECHO',flush=True)\n"
                        "try:\n"
                        + ("    raise RuntimeError('simulated preview error')\n" if fail else "    pass\n")
                        + "finally:\n    guard.restore()\n"
                    )
                    process = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=ROOT,
                                               stdin=slave, stdout=slave, stderr=slave)
                    self.read_until(master, b"ACTIVE_NO_ECHO")
                    self.assertEqual(process.wait(timeout=5), 1 if fail else 0)
                    flags = termios.tcgetattr(slave)[3]
                    self.assertTrue(flags & termios.ECHO, "GUI exit leaked readline's no-echo mode")
                    self.assertTrue(flags & termios.ICANON, "GUI exit leaked readline's raw mode")
                finally:
                    if process is not None and process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    os.close(master)
                    os.close(slave)

    @unittest.skipUnless(os.name == "posix" and importlib.util.find_spec("readline"), "Requires POSIX readline")
    def test_startup_repairs_no_echo_and_chinese_is_visible_before_enter(self):
        import pty
        import termios

        master, slave = pty.openpty()
        process = None
        try:
            attributes = termios.tcgetattr(slave)
            attributes[3] &= ~(termios.ECHO | termios.ICANON)
            attributes[0] &= ~getattr(termios, "IUTF8", 0x4000)
            termios.tcsetattr(slave, termios.TCSANOW, attributes)
            code = (
                "import readline\n"
                "from utils.terminal_utils import TerminalStateGuard\n"
                "guard=TerminalStateGuard()\n"
                "try:\n"
                "    value=input('PROMPT_READY:')\n"
                "    print('\\nRESULT='+ascii(value),flush=True)\n"
                "finally:\n    guard.restore()\n"
            )
            process = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=ROOT,
                                       stdin=slave, stdout=slave, stderr=slave)
            self.read_until(master, b"PROMPT_READY:")
            text = "抓取左边的香蕉"
            os.write(master, text.encode())
            # This must be terminal echo while input() is still waiting, not a
            # log printed after submitting the instruction.
            self.read_until(master, text.encode())
            self.assertIsNone(process.poll())
            os.write(master, "错误".encode() + b"\x7f\x7f\r")
            captured = self.read_until(master, b"RESULT=")
            process.wait(timeout=5)
            # The line may arrive in more than one PTY read.
            if ascii(text).encode() not in captured:
                captured += self.read_until(master, ascii(text).encode())
            self.assertIn(ascii(text).encode(), captured)
            self.assertEqual(process.returncode, 0)
            self.assertTrue(termios.tcgetattr(slave)[3] & termios.ECHO)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
            os.close(slave)

    def read_until(self, master, marker):
        captured = bytearray()
        deadline = time.monotonic() + 8
        while marker not in captured and time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 8192)
                except OSError:
                    break
                if not chunk:
                    break
                captured.extend(chunk)
        self.assertIn(marker, captured, captured.decode(errors="replace"))
        return captured


if __name__ == "__main__":
    unittest.main()
