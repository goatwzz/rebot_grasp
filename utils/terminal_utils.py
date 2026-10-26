"""Restore an echoed terminal after a GUI exits with readline still waiting."""

from __future__ import annotations

import atexit
import os
import sys

try:
    import termios
except ImportError:
    termios = None


class TerminalStateGuard:
    """Keep a sane baseline instead of inheriting readline's no-echo mode.

    A daemon thread blocked in input() does not reliably run readline cleanup
    when the main GUI exits. Restore explicitly after closing the GUI and again
    at interpreter exit. Do not flush input queued by the user.
    """

    def __init__(self, stream=None) -> None:
        self._fd = None
        self._attributes = None
        stream = sys.stdin if stream is None else stream
        if termios is None:
            return
        try:
            fd = stream.fileno()
            if not os.isatty(fd):
                return
            attributes = termios.tcgetattr(fd)
            attributes[6] = attributes[6][:]
            # Repair a previous abnormal exit too: saving a no-echo baseline
            # would cause GNU readline to hide input in every following prompt.
            attributes[3] |= termios.ECHO | termios.ICANON | termios.ISIG
            attributes[0] |= termios.ICRNL
            if sys.platform.startswith("linux"):
                attributes[0] |= getattr(termios, "IUTF8", 0x4000)
            self._fd, self._attributes = fd, attributes
            self.restore()
            atexit.register(self.restore)
        except (AttributeError, OSError, ValueError, termios.error):
            self._fd = self._attributes = None

    def restore(self) -> None:
        if self._fd is None or self._attributes is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._attributes)
        except (OSError, ValueError, termios.error):
            pass  # Closed/replaced stdin must not prevent GUI cleanup.
