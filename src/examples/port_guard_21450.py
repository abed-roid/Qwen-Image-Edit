#!/usr/bin/env python3
import subprocess
import time
import logging
import signal
import sys
from typing import Tuple

# ====== Config ======
PORT = 21450
POLL_INTERVAL_SEC = 5
MISSING_WINDOW_SEC = 30
COOLDOWN_AFTER_ACTION_SEC = 60
LOG_FILE = "/root/port_guard_21450.log"
USE_LISTEN_FILTER = False

# ====== Logging ======
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("port_guard_21450")

# ====== Graceful shutdown ======
_running = True
def _handle_sigterm(_sig, _frm):
    global _running
    _running = False
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)

# ====== Helpers ======
def run_cmd(cmd: list[str], label: str):
    """Run shell command with logging."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        logger.info(f"[{label}] rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}")
    except Exception as e:
        logger.exception(f"Failed running {label}: {e}")

def check_port_active() -> Tuple[bool, str]:
    """Return (is_active, raw_output) using `sudo lsof -i`."""
    cmd = ["sudo", "lsof", "-nP", "-i", f":{PORT}"]
    if USE_LISTEN_FILTER:
        cmd.extend(["-sTCP:LISTEN"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = (proc.stdout or "") + (proc.stderr or "")
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        is_active = len(lines) > 1  # header + entries
        return is_active, out
    except Exception as e:
        logger.exception(f"lsof check failed: {e}")
        return False, ""

def recover_services():
    """Start supervisor and qwen_edit_demo."""
    logger.warning("No process detected — starting supervisor and qwen_edit_demo.")
    run_cmd(["sudo", "service", "supervisor", "start"], "service supervisor start")
    run_cmd(["sudo", "supervisorctl", "start", "qwen_edit_demo"], "supervisorctl start qwen_edit_demo")

# ====== Main loop ======
def main():
    logger.info(f"Port guard started on port {PORT}")
    last_seen_active = time.monotonic()
    last_action_time = 0.0

    while _running:
        is_active, _ = check_port_active()
        now = time.monotonic()

        if is_active:
            last_seen_active = now
        else:
            missing_for = now - last_seen_active
            if missing_for >= MISSING_WINDOW_SEC:
                if now - last_action_time >= COOLDOWN_AFTER_ACTION_SEC:
                    recover_services()
                    last_action_time = now
                else:
                    logger.info("Cooldown active, not restarting again yet.")

        time.sleep(POLL_INTERVAL_SEC)

    logger.info("Port guard exiting gracefully.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception; exiting.")
        sys.exit(1)
