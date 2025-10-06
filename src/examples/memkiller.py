#!/usr/bin/env python3
"""
memkiller.py — watchdog that terminates any process that sustains memory usage
above a threshold of total system RAM.

Behavior:
- Samples processes every INTERVAL_SEC.
- Tracks per-PID RSS over a sliding WINDOW_SEC.
- If a process's RSS/total_ram >= THRESHOLD for >= GRACE_SEC, send SIGTERM.
  If still above threshold or not exited after KILL_WAIT_SEC, send SIGKILL.
- Logs to file and syslog.
- All options below are embedded here as requested.

Requires: psutil (pip install psutil)
"""

import os
import time
import signal
import logging
import logging.handlers
from collections import deque, defaultdict

try:
    import psutil
except ImportError:
    raise SystemExit("psutil is required. Install:  pip install psutil")

# -----------------------------
# Tunables (edit here)
# -----------------------------
THRESHOLD = 0.8          # 85% of total system RAM
INTERVAL_SEC = 1.0        # sampling interval
WINDOW_SEC = 15.0         # sliding window size to smooth spikes
GRACE_SEC = 60.0           # must be above threshold for at least this long
KILL_WAIT_SEC = 5.0       # wait time after SIGTERM before SIGKILL
DRY_RUN = False           # if True, never actually kill; just log what would happen
LOG_PATH = "/var/log/memkiller.log"

# Skip killing these exact PIDs (script, systemd, etc.) and name prefixes
EXCLUDE_PIDS = {1}        # PID 1 (systemd), add more if needed
EXCLUDE_NAME_PREFIXES = (
    "systemd", "kthreadd", "rcu_", "idle", "memkiller", "journal", "containerd-shim",
)
# You may also exclude critical daemons by name if you like:
EXCLUDE_NAMES = {
    # "mongod", "mysqld", "postgres", "dockerd",
}

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("memkiller")
logger.setLevel(logging.INFO)

# File log with rotation
file_handler = logging.handlers.RotatingFileHandler(
    LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# Syslog (optional but helpful with journalctl)
try:
    syslog_handler = logging.handlers.SysLogHandler(address="/dev/log")
    syslog_handler.setFormatter(logging.Formatter("memkiller: %(message)s"))
    logger.addHandler(syslog_handler)
except Exception:
    # Not fatal if syslog socket isn't available in the environment
    pass

# -----------------------------
# Helpers
# -----------------------------
def is_excluded(proc: psutil.Process) -> bool:
    if proc.pid in EXCLUDE_PIDS:
        return True
    try:
        name = (proc.name() or "").lower()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return True
    if name in (n.lower() for n in EXCLUDE_NAMES):
        return True
    for pref in EXCLUDE_NAME_PREFIXES:
        if name.startswith(pref.lower()):
            return True
    # Exclude kernel threads (no executable, typically ppid==2/kthreadd family)
    try:
        exe = proc.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        exe = ""
    if not exe:
        # Heuristic: most killable user processes have a real exe path.
        return True
    return False


def percent_of_total(rss_bytes: int, total_bytes: int) -> float:
    return rss_bytes / total_bytes if total_bytes else 0.0


def safe_process_iter():
    for p in psutil.process_iter(["pid", "name", "exe", "username"]):
        yield p


def kill_process(proc: psutil.Process, reason: str):
    try:
        info = f"pid={proc.pid} name={proc.name()} user={proc.username()}"
    except Exception:
        info = f"pid={getattr(proc, 'pid', '?')}"

    if DRY_RUN:
        logger.warning(f"[DRY-RUN] Would terminate {info} — {reason}")
        return

    try:
        logger.warning(f"Sending SIGTERM to {info} — {reason}")
        proc.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.info(f"Could not send SIGTERM to {info}: {e}")
        return

    # Wait a bit; if it's still around, escalate
    try:
        gone, alive = psutil.wait_procs([proc], timeout=KILL_WAIT_SEC)
        if alive:
            for p in alive:
                try:
                    logger.warning(f"Sending SIGKILL to pid={p.pid} name={p.name()}")
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.info(f"Could not SIGKILL pid={p.pid}: {e}")
    except Exception as e:
        logger.info(f"wait_procs error: {e}")


def main():
    me = psutil.Process(os.getpid())
    EXCLUDE_PIDS.add(me.pid)  # don't kill ourselves

    total_ram = psutil.virtual_memory().total
    logger.info(
        f"Start threshold={THRESHOLD*100:.1f}% grace={GRACE_SEC:.1f}s "
        f"interval={INTERVAL_SEC:.1f}s dry_run={DRY_RUN} "
        f"total_ram={total_ram/1024/1024:.1f} MiB"
    )

    # PID -> timestamp when it first exceeded the threshold
    exceed_since: dict[int, float] = {}

    while True:
        now = time.time()

        for proc in safe_process_iter():
            pid = proc.pid

            # Skip excluded and vanished processes early
            if is_excluded(proc):
                exceed_since.pop(pid, None)
                continue

            try:
                rss = proc.memory_info().rss
                pct = percent_of_total(rss, total_ram)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                exceed_since.pop(pid, None)
                continue

            if pct >= THRESHOLD:
                # First time crossing: start grace timer
                if pid not in exceed_since:
                    exceed_since[pid] = now
                    try:
                        logger.info(
                            f"PID {pid} ({proc.name()}) crossed {THRESHOLD*100:.1f}% "
                            f"— starting {GRACE_SEC:.1f}s grace (RSS={rss/1024/1024:.1f} MiB, "
                            f"{pct*100:.1f}%)"
                        )
                    except Exception:
                        pass

                # If continuously above for GRACE_SEC, terminate
                elif now - exceed_since[pid] >= GRACE_SEC:
                    try:
                        # Refresh metrics before action
                        proc = psutil.Process(pid)
                        cur_rss = proc.memory_info().rss
                        cur_pct = percent_of_total(cur_rss, total_ram)
                        if cur_pct >= THRESHOLD:
                            reason = (
                                f"exceeded {THRESHOLD*100:.1f}% for >= {GRACE_SEC:.1f}s "
                                f"(RSS={cur_rss/1024/1024:.1f} MiB, {cur_pct*100:.1f}%)"
                            )
                            kill_process(proc, reason)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                    finally:
                        exceed_since.pop(pid, None)
            else:
                # Dropped back below threshold — reset grace timer if present
                if pid in exceed_since:
                    try:
                        logger.info(
                            f"PID {pid} ({proc.name()}) fell below threshold — "
                            f"resetting grace timer"
                        )
                    except Exception:
                        pass
                    exceed_since.pop(pid, None)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    # Make sure SIGTERM cleanly stops the loop (systemd stop)
    signal.signal(signal.SIGTERM, lambda *_: exit(0))
    try:
        main()
    except KeyboardInterrupt:
        pass
