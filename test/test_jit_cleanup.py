"""
Tests for the JIT teardown path.

A ``jit`` killed by SIGTERM (e.g. ``timeout``) or crashing must reliably tear
down everything it started: the gkfs daemon (which holds a RAM-backed tmpfs
rootdir) and the whole ``predictor_jit`` worker pool. These tests exercise the
cleanup logic in isolation:

* ``kill_process_tree`` reaps a process *and its children*, is idempotent, and
  never raises when a PID is already gone or invalid;
* ``kill_process_tree`` never signals the JIT's own process group;
* ``shut_down`` routes through ``kill_process_tree``;
* ``hard_kill`` targets only the recorded PIDs (no blanket name matching);
* ``install_signal_handlers`` wires SIGINT, SIGTERM, and SIGHUP to the handler.

No gkfs daemon is ever spawned -- only short-lived ``sleep`` helpers.

Author: Ahmad Tarraf
Copyright (c) 2024-2026 TU Darmstadt, Germany
Date: Jul 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ftio.api.gekkoFs.jit import setup_helper
from ftio.api.gekkoFs.jit.setup_helper import (
    hard_kill,
    install_signal_handlers,
    kill_process_tree,
    shut_down,
)


def _pid_alive(pid: int) -> bool:
    """True while ``pid`` exists (signal 0 probes without killing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    """Poll until ``pid`` disappears or ``timeout`` elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.02)
    return not _pid_alive(pid)


def _spawn_tree_leader():
    """Start a session-leader shell that forks a child ``sleep`` and waits.

    Mirrors how ``execute_background`` launches components (start_new_session),
    so process.pid leads its own group and the child models a pool worker.
    Returns the ``subprocess.Popen`` handle and the child PID.
    """
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 60 & echo $! ; wait"],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(proc.stdout.readline().strip())
    return proc, child_pid


# --------------------------------------------------------------------------- #
# kill_process_tree
# --------------------------------------------------------------------------- #
def test_kill_process_tree_reaps_leader_and_child():
    proc, child_pid = _spawn_tree_leader()
    assert _pid_alive(proc.pid)
    assert _pid_alive(child_pid)

    kill_process_tree(proc.pid, signal.SIGTERM)

    assert _wait_gone(child_pid), "pool-worker child should be reaped with the group"
    # The leader is our direct child; wait() reaps it (a signalled-but-unreaped
    # process lingers as a zombie, so probe its exit status rather than its PID).
    assert proc.wait(timeout=5) is not None, "group leader should be reaped"


def test_kill_process_tree_is_idempotent():
    proc, child_pid = _spawn_tree_leader()
    kill_process_tree(proc.pid, signal.SIGTERM)
    assert _wait_gone(child_pid)
    proc.wait(timeout=5)
    # Second call on the now-dead group leader must not raise.
    kill_process_tree(proc.pid, signal.SIGTERM)


@pytest.mark.parametrize("pid", [0, None, "", 1, -5, "not-a-pid"])
def test_kill_process_tree_ignores_bogus_pids(pid):
    # Must never raise and never touch init (pid 1) or invalid PIDs.
    kill_process_tree(pid, signal.SIGTERM)


def test_kill_process_tree_missing_pid_does_not_raise():
    # A very high, almost-certainly-free PID: no process, no exception.
    kill_process_tree(2**22, signal.SIGTERM)


def test_kill_process_tree_never_signals_own_group(monkeypatch):
    """If the target's group equals the JIT's own group, skip killpg entirely."""
    own_group = os.getpgrp()

    killpg_calls = []
    kill_calls = []
    monkeypatch.setattr(os, "getpgid", lambda pid: own_group)
    monkeypatch.setattr(os, "getpgrp", lambda: own_group)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killpg_calls.append(pgid))
    monkeypatch.setattr(os, "kill", lambda pid, sig: kill_calls.append(pid))
    # No descendants so the fallback stays contained to the target pid.
    monkeypatch.setattr(setup_helper, "_descendant_pids", lambda pid: [])

    kill_process_tree(own_group, signal.SIGTERM)

    assert killpg_calls == [], "must not killpg the JIT's own group"
    assert kill_calls == [own_group], "falls back to signalling only the target pid"


def test_kill_process_tree_walks_descendants_when_not_group_leader(monkeypatch):
    """When pid is not a group leader, descendants are signalled individually."""
    signalled = []
    # pid 500 is not its own group leader (group 400), so killpg is skipped.
    monkeypatch.setattr(os, "getpgid", lambda pid: 400)
    monkeypatch.setattr(os, "getpgrp", lambda: 999)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signalled.append(("pg", pgid)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append(("kill", pid)))
    monkeypatch.setattr(setup_helper, "_descendant_pids", lambda pid: [600, 601])

    kill_process_tree(500, signal.SIGKILL)

    assert ("pg", 400) not in signalled
    assert ("kill", 600) in signalled
    assert ("kill", 601) in signalled
    assert ("kill", 500) in signalled


# --------------------------------------------------------------------------- #
# shut_down
# --------------------------------------------------------------------------- #
def test_shut_down_routes_through_kill_process_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(
        setup_helper, "kill_process_tree", lambda pid, sig: calls.append((pid, sig))
    )
    shut_down(SimpleNamespace(), "GEKKO", 4242)
    assert calls == [(4242, signal.SIGTERM)]


def test_shut_down_ignores_zero_pid(monkeypatch):
    # Real kill_process_tree short-circuits on a falsy pid without raising.
    shut_down(SimpleNamespace(), "GEKKO", 0)


# --------------------------------------------------------------------------- #
# hard_kill
# --------------------------------------------------------------------------- #
def _hard_kill_settings(**over):
    base = {
        "hard_kill": True,
        "cluster": False,
        "static_allocation": False,
        "job_id": 0,
        "gkfs_daemon_pid": 101,
        "gkfs_proxy_pid": 102,
        "gkfs_fuse_pid": 103,
        "ftio_pid": 104,
        "cargo_pid": 105,
        "app_pid": 106,
        "gkfs_hostfile": "",
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_hard_kill_targets_only_tracked_pids(monkeypatch):
    calls = []
    monkeypatch.setattr(
        setup_helper, "kill_process_tree", lambda pid, sig: calls.append((pid, sig))
    )
    # Guard against any accidental blanket process matching.
    monkeypatch.setattr(
        setup_helper.subprocess,
        "run",
        lambda *a, **k: pytest.fail("hard_kill must not shell out to ps/grep"),
    )

    hard_kill(_hard_kill_settings())

    killed = {pid for pid, _ in calls}
    assert killed == {101, 102, 103, 104, 105, 106}
    assert all(sig == signal.SIGKILL for _, sig in calls)


def test_hard_kill_disabled_is_a_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(
        setup_helper, "kill_process_tree", lambda pid, sig: calls.append(pid)
    )
    hard_kill(_hard_kill_settings(hard_kill=False))
    assert calls == []


# --------------------------------------------------------------------------- #
# install_signal_handlers
# --------------------------------------------------------------------------- #
def test_install_signal_handlers_covers_sigint_sigterm_sighup(monkeypatch):
    registered = {}
    monkeypatch.setattr(
        setup_helper.signal,
        "signal",
        lambda sig, handler: registered.__setitem__(sig, handler),
    )
    settings = SimpleNamespace()
    install_signal_handlers(settings)

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in registered


def test_install_signal_handlers_handler_invokes_handle_sigint(monkeypatch):
    registered = {}
    monkeypatch.setattr(
        setup_helper.signal,
        "signal",
        lambda sig, handler: registered.__setitem__(sig, handler),
    )
    seen = []
    monkeypatch.setattr(setup_helper, "handle_sigint", lambda s: seen.append(s))

    settings = SimpleNamespace()
    install_signal_handlers(settings)
    registered[signal.SIGTERM](signal.SIGTERM, None)

    assert seen == [settings]


# --------------------------------------------------------------------------- #
# End-to-end: a killed jit-like process leaves no orphaned subtree
# --------------------------------------------------------------------------- #
def test_sigterm_handler_tears_down_started_subtree(tmp_path):
    """A process that installs the handler and is sent SIGTERM must clean up.

    Models jit: a parent starts a background 'daemon' (session leader with a
    child, like the predictor pool), installs the signal handlers, then is hit
    with SIGTERM. On exit the started subtree must be gone.
    """
    pid_file = tmp_path / "pids.txt"
    script = f"""
import os, signal, subprocess, sys, time
sys.path.insert(0, {os.getcwd()!r})
from ftio.api.gekkoFs.jit.setup_helper import kill_process_tree

daemon = subprocess.Popen(
    ["bash", "-c", "sleep 60 & echo $! ; wait"],
    stdout=subprocess.PIPE, text=True, start_new_session=True,
)
child = daemon.stdout.readline().strip()
open({str(pid_file)!r}, "w").write(f"{{daemon.pid}} {{child}}")

def handler(signum, frame):
    kill_process_tree(daemon.pid, signal.SIGTERM)
    sys.exit(0)

signal.signal(signal.SIGTERM, handler)
print("ready", flush=True)
time.sleep(60)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "ready"
    daemon_pid, child_pid = (int(x) for x in pid_file.read_text().split())
    assert _pid_alive(daemon_pid)
    assert _pid_alive(child_pid)

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)

    assert _wait_gone(child_pid), "daemon child (pool worker) must be reaped"
    assert _wait_gone(daemon_pid), "background daemon must be reaped"
