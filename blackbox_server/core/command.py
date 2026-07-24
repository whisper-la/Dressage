from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

from blackbox_server.core.models import ExecuteCmdResult, utcnow


ProcessHook = Callable[[asyncio.subprocess.Process], None]


class CommandStartError(Exception):
    pass


async def execute_shell_command(
    cmd: str,
    *,
    timeout: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    on_process_start: ProcessHook | None = None,
    on_process_end: ProcessHook | None = None,
) -> ExecuteCmdResult:
    started_at = utcnow()
    started = perf_counter()
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="blackbox-execute-",
            suffix=".sh",
            delete=False,
        ) as script:
            script.write(cmd)
            script_path = Path(script.name)
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-l",
            str(script_path),
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        if script_path is not None:
            script_path.unlink(missing_ok=True)
        raise CommandStartError(f"Failed to start execute_cmd process: {exc}") from exc

    if on_process_start is not None:
        on_process_start(proc)

    timed_out = False
    communicate_task = asyncio.create_task(proc.communicate())
    try:
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            await terminate_process_group(proc)
            stdout_bytes, stderr_bytes = await communicate_task
    finally:
        try:
            if on_process_end is not None:
                on_process_end(proc)
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)

    return ExecuteCmdResult(
        cmd=cmd,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        returncode=proc.returncode,
        timed_out=timed_out,
        duration_seconds=perf_counter() - started,
        started_at=started_at,
        finished_at=utcnow(),
    )


async def terminate_process_group(
    proc: asyncio.subprocess.Process | None,
    *,
    terminate_timeout: float = 2.0,
) -> None:
    if proc is None or proc.returncode is not None:
        return
    _signal_process_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=terminate_timeout)
        return
    except asyncio.TimeoutError:
        pass

    if proc.returncode is not None:
        return
    _signal_process_group(proc, signal.SIGKILL)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=terminate_timeout)


def _signal_process_group(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)
