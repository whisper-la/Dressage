from __future__ import annotations

import asyncio
import os

from blackbox_server.core.command import execute_shell_command


def test_execute_shell_command_captures_output_and_returncode(tmp_path):
    async def run_test() -> None:
        result = await execute_shell_command(
            "printf out && printf err >&2 && exit 7",
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=5.0,
        )

        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.returncode == 7
        assert result.timed_out is False
        assert result.duration_seconds >= 0

    asyncio.run(run_test())


def test_execute_shell_command_times_out_and_kills_process_group(tmp_path):
    async def run_test() -> None:
        result = await execute_shell_command(
            "python -c 'import time; time.sleep(30)'",
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=0.05,
        )

        assert result.timed_out is True
        assert result.returncode is not None

    asyncio.run(run_test())


def test_execute_shell_command_accepts_scripts_larger_than_one_argv(tmp_path):
    async def run_test() -> None:
        command = "# padding to exceed Linux's per-argument limit\n" * 5000
        command += "printf large-command-ok"
        assert len(command.encode()) > 131072

        result = await execute_shell_command(
            command,
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=5.0,
        )

        assert result.stdout == "large-command-ok"
        assert result.stderr == ""
        assert result.returncode == 0
        assert result.timed_out is False

    asyncio.run(run_test())
