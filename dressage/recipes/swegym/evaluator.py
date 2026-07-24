"""Fresh-sandbox SWE-Gym evaluation for blackbox rollouts."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import time
import uuid
import zlib
from pathlib import PurePosixPath
from typing import Any

import httpx

from dressage.config import proxy_url
from dressage.paddock.blackbox.execute_hooks import (
    BlackboxExecuteCmd,
    BlackboxExecuteCmdSchedule,
    execute_blackbox_cmds_for_stage,
    maybe_await,
)
from dressage.paddock.lifecycle import terminate_paddock_best_effort

logger = logging.getLogger(__name__)

REWARD_MARKER = "DRESSAGE_SWEGYM_REWARD_JSON="
_POLL_PENDING_MARKER = "__DRESSAGE_SWEGYM_EVAL_PENDING__"
_POLL_DONE_MARKER = "__DRESSAGE_SWEGYM_EVAL_DONE__="
_POLL_TRUNCATED_MARKER = "__DRESSAGE_SWEGYM_EVAL_TRUNCATED__"
_DEFAULT_POLL_OUTPUT_LIMIT = 65536
_METADATA_OUTPUT_LIMIT = 16384


def use_fresh_swegym_eval(metadata: dict[str, Any]) -> bool:
    if str(metadata.get("reward_fn") or "") != "swegym_harness_marker":
        return False
    value = metadata.get("swegym_fresh_eval")
    if value is None:
        value = os.environ.get("DRESSAGE_SWEGYM_FRESH_EVAL", "1")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


async def execute_fresh_swegym_eval(
    paddock: Any,
    source_state: Any,
    metadata: dict[str, Any],
    *,
    schedule: BlackboxExecuteCmdSchedule,
    source_session_id: str,
    instance_id: str,
    env_type: str | None,
    env_args: dict[str, Any],
    blackbox_type: str,
    backend_options: Any,
) -> None:
    """Extract the agent patch and run the configured hook in a clean sandbox."""

    workdir = str(metadata.get("workdir") or "/testbed")
    base_commit = str(metadata.get("base_commit") or "").strip()
    patch_timeout = _positive_float(
        metadata.get("swegym_patch_timeout_sec"),
        env_name="DRESSAGE_SWEGYM_PATCH_TIMEOUT_SEC",
        default=120.0,
    )
    eval_session_id = f"swegym-eval-{uuid.uuid4().hex}"
    eval_state = None
    metadata["swegym_eval"] = {
        "mode": "fresh_sandbox",
        "source_session_id": source_session_id,
        "eval_session_id": eval_session_id,
        "base_commit": base_commit or None,
    }

    try:
        patch_result = await _extract_source_patch_with_retry(
            paddock,
            source_state,
            metadata,
            session_id=source_session_id,
            cmd=_extract_patch_command(workdir, base_commit=base_commit),
            timeout=patch_timeout,
        )
        if _result_failed(patch_result):
            _record_failure(metadata, "patch_extract_failed", patch_result)
            return
        patch = _stdout(patch_result)
        metadata["swegym_eval"]["patch_bytes"] = len(patch.encode())
        if not patch.strip():
            _record_failure(metadata, "empty_generation")
            return
        prohibited_paths = _prohibited_patch_paths(patch)
        if prohibited_paths:
            metadata["swegym_eval"]["prohibited_paths"] = prohibited_paths
            _record_failure(metadata, "prohibited_test_modification")
            return

        eval_state = await maybe_await(
            paddock.init(eval_session_id, env_type, dict(env_args))
        )
        await maybe_await(
            paddock.register_agent(
                eval_state,
                instance_id=instance_id,
                session_id=eval_session_id,
                router_url=proxy_url(),
                blackbox_type=blackbox_type,
                backend_options=backend_options,
            )
        )
        apply_result = await maybe_await(
            paddock.execute_cmd(
                eval_state,
                session_id=eval_session_id,
                cmd=_apply_patch_command(
                    patch,
                    workdir,
                    base_commit=base_commit,
                ),
                timeout=patch_timeout,
            )
        )
        if _result_failed(apply_result):
            _record_failure(metadata, "failed_apply_patch", apply_result)
            return

        await _execute_after_agent_commands(
            paddock,
            eval_state,
            metadata,
            schedule=schedule,
            session_id=eval_session_id,
        )
    except (httpx.HTTPError, TimeoutError) as exc:
        logger.warning(
            "fresh SWE-Gym evaluation failed for instance=%s: %s",
            instance_id,
            exc,
            exc_info=True,
        )
        _record_failure(
            metadata,
            "fresh_eval_exception",
            error={
                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc),
            },
        )
    finally:
        if eval_state is not None:
            await terminate_paddock_best_effort(
                paddock,
                session_id=eval_session_id,
                env_args=dict(env_args),
            )


async def _execute_after_agent_commands(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    schedule: BlackboxExecuteCmdSchedule,
    session_id: str,
) -> None:
    """Run the long official grade without one long-lived HTTP request."""

    for command in schedule.get("after_agent", ()):
        if command.name == "swegym_harness":
            await _execute_polled_swegym_harness(
                paddock,
                state,
                metadata,
                session_id=session_id,
                command=command,
            )
            continue
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule={"before_agent": (), "after_agent": (command,)},
            session_id=session_id,
            stage="after_agent",
        )


async def _execute_polled_swegym_harness(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    session_id: str,
    command: BlackboxExecuteCmd,
) -> None:
    """Start the grade in the background and retrieve its result by polling."""

    timeout = command.timeout or 1260.0
    poll_interval = _positive_float(
        metadata.get("swegym_eval_poll_interval_sec"),
        env_name="DRESSAGE_SWEGYM_EVAL_POLL_INTERVAL_SEC",
        default=15.0,
    )
    token = uuid.uuid4().hex
    output_path = f"/tmp/dressage_swegym_eval_{token}.out"
    done_path = f"/tmp/dressage_swegym_eval_{token}.done"
    runner_path = f"/tmp/dressage_swegym_eval_{token}.sh"
    output_limit = _DEFAULT_POLL_OUTPUT_LIMIT
    runner = (
        f"{command.cmd}; rc=$?; "
        f'printf \'%s\\n\' "$rc" > {shlex.quote(done_path)}; exit "$rc"'
    )
    start_cmd = (
        f"rm -f {shlex.quote(output_path)} {shlex.quote(done_path)} "
        f"{shlex.quote(runner_path)}; "
        f"printf %s {shlex.quote(runner)} > {shlex.quote(runner_path)}; "
        f"nohup bash {shlex.quote(runner_path)} "
        f"> {shlex.quote(output_path)} 2>&1 < /dev/null & echo $!"
    )
    start_result = await maybe_await(
        paddock.execute_cmd(
            state,
            session_id=session_id,
            cmd=start_cmd,
            timeout=60.0,
        )
    )
    if _result_failed(start_result) or not _stdout(start_result).strip():
        raise RuntimeError(
            "failed to start background SWE-Gym evaluator: " f"result={start_result!r}"
        )

    poll_cmd = (
        f"if test -f {shlex.quote(done_path)}; then "
        f"rc=$(cat {shlex.quote(done_path)}); "
        f"printf '{_POLL_DONE_MARKER}%s\\n' \"$rc\"; "
        f"size=$(wc -c < {shlex.quote(output_path)}); "
        f'if test "$size" -gt {output_limit}; then '
        f"printf '{_POLL_TRUNCATED_MARKER}\\n'; "
        f"tail -c {output_limit} {shlex.quote(output_path)}; "
        f"else cat {shlex.quote(output_path)}; fi; "
        f"else printf '{_POLL_PENDING_MARKER}\\n'; fi"
    )
    deadline = time.monotonic() + timeout
    attempts = 0
    poll_errors: list[str] = []
    poll_output = ""
    while (remaining := deadline - time.monotonic()) > 0:
        attempts += 1
        try:
            poll_result = await maybe_await(
                paddock.execute_cmd(
                    state,
                    session_id=session_id,
                    cmd=poll_cmd,
                    timeout=min(60.0, remaining),
                )
            )
            if _result_failed(poll_result):
                raise RuntimeError(f"poll command failed: result={poll_result!r}")
            poll_output = _stdout(poll_result)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if not _is_retryable_request_error(exc):
                raise
            poll_errors.append(f"{type(exc).__module__}.{type(exc).__name__}: {exc}")
            logger.warning(
                "SWE-Gym evaluator poll failed; retrying: session_id=%s "
                "attempt=%s error=%s",
                session_id,
                attempts,
                exc,
            )
        else:
            if poll_output.startswith(_POLL_DONE_MARKER):
                break
            if not poll_output.startswith(_POLL_PENDING_MARKER):
                raise RuntimeError(
                    "unexpected background evaluator poll response: "
                    f"{poll_output[:500]!r}"
                )

        await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    else:
        raise TimeoutError(
            f"background SWE-Gym evaluator did not finish within {timeout:.1f}s; "
            f"poll_attempts={attempts} poll_errors={poll_errors[-5:]!r}"
        )

    header, _, stdout = poll_output.partition("\n")
    returncode = int(header[len(_POLL_DONE_MARKER) :].strip())
    truncated_prefix = _POLL_TRUNCATED_MARKER + "\n"
    stdout_truncated = stdout.startswith(truncated_prefix)
    if stdout_truncated:
        stdout = stdout[len(truncated_prefix) :]
    stdout, metadata_truncated = _truncated_metadata_text(
        stdout,
        limit=_METADATA_OUTPUT_LIMIT,
    )
    stdout_truncated = stdout_truncated or metadata_truncated
    diagnostics = {
        "background_polled": True,
        "poll_attempts": attempts,
        "poll_errors": poll_errors[-5:],
    }
    metadata.setdefault("execute_cmds", []).append(
        {
            "stage": "after_agent",
            "name": command.name,
            "cmd": command.cmd,
            "timeout": command.timeout,
            "required": command.required,
            "cmd_result": {
                "stdout": stdout,
                "stderr": "",
                "returncode": returncode,
                "timed_out": False,
                "stdout_truncated": stdout_truncated,
                **diagnostics,
            },
        }
    )
    metadata.setdefault("swegym_eval", {}).update(diagnostics)


def _truncated_metadata_text(value: Any, *, limit: int) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def _extract_patch_command(workdir: str, *, base_commit: str = "") -> str:
    wd = shlex.quote(workdir)
    # The reference run captures staged + unstaged + untracked changes.
    # Staging in the source runtime is harmless because it is destroyed after
    # rollout.
    command = f"cd {wd} && git add -A && git diff --cached --binary --submodule=diff"
    if base_commit:
        # Diff against the task's declared baseline instead of the image's
        # current HEAD/index. This includes agent-created commits and avoids
        # silently dropping committed work.
        command += f" {shlex.quote(base_commit)} --"
    return command


async def _extract_source_patch_with_retry(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    session_id: str,
    cmd: str,
    timeout: float,
) -> Any:
    """Retry transient source-sandbox conflicts before grading the patch."""

    max_attempts = _positive_int(
        metadata.get("swegym_patch_extract_max_attempts"),
        env_name="DRESSAGE_SWEGYM_PATCH_EXTRACT_MAX_ATTEMPTS",
        default=5,
    )
    retry_delay = _positive_float(
        metadata.get("swegym_patch_extract_retry_delay_sec"),
        env_name="DRESSAGE_SWEGYM_PATCH_EXTRACT_RETRY_DELAY_SEC",
        default=2.0,
    )
    errors: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            result = await maybe_await(
                paddock.execute_cmd(
                    state,
                    session_id=session_id,
                    cmd=cmd,
                    timeout=timeout,
                )
            )
        except Exception as exc:
            error = _source_patch_exception_diagnostics(exc)
            errors.append(error)
            metadata["swegym_eval"].update(
                {
                    "patch_extract_attempts": attempt,
                    "patch_extract_errors": errors[-5:],
                }
            )
            if attempt >= max_attempts or not error["retryable"]:
                raise
            delay = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "source patch extraction failed transiently; retrying: "
                "session_id=%s attempt=%s/%s delay=%.1fs status=%s body=%r error=%s",
                session_id,
                attempt,
                max_attempts,
                delay,
                error.get("status_code"),
                error.get("response_body"),
                error["message"],
            )
            await asyncio.sleep(delay)
            continue

        metadata["swegym_eval"].update(
            {
                "patch_extract_attempts": attempt,
                "patch_extract_errors": errors[-5:],
            }
        )
        return result

    raise AssertionError("unreachable source patch extraction retry state")


def _source_patch_exception_diagnostics(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    response_body = ""
    if response is not None:
        try:
            response_body = str(response.text or "")[-2000:]
        except Exception:
            response_body = "<unavailable>"
    retryable = _is_retryable_request_error(exc)
    return {
        "type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
        "status_code": status_code,
        "response_body": response_body,
        "retryable": retryable,
    }


def _is_retryable_request_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {409, 502, 503, 504} or isinstance(exc, httpx.TransportError)


def _apply_patch_command(
    patch: str,
    workdir: str,
    *,
    base_commit: str = "",
) -> str:
    patch_b64 = base64.b64encode(zlib.compress(patch.encode(), level=9)).decode()
    patch_path = "/tmp/dressage_swegym_model.patch"
    decoder = (
        "import base64,zlib,sys;"
        "open(sys.argv[1],'wb').write("
        "zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))"
    )
    decode_patch = (
        f"printf %s {shlex.quote(patch_b64)} | "
        f"python3 -c {shlex.quote(decoder)} {shlex.quote(patch_path)}"
    )
    prepare_worktree = f"cd {shlex.quote(workdir)}"
    if base_commit:
        # SWE-Gym images can contain build-time tracked/untracked changes.
        # Applying a patch that includes that baseline dirt onto another dirty
        # image conflicts (for example MONAI-6924 requirements-dev.txt). Reset
        # the disposable evaluation sandbox to the dataset baseline first.
        base = shlex.quote(base_commit)
        prepare_worktree += f" && git reset --hard {base} && git clean -fd"
    return " && ".join(
        (
            decode_patch,
            prepare_worktree,
            f"git apply --binary -v {shlex.quote(patch_path)}",
        )
    )


def _stdout(result: Any) -> str:
    if not isinstance(result, dict):
        raise TypeError(f"execute_cmd returned {type(result).__name__}; expected dict")
    stdout = result.get("stdout", "")
    if not isinstance(stdout, str):
        raise TypeError("execute_cmd result.stdout must be a string")
    return stdout


def _prohibited_patch_paths(patch: str) -> list[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            fields = shlex.split(line)
        except ValueError:
            continue
        if len(fields) < 4:
            continue
        for path in fields[2:4]:
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if _is_test_or_harness_path(path):
                paths.add(path)
    return sorted(paths)


def _is_test_or_harness_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip("/")
    parsed = PurePosixPath(normalized)
    lowered_parts = tuple(part.lower() for part in parsed.parts)
    basename = parsed.name.lower()
    return bool(
        any(part in {"test", "tests", "testing", "r2e_tests"} for part in lowered_parts)
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or basename
        in {
            "conftest.py",
            "pytest.ini",
            "tox.ini",
            "setup.cfg",
            "pyproject.toml",
        }
    )


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        raise TypeError(f"execute_cmd returned {type(result).__name__}; expected dict")
    return result.get("returncode") != 0 or result.get("timed_out") is True


def _record_failure(
    metadata: dict[str, Any],
    reason: str,
    result: Any = None,
    *,
    error: dict[str, Any] | None = None,
) -> None:
    diag = {
        "resolved": False,
        "failure_reason": reason,
        "harness": "swegym",
        "fresh_sandbox": True,
    }
    if isinstance(result, dict):
        diag["returncode"] = result.get("returncode")
        diag["timed_out"] = bool(result.get("timed_out"))
        stderr = str(result.get("stderr") or "")
        if stderr:
            diag["stderr_tail"] = stderr[-2000:]
    if error is not None:
        diag["error"] = error
    marker = REWARD_MARKER + json.dumps(diag, ensure_ascii=False, separators=(",", ":"))
    metadata.setdefault("execute_cmds", []).append(
        {
            "stage": "after_agent",
            "name": "swegym_harness",
            "cmd": "<fresh-sandbox-evaluator>",
            "timeout": None,
            "required": False,
            "cmd_result": {
                "stdout": marker,
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
            },
        }
    )
    metadata.setdefault("swegym_eval", {})["failure_reason"] = reason


def _positive_float(value: Any, *, env_name: str, default: float) -> float:
    raw = value if value not in (None, "") else os.environ.get(env_name, default)
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive number; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be positive; got {raw!r}")
    return parsed


def _positive_int(value: Any, *, env_name: str, default: int) -> int:
    raw = value if value not in (None, "") else os.environ.get(env_name, default)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive integer; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be positive; got {raw!r}")
    return parsed
