#!/usr/bin/env python3
"""Restore Dressage-Claw data and generate Dressage-compatible JSONL.

Modes:
  --parquet PATH   Restore a local archive, defaulting to its parent directory.
  --output DIR     Download the configured Hugging Face archive into DIR.
  --dataset-dir PATH
                   Convert an existing local dataset directory.

Parquet modes restore the dataset tree first, then reuse the existing local
directory-to-JSONL conversion. The output file is ``dressage_claw_e2b.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---- config ----

# Replace this placeholder with the Hugging Face dataset repo ID or dataset
# URL after upload. This is the only Hugging Face source configuration.
HF_DATASET_SOURCE = "https://huggingface.co/datasets/huang3eng/Dressage-Claw"
HF_ARCHIVE_PATH = "archive/Dressage-Claw.parquet"

# LLM Judge defaults (CLI > env > these values)
_JUDGE_DEFAULTS = {
    "OMNI_CLAW_EVAL_ENABLE_JUDGE": "1",
    "OMNI_CLAW_EVAL_JUDGE_API_KEY": "",
    "OMNI_CLAW_EVAL_JUDGE_MODEL": "gpt-4o",
    "OMNI_CLAW_EVAL_JUDGE_BASE_URL": "https://api.openai.com/v1",
}

# MCP server config
MCP_SERVER_NAME = "claw-tools"
# Use single-quoted JSON so shell vars inside are literal
_MCP_CONFIG_JSON = (
    '{"command":"python3",'
    '"args":["/tmp/omni_task/runtime/claw_tools_mcp.py"],'
    '"env":{"OMNI_TOOL_SPECS_DIR":"/tmp/omni_task/tools",'
    '"OMNI_TRACE_DIR":"/tmp/omni_task/trace",'
    '"NO_PROXY":"127.0.0.1,localhost,0.0.0.0,::1",'
    '"no_proxy":"127.0.0.1,localhost,0.0.0.0,::1"},'
    '"cwd":"/tmp/omni_task"}'
)

# Resolve BBS runtime paths inside the E2B sandbox
_FIND_OC_HOME = (
    'OC_HOME=$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home '
    "2>/dev/null | head -1)"
)
_FIND_OC_WS = (
    'OC_WS=$(ls -d "${BBS_RUNTIME_ROOT:-/workspace_sandbox/blackbox_server_runtime}"/bbs-*/home/.openclaw/workspace '
    "2>/dev/null | head -1)"
)

MCP_REGISTER_CMD = (
    f"{_FIND_OC_HOME}; "
    'HOME="$OC_HOME" openclaw mcp set ' + MCP_SERVER_NAME + " " + "'" + _MCP_CONFIG_JSON + "'"
)
GENERATE_TRACE_CMD = "python3 /tmp/omni_task/runtime/gen_trace.py"
EXPOSE_TOOL_INSTRUCTIONS_CMD = (
    "OMNI_TOOL_SPECS_DIR=/tmp/omni_task/tools "
    "python3 /tmp/omni_task/runtime/claw_tools_mcp.py --append-agents-md"
)
PATCH_LLM_JUDGE_CMD = "python3 /tmp/omni_task/runtime/patch_llm_judge.py /tmp/omni_grader"

# Runtime helper scripts that live outside the Claw dataset.
RUNTIME_HELPER_MCP = "claw_tools_mcp.py"
RUNTIME_HELPER_TRACE = "gen_trace.py"
RUNTIME_HELPER_PATCH = "patch_llm_judge.py"

# Copy workspace files into OpenClaw's internal workspace (the only shell
# bridge left: the BBS runtime dir is created at register time and its name
# is only discoverable from inside the sandbox)
MAP_WORKSPACE_CMD_TEMPLATE = (
    "mkdir -p __WS_TARGET__; "
    f"{_FIND_OC_WS}; "
    'if [ -n "$OC_WS" ] && [ -n "$(ls -A __WS_TARGET__ 2>/dev/null)" ]; then '
    'cp -r __WS_TARGET__/* "$OC_WS/"; fi; '
    "ls -la __WS_TARGET__/"
)

# Exclude OS junk and caches from file injection

_EXCLUDED_NAMES = {".DS_Store", "__pycache__"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


# ---- data structures ----


@dataclass
class TaskInfo:
    task_id: str
    path: str
    dataset: str
    subset: str
    original_id: str

    prompt_text: str = ""
    workspace_mount_path: str = "/root"
    has_services: bool = False
    has_llm_judge: bool = False
    has_tools: bool = False
    has_system_prompt: bool = False
    needs_patch: bool = False

    files_task_mappings: list[dict[str, Any]] = field(default_factory=list)
    grader_files_mappings: list[dict[str, Any]] = field(default_factory=list)

    grader_run: str = "grader/run.sh"
    grader_cwd: str = "/tmp/omni_grader"
    grader_timeout: float = 300.0
    setup_script: str | None = None
    start_services_script: str | None = None
    stop_services_script: str | None = None

    tools_list: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ---- parsing ----


def load_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[error] manifest.json not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("tasks") or []


def _detect_patch_dependency(task_dir: Path, task_subdir: Path) -> bool:
    import re

    pattern = re.compile(r"""\bpatch\b""")
    scan_dirs = [task_dir / "grader", task_subdir / "scripts"]
    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for fpath in scan_dir.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix not in (".py", ".sh", ".bash"):
                continue
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                if pattern.search(content):
                    return True
            except OSError:
                continue
    return False


def _validate_task_file(task_dir: Path, relative_path: str, task_id: str) -> None:
    full_path = task_dir / relative_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"task {task_id}: file referenced in task.json not found: {relative_path} "
            f"(expected at {full_path})"
        )


def parse_task(
    dataset_dir: Path, entry: dict[str, Any], warnings: list[str] | None = None
) -> TaskInfo:
    task_dir = dataset_dir / entry["path"]
    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        raise FileNotFoundError(
            f"task {entry['task_id']}: task.json not found at {task_json_path}"
        )

    task = json.loads(task_json_path.read_text(encoding="utf-8"))
    task_id = entry["task_id"]

    # Non-grader files live in task/ subdir
    task_subdir = task_dir / "task"
    prompt_ref = task.get("prompt", "prompt.md")
    prompt_path = task_subdir / prompt_ref
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"task {task_id}: prompt file not found: {prompt_ref} "
            f"(expected at {prompt_path})"
        )
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    runtime = task.get("runtime") or {}
    workspace_mount = runtime.get("workspace_mount_path", "/root")

    scripts = task.get("scripts") or {}
    setup_script = scripts.get("setup")
    start_services = scripts.get("start_services")
    stop_services = scripts.get("stop_services")
    if setup_script:
        _validate_task_file(task_subdir, setup_script, task_id)
    if start_services:
        _validate_task_file(task_subdir, start_services, task_id)
    if stop_services:
        _validate_task_file(task_subdir, stop_services, task_id)

    has_services = bool(start_services)

    files_task = (task.get("files") or {}).get("task") or []
    files_task_mappings: list[dict[str, Any]] = []
    for ft in files_task:
        source = ft.get("source", "")
        target = ft.get("target", "")
        copy_mode = ft.get("copy_mode", "contents")
        required = ft.get("required", False)
        if not source or not target:
            continue
        source_dir = task_subdir / source
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"task {task_id}: source directory not found: {source} "
                f"(expected at {source_dir})"
            )
        files_task_mappings.append({
            "source": source,
            "target": target,
            "copy_mode": copy_mode,
            "required": required,
        })

    grader = task.get("grader") or {}
    items = grader.get("items") or []
    has_llm_judge = any(
        i.get("type") in ("llm_judge", "llm-judge") for i in items
    )

    grader_files = grader.get("files") or []
    grader_files_mappings: list[dict[str, Any]] = []
    for gf in grader_files:
        source = gf.get("source", "")
        target = gf.get("target", "")
        copy_mode = gf.get("copy_mode", "contents")
        required = gf.get("required", False)
        if not source or not target:
            continue
        # grader source is relative to task_dir (grader/ stays at task root)
        source_dir = task_dir / source
        if not source_dir.exists():
            raise FileNotFoundError(
                f"task {task_id}: grader source not found: {source} "
                f"(expected at {source_dir})"
            )
        grader_files_mappings.append({
            "source": source,
            "target": target,
            "copy_mode": copy_mode,
            "required": required,
        })

    tools_list = task.get("tools") or []
    has_tools = len(tools_list) > 0
    if has_tools:
        for tool_entry in tools_list:
            if isinstance(tool_entry, str):
                _validate_task_file(task_subdir, tool_entry, task_id)
        # Check that tools/ directory files match declared tools
        tools_dir = task_subdir / "tools"
        if tools_dir.is_dir():
            actual_tool_files = {f.name for f in tools_dir.glob("*.json")}
            declared_tool_files = {
                Path(entry).name if isinstance(entry, str) else entry.get("name", "")
                for entry in tools_list
            }
            declared_tool_files = {f for f in declared_tool_files if f}
            # Warn about mismatches (not fatal — extra files are harmless)
            undeclared = actual_tool_files - declared_tool_files
            if undeclared:
                msg = (
                    f"task {task_id}: tools/ has {len(undeclared)} files not "
                    f"declared in task.json: {sorted(undeclared)}"
                )
                if warnings is not None:
                    warnings.append(msg)
                else:
                    print(f"[warn] {msg}", file=sys.stderr)

    spp_field = runtime.get("system_prompt_prefix_file")
    has_system_prompt = bool(spp_field)
    if has_system_prompt:
        _validate_task_file(task_subdir, spp_field, task_id)

    needs_patch = _detect_patch_dependency(task_dir, task_subdir)

    grader_run = grader.get("run", "grader/run.sh")
    grader_cwd = grader.get("cwd", "/tmp/omni_grader")
    timeouts = task.get("timeouts") or {}
    grader_timeout = float(timeouts.get("grader_seconds", 300))

    return TaskInfo(
        task_id=task_id,
        path=entry["path"],
        dataset=entry.get("dataset", ""),
        subset=entry.get("subset", ""),
        original_id=entry.get("original_id") or task_id,
        prompt_text=prompt_text,
        workspace_mount_path=workspace_mount,
        has_services=has_services,
        has_llm_judge=has_llm_judge,
        has_tools=has_tools,
        has_system_prompt=has_system_prompt,
        needs_patch=needs_patch,
        files_task_mappings=files_task_mappings,
        grader_files_mappings=grader_files_mappings,
        grader_run=grader_run,
        grader_cwd=grader_cwd,
        grader_timeout=grader_timeout,
        setup_script=setup_script,
        start_services_script=start_services,
        stop_services_script=stop_services,
        tools_list=[t if isinstance(t, str) else t.get("name", "") for t in tools_list],
        tags=task.get("tags") or [],
    )


# ---- file collection ----


def _iter_injectable_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        if any(part in _EXCLUDED_NAMES for part in fpath.parts):
            continue
        if fpath.suffix in _EXCLUDED_SUFFIXES:
            continue
        files.append(fpath)
    return files


RUNTIME_HELPERS = (RUNTIME_HELPER_MCP, RUNTIME_HELPER_TRACE, RUNTIME_HELPER_PATCH)


def detect_runtime_helpers(task_dir: Path) -> set[str]:
    task_runtime = task_dir / "task" / "runtime"
    return {
        helper for helper in RUNTIME_HELPERS if (task_runtime / helper).is_file()
    }


def _mapping_target(
    mappings: list[dict[str, Any]], source: str, default: str
) -> str:
    """Final mount target for a files.task mapping source."""
    for mapping in mappings:
        if mapping.get("source") == source and mapping.get("target"):
            return mapping["target"]
    return default


def collect_before_agent_files(
    task_dir: Path, task: TaskInfo
) -> list[dict[str, str]]:
    """Write each files.task source dir straight to its declared target."""
    entries: list[dict[str, str]] = []
    task_subdir = task_dir / "task"
    for mapping in task.files_task_mappings:
        source = mapping["source"]
        target = mapping["target"].rstrip("/")
        copy_mode = mapping.get("copy_mode", "contents")
        source_dir = task_subdir / source
        for fpath in _iter_injectable_files(source_dir):
            rel = fpath.relative_to(source_dir).as_posix()
            if copy_mode == "file":
                dest = f"{target}/{source}/{rel}"
            else:
                dest = f"{target}/{rel}"
            entries.append({"path": dest, "local_path": str(fpath.resolve())})
    return entries


def collect_after_agent_files(
    task_dir: Path, task: TaskInfo
) -> list[dict[str, str]]:
    """Write grader files straight to their declared targets (post-agent)."""
    entries: list[dict[str, str]] = []
    for mapping in task.grader_files_mappings:
        source = mapping["source"]
        target = mapping["target"].rstrip("/")
        copy_mode = mapping.get("copy_mode", "contents")
        source_dir = task_dir / source
        for fpath in _iter_injectable_files(source_dir):
            rel = fpath.relative_to(source_dir).as_posix()
            if copy_mode == "file":
                dest = f"{target}/{source}/{rel}"
            else:
                dest = f"{target}/{rel}"
            entries.append({"path": dest, "local_path": str(fpath.resolve())})
    return entries


# ---- command builders ----


def _build_judge_exports(args: argparse.Namespace) -> str:
    cli_values = {
        "OMNI_CLAW_EVAL_JUDGE_API_KEY": args.judge_api_key,
        "OMNI_CLAW_EVAL_JUDGE_MODEL": args.judge_model,
        "OMNI_CLAW_EVAL_JUDGE_BASE_URL": args.judge_base_url,
    }
    env_vars: dict[str, str] = {}
    for name, default in _JUDGE_DEFAULTS.items():
        value = cli_values.get(name) or os.environ.get(name) or default
        if value:
            env_vars[name] = value
    return "".join(
        f"export {k}={shlex.quote(str(v))}; " for k, v in env_vars.items()
    )


def build_before_agent_cmds(
    task: TaskInfo, runtime_helpers: set[str]
) -> list[dict[str, Any]]:
    cmds: list[dict[str, Any]] = []

    scripts_target = _mapping_target(
        task.files_task_mappings, "scripts", "/tmp/omni_task/scripts"
    )

    # workspace is the only mapping that still needs a shell bridge: files are
    # written to the declared target directly, then mirrored into OpenClaw's
    # dynamic runtime workspace.
    for mapping in task.files_task_mappings:
        if mapping["source"] != "workspace":
            continue
        cmds.append({
            "name": "map_workspace",
            "cmd": MAP_WORKSPACE_CMD_TEMPLATE.replace("__WS_TARGET__", mapping["target"]),
            "required": mapping.get("required", False),
            "timeout": 15.0,
        })

    if task.needs_patch:
        cmds.append({
            "name": "ensure_patch",
            "cmd": "which patch || (apt-get update -qq && apt-get install -y -qq patch)",
            "required": True,
            "timeout": 60.0,
        })

    if task.setup_script:
        cmds.append({
            "name": "setup_task",
            "cmd": f"bash {scripts_target}/setup.sh",
            "required": True,
            "timeout": 120.0,
        })

    if task.start_services_script:
        cmds.append({
            "name": "start_services",
            "cmd": f"bash {scripts_target}/start_services.sh",
            "required": True,
            "timeout": 60.0,
        })

    if task.has_tools and RUNTIME_HELPER_MCP in runtime_helpers:
        cmds.append({
            "name": "register_mcp",
            "cmd": MCP_REGISTER_CMD,
            "required": True,
            "timeout": 120.0,
        })
        cmds.append({
            "name": "expose_tool_instructions",
            "cmd": EXPOSE_TOOL_INSTRUCTIONS_CMD,
            "required": True,
            "timeout": 15.0,
        })

    return cmds


def build_after_agent_cmds(
    task: TaskInfo,
    args: argparse.Namespace,
    runtime_helpers: set[str],
) -> list[dict[str, Any]]:
    cmds: list[dict[str, Any]] = []

    scripts_target = _mapping_target(
        task.files_task_mappings, "scripts", "/tmp/omni_task/scripts"
    )

    if task.has_llm_judge and RUNTIME_HELPER_PATCH in runtime_helpers:
        cmds.append({
            "name": "patch_llm_judge",
            "cmd": PATCH_LLM_JUDGE_CMD,
            "required": True,
            "timeout": 30.0,
        })

    if task.has_tools and RUNTIME_HELPER_TRACE in runtime_helpers:
        cmds.append({
            "name": "generate_trace",
            "cmd": GENERATE_TRACE_CMD,
            "required": True,
            "timeout": 15.0,
        })

    if task.stop_services_script:
        cmds.append({
            "name": "stop_services",
            "cmd": f"bash {scripts_target}/stop_services.sh",
            "required": True,
            "timeout": 30.0,
        })

    grader_cmd = (
        f"{_FIND_OC_WS}; "
        f'[ -n "$OC_WS" ] && cp -r "$OC_WS/"* {task.workspace_mount_path}/ 2>/dev/null; '
        f"{_build_judge_exports(args) if task.has_llm_judge and not args.no_judge else ''}"
        f"bash {task.grader_cwd}/run.sh"
    )
    cmds.append({
        "name": "run_grader",
        "cmd": grader_cmd,
        "required": True,
        "timeout": task.grader_timeout,
    })

    return cmds


# ---- JSONL generation ----

ARCHIVE_PARQUET_COLUMNS = frozenset(
    {"path", "kind", "content", "mode", "sha256"}
)
RESTORE_WORKERS = min(16, max(4, (os.cpu_count() or 4) * 2))
RESTORE_BATCH_SIZE = 128


def restore_parquet_to_local(
    parquet_paths: list[Path], output_dir: Path,
) -> int:
    """Restore a Parquet dataset archive to its original local tree."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required to read Parquet; install it with `pip install pyarrow`"
        ) from exc

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    directory_entries: list[tuple[Path, int]] = []
    file_entries: list[tuple[Path, bytes, int, str, Path]] = []

    for parquet_path in sorted(parquet_paths):
        table = pq.read_table(
            parquet_path, columns=["path", "kind", "content", "mode", "sha256"]
        )
        columns = table.to_pydict()
        for relative_path, kind, content, mode, expected_sha256 in zip(
            columns["path"], columns["kind"], columns["content"],
            columns["mode"], columns["sha256"]
        ):
            if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
                raise ValueError(f"{parquet_path}: unsafe archive path {relative_path!r}")
            target = (output_dir / relative_path).resolve()
            if output_dir not in target.parents:
                raise ValueError(f"{parquet_path}: archive path escapes output dir: {relative_path}")
            if kind == "directory":
                directory_entries.append((target, int(mode) & 0o777))
            elif kind == "file":
                if not isinstance(content, (bytes, bytearray)):
                    raise ValueError(f"{parquet_path}: content for {relative_path} is not binary")
                file_entries.append(
                    (
                        target,
                        bytes(content),
                        int(mode) & 0o777,
                        str(expected_sha256 or ""),
                        parquet_path,
                    )
                )
            else:
                raise ValueError(f"{parquet_path}: unsupported archive entry kind {kind!r}")

    total = len(directory_entries) + len(file_entries)
    restored = 0
    last_progress_at = 0.0

    def show_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and now - last_progress_at < 0.1:
            return
        percent = restored * 100 // total if total else 100
        bar_len = 30
        filled = bar_len * percent // 100
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stderr.write(
            f"\rRestoring: [{bar}] {restored}/{total} ({percent}%)"
        )
        sys.stderr.flush()
        last_progress_at = now

    # Create every directory before concurrent file writes. Apply directory
    # modes after writing so read-only directories do not block child files.
    for target, _ in directory_entries:
        target.mkdir(parents=True, exist_ok=True)
        restored += 1
        show_progress()

    def write_batch(
        batch: list[tuple[Path, bytes, int, str, Path]],
    ) -> int:
        for target, content, mode, expected_sha256, parquet_path in batch:
            actual = hashlib.sha256(content).hexdigest()
            if expected_sha256 and actual != expected_sha256:
                raise ValueError(
                    f"{parquet_path}: checksum mismatch for {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(mode)
        return len(batch)

    batches = [
        file_entries[index:index + RESTORE_BATCH_SIZE]
        for index in range(0, len(file_entries), RESTORE_BATCH_SIZE)
    ]
    if batches:
        with ThreadPoolExecutor(max_workers=RESTORE_WORKERS) as executor:
            futures = [executor.submit(write_batch, batch) for batch in batches]
            for future in as_completed(futures):
                restored += future.result()
                show_progress()

    for target, mode in sorted(
        directory_entries,
        key=lambda item: len(item[0].parts),
        reverse=True,
    ):
        target.chmod(mode)

    if total:
        restored = total
    show_progress(force=True)
    sys.stderr.write("\n")
    return restored


def find_parquet_files(path: Path) -> list[Path]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required to inspect Parquet files; install it with "
            "`pip install pyarrow`"
        ) from exc

    if path.is_file() and path.suffix == ".parquet":
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.parquet"))
    else:
        raise FileNotFoundError(f"no .parquet files found under {path}")

    archive_files: list[Path] = []
    skipped: list[Path] = []
    for candidate in candidates:
        columns = set(pq.ParquetFile(candidate).schema_arrow.names)
        if ARCHIVE_PARQUET_COLUMNS.issubset(columns):
            archive_files.append(candidate)
        else:
            skipped.append(candidate)

    for candidate in skipped:
        print(
            f"[restore] skipping non-archive Parquet: {candidate}",
            file=sys.stderr,
        )
    if not archive_files:
        raise ValueError(
            f"no Dressage-Claw archive Parquet found under {path}; required "
            f"columns: {sorted(ARCHIVE_PARQUET_COLUMNS)}"
        )
    return archive_files


def download_parquet(repo_id_or_url: str, revision: str, cache_dir: Path | None) -> list[Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "huggingface_hub is required to download the configured dataset; install it with "
            "`pip install huggingface_hub`"
        ) from exc
    repo_id = repo_id_or_url
    if "://" in repo_id_or_url:
        parsed = urlparse(repo_id_or_url)
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"} or len(parts) < 3 or parts[0] != "datasets":
            raise ValueError(
                "HF_DATASET_SOURCE must be a Hugging Face dataset URL, "
                "for example https://huggingface.co/datasets/org/name"
            )
        repo_id = "/".join(parts[1:3])
        if len(parts) >= 5 and parts[3] in {"resolve", "blob"}:
            revision = parts[4]

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=[HF_ARCHIVE_PATH],
    )
    return find_parquet_files(Path(local_dir) / HF_ARCHIVE_PATH)


def task_to_jsonl_record(
    task: TaskInfo,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_dir = dataset_dir / task.path
    runtime_helpers = detect_runtime_helpers(task_dir)

    metadata: dict[str, Any] = {
        "instance_id": f"claw-{task.original_id}".replace("_", "-"),
        "task_id": task.task_id,
        "blackbox_type": args.blackbox_type,
        "sandbox_image": args.sandbox_image,
        "sandbox_timeout_sec": args.sandbox_timeout_sec,
    }

    if task.has_system_prompt:
        runtime_target = _mapping_target(
            task.files_task_mappings, "runtime", "/tmp/omni_task/runtime"
        )
        metadata["system_prompt_file"] = f"{runtime_target}/system_prompt_prefix.txt"

    before_agent_files = collect_before_agent_files(task_dir, task)
    after_agent_files = collect_after_agent_files(task_dir, task)
    if before_agent_files:
        metadata["before_agent_files"] = before_agent_files
    if after_agent_files:
        metadata["after_agent_files"] = after_agent_files

    metadata["blackbox_execute_cmds"] = {
        "before_agent": build_before_agent_cmds(task, runtime_helpers),
        "after_agent": build_after_agent_cmds(task, args, runtime_helpers),
    }

    if args.sandbox_cmd:
        metadata["sandbox_cmd"] = [args.sandbox_cmd]

    extra: dict[str, Any] = {}
    if args.e2b_envs_json:
        extra["e2b_envs"] = json.loads(args.e2b_envs_json)
    if args.e2b_metadata_json:
        extra["e2b_metadata"] = json.loads(args.e2b_metadata_json)
    if extra:
        metadata["sandbox_extra_params"] = extra

    return {
        "prompt": task.prompt_text,
        "label": "",
        "reward_fn": "claw_grader",
        "metadata": metadata,
    }


# ---- CLI ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore Dressage-Claw from Parquet and generate its JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to Claw dataset directory (must contain manifest.json and tasks/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory; for local Parquet, defaults to its parent directory",
    )
    parser.add_argument("--blackbox-type", default="openclaw", help="Blackbox agent type")
    parser.add_argument("--sandbox-image", default="e2b-dressage-claw-blackbox", help="E2B template name")
    parser.add_argument("--sandbox-timeout-sec", type=int, default=3600)
    parser.add_argument("--sandbox-cmd", default=None, help="Optional shell command at sandbox creation")

    parser.add_argument("--e2b-envs-json", help="JSON for e2b_envs in sandbox_extra_params")
    parser.add_argument("--e2b-metadata-json", help="JSON for e2b_metadata in sandbox_extra_params")

    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default=None)

    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Randomly sample N tasks",
    )
    parser.add_argument("--no-require-prompt", action="store_true", help="Include tasks even without prompt.md")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing output")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge even for tasks that use it")
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Read one Parquet file or a directory of Parquet files and restore the local dataset tree",
    )
    args = parser.parse_args(argv)

    if args.parquet and args.dataset_dir:
        parser.error("--parquet and --dataset-dir are mutually exclusive")

    if args.parquet or args.dataset_dir is None:
        if args.dry_run:
            parser.error("--dry-run is only supported for local dataset conversion")

        if args.parquet:
            parquet_source = args.parquet.resolve()
            parquet_paths = find_parquet_files(parquet_source)
            if args.output:
                output_dir = args.output
            elif parquet_source.is_file():
                output_dir = parquet_source.parent
            else:
                output_dir = parquet_source
        else:
            if args.output is None:
                parser.error("--output is required when downloading from Hugging Face")
            output_dir = args.output
            parquet_paths = download_parquet(HF_DATASET_SOURCE, "main", None)

        count = restore_parquet_to_local(parquet_paths, output_dir.resolve())
        print(f"[restore] read {len(parquet_paths)} Parquet file(s)")
        print(f"[restore] restored {count} files to {output_dir.resolve()}")
        jsonl_args = [
            "--dataset-dir", str(output_dir.resolve()),
            "--output", str(output_dir.resolve()),
            "--blackbox-type", args.blackbox_type,
            "--sandbox-image", args.sandbox_image,
            "--sandbox-timeout-sec", str(args.sandbox_timeout_sec),
        ]
        if args.sandbox_cmd:
            jsonl_args.extend(["--sandbox-cmd", args.sandbox_cmd])
        if args.e2b_envs_json:
            jsonl_args.extend(["--e2b-envs-json", args.e2b_envs_json])
        if args.e2b_metadata_json:
            jsonl_args.extend(["--e2b-metadata-json", args.e2b_metadata_json])
        if args.judge_api_key:
            jsonl_args.extend(["--judge-api-key", args.judge_api_key])
        if args.judge_base_url:
            jsonl_args.extend(["--judge-base-url", args.judge_base_url])
        if args.judge_model:
            jsonl_args.extend(["--judge-model", args.judge_model])
        if args.limit is not None:
            jsonl_args.extend(["--limit", str(args.limit)])
        if args.no_require_prompt:
            jsonl_args.append("--no-require-prompt")
        if args.no_judge:
            jsonl_args.append("--no-judge")
        return main(jsonl_args)

    if args.dataset_dir is None:
        parser.error("dataset_dir is required unless --parquet is used")
    dataset_dir: Path = args.dataset_dir.resolve()

    if not dataset_dir.is_dir():
        print(f"[error] dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    output_dir = args.output or Path("examples/data") / f"dressage_claw_{dataset_dir.name}"
    output_path = output_dir / "dressage_claw_e2b.jsonl"
    print(f"[converter] dataset: {dataset_dir.name}")
    print(f"[converter] output: {output_path}")

    manifest_tasks = load_manifest(dataset_dir)
    print(f"[converter] manifest contains {len(manifest_tasks)} tasks")

    parsed: list[TaskInfo] = []
    skipped_no_prompt = 0
    conversion_warnings: list[str] = []
    for entry in manifest_tasks:
        info = parse_task(dataset_dir, entry, warnings=conversion_warnings)
        if not args.no_require_prompt and not info.prompt_text:
            skipped_no_prompt += 1
            continue
        parsed.append(info)

    print(f"[converter] parsed {len(parsed)} tasks "
          f"(skipped: {skipped_no_prompt} no prompt)")
    if conversion_warnings:
        print(f"[converter] {len(conversion_warnings)} tools mismatch warnings "
              "(see conversion log)")
    else:
        print("[converter] no tools mismatch warnings")

    # Warn about tasks missing required runtime helpers
    helper_sets = {
        t.task_id: detect_runtime_helpers(dataset_dir / t.path)
        for t in parsed
    }
    tool_no_mcp = sum(
        1 for t in parsed
        if t.has_tools and RUNTIME_HELPER_MCP not in helper_sets[t.task_id]
    )
    if tool_no_mcp:
        print(f"[warn] {tool_no_mcp} tasks declare MCP tools but "
              f"{RUNTIME_HELPER_MCP} is unavailable; MCP commands skipped",
              file=sys.stderr)
    judge_no_patch = sum(
        1 for t in parsed
        if t.has_llm_judge and RUNTIME_HELPER_PATCH not in helper_sets[t.task_id]
    )
    if judge_no_patch:
        print(f"[warn] {judge_no_patch} tasks use LLM judge but "
              f"{RUNTIME_HELPER_PATCH} is unavailable; patch step skipped",
              file=sys.stderr)

    filtered = parsed
    if args.limit is not None and args.limit > 0:
        filtered = random.sample(parsed, min(args.limit, len(parsed)))
    print(f"[converter] selected {len(filtered)} tasks")

    ds_counter = Counter(t.dataset for t in filtered)
    print(f"[converter] dataset breakdown: {dict(ds_counter)}")
    svc_count = sum(1 for t in filtered if t.has_services)
    llm_count = sum(1 for t in filtered if t.has_llm_judge)
    tool_count = sum(1 for t in filtered if t.has_tools)
    sp_count = sum(1 for t in filtered if t.has_system_prompt)
    print(f"[converter] features: services={svc_count} llm_judge={llm_count} "
          f"tools={tool_count} system_prompt={sp_count}")

    if llm_count and not args.no_judge and not (
        args.judge_api_key or os.environ.get("OMNI_CLAW_EVAL_JUDGE_API_KEY")
    ):
        print("[warn] LLM judge tasks present but no judge API key configured; "
              "set --judge-api-key or OMNI_CLAW_EVAL_JUDGE_API_KEY",
              file=sys.stderr)

    if args.dry_run:
        print("[converter] --dry-run: not writing output")
        for t in filtered[:10]:
            print(f"  {t.task_id} ({t.dataset}/{t.subset})")
        if len(filtered) > 10:
            print(f"  ... and {len(filtered) - 10} more")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(filtered)
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, task in enumerate(filtered, 1):
            record = task_to_jsonl_record(task, dataset_dir, args)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            pct = idx * 100 // total if total else 100
            bar_len = 30
            filled = bar_len * pct // 100
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stderr.write(f"\rConverting: [{bar}] {idx}/{total} ({pct}%)")
            sys.stderr.flush()

    sys.stderr.write("\n")
    print(f"[converter] wrote {len(filtered)} records to {output_path}")
    print("[converter] ensure the dataset directory is accessible from training nodes",
          file=sys.stderr)

    log_path = output_path.with_suffix(".convert.log")
    with log_path.open("w", encoding="utf-8") as logfh:
        logfh.write(f"Conversion log for {dataset_dir.name}\n")
        logfh.write(f"Output: {output_path}\n")
        logfh.write(f"Total tasks: {len(manifest_tasks)}, parsed: {len(parsed)}, "
                    f"filtered: {len(filtered)}, skipped: {skipped_no_prompt}\n")
        logfh.write(f"Warnings: {len(conversion_warnings)}\n")
        logfh.write("\n" + "=" * 80 + "\n")
        if conversion_warnings:
            logfh.write("Tools mismatch warnings (tools/ has files not declared in task.json):\n\n")
            for msg in conversion_warnings:
                logfh.write(f"  [warn] {msg}\n")
        else:
            logfh.write("No warnings.\n")
    print(f"[converter] conversion log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
