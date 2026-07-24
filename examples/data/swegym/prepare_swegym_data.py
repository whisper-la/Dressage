#!/usr/bin/env python3
"""Convert the 293-task SWE-Gym Parquet into final Dressage agent JSONL.

The converter deliberately uses the harness commit pinned by the reference run to
build each repository-specific evaluation script.  The script is executed only
after the coding agent finishes and emits a compact reward marker parsed by
``dressage.recipes.swegym.reward``.

Claude Code backend options, sandbox bootstrap commands, and provider-specific
image mapping are applied in the same pass, so no second metadata-overlay
conversion is required.

Install the pinned harness when it is not already importable::

    pip install \
      'swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7'
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Callable

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"
SWEGYM_HARNESS_COMMIT = "16dd480cce9b27bf111a362d280881c6def5d2a7"
EXPECTED_SPLIT_SIZES = {"train": 293, "validation": 23}
REWARD_MARKER = "DRESSAGE_SWEGYM_REWARD_JSON="
GIT_SANITIZE_CMD_NAME = "swegym_git_sanitize"
DEFAULT_GIT_SANITIZE_TIMEOUT = 120
SANDBOX_PROVIDERS = ("custom", "e2b")
PERMISSION_MODES = (
    "acceptEdits",
    "auto",
    "bypassPermissions",
    "default",
    "dontAsk",
    "plan",
)

LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}

TEST_STATUS_IMPORT = "from swegym.harness.constants import TestStatus"
STANDALONE_TEST_STATUS = '''class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"'''

# This driver is appended to the exact log_parsers.py source from the pinned
# pinned SWE-Gym package.  The generated command is self-contained because the
# task images do not install the host-side swegym Python package.
SANDBOX_GRADING_DRIVER = r"""
import base64,json,sys,zlib

log_path,expected_b64,repo,eval_rc=sys.argv[1:]
content=open(log_path,encoding="utf-8",errors="replace").read()
expected=json.loads(zlib.decompress(base64.b64decode(expected_b64)).decode())
repo_key=repo.lower()
parser=MAP_REPO_TO_PARSER.get(repo_key)
if parser is None:
    statuses={}
    parser_error="missing_official_log_parser"
else:
    try:
        statuses=parser(content)
        parser_error=""
    except Exception as exc:
        statuses={}
        parser_error="official_log_parser_exception:"+type(exc).__name__+":"+str(exc)

passing={"PASSED","XFAIL"}
f2p=list(expected.get("FAIL_TO_PASS") or [])
p2p=list(expected.get("PASS_TO_PASS") or [])
f2p_ok=[case for case in f2p if statuses.get(case) in passing]
p2p_ok=[case for case in p2p if statuses.get(case) in passing]
resolved=len(f2p_ok)==len(f2p) and len(p2p_ok)==len(p2p)
diag={
    "resolved":resolved,
    "fail_to_pass_success":len(f2p_ok),
    "fail_to_pass_total":len(f2p),
    "pass_to_pass_success":len(p2p_ok),
    "pass_to_pass_total":len(p2p),
    "parsed_tests":len(statuses),
    "eval_returncode":int(eval_rc),
    "harness":"swegym",
    "log_parser_repo":repo_key,
}
if not resolved:
    if parser_error:
        diag["failure_reason"]=parser_error
    elif not statuses:
        diag["failure_reason"]="test_log_parse_failure"
    elif len(f2p_ok)!=len(f2p):
        diag["failure_reason"]="fail_to_pass_failure"
    else:
        diag["failure_reason"]="pass_to_pass_regression"
if not statuses:
    diag["test_log_tail"]=content[-4000:]
print("DRESSAGE_SWEGYM_REWARD_JSON="+json.dumps(diag,separators=(",",":")))
"""


def standalone_log_parser_source(official_source: str) -> str:
    """Make the pinned SWE-Gym log parser executable inside task images."""

    if TEST_STATUS_IMPORT not in official_source:
        raise RuntimeError(
            "unexpected SWE-Gym log_parsers.py: TestStatus import missing"
        )
    standalone = official_source.replace(TEST_STATUS_IMPORT, STANDALONE_TEST_STATUS, 1)
    if "MAP_REPO_TO_PARSER" not in standalone:
        raise RuntimeError("unexpected SWE-Gym log_parsers.py: parser map missing")
    # Several SWE-Gym images still use Python 3.7/3.8, while the pinned parser
    # annotates functions with PEP 585 built-in generics such as dict[str, str].
    # Postponing annotation evaluation keeps the exact parser implementation
    # executable in those older task environments.
    if not standalone.startswith("from __future__ import annotations\n"):
        standalone = "from __future__ import annotations\n" + standalone
    return standalone.rstrip() + "\n\n" + SANDBOX_GRADING_DRIVER


def format_prompt(problem_statement: str) -> str:
    return (
        problem_statement.strip() + "\n\nIntegrity rules:\n"
        "- Do not modify tests, pytest configuration, or evaluation harness files.\n"
        "- Do not retrieve a solution, patch, commit, or pull request from the "
        "task repository or its mirrors."
    )


def load_sandbox_image_map(path: Path | None) -> dict[str, str] | None:
    """Load a raw task-image to provider-image mapping."""

    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sandbox image map {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"sandbox image map must be a JSON object: {path}")
    mapping: dict[str, str] = {}
    for raw_source, raw_target in value.items():
        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source or not target:
            raise ValueError(
                f"sandbox image map contains an empty key or value: {path}"
            )
        mapping[source] = target
    if not mapping:
        raise ValueError(f"sandbox image map is empty: {path}")
    return mapping


def claude_code_backend_options(
    *,
    max_turns: int,
    permission_mode: str,
    working_directory: str,
) -> dict[str, Any]:
    """Return the Claude Code options used by this recipe."""

    return {
        "working_directory": working_directory,
        "max_turns": int(max_turns),
        "permission_mode": permission_mode,
        "setting_sources": "user",
        "system_prompt_mode": "append",
        "compaction": {"auto": True},
        "compat": {
            "disable_prompt_caching": True,
            "disable_nonessential_traffic": True,
        },
        "subagents": {"enabled": False},
    }


def _claude_install_command(workdir: str) -> str:
    script = (
        "set -euo pipefail; "
        "(curl -fsSL https://claude.ai/install.sh | bash -s stable || "
        "npm install -g @anthropic-ai/claude-code); "
        'CLAUDE_CODE_BIN="$(command -v claude || true)"; '
        'if [ -z "$CLAUDE_CODE_BIN" ] && [ -x "$HOME/.local/bin/claude" ]; then '
        'CLAUDE_CODE_BIN="$HOME/.local/bin/claude"; fi; '
        'if [ -z "$CLAUDE_CODE_BIN" ] && [ -x "$HOME/.claude/local/claude" ]; then '
        'CLAUDE_CODE_BIN="$HOME/.claude/local/claude"; fi; '
        'test -n "$CLAUDE_CODE_BIN"; '
        'ln -sf "$CLAUDE_CODE_BIN" /usr/local/bin/claude; '
        "/usr/local/bin/claude --version"
    )
    return f"cd {shlex.quote(workdir)} && bash -lc " + shlex.quote(script)


def wheel_bootstrap(wheel_url: str, *, workdir: str) -> list[str]:
    """Install Claude Code and BBS into a compatible custom sandbox."""

    return [
        _claude_install_command(workdir),
        "pip install --no-cache-dir " + shlex.quote(wheel_url),
        "env BBS_PORT=31000 "
        "python -m blackbox_server.main > /tmp/blackbox-server.log 2>&1",
    ]


def packaged_bootstrap(
    binary_url: str,
    runtime_url: str,
    *,
    workdir: str,
) -> list[str]:
    """Start Claude Code and BBS from immutable portable artifacts."""

    script = (
        "set -euo pipefail; "
        "curl --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 2 -fsSL "
        + shlex.quote(binary_url)
        + " -o /usr/local/bin/claude; "
        "chmod 0755 /usr/local/bin/claude; "
        "rm -rf /opt/cc-runtime; mkdir -p /opt/cc-runtime; "
        "curl --connect-timeout 15 --max-time 120 --retry 3 --retry-delay 2 -fsSL "
        + shlex.quote(runtime_url)
        + " | tar -xz -C /opt/cc-runtime --strip-components=1; "
        "/usr/local/bin/claude --version; "
        "export BBS_PORT=31000; "
        "exec /opt/cc-runtime/python/bin/python3 -c "
        + shlex.quote(
            "import sys; "
            "sys.path.insert(0, '/opt/cc-runtime/bbs-site'); "
            "from blackbox_server.main import main; main()"
        )
        + " > /tmp/blackbox-server.log 2>&1"
    )
    return [f"cd {shlex.quote(workdir)} && bash -lc " + shlex.quote(script)]


def build_sandbox_cmd(
    *,
    bbs_wheel_url: str,
    binary_url: str,
    runtime_url: str,
    workdir: str,
) -> list[str] | None:
    if binary_url or runtime_url:
        if not binary_url or not runtime_url:
            raise ValueError("--binary-url and --runtime-url must be set together")
        return packaged_bootstrap(binary_url, runtime_url, workdir=workdir)
    if bbs_wheel_url:
        return wheel_bootstrap(bbs_wheel_url, workdir=workdir)
    return None


def registry_image_for_instance(instance_id: str, repo: str) -> str:
    """Return the image naming convention used by the reference run."""

    owner, repo_with_issue = instance_id.split("__", 1)
    image_repo, issue_id = repo_with_issue.rsplit("-", 1)
    if repo in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{image_repo}-{issue_id}:latest"
    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            normalized[key] = json.loads(value)
    required = (
        "instance_id",
        "repo",
        "problem_statement",
        "base_commit",
        "version",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    )
    missing = [key for key in required if normalized.get(key) is None]
    if missing:
        raise ValueError(
            f"SWE-Gym instance {normalized.get('instance_id', '?')} missing fields: "
            + ", ".join(missing)
        )
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if not isinstance(normalized[key], list):
            raise ValueError(f"{key} must be a list for {normalized['instance_id']}")
    return normalized


def load_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to read the SWE-Gym Parquet") from exc

    rows = []
    for row in pq.read_table(path).to_pylist():
        instance = row.get("instance")
        if not isinstance(instance, dict):
            raise ValueError(f"row in {path} does not contain an instance object")
        rows.append(_normalize_instance(instance))
    return rows


def download_split(split: str, output_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required for --download") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=DATASET_NAME,
            filename=f"{split}.parquet",
            repo_type="dataset",
        )
    )
    target = output_dir / f"{split}.parquet"
    if cached.resolve() != target.resolve():
        target.write_bytes(cached.read_bytes())
    return target


def _activate_pinned_package_root(package_root: Path | None) -> None:
    """Expose a pinned checkout without executing swegym's heavy top-level import."""

    if package_root is None:
        return
    package_dir = package_root.resolve() / "swegym"
    if not package_dir.is_dir():
        raise RuntimeError(f"invalid SWE-Gym package root: {package_root}")

    loaded = sys.modules.get("swegym")
    if loaded is not None:
        if str(package_dir) not in getattr(loaded, "__path__", ()):
            raise RuntimeError(
                "a different swegym package is already imported; "
                f"expected {package_dir}"
            )
        return

    # Avoid swegym/__init__.py: it imports optional Docker/collection packages
    # that are irrelevant to this converter.
    package = types.ModuleType("swegym")
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = "swegym"
    package.__path__ = [str(package_dir)]
    sys.modules["swegym"] = package


def load_test_spec_builder(
    package_root: Path | None = None,
) -> Callable[[dict[str, Any]], Any]:
    _activate_pinned_package_root(package_root)
    try:
        from swegym.harness.test_spec import make_test_spec
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pinned swegym harness is required; install SWE-Bench-Package "
            f"at commit {SWEGYM_HARNESS_COMMIT} or pass --swegym-package-root"
        ) from exc
    return make_test_spec


def load_official_log_parser_source(package_root: Path | None = None) -> str:
    """Load and make self-contained the pinned repo-specific parser map."""

    _activate_pinned_package_root(package_root)
    try:
        from swegym.harness import log_parsers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pinned swegym log parsers are required; install "
            "SWE-Bench-Package or pass --swegym-package-root"
        ) from exc
    source_path = Path(str(log_parsers.__file__))
    return standalone_log_parser_source(source_path.read_text(encoding="utf-8"))


def build_eval_command(
    *,
    eval_script: str,
    instance: dict[str, Any],
    timeout: int,
    log_parser_source: str,
) -> str:
    script_b64 = base64.b64encode(zlib.compress(eval_script.encode(), level=9)).decode()
    parser_b64 = base64.b64encode(
        zlib.compress(log_parser_source.encode(), level=9)
    ).decode()
    expected_b64 = base64.b64encode(
        zlib.compress(
            json.dumps(
                {
                    "FAIL_TO_PASS": instance["FAIL_TO_PASS"],
                    "PASS_TO_PASS": instance["PASS_TO_PASS"],
                },
                separators=(",", ":"),
            ).encode(),
            level=9,
        )
    ).decode()
    instance_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance["instance_id"]))
    eval_path = f"/tmp/dressage_swegym_eval_{instance_slug}.sh"
    log_path = f"/tmp/dressage_swegym_eval_{instance_slug}.log"
    decoder = (
        "import base64,zlib,sys;"
        "open(sys.argv[1],'wb').write(zlib.decompress(base64.b64decode(sys.argv[2])))"
    )
    parser = (
        "import base64,zlib;exec(zlib.decompress(base64.b64decode("
        + repr(parser_b64)
        + ")))"
    )
    return (
        f"python3 -c {shlex.quote(decoder)} {shlex.quote(eval_path)} {shlex.quote(script_b64)}; "
        f"chmod 0700 {shlex.quote(eval_path)}; "
        f"timeout {int(timeout)} bash {shlex.quote(eval_path)} >{shlex.quote(log_path)} 2>&1; "
        "RC=$?; "
        f"python3 -c {shlex.quote(parser)} {shlex.quote(log_path)} "
        f"{shlex.quote(expected_b64)} {shlex.quote(str(instance['repo']))} \"$RC\""
    )


def build_git_sanitizer_command(
    *,
    base_commit: str,
    workdir: str,
    timeout: int = DEFAULT_GIT_SANITIZE_TIMEOUT,
) -> dict[str, Any]:
    """Build the before-agent cleanup that removes task-history leakage."""

    target = str(base_commit).strip()
    if not target:
        raise ValueError("base_commit is required for SWE-Gym git sanitization")

    quoted_workdir = shlex.quote(str(workdir).strip() or "/testbed")
    script = "; ".join(
        [
            "set -e",
            f"cd {quoted_workdir}",
            "if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
            "echo 'SWE-Gym git sanitize failed: not a git worktree' >&2; exit 2; fi",
            f"TARGET={shlex.quote(target)}",
            "git config --global --add safe.directory "
            + quoted_workdir
            + " >/dev/null 2>&1 || true",
            'if ! git cat-file -e "$TARGET^{commit}" >/dev/null 2>&1; then '
            'echo "SWE-Gym git sanitize failed: missing base commit $TARGET" >&2; '
            "exit 2; fi",
            'RESOLVED=$(git rev-parse "$TARGET^{commit}")',
            'git reset --hard "$RESOLVED"',
            # Keep ignored dependencies/caches intact while removing any
            # untracked source or test artifacts baked into the task image.
            "git clean -ffd",
            'git checkout --detach "$RESOLVED"',
            'for remote in $(git remote); do git remote remove "$remote" || true; done',
            # With a detached HEAD, every named ref is unnecessary for solving
            # the task and may retain a future fix, stash, tag, or alternate
            # branch. Delete them uniformly before expiring reflogs.
            "git for-each-ref --format='delete %(refname)' | git update-ref --stdin",
            "git reflog expire --expire=now --expire-unreachable=now --all",
            "git gc --prune=now",
            "git prune --expire=now",
            'test "$(git rev-parse HEAD)" = "$RESOLVED"',
            'test -z "$(git status --porcelain --untracked-files=all)"',
            'echo "SWE-Gym git sanitize completed at $RESOLVED"',
        ]
    )
    return {
        "name": GIT_SANITIZE_CMD_NAME,
        "cmd": "bash -lc " + shlex.quote(script),
        "timeout": int(timeout),
        "required": True,
    }


def output_row(
    instance: dict[str, Any],
    *,
    split: str,
    make_test_spec: Callable[[dict[str, Any]], Any],
    workdir: str,
    blackbox_type: str,
    sandbox_image_map: dict[str, str] | None = None,
    sandbox_cmd: list[str] | None = None,
    claude_max_turns: int = 80,
    claude_permission_mode: str = "acceptEdits",
    test_timeout: int,
    command_timeout: int,
    sandbox_timeout: int,
    log_parser_source: str,
    git_sanitize_timeout: int = DEFAULT_GIT_SANITIZE_TIMEOUT,
) -> dict[str, Any]:
    # make_test_spec lowercases a few instance IDs internally. Keep the source
    # ID for image lookup and metadata, but use its official eval script.
    test_spec = make_test_spec(dict(instance))
    eval_script = str(test_spec.eval_script)
    instance_id = str(instance["instance_id"])
    repo = str(instance["repo"])
    base_commit = str(instance["base_commit"])
    git_sanitizer = build_git_sanitizer_command(
        base_commit=base_commit,
        workdir=workdir,
        timeout=git_sanitize_timeout,
    )
    docker_image = registry_image_for_instance(instance_id, repo)
    sandbox_image = docker_image
    if sandbox_image_map is not None:
        try:
            sandbox_image = sandbox_image_map[docker_image]
        except KeyError as exc:
            raise ValueError(
                f"sandbox image map has no entry for {docker_image!r} "
                f"(instance {instance_id})"
            ) from exc
    metadata: dict[str, Any] = {
        "agent_mode": "blackbox",
        "blackbox_type": blackbox_type,
        "instance_id": instance_id,
        "repo": repo,
        "repo_name": repo.rsplit("/", 1)[-1],
        "base_commit": base_commit,
        "commit_hash": base_commit,
        "workdir": workdir,
        "task_type": "swe-gym",
        "dataset": DATASET_NAME,
        "dataset_split": split,
        "reward_fn": "swegym_harness_marker",
        "swegym_harness_commit": SWEGYM_HARNESS_COMMIT,
        "swegym_eval_mode": "official_script_in_fresh_sandbox",
        "swegym_fresh_eval": True,
        "swegym_git_sanitizer": {
            "enabled": True,
            "target_commit": base_commit,
            "workdir": workdir,
        },
        "FAIL_TO_PASS": list(instance["FAIL_TO_PASS"]),
        "PASS_TO_PASS": list(instance["PASS_TO_PASS"]),
        "sandbox_image": sandbox_image,
        "sandbox_timeout_sec": sandbox_timeout,
        "blackbox_execute_cmds": {
            "before_agent": [git_sanitizer],
            "after_agent": [
                {
                    "name": "swegym_harness",
                    "cmd": build_eval_command(
                        eval_script=eval_script,
                        instance=instance,
                        timeout=test_timeout,
                        log_parser_source=log_parser_source,
                    ),
                    "timeout": command_timeout,
                    "required": False,
                }
            ],
        },
    }
    if sandbox_image != docker_image:
        metadata["docker_image"] = docker_image
    if blackbox_type == "claude_code":
        metadata["backend_options"] = claude_code_backend_options(
            max_turns=claude_max_turns,
            permission_mode=claude_permission_mode,
            working_directory=workdir,
        )
    if sandbox_cmd is not None:
        metadata["sandbox_cmd"] = list(sandbox_cmd)
    return {
        "prompt": [
            {
                "role": "user",
                "content": format_prompt(str(instance["problem_statement"])),
            }
        ],
        "label": "",
        "metadata": metadata,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--split", choices=tuple(EXPECTED_SPLIT_SIZES), default="train")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=Path("data/swegym_raw"))
    parser.add_argument("--swegym-package-root", type=Path)
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--blackbox-type", default="claude_code")
    parser.add_argument(
        "--provider",
        choices=SANDBOX_PROVIDERS,
        default="e2b",
        help="Sandbox provider targeted by metadata.sandbox_image.",
    )
    parser.add_argument(
        "--sandbox-image-map",
        type=Path,
        help="JSON object mapping raw Docker task images to provider image IDs.",
    )
    parser.add_argument(
        "--bbs-wheel-url",
        default=os.environ.get("CLAUDE_CODE_BBS_WHEEL_URL", ""),
        help="Optional BBS wheel URL for a compatible custom sandbox.",
    )
    parser.add_argument(
        "--binary-url",
        default=os.environ.get("CLAUDE_CODE_BINARY_URL", ""),
        help="Optional Claude Code binary URL; requires --runtime-url.",
    )
    parser.add_argument(
        "--runtime-url",
        default=os.environ.get("CLAUDE_CODE_RUNTIME_URL", ""),
        help="Optional portable BBS runtime URL; requires --binary-url.",
    )
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument(
        "--permission-mode",
        choices=PERMISSION_MODES,
        default="acceptEdits",
    )
    parser.add_argument("--test-timeout", type=int, default=1200)
    parser.add_argument("--command-timeout", type=int, default=1260)
    parser.add_argument("--sandbox-timeout", type=int, default=3600)
    parser.add_argument(
        "--git-sanitize-timeout",
        type=int,
        default=DEFAULT_GIT_SANITIZE_TIMEOUT,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.provider == "e2b":
        if args.sandbox_image_map is None:
            parser.error("--sandbox-image-map is required for --provider e2b")
        if args.bbs_wheel_url or args.binary_url or args.runtime_url:
            parser.error(
                "--provider e2b expects Claude Code and BBS in the prebuilt "
                "template; do not pass runtime bootstrap URLs"
            )
    if args.blackbox_type != "claude_code" and (
        args.bbs_wheel_url or args.binary_url or args.runtime_url
    ):
        parser.error("Claude Code bootstrap URLs require --blackbox-type claude_code")
    try:
        build_sandbox_cmd(
            bbs_wheel_url=args.bbs_wheel_url,
            binary_url=args.binary_url,
            runtime_url=args.runtime_url,
            workdir=args.workdir,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input is None:
        if not args.download:
            raise SystemExit("--input is required unless --download is used")
        args.input = download_split(args.split, args.download_dir)

    instances = load_parquet(args.input)
    expected = EXPECTED_SPLIT_SIZES[args.split]
    if len(instances) != expected:
        raise RuntimeError(
            f"expected {expected} {args.split} rows, got {len(instances)}"
        )
    if args.limit is not None:
        instances = instances[: args.limit]

    make_test_spec = load_test_spec_builder(args.swegym_package_root)
    log_parser_source = load_official_log_parser_source(args.swegym_package_root)
    sandbox_image_map = load_sandbox_image_map(args.sandbox_image_map)
    sandbox_cmd = build_sandbox_cmd(
        bbs_wheel_url=args.bbs_wheel_url,
        binary_url=args.binary_url,
        runtime_url=args.runtime_url,
        workdir=args.workdir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for instance in instances:
                row = output_row(
                    instance,
                    split=args.split,
                    make_test_spec=make_test_spec,
                    workdir=args.workdir,
                    blackbox_type=args.blackbox_type,
                    sandbox_image_map=sandbox_image_map,
                    sandbox_cmd=sandbox_cmd,
                    claude_max_turns=args.max_turns,
                    claude_permission_mode=args.permission_mode,
                    test_timeout=args.test_timeout,
                    command_timeout=args.command_timeout,
                    sandbox_timeout=args.sandbox_timeout,
                    git_sanitize_timeout=args.git_sanitize_timeout,
                    log_parser_source=log_parser_source,
                )
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"wrote {len(instances)} {args.split} rows from {args.input} to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
