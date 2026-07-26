#!/usr/bin/env python3
"""Prepare and validate E2B templates for the public SWE-Gym recipe.

The commands in this utility keep E2B SDK implementation details out of the
experiment guide:

* ``list-images`` downloads or reads a SWE-Gym Parquet split and writes the
  unique task images required by that split.
* ``build`` creates one E2B template containing Claude Code and the isolated
  Blackbox Server runtime produced by
  ``dressage/recipes/swegym/prepare_claude_code_sandbox_artifacts.sh``.
* ``smoke`` resumes one template and verifies the Blackbox Server health
  endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from prepare_swegym_data import (
    EXPECTED_SPLIT_SIZES,
    download_split,
    load_parquet,
    registry_image_for_instance,
)

DEFAULT_CLAUDE_CODE_VERSION = "2.1.207"
DEFAULT_PORTABLE_PYTHON_VERSION = "3.10.20"
DEFAULT_BLACKBOX_PORT = 31000


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def list_images(args: argparse.Namespace) -> int:
    input_path = args.input
    if input_path is None:
        input_path = download_split(args.split, args.download_dir)

    instances = load_parquet(input_path)
    expected = EXPECTED_SPLIT_SIZES[args.split]
    if len(instances) != expected:
        raise RuntimeError(
            f"expected {expected} {args.split} rows, got {len(instances)}"
        )

    images = list(
        dict.fromkeys(
            registry_image_for_instance(
                str(instance["instance_id"]),
                str(instance["repo"]),
            )
            for instance in instances
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(f"{image}\n" for image in images),
        encoding="utf-8",
    )
    print(
        f"wrote {len(images)} unique images from {input_path} to {args.output}",
        file=sys.stderr,
    )
    return 0


def _required_text(
    parser: argparse.ArgumentParser,
    value: str | None,
    option: str,
    environment: str,
) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        parser.error(f"{option} is required (or set {environment})")
    return normalized


def _artifact_names(args: argparse.Namespace) -> tuple[str, str]:
    claude_binary = f"claude-{args.claude_code_version}-linux-x64"
    bbs_runtime = (
        "claude-code-runtime-python-"
        f"{args.portable_python_version}-bbs-{args.bbs_version}.tar.gz"
    )
    return claude_binary, bbs_runtime


def create_template(args: argparse.Namespace) -> Any:
    try:
        from e2b import Template, wait_for_url
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "e2b is required; install the repository before building templates"
        ) from exc

    claude_binary, bbs_runtime = _artifact_names(args)
    for artifact in (claude_binary, bbs_runtime):
        artifact_path = args.artifact_dir / artifact
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing sandbox artifact: {artifact_path}")

    bbs_start = f"""
cd /testbed
export BBS_HOST=0.0.0.0
export BBS_PORT={args.blackbox_port}
export BBS_RUNTIME_ROOT=/tmp/blackbox_server
exec /opt/cc-runtime/python/bin/python3 -c \
  "import sys; sys.path.insert(0, '/opt/cc-runtime/bbs-site'); \
from blackbox_server.main import main; main()" \
  > /tmp/blackbox-server.log 2>&1
""".strip()

    return (
        Template(file_context_path=args.artifact_dir)
        .from_image(args.task_image)
        .set_user("root")
        .apt_install(["bash", "curl", "ca-certificates", "git", "patch", "tar"])
        .copy(
            claude_binary,
            "/usr/local/bin/claude",
            mode=0o755,
        )
        .copy(
            bbs_runtime,
            "/tmp/cc-runtime.tar.gz",
        )
        .run_cmd(
            "rm -rf /opt/cc-runtime && mkdir -p /opt/cc-runtime && "
            "tar -xzf /tmp/cc-runtime.tar.gz -C /opt/cc-runtime "
            "--strip-components=1 && "
            "/usr/local/bin/claude --version"
        )
        .set_start_cmd(
            bbs_start,
            wait_for_url(
                f"http://127.0.0.1:{args.blackbox_port}/health",
            ),
        )
    )


def build_template(args: argparse.Namespace) -> int:
    try:
        from e2b import Template, default_build_logger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "e2b is required; install the repository before building templates"
        ) from exc

    template = create_template(args)
    build = Template.build(
        template,
        args.template_name,
        cpu_count=args.cpu_count,
        memory_mb=args.memory_mb,
        skip_cache=args.skip_cache,
        on_build_logs=default_build_logger(),
    )
    print(build)
    return 0


async def _smoke_template(args: argparse.Namespace) -> None:
    try:
        from e2b import AsyncSandbox
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "e2b is required; install the repository before testing templates"
        ) from exc

    sandbox = await AsyncSandbox.create(
        template=args.template_name,
        timeout=args.sandbox_timeout,
    )
    try:
        print(f"blackbox_endpoint={sandbox.get_host(args.blackbox_port)}")
        result = await sandbox.commands.run(
            f"curl -sf http://127.0.0.1:{args.blackbox_port}/health",
            timeout=args.command_timeout,
        )
        output = str(result.stdout or "").strip()
        if output:
            print(output)
    finally:
        sandbox.kill()


def smoke_template(args: argparse.Namespace) -> int:
    asyncio.run(_smoke_template(args))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    images_parser = subparsers.add_parser(
        "list-images",
        help="Write the unique Docker task images required by a SWE-Gym split.",
    )
    source_group = images_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path)
    source_group.add_argument("--download", action="store_true")
    images_parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_SPLIT_SIZES),
        default="train",
    )
    images_parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/swegym-source"),
    )
    images_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/swegym-images.txt"),
    )
    images_parser.set_defaults(handler=list_images)

    build_parser = subparsers.add_parser(
        "build",
        help="Build one Claude Code SWE-Gym E2B template.",
    )
    build_parser.add_argument("--task-image", default=os.environ.get("TASK_IMAGE"))
    build_parser.add_argument(
        "--template-name",
        default=os.environ.get("TEMPLATE_NAME"),
    )
    build_parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=os.environ.get("CLAUDE_CODE_ARTIFACT_DIR"),
    )
    build_parser.add_argument(
        "--bbs-version",
        default=os.environ.get("CLAUDE_CODE_BBS_VERSION"),
    )
    build_parser.add_argument(
        "--claude-code-version",
        default=os.environ.get(
            "CLAUDE_CODE_VERSION",
            DEFAULT_CLAUDE_CODE_VERSION,
        ),
    )
    build_parser.add_argument(
        "--portable-python-version",
        default=os.environ.get(
            "CLAUDE_CODE_PORTABLE_PYTHON_VERSION",
            DEFAULT_PORTABLE_PYTHON_VERSION,
        ),
    )
    build_parser.add_argument(
        "--blackbox-port",
        type=_positive_int,
        default=DEFAULT_BLACKBOX_PORT,
    )
    build_parser.add_argument("--cpu-count", type=_positive_int, default=4)
    build_parser.add_argument("--memory-mb", type=_positive_int, default=8192)
    build_parser.add_argument("--skip-cache", action="store_true")
    build_parser.set_defaults(handler=build_template)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Resume one E2B template and check the Blackbox Server health endpoint.",
    )
    smoke_parser.add_argument(
        "--template-name",
        default=os.environ.get("TEMPLATE_NAME"),
    )
    smoke_parser.add_argument(
        "--blackbox-port",
        type=_positive_int,
        default=DEFAULT_BLACKBOX_PORT,
    )
    smoke_parser.add_argument(
        "--sandbox-timeout",
        type=_positive_int,
        default=300,
    )
    smoke_parser.add_argument(
        "--command-timeout",
        type=_positive_int,
        default=60,
    )
    smoke_parser.set_defaults(handler=smoke_template)

    args = parser.parse_args(argv)
    if args.command == "build":
        args.task_image = _required_text(
            build_parser,
            args.task_image,
            "--task-image",
            "TASK_IMAGE",
        )
        args.template_name = _required_text(
            build_parser,
            args.template_name,
            "--template-name",
            "TEMPLATE_NAME",
        )
        artifact_dir = _required_text(
            build_parser,
            str(args.artifact_dir or ""),
            "--artifact-dir",
            "CLAUDE_CODE_ARTIFACT_DIR",
        )
        args.artifact_dir = Path(artifact_dir)
        args.bbs_version = _required_text(
            build_parser,
            args.bbs_version,
            "--bbs-version",
            "CLAUDE_CODE_BBS_VERSION",
        )
    elif args.command == "smoke":
        args.template_name = _required_text(
            smoke_parser,
            args.template_name,
            "--template-name",
            "TEMPLATE_NAME",
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
