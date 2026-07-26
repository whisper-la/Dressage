"""Build an E2B dressage_claw blackbox template with an isolated BBS runtime.

The template replicates the dressage_claw sandbox runtime: python3, curl, patch,
openclaw, BlackboxServer (started on port 31000), and the Python packages the
mock services need (fastapi/uvicorn/pydantic/httpx/openai).

Usage::

    E2B_API_KEY=e2b_... \
    BUILD_FROM_PUBLIC_IMAGE=1 \
    TASK_IMAGE='python:3.11-slim-bookworm' \
    TEMPLATE_NAME='e2b-dressage-claw-blackbox' \
    python3 examples/data/dressage_claw/build_e2b_blackbox_template.py

Only ``blackbox_server`` (the HTTP adapter) is installed into the isolated
venv — the full ``dressage`` package is not needed inside the sandbox since
it runs on the training machine.

Environment variables baked into the template:
    BBS_HOST=0.0.0.0          BlackboxServer bind address
    BBS_PORT=31000            BlackboxServer listen port
    BBS_RUNTIME_ROOT=/workspace_sandbox/blackbox_server_runtime
                              Runtime directory root

Optional build-time variables:
    BBS_PYTHON_VERSION        Python version for the BBS venv (default: 3.11)
    E2B_TEMPLATE_CPU_COUNT    Template CPU cores (default: 2)
    E2B_TEMPLATE_MEMORY_MB    Template memory in MB (default: 4096)
    OPENCLAW_VERSION          openclaw version for public-image mode
"""

from __future__ import annotations

import os
from pathlib import Path

from e2b import Template, default_build_logger, wait_for_port

try:
    # E2B renamed this exception in different Python SDK releases.  Keep the
    # builder usable with both the older SDK used by the Dressage images and
    # the current SDK.
    from e2b import TemplateBuildError as _E2BBuildError
except ImportError:  # pragma: no cover - depends on the installed E2B SDK
    from e2b import BuildException as _E2BBuildError


DEFAULT_BBS_PYTHON_VERSION = "3.11"


def _env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def build_template() -> object:
    task_image = _env("TASK_IMAGE")
    template_name = _env("TEMPLATE_NAME")
    python_version = _env("BBS_PYTHON_VERSION", default=DEFAULT_BBS_PYTHON_VERSION)
    from_public = os.environ.get("BUILD_FROM_PUBLIC_IMAGE") == "1"
    openclaw_version = _env("OPENCLAW_VERSION", default="2026.6.6")

    if not Path("blackbox_server").is_dir():
        raise SystemExit(
            "run this script from the Dressage repository root "
            "(expected ./blackbox_server)"
        )

    template = (
        Template(file_context_path=".")
        .from_image(task_image)
        .set_user("root")
        .set_envs(
            {
                # Match the local bwrap runner's blackbox environment.  The
                # E2B sandbox has no supervisor slot, but these names are
                # intentionally kept stable for task/setup scripts.
                "HOME": "/home/blackbox",
                "XDG_CONFIG_HOME": "/home/blackbox/.config",
                "XDG_CACHE_HOME": "/home/blackbox/.cache",
                "TMPDIR": "/tmp",
                # The official python:slim image has a second Debian
                # /usr/bin/python3 without pip. Keep /usr/local first for
                # task-side `python3` after uv installs its dependencies.
                "PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
                "OPENCLAW_BIN": "/usr/local/bin/openclaw",
                "BBS_HOST": "0.0.0.0",
                "BBS_PORT": "31000",
                "BBS_RUNTIME_ROOT": "/workspace_sandbox/blackbox_server_runtime",
                "DRESSAGE_BLACKBOX_RUNTIME_ROOT": "/workspace_sandbox/blackbox_server_runtime",
                "DRESSAGE_BLACKBOX_SLOT_ID": "e2b",
                "DRESSAGE_BLACKBOX_SLOT_DIR": "/workspace_sandbox",
                "DRESSAGE_BLACKBOX_SLOT_GENERATION": "0",
                "DRESSAGE_BLACKBOX_SLOT_PORT": "31000",
                "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID": "e2b-template",
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "void",
            }
        )
    )
    if from_public:
        # Install openclaw inside the template build (normally baked into
        # the base image by docker/build.sh).
        template = template.run_cmd(
            "install -d -m 0777 /home/blackbox /workspace_sandbox /data && "
            "touch /home/blackbox/.bashrc && "
            "apt-get update && "
            "apt-get install -y --no-install-recommends bash ca-certificates curl patch && "
            "rm -rf /var/lib/apt/lists/* && "
            "curl -fsSL https://openclaw.ai/install.sh | "
            f"bash -s -- --version {openclaw_version} --no-onboard && "
            'ln -sf "$(command -v openclaw)" /usr/local/bin/openclaw'
        )
    template = (
        template
        .run_cmd(
            "apt-get update && "
            "apt-get install -y --no-install-recommends bash ca-certificates curl patch && "
            "rm -rf /var/lib/apt/lists/*"
        )
        # Create an isolated venv for BlackboxServer so the task image's
        # Python environment stays untouched. Use uv for both the isolated BBS
        # environment and task dependencies: slim images commonly omit pip.
        .run_cmd(
            "install -d -m 0777 /home/blackbox /workspace_sandbox /data "
            "/workspace_sandbox/blackbox_server_runtime && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            f"/home/blackbox/.local/bin/uv python install {python_version} && "
            f"/home/blackbox/.local/bin/uv venv --python {python_version} "
            "/opt/dressage-bbs-venv && "
            "/home/blackbox/.local/bin/uv pip install "
            "--python /opt/dressage-bbs-venv/bin/python setuptools wheel && "
            "/home/blackbox/.local/bin/uv pip install --system --break-system-packages "
            "--python \"$(command -v python3)\" "
            "fastapi uvicorn pydantic httpx openai"
        )
        .copy(".", "/root/Dressage")
        .run_cmd(
            "/home/blackbox/.local/bin/uv pip install "
            "--python /opt/dressage-bbs-venv/bin/python "
            "--no-cache-dir --no-build-isolation "
            "/root/Dressage/blackbox_server"
        )
        # Verify openclaw is reachable from PATH (it ships in the task image).
        .run_cmd(
            "openclaw_resolved=\"$(command -v openclaw)\" && "
            "ln -sf \"$openclaw_resolved\" /usr/local/bin/openclaw && "
            "test -x /usr/local/bin/openclaw || "
            "{ echo 'ERROR: openclaw not found in PATH'; exit 1; } && "
            "/usr/local/bin/openclaw --version"
        )
        .set_start_cmd(
            "install -d -m 0777 /home/blackbox /workspace_sandbox /data "
            "/workspace_sandbox/blackbox_server_runtime && "
            "cd /workspace_sandbox && "
            "BBS_HOST=0.0.0.0 BBS_PORT=31000 "
            "BBS_RUNTIME_ROOT=/workspace_sandbox/blackbox_server_runtime "
            "DRESSAGE_BLACKBOX_RUNTIME_ROOT=/workspace_sandbox/blackbox_server_runtime "
            "DRESSAGE_BLACKBOX_SLOT_ID=e2b DRESSAGE_BLACKBOX_SLOT_DIR=/workspace_sandbox "
            "DRESSAGE_BLACKBOX_SLOT_GENERATION=0 DRESSAGE_BLACKBOX_SLOT_PORT=31000 "
            "DRESSAGE_BLACKBOX_SUPERVISOR_RUN_ID=e2b-template "
            "HOME=/home/blackbox XDG_CONFIG_HOME=/home/blackbox/.config "
            "XDG_CACHE_HOME=/home/blackbox/.cache TMPDIR=/tmp "
            "OPENCLAW_BIN=/usr/local/bin/openclaw "
            "/opt/dressage-bbs-venv/bin/python -m blackbox_server.main",
            wait_for_port(31000),
        )
    )

    try:
        return Template.build(
            template,
            template_name,
            cpu_count=int(os.environ.get("E2B_TEMPLATE_CPU_COUNT", "2")),
            memory_mb=int(os.environ.get("E2B_TEMPLATE_MEMORY_MB", "4096")),
            on_build_logs=default_build_logger(),
        )
    except _E2BBuildError as exc:
        raise SystemExit(f"failed to build {template_name}: {exc}") from exc


if __name__ == "__main__":
    info = build_template()
    print({"docker_image": os.environ["TASK_IMAGE"], "template": info.name})
