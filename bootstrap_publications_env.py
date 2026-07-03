#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tardis.utils import find_project_root


def log(message: str) -> None:
    print(message, file=sys.stderr)


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_checked(cmd: list[str]) -> None:
    # Keep stdout reserved for emitted shell exports so bash eval stays parse-safe.
    subprocess.run(cmd, check=True, stdout=sys.stderr, stderr=sys.stderr)


def shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap publications Python environment.")
    parser.add_argument("--emit-shell", action="store_true", help="Emit shell exports on stdout.")
    args = parser.parse_args()

    self_dir = Path(__file__).resolve().parent

    try:
        root_dir = find_project_root(self_dir)
    except FileNotFoundError as exc:
        log(f"[PUB] error: {exc}")
        return 1

    venv_dir = root_dir / "venv"
    if not venv_dir.is_dir():
        log(f"[PUB] creating venv at {venv_dir}")
        run_checked([sys.executable, "-m", "venv", str(venv_dir)])

    py_bin = venv_python_path(venv_dir)
    if not py_bin.is_file():
        log(f"[PUB] error: venv python not found at {py_bin}")
        return 1

    requirements = root_dir / "requirements.txt"
    run_checked([str(py_bin), "-m", "pip", "install", "--upgrade", "pip"])
    if requirements.is_file():
        run_checked([str(py_bin), "-m", "pip", "install", "-r", str(requirements)])
        log(f"[PUB] installed requirements from {requirements}")
    else:
        log(f"[PUB] requirements.txt not found at {requirements}; skipping dependency install")

    run_checked([str(py_bin), "-m", "pip", "install", "ipykernel"])

    kernel_name = os.getenv("PUBLICATIONS_KERNEL_NAME", "tardisproject-venv")
    kernel_display = os.getenv("PUBLICATIONS_KERNEL_DISPLAY_NAME", "Python (TardisProject venv)")
    run_checked(
        [
            str(py_bin),
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            kernel_name,
            "--display-name",
            kernel_display,
        ]
    )
    log(f"[PUB] registered Jupyter kernel: {kernel_display} ({kernel_name})")
    log(f"[PUB] root: {root_dir}")

    if args.emit_shell:
        print(shell_export("ROOT_DIR", str(root_dir)))
        print(shell_export("PUBLICATIONS_VENV_DIR", str(venv_dir)))
        print(shell_export("PUBLICATIONS_VENV_PYTHON", str(py_bin)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
