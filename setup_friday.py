"""Create or reuse a project-local virtual environment and install FRIDAY requirements."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
DEFAULT_ENV = PROJECT_DIR / ".venv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=DEFAULT_ENV,
        help="environment directory (default: .venv beside this script)",
    )
    parser.add_argument(
        "--requirements-file",
        type=Path,
        default=REQUIREMENTS,
        help="requirements file to install (default: requirements.txt)",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("FRIDAY setup requires Python 3.11 or newer")

    env_dir = args.venv_dir.expanduser()
    if not env_dir.is_absolute():
        env_dir = (Path.cwd() / env_dir).resolve()
    requirements = args.requirements_file.expanduser()
    if not requirements.is_absolute():
        requirements = (PROJECT_DIR / requirements).resolve()
    if not requirements.is_file():
        parser.error(f"requirements file not found: {requirements}")
    env_python = env_dir / "Scripts" / "python.exe" if sys.platform == "win32" else env_dir / "bin" / "python"

    if env_dir.exists():
        if not env_python.is_file():
            print(f"Refusing to replace existing directory without a venv interpreter: {env_dir}", file=sys.stderr)
            return 2
        print(f"Reusing existing virtual environment: {env_dir}")
    else:
        print(f"Creating virtual environment: {env_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], cwd=PROJECT_DIR, check=True)

    if not env_python.is_file():
        print(f"Virtual environment did not create its Python executable: {env_python}", file=sys.stderr)
        return 2

    print(f"Installing {requirements.name} into the selected virtual environment...")
    install = subprocess.run(
        [str(env_python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    if install.stdout:
        print(install.stdout, end="")
    if install.stderr:
        print(install.stderr, end="", file=sys.stderr)
    if install.returncode:
        return install.returncode
    print(f"Setup complete. Run FRIDAY with: \"{env_python}\" -m app.main")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode or 1)
