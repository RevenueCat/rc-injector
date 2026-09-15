import re
from pathlib import Path

import nox
from nox import Session, session

nox.options.sessions = "lint", "format", "types", "tests", "check_version"
locations = "src", "tests", "noxfile.py"
DEFAULT_VERSION = "3.13"
# Every version supported by `requires-python` on pyproject.toml
VERSIONS = ["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"]
# Pinned, so a new ruff release can't turn CI red on its own
RUFF = "ruff@0.16.5"
INIT_FILE = Path("src/rc_injector/__init__.py")

# Default to uv backend:
nox.options.default_venv_backend = "uv|virtualenv"


@session(python=DEFAULT_VERSION)
def format(session: Session) -> None:
    """Check formatting using ruff."""
    args = session.posargs or locations
    session.run("uvx", RUFF, "format", "--diff", *args, external=True)


@session(python=DEFAULT_VERSION)
def lint(session: Session) -> None:
    """Lint using ruff."""
    args = session.posargs or locations
    session.run("uvx", RUFF, "check", *args, external=True)


@session(python=DEFAULT_VERSION)
def fix(session: Session) -> None:
    """Apply ruff formatting and the fixable lint findings."""
    args = session.posargs or locations
    session.run("uvx", RUFF, "format", *args, external=True)
    session.run("uvx", RUFF, "check", "--fix", *args, external=True)


@session(python=DEFAULT_VERSION)
def types(session: Session) -> None:
    """Type-check using mypy."""
    session.install("mypy", "pytest", ".")
    session.run("mypy", "--strict", "src/")
    session.run("mypy", "--strict", "tests/")


@session(python=None)
def check_version(session: Session) -> None:
    """Check the version of the package matches the module."""
    pyproject_version = nox.project.load_toml("pyproject.toml")["project"]["version"]
    module_version_match = re.search(
        r'^__version__ = "(.*)"$', INIT_FILE.read_text(), re.MULTILINE
    )
    if module_version_match is None:
        session.error(f"Unable to find __version__ on {INIT_FILE}")
    module_version = module_version_match.group(1)
    print(f"project version: {pyproject_version} (project.version on pyproject.toml)")
    print(f"module version: {module_version}  (__version__ on {INIT_FILE})")

    if pyproject_version != module_version:
        session.error("Version mismatch!")


@session(python=VERSIONS)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.install(
        "pytest",
        "pytest-cov",
        ".",
    )
    session.run("pytest", *args, env={"PYTHONHASHSEED": "0"})
