import nox
from nox import session, Session


package = "rc_injector"
nox.options.sessions = "lint", "format", "types", "tests", "check_version"
locations = "src", "tests", "noxfile.py"
DEFAULT_VERSION = "3.11"
VERSIONS = ["3.13", "3.12", "3.11"]

# Default to uv backend:
nox.options.default_venv_backend = "uv|virtualenv"


@session(python=DEFAULT_VERSION)
def format(session: Session) -> None:
    """Format code using ruff."""
    args = session.posargs or locations
    session.install("ruff", ".")
    session.run("ruff", "format", "--diff", *args)


@session(python=DEFAULT_VERSION)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install("ruff", ".")
    session.run("ruff", "check", *args)


@session(python=DEFAULT_VERSION)
def types(session: Session) -> None:
    """Type-check using mypy."""
    session.install("mypy", ".")
    session.run("mypy", "--strict", "src/")
    session.run("mypy", "--strict", "tests/")


@session(python=None)
def check_version(session: Session) -> None:
    """Check the version of the package."""
    pyproject_version = nox.project.load_toml("pyproject.toml")["project"]["version"]
    module_version = session.run(
        "sed",
        "-n",
        's/^__version__ = "\(.*\)"$/\\1/p',
        "src/rc_injector/__init__.py",
        silent=True,
        external=True,
    ).strip()
    print(f"project version: {pyproject_version} (project.version on pyproject.toml)")
    print(
        f"module version: {module_version}  (__version__ on src/rc_injector/__init__.py)"
    )

    if pyproject_version != module_version:
        session.error("Version mismatch!")


@session(python=VERSIONS)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.install(
        # "coverage[toml]",
        "pytest",
        "pytest-cov",
        ".",
    )
    session.run("pytest", *args, env={"PYTHONHASHSEED": "0"})
