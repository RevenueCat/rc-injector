import nox
from nox import session, Session


package = "rc_injector"
nox.options.sessions = "lint", "types", "tests"
locations = "src", "tests", "noxfile.py"
DEFAULT_VERSION = "3.8"
VERSIONS = ["3.8", "3.10"]


@session(python=DEFAULT_VERSION)
def black(session: Session) -> None:
    """Run black code formatter."""

    args = session.posargs or locations
    session.install("black", ".")
    session.run("black", *args)


@session(python=VERSIONS)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-docstrings",
        "bandit",
        ".",
    )
    session.run("flake8", *args)


@session(python=DEFAULT_VERSION)
def types(session: Session) -> None:
    """Type-check using mypy."""
    session.install("mypy", ".")
    session.run("mypy", "--strict", "src/")
    session.run("mypy", "--strict", "tests/")


@session(python=VERSIONS)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    session.install(
        # "coverage[toml]",
        "pytest",
        "pytest-cov",
    )
    session.run("pytest", *args, env={"PYTHONHASHSEED": "0"})
