import nox

nox.options.reuse_existing_virtualenvs = True

SRC = ["aidial_adapter_bedrock", "tests", "noxfile.py"]


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


@nox.session
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", "--with", "lint", external=True)
        session.run("poetry", "check", "--lock", "--strict", external=True)
        session.run("ruff", "check", *SRC)
        session.run("ruff", "format", "--check", *SRC)
        session.run("pyright", *SRC)
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", "--only", "lint", external=True)
    session.run("ruff", "check", "--fix", *SRC)
    session.run("ruff", "format", *SRC)


def run_tests(session: nox.Session, *args):
    session.run("poetry", "install", external=True)
    session.run("pytest", *args, *session.posargs)


@nox.session
def test(session: nox.Session):
    run_tests(session, "tests/unit_tests/")


@nox.session
def integration_tests(session: nox.Session):
    run_tests(session, "tests/integration_tests/")
