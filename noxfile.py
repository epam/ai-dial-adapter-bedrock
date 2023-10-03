import nox

nox.options.reuse_existing_virtualenvs = True

SRC = "."


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


@nox.session
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", external=True)
        session.run("poetry", "check", "--lock", external=True)
        session.run("pyright", SRC)
        session.run("flake8", SRC)
        format_with_args(session, SRC, "--check")
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", external=True)
    format_with_args(session, SRC)


def run_tests(session: nox.Session, default_dir: str):
    session.run("poetry", "install", external=True)

    if session.posargs:
        test_files = session.posargs
    else:
        test_files = [default_dir]

    session.run("pytest", *test_files)


@nox.session
def test(session: nox.Session):
    run_tests(session, "tests/unit_tests/")


@nox.session
def integration_tests(session: nox.Session):
    run_tests(session, "tests/integration_tests/")
