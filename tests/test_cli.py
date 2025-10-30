from click.testing import CliRunner, Result
from hpmcm.cli.commands import cli


def checkResult(
    result: Result,
) -> None:
    """Check the result of an invocation"""
    if not result.exit_code == 0:
        raise ValueError(f"{result} failed with {result.exit_code} {result.output}")


def testCliHelp() -> None:
    """Make sure the hpmcm --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "--help")
    checkResult(result)


def testCliShear() -> None:
    """Make sure the hpmcm shear --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "shear --help")
    checkResult(result)


def testCliShearMatch() -> None:
    """Make sure the hpmcm shear match --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "shear match --help")
    checkResult(result)
