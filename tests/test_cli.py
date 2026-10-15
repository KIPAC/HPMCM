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
    assert "--joined-input" in result.output
    assert "--catalog-format" in result.output
    assert "--shear-type-col" in result.output
    assert "--tract" in result.output


def testCliShearMatchMutualExclusion() -> None:
    """Both positional inputs and --joined-input should be rejected."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "shear", "match",
            "file1.pq",
            "--joined-input", "joined.pq",
            "--output-file", "out.pq",
        ],
    )
    assert result.exit_code != 0
    assert "Cannot use both" in result.output


def testCliShearMatchNoInput() -> None:
    """No inputs at all should be rejected."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["shear", "match", "--output-file", "out.pq"],
    )
    assert result.exit_code != 0
    assert "Must provide" in result.output


def testCliShearMatchRubinMDRequiresTract() -> None:
    """--catalog-format rubin-md without --tract should be rejected."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "shear", "match",
            "--joined-input", "raw.pq",
            "--catalog-format", "rubin-md",
            "--output-file", "out.pq",
        ],
    )
    assert result.exit_code != 0
    assert "--tract" in result.output
