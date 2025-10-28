import os

import pytest
from click.testing import CliRunner, Result

from hpmcm.cli.commands import cli


def check_result(
    result: Result,
) -> None:
    """Check the result of an invocation"""
    if not result.exit_code == 0:
        raise ValueError(f"{result} failed with {result.exit_code} {result.output}")


def test_cli_help() -> None:
    """Make sure the hpmcm --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "--help")
    check_result(result)


def test_cli_shear() -> None:
    """Make sure the hpmcm shear --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "shear --help")
    check_result(result)


def test_cli_shear_match() -> None:
    """Make sure the hpmcm shear match --help command works"""
    runner = CliRunner()

    result = runner.invoke(cli, "shear match --help")
    check_result(result)

