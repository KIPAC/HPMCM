from typing import Any

import numpy as np
import tables_io

import click
import yaml
import hpmcm
from hpmcm import __version__

from . import options


__all__ = [
    "cli",
    "match_group",
    "match_shear_command",
]


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """HPMCM Command line interface
    """


@cli.group(name="shear")
def shear_group() -> None:
    """Operations on shear catalogs"""
    

@shear_group.command(name="match")
@options.inputs()
@options.output_file_base()
@options.shear()
@options.pixel_r2_cut()
@options.pixel_match_scale()
@options.deshear()
def shear_match_command(
    inputs: list[str],
    output_file_base: click.Path,
    shear: float|None,
    pixel_r2_cut: float|None,
    pixel_match_scale: int,
    deshear: bool,
) -> None:
    """Match shear catalogs
    """
    deshear_value: float | None = None
    if deshear:
        assert shear is not None
        deshear_value = -1*shear
    matcher = hpmcm.ShearMatch.createShearMatch(pixelR2Cut=pixel_r2_cut, pixelMatchScale=pixel_match_scale, deshear=deshear_value)
    input_files = list(inputs)
    input_files.reverse()
    visit_ids = list(np.arange(len(input_files)))
    matcher.reduceData(input_files, visit_ids)
    print("Matching catalogs:")
    matcher.analysisLoop()
    print("Extracting stats.")
    stats = matcher.extractStats()
    print("Extracting shear stats.")
    shear_stats = matcher.extractShearStats()
    print("Writing output.")
    out_dict = dict(
        _cluster_assoc=stats[0],
        _object_assoc=stats[1],
        _cluster_stats=stats[2],
        _object_stats=stats[3],
        _cluster_shear=shear_stats[0],
        _object_shear=shear_stats[1],
    )
    tables_io.write(out_dict, output_file_base)
    print("Success!")


@shear_group.command(name="split")
@options.basefile()
@options.tract()
@options.shear(required=True)
def shear_split_command(
    basefile: click.Path,
    tract: int,
    shear: float,
) -> None:
    """split input shear catalogs
    """
    hpmcm.ShearMatch.splitByTypeAndClean(basefile, tract, shear)


@shear_group.command(name="report")
@options.basefile()
@options.shear(required=True)
def shear_report_command(
    basefile: click.Path,
    shear: float,
) -> None:
    """split input shear catalogs
    """
    hpmcm.ShearMatch.shear_report(basefile, shear)

    
