import click
import numpy as np
import tables_io

import hpmcm
from hpmcm import __version__

from . import options

__all__ = [
    "cli",
    "wcsGroup",
    "wcsMatchCommand",
    "shearGroup",
    "shearMatchCommand",
    "shearSplitCommand",
    "shearReportCommand",
]


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """HPMCM Command line interface"""


@cli.group(name="wcs")
def wcsGroup() -> None:
    """Operations on catalogs using single WCS"""


@wcsGroup.command(name="match")
@options.inputs()
@options.output_file()
@options.ra_ref(required=True)
@options.dec_ref(required=True)
@options.ra_size(required=True)
@options.dec_size(required=True)
@options.pixel_size(required=True)
@options.cell_size()
@options.pixel_r2_cut()
def wcsMatchCommand(
    inputs: list[str],
    output_file: click.Path,
    ra_ref: float,
    dec_ref: float,
    ra_size: float,
    dec_size: float,
    pixel_size: float,
    cell_size: int,
    pixel_r2_cut: float | None,
) -> None:
    """Match catalogs using a global WCS map"""
    matcher = hpmcm.WcsMatch.create(
        ref_dir=(ra_ref, dec_ref),
        region_size=(ra_size, dec_size),
        pixel_size=pixel_size,
        cell_size=cell_size,
        pixel_R2_cut=pixel_r2_cut,
    )
    input_files = list(inputs)
    visit_ids: list[int] = list(np.arange(len(input_files)))
    matcher.reduceData(input_files, visit_ids)
    print("Matching catalogs:")
    matcher.analysisLoop()
    print("Extracting stats.")
    stats = matcher.extractStats()
    print("Writing output.")
    out_dict = dict(
        _cluster_assoc=stats[0],
        _object_assoc=stats[1],
        _cluster_stats=stats[2],
        _object_stats=stats[3],
    )
    tables_io.write(out_dict, output_file)
    print("Success!")


@cli.group(name="shear")
def shearGroup() -> None:
    """Operations on shear catalogs"""


@shearGroup.command(name="match")
@options.inputs()
@options.output_file()
@options.shear()
@options.pixel_r2_cut()
@options.pixel_match_scale()
@options.deshear()
def shearMatchCommand(
    inputs: list[str],
    output_file: click.Path,
    shear: float | None,
    pixel_r2_cut: float | None,
    pixel_match_scale: int,
    deshear: bool,
) -> None:
    """Match shear catalogs"""
    deshear_value: float | None = None
    if deshear:
        assert shear is not None
        deshear_value = -1 * shear
    matcher = hpmcm.ShearMatch.createShearMatch(
        pixelR2Cut=pixel_r2_cut,
        pixelMatchScale=pixel_match_scale,
        deshear=deshear_value,
    )
    input_files = list(inputs)
    input_files.reverse()
    visit_ids: list[int] = list(np.arange(len(input_files)))
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
    tables_io.write(out_dict, output_file)
    print("Success!")


@shearGroup.command(name="split")
@options.basefile()
@options.tract()
@options.catalog_type()
@options.shear(required=True)
def shearSplitCommand(
    basefile: str,
    tract: int,
    shear: float,
    catalog_type: str,
) -> None:
    """Split input shear catalogs"""
    hpmcm.shear_utils.splitByTypeAndClean(basefile, tract, shear, catalog_type)


@shearGroup.command(name="report")
@options.basefile()
@options.output_file()
@options.shear(required=True)
@options.snr_cut()
def shearReportCommand(
    basefile: str,
    output_file: str,
    shear: float,
    snr_cut: float,
) -> None:
    """Build shear calibration reports"""
    tokens = basefile.split("_")
    tract = int(tokens[-1])
    cat_type = tokens[-4]
    hpmcm.shear_utils.shearReport(
        basefile, output_file, shear, cat_type=cat_type, tract=tract, snr_cut=snr_cut
    )


@shearGroup.command(name="merge-reports")
@options.inputs()
@options.output_file()
def shearMergeReportsCommand(
    inputs: list[str],
    output_file: str,
) -> None:
    """Build shear calibration reports"""
    hpmcm.shear_utils.mergeShearReports(inputs, output_file)
