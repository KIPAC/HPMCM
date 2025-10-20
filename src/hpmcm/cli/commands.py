import click
import numpy as np
import tables_io

import hpmcm
from hpmcm import __version__

from . import options

__all__ = [
    "cli",
    "wcs_group",
    "wcs_match_command",
    "shear_group",
    "shear_match_command",
    "shear_split_command",
    "shear_report_command",
]


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """HPMCM Command line interface"""


@cli.group(name="wcs")
def wcs_group() -> None:
    """Operations on catalogs using single WCS"""


@wcs_group.command(name="match")
@options.inputs()
@options.output_file()
@options.ra_ref(required=True)
@options.dec_ref(required=True)
@options.ra_size(required=True)
@options.dec_size(required=True)
@options.pixel_size(required=True)
@options.pixel_r2_cut()
def wcs_match_command(
    inputs: list[str],
    output_file: click.Path,
    ra_ref: float,
    dec_ref: float,
    ra_size: float,
    dec_size: float,
    pixel_size: float,
    pixel_r2_cut: float | None,
) -> None:
    """Match catalogs using a global WCS map"""
    matcher = hpmcm.WcsMatch.create(
        refDir=(ra_ref, dec_ref),
        regionSize=(ra_size, dec_size),
        pixelSize=pixel_size,
        pixelR2Cut=pixel_r2_cut,
    )
    input_files = list(inputs)
    visit_ids = list(np.arange(len(input_files)))
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
def shear_group() -> None:
    """Operations on shear catalogs"""


@shear_group.command(name="match")
@options.inputs()
@options.output_file()
@options.shear()
@options.pixel_r2_cut()
@options.pixel_match_scale()
@options.deshear()
def shear_match_command(
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
    tables_io.write(out_dict, output_file)
    print("Success!")


@shear_group.command(name="split")
@options.basefile()
@options.tract()
@options.catalog_type()
@options.shear(required=True)
def shear_split_command(
    basefile: str,
    tract: int,
    shear: float,
    catalog_type: str,
) -> None:
    """Split input shear catalogs"""
    hpmcm.shear_utils.splitByTypeAndClean(basefile, tract, shear, catalog_type)


@shear_group.command(name="report")
@options.basefile()
@options.output_file()
@options.shear(required=True)
@options.snr_cut()
def shear_report_command(
    basefile: str,
    output_file: str,
    shear: float,
    snr_cut: float,
) -> None:
    """Build shear calibration reports"""
    tokens = basefile.split("_")
    tract = int(tokens[-1])
    catType = tokens[-4]
    hpmcm.shear_utils.shearReport(
        basefile, output_file, shear, catType=catType, tract=tract, snrCut=snr_cut
    )


@shear_group.command(name="merge-reports")
@options.inputs()
@options.output_file()
def shear_merge_reports_command(
    inputs: list[str],
    output_file: str,
) -> None:
    """Build shear calibration reports"""
    hpmcm.shear_utils.mergeShearReports(inputs, output_file)
