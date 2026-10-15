import click
import numpy as np
import pandas
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
    "shearSplitRubinCommand",
    "shearSplitDESCCommand",
    "shearReportCommand",
    "shearMergeReportsCommand",
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
    out_dict = {f"_{k}": v for k, v in stats.items()}
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
@options.joined_input()
@options.catalog_format()
@options.shear_type_col()
@options.tract(required=False, default=None)
def shearMatchCommand(
    inputs: list[str],
    output_file: click.Path,
    shear: float | None,
    pixel_r2_cut: float | None,
    pixel_match_scale: int,
    deshear: bool,
    joined_input: str | None,
    catalog_format: str,
    shear_type_col: str,
    tract: int | None,
) -> None:
    """Match shear catalogs.

    Input catalogs can be supplied in two ways:

    \b
    1. As multiple positional INPUTS (one file per shear type, pre-split):
         hpmcm shear match ns.pq 1p.pq 1m.pq 2p.pq 2m.pq --output-file out.pq

    \b
    2. As a single joined catalog via --joined-input (all types in one file).
       Use --catalog-format to specify the input format:

    \b
         # already-processed catalog (x_cell_coadd etc. present):
         hpmcm shear match --joined-input all.pq --output-file out.pq

    \b
         # raw joined Rubin MD catalog (requires --tract):
         hpmcm shear match --joined-input raw.pq --catalog-format rubin-md \\
             --tract 10463 --output-file out.pq

    \b
         # raw joined DESC MD catalog:
         hpmcm shear match --joined-input raw.pq --catalog-format desc-md \\
             --output-file out.pq
    """
    if inputs and joined_input is not None:
        raise click.UsageError(
            "Cannot use both positional INPUTS and --joined-input. Choose one."
        )
    if not inputs and joined_input is None:
        raise click.UsageError(
            "Must provide either positional INPUTS or --joined-input."
        )
    if catalog_format == "rubin-md" and tract is None:
        raise click.UsageError(
            "--tract is required when --catalog-format is 'rubin-md'."
        )

    deshear_value: float | None = None
    if deshear:
        if shear is None:
            raise click.UsageError("--shear is required when --deshear is set.")
        deshear_value = -1 * shear
    matcher = hpmcm.ShearMatch.createShearMatch(
        pixelR2Cut=pixel_r2_cut,
        pixelMatchScale=pixel_match_scale,
        deshear=deshear_value,
    )

    if joined_input is not None:
        df = pandas.read_parquet(joined_input)
        if catalog_format == "rubin-md":
            assert tract is not None  # validated above
            df = hpmcm.shear_utils.prepareRubinMDForMatch(df, tract)
            matcher.splitAndReduce(df)
        elif catalog_format == "desc-md":
            df = hpmcm.shear_utils.prepareDESCMDForMatch(df)
            matcher.splitAndReduce(df)
        else:
            matcher.splitAndReduce(df, shear_type_col)
    else:
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
    out_dict = {f"_{k}": v for k, v in {**stats, **shear_stats}.items()}
    tables_io.write(out_dict, output_file)
    print("Success!")


@shearGroup.command(name="split-rubin")
@options.basefile()
@options.tract()
@options.shear(required=True)
def shearSplitRubinCommand(
    basefile: str,
    tract: int,
    shear: float,
) -> None:
    """Split input shear catalogs"""
    hpmcm.shear_utils.splitRubinMDByTypeAndClean(basefile, tract, shear)


@shearGroup.command(name="split-desc")
@options.basefile()
@options.tract()
@options.shear(required=True)
def shearSplitDESCCommand(
    basefile: str,
    tract: int,
    shear: float,
) -> None:
    """Split input shear catalogs"""
    hpmcm.shear_utils.splitDESCMDByTypeAndClean(basefile, tract, shear)


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
