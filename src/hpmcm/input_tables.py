"""Expected schema for various input tables used by hpmcm"""

from __future__ import annotations

from .table import TableColumnInfo, TableInterface


class SourceTable(TableInterface):
    """Basic input table Using RA and DEC"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        id=TableColumnInfo(int, "Unique ID for source"),
        ra=TableColumnInfo(float, "RA of source"),
        dec=TableColumnInfo(float, "DEC of source"),
        snr=TableColumnInfo(float, "Signal-to-noise of source"),
    )


class CoaddSourceTable(TableInterface):
    """Basic input table using cell-positions"""

    _schema = TableInterface._schema.copy()
    _schema.update(
        id=TableColumnInfo(int, "Unique ID for source"),
        tract=TableColumnInfo(int, "Tract"),
        x_cell_coadd=TableColumnInfo(
            float, "X-postion in cell-based coadd used for metadetect"
        ),
        y_cell_coadd=TableColumnInfo(
            float, "Y-postion in cell-based coadd used for metadetect"
        ),
        snr=TableColumnInfo(float, "Signal-to-noise of source"),
        cell_idx_x=TableColumnInfo(int, "Cell x-index within Tract"),
        cell_idx_y=TableColumnInfo(int, "Cell y-index within Tract"),
    )


class ShearCoaddSourceTable(CoaddSourceTable):
    """Shear calibration input table using cell-positions"""

    _schema = CoaddSourceTable._schema.copy()
    _schema.update(
        g_1=TableColumnInfo(float, "Shear g1 component"),
        g_2=TableColumnInfo(float, "Shear g2 component"),
    )
