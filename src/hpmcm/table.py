from __future__ import annotations

from typing import Any

import numpy as np
import pandas
import pyarrow.parquet as pq


class TableColumnInfo:
    """Helper class to manage a column in a table

    This provides a mechanism to document the
    column in the class docstring, and
    to validate input data
    """

    def __init__(self, dtype: type, msg: str):
        self.dtype = dtype
        self.msg = msg

    def __repr__(self) -> str:
        # return f"{self.dtype:10}\n    {self.msg}"
        return f"{self.dtype.__name__:8} | {self.msg:50}"

    def validate(self, val: np.ndarray) -> None:
        """Validate data used to fill a column is of the correct type"""
        assert isinstance(val, np.ndarray)
        assert val.dtype == self.dtype


class TableInterface:
    """Table Schema"""

    _schema: dict[str, TableColumnInfo] = dict()

    def __init__(self, df: pandas.DataFrame | None = None, **kwargs: Any):
        if df is None:
            self._data = self.toPandas(**kwargs)
        else:
            self._data = df

    @classmethod
    def _describeSchema(cls) -> str:
        """Describe the columns in this table"""
        s = []
        for name, val in cls._schema.items():
            assert isinstance(val, TableColumnInfo)
            s.append(f"| {name:15} | {val} |")
        return "\n+-----------------+----------+----------------------------------------------------+\n".join(
            s
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        config_text = cls._describeSchema()
        if cls.__doc__ is None:
            cls.__doc__ = f"\nNotes\n-----\n{cls.__name__} schema\n\n"
            cls.__doc__ += "+-----------------+----------+----------------------------------------------------+\n"
            cls.__doc__ += "| Column          | Type     | Description                                        |\n"
            cls.__doc__ += "+=================+==========+====================================================+\n"
            cls.__doc__ += config_text
            cls.__doc__ += "\n+-----------------+----------+----------------------------------------------------+\n"
        else:
            # strip any existing configuration text from parent classes that is at the end of the doctring
            cls.__doc__ = cls.__doc__.split("Notes")[0]
            cls.__doc__ += f"\nNotes\n-----\n{cls.__name__} schema\n\n"
            cls.__doc__ += "+-----------------+----------+----------------------------------------------------+\n"
            cls.__doc__ += "| Column          | Type     | Description                                        |\n"
            cls.__doc__ += "+=================+==========+====================================================+\n"
            cls.__doc__ += config_text
            cls.__doc__ += "\n+-----------------+----------+----------------------------------------------------+\n"

    @property
    def data(self) -> pandas.DataFrame:
        """Return the underlying data"""
        return self._data

    @classmethod
    def validate(cls, **kwargs: Any) -> None:
        """Validate that data match the schema

        Parameters
        ----------
        kwargs:
            The input data

        Raises
        ------
        ValueError:
            The number of columns don't match the schema

        KeyError:
            An input column is not in the schema
        """
        table_size: int = -1
        if len(kwargs) != len(cls._schema):
            raise ValueError(f"{len(kwargs)} != {len(cls._schema)}")
        for key, val in kwargs.items():
            if key not in cls._schema:
                raise KeyError(f"{key} not in {list(cls._schema.keys())}")
            colInfo = cls._schema[key]
            colInfo.validate(val)
            if table_size < 0:
                table_size = val.size
            else:
                assert val.size == table_size

    @classmethod
    def read(cls, filePath: str, extraCols: list[str]) -> pandas.DataFrame:
        """Read a dataframe from a file"""
        readList = list(cls._schema.keys())
        readList += extraCols
        parq = pq.read_pandas(filePath, columns=readList)
        df = parq.to_pandas()
        return df

    @classmethod
    def toPandas(cls, **kwargs: Any) -> pandas.DataFrame:
        """Convert data to a pandas DataFrame

        Parameters
        ----------
        kwargs:
            The input data

        """
        cls.validate(**kwargs)
        return pandas.DataFrame(kwargs)

    @classmethod
    def emtpyNumpyDict(cls, n: int) -> dict[str, np.ndarray]:
        """Create a dict of empty numpy arrays

        Parameter
        ---------
        n:
            Length of the arrays
        """
        return {key: np.zeros((n), dtype=val.dtype) for key, val in cls._schema.items()}
