from __future__ import annotations

from typing import Any

import numpy as np
import pandas


class TableColumnInfo:
    """Class to document a column in a table"""

    def __init__(self, dtype: type, msg: str):
        self.dtype = dtype
        self.msg = msg

    def __repr__(self) -> str:
        return f"{self.dtype}\n    {self.msg}"

    def validate(self, val: np.ndarray) -> None:
        """Validate that a column is of the correct type"""
        assert isinstance(val, np.ndarray)
        assert val.dtype == self.dtype


class TableInterface:
    """Helper class to manage table schema"""

    _schema: dict[str, TableColumnInfo] = dict()

    def __init__(self, **kwargs: Any):
        self._data = self.toPandas(**kwargs)

    @classmethod
    def _describe_schema(cls):
        """Describe the columns in this table"""
        s = []
        for name, val in cls._schema.items():
            assert isinstance(val, TableColumnInfo)
            s.append(f"{name}: {val}")
        return '\n\n'.join(s)

    def __init_subclass__(cls, **kwargs):
        config_text = cls._describe_schema()
        if cls.__doc__ is None:                 
            cls.__doc__ = f"Stage {cls.name}\n\nParameters\n----------\n{config_text}"
        else:
            # strip any existing configuration text from parent classes that is at the end of the doctring
            cls.__doc__ = cls.__doc__.split("Parameters")[0]
            cls.__doc__ += f"\n\nParameters\n----------\n{config_text}"    
        
    @property
    def data(self) -> pandas.DataFrame:
        """Return the underlying data"""
        return self._data

    @classmethod
    def validate(cls, **kwargs: Any) -> None:
        """Validate that data match the schema"""
        table_size: int = -1
        if len(kwargs) != len(cls._schema):
            raise ValueError(f"{len(kwargs)} != {len(cls.schema)}")
        for key, val in kwargs.items():
            if key not in cls._schema:
                raise KeyError(f"{key} not in {list(cls.schema.keys())}")
            colInfo = cls._schema[key]
            colInfo.validate(val)
            if table_size < 0:
                table_size = val.size
            else:
                assert val.size == table_size

    @classmethod
    def toPandas(cls, **kwargs: Any) -> pandas.DataFrame:
        """Convert data to a pandas DataFrame"""
        cls.validate(**kwargs)
        return pandas.DataFrame(kwargs)

    @classmethod
    def emtpyNumpyDict(cls, n: int) -> dict[str, np.ndarray]:
        """Create a dict of empty number array"""
        return {key: np.zeros((n), dtype=val.dtype) for key, val in cls._schema.items()}
