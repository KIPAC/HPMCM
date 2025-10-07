import enum
from functools import partial
from typing import Any, Type, TypeVar

import click

__all__: list[str] = [
    "basefile",
    "deshear",
    "inputs",
    "output_file_base",
    "pixel_match_scale",
    "pixel_r2_cut",
    "shear",
    "tract",
]

EnumType_co = TypeVar("EnumType_co", bound=Type[enum.Enum], covariant=True)


class EnumChoice(click.Choice):
    """A version of click.Choice specialized for enum types"""

    def __init__(self, the_enum: EnumType_co, case_sensitive: bool = True) -> None:
        self._enum = the_enum
        super().__init__(
            list(the_enum.__members__.keys()), case_sensitive=case_sensitive
        )

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> enum.Enum:  # pragma: no cover
        converted_str = super().convert(value, param, ctx)
        return self._enum.__members__[converted_str]

    
class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: str, **kwargs: Any) -> None:
        self._partial = partial(click.option, *param_decls, cls=click.Option, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=click.Argument, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)

basefile = PartialOption(
    "--basefile",
    type=click.Path(),
    help="Input file basename",
    required=True,
)

deshear = PartialOption(
    "--deshear",
    help="Deshear postions before match",
    is_flag=True,
)

output_file_base = PartialOption(
    "--output_file_base",
    type=click.Path(),
    help="Path for output yaml",
    required=True,
)

pixel_r2_cut = PartialOption(
    "--pixel_r2_cut",
    help="Matching cut to apply in pixel distance squared",
    type=float,
    default=4.,
)

pixel_match_scale = PartialOption(
    "--pixel_match_scale",
    help="Scale factor to apply for pixel in matching",
    type=int,
    default=1,
)

shear = PartialOption(
    "--shear",
    help="Shear value",
    type=float,
    default=None,
)

tract = PartialOption(
    "--tract",
    help="Tract to select",
    type=int,
    required=True
)

inputs = PartialArgument("inputs", nargs=-1)
