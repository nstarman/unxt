"""Unit system utils."""

__all__: list[str] = []

import re
from typing import (  # type: ignore[attr-defined]
    Annotated,
    Any,
    TypeGuard,
    _AnnotatedAlias,
)

from astropy.units import PhysicalType as Dimension
from plum import dispatch

from .builtin_dimensions import speed
from unxt._unxt.typing_ext import Unit

AnnotationType = type(Annotated[int, "_"])


def is_annotated(hint: Any) -> TypeGuard[_AnnotatedAlias]:
    """Check if a type hint is an `Annotated` type.

    Examples
    --------
    >>> from unxt._unxt.unitsystems.utils import is_annotated

    >>> is_annotated(int)
    False

    >>> from typing import Annotated
    >>> is_annotated(Annotated[int, "2"])
    True

    """
    return type(hint) is AnnotationType  # pylint: disable=unidiomatic-typecheck


# ------------------------------------


@dispatch.abstract
def get_dimension_name(pt: Any, /) -> str:
    """Get the dimension name from the object."""
    raise NotImplementedError  # pragma: no cover


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: str, /) -> str:
    """Return the dimension name.

    Note that this does not check for the existence of that dimension.

    Examples
    --------
    >>> from unxt._unxt.unitsystems.utils import get_dimension_name

    >>> get_dimension_name("length")
    'length'

    >>> get_dimension_name("not real")
    'not real'

    >>> try: get_dimension_name("*62")
    ... except ValueError as e: print(e)
    Input contains non-letter characters

    """
    # A regex search to match anything that's not a letter or a whitespace.
    if re.search(r"[^a-zA-Z_ ]", pt):
        msg = "Input contains non-letter characters"
        raise ValueError(msg)

    return pt


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: Dimension, /) -> str:
    """Return the dimension name from a dimension.

    Examples
    --------
    >>> from unxt._unxt.unitsystems.utils import get_dimension_name
    >>> import astropy.units as u
    >>> get_dimension_name(u.get_physical_type("length"))
    'length'

    >>> get_dimension_name(u.get_physical_type("speed"))
    'speed'

    """
    # Note: this is not deterministic b/c ``_physical_type`` is a set
    #       that's why the `if` statement is needed.
    if pt == speed:
        return "speed"
    return get_dimension_name(next(iter(pt._physical_type)))  # noqa: SLF001


@dispatch  # type: ignore[no-redef]
def get_dimension_name(pt: Unit, /) -> str:
    """Return the dimension name from a unit.

    Examples
    --------
    >>> from unxt._unxt.unitsystems.utils import get_dimension_name
    >>> import astropy.units as u
    >>> get_dimension_name(u.km)
    'length'

    """
    return get_dimension_name(pt.physical_type)