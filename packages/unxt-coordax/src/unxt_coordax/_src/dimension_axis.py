"""DimensionAxis: a coordax Coordinate that tracks physical dimensions."""

from __future__ import annotations

__all__ = ["DimensionAxis", "DimensionMismatchError"]

import dataclasses
from typing import Any

import astropy.units as apyu
import jax
from coordax import Coordinate
from coordax.coordinate_systems import Scalar


class DimensionMismatchError(ValueError):
    """Raised when two fields have incompatible physical dimensions."""


def _parse_dimension(obj: apyu.PhysicalType | str) -> apyu.PhysicalType:
    """Coerce *obj* to an :class:`astropy.units.PhysicalType`."""
    if isinstance(obj, apyu.PhysicalType):
        return obj
    return apyu.get_physical_type(obj)


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class DimensionAxis(Coordinate):
    """A one-dimensional coordinate with a physical dimension label.

    ``DimensionAxis`` extends :class:`coordax.Coordinate` to carry an
    :class:`astropy.units.PhysicalType` that describes the physical dimension
    of the values associated with this axis (e.g. *length*, *time*, *mass*).

    The dimension information is used by the dimension-aware arithmetic helpers
    in :mod:`unxt_coordax` (:func:`~unxt_coordax.dadd`,
    :func:`~unxt_coordax.dmul`, etc.) to enforce dimensional consistency and
    compute the resulting dimension after an operation.

    Fields backed by plain NumPy arrays, JAX arrays **or** :class:`unxt.Quantity`
    arrays can all be used with this coordinate type.

    Parameters
    ----------
    name:
        The dimension name of this axis (e.g. ``"x"`` or ``"time"``).
    size:
        Number of elements along this axis.
    dimension:
        The physical dimension of values stored along this axis.  May be given
        as an :class:`astropy.units.PhysicalType` instance or as a plain string
        that :func:`astropy.units.get_physical_type` can parse (e.g.
        ``"length"``, ``"time"``).

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    Create an axis labelled *length*:

    >>> x = ucx.DimensionAxis("x", 5, "length")
    >>> x.name
    'x'
    >>> x.size
    5
    >>> x.dimension
    PhysicalType('length')
    >>> x.dims
    ('x',)
    >>> x.shape
    (5,)

    Use it to annotate a coordax field:

    >>> f = cx.field(np.ones(5), x)
    >>> f.axes["x"].dimension
    PhysicalType('length')
    """

    name: str
    size: int
    dimension: apyu.PhysicalType = dataclasses.field(
        default=apyu.get_physical_type("dimensionless")
    )

    def __post_init__(self) -> None:
        # Coerce string dimensions to PhysicalType
        object.__setattr__(self, "dimension", _parse_dimension(self.dimension))

    # ------------------------------------------------------------------
    # Abstract Coordinate interface

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names of the coordinate."""
        return (self.name,)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the coordinate."""
        return (self.size,)

    # ------------------------------------------------------------------
    # Indexing support

    def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
        key = self.name if self.name in indexers else self
        indexer = indexers[key]
        if isinstance(indexer, int):
            return Scalar()
        if isinstance(indexer, slice):
            start, stop, step = indexer.indices(self.size)
            new_size = len(range(start, stop, step))
            return DimensionAxis(self.name, new_size, self.dimension)
        if hasattr(indexer, "__len__"):
            return DimensionAxis(self.name, len(indexer), self.dimension)
        raise ValueError(f"Unsupported indexer type {type(indexer)}")

    # ------------------------------------------------------------------
    # Representation

    def __repr__(self) -> str:
        return (
            f"DimensionAxis({self.name!r}, size={self.size}, "
            f"dimension={self.dimension!r})"
        )
