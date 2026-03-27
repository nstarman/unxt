"""DimensionAxis: a coordax Coordinate that tracks physical dimensions."""

from __future__ import annotations

__all__ = ["DimensionAxis", "DimensionOperationError"]

import dataclasses
from typing import Any

import astropy.units as apyu
import jax
from coordax import Coordinate
from coordax.coordinate_systems import Scalar

from unxt.dims import dimension


class DimensionOperationError(TypeError):
    """Raised when a dimension-unsafe arithmetic operation is attempted.

    Multiplication and division on :class:`coordax.Field` objects are not
    automatically tracked when either operand carries a
    :class:`DimensionAxis`.  Specifically:

    * **Multiply / divide**: raises if the *other* operand is a
      :class:`coordax.Field` that contains a :class:`DimensionAxis`.
      Scaling by a plain number or a field *without* a ``DimensionAxis``
      is allowed.
    * **Power**: always raises when the base field contains a
      :class:`DimensionAxis`, because the resulting physical dimension
      cannot be recorded automatically.

    To work with the resulting dimension manually, compute it explicitly::

        import unxt as u
        dim_result = u.dimension_of(f_a) * u.dimension_of(f_b)
    """


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class DimensionAxis(Coordinate):
    """A one-dimensional coordinate with a physical dimension label.

    ``DimensionAxis`` extends :class:`coordax.Coordinate` to carry an
    :class:`astropy.units.PhysicalType` that describes the physical dimension
    of the values associated with this axis (e.g. *length*, *time*, *mass*).

    The physical dimension label can be retrieved by calling
    :func:`unxt.dimension_of` on a :class:`coordax.Field` that uses this
    coordinate.

    .. rubric:: Arithmetic restrictions

    Addition and subtraction are dimension-safe by construction: coordax
    enforces that operands sharing the same axis name carry identical
    coordinate objects, so adding two fields whose ``DimensionAxis`` instances
    carry different physical dimensions raises a ``ValueError`` from coordax.

    **Multiplication and division** raise :class:`DimensionOperationError` when
    the *other* operand is a :class:`~coordax.Field` with a ``DimensionAxis``
    (because the resulting physical dimension cannot be automatically
    propagated).  Scaling by a plain number or a field *without* a
    ``DimensionAxis`` is allowed.

    **Powers** always raise :class:`DimensionOperationError` when the base
    field contains a ``DimensionAxis``.

    To compute the result dimension explicitly::

        dim_result = u.dimension_of(f_a) * u.dimension_of(f_b)

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
        that :func:`unxt.dimension` can parse (e.g. ``"length"``, ``"time"``).

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

    Use it to annotate a coordax field and query its dimension:

    >>> import unxt as u
    >>> f = cx.field(np.ones(5), x)
    >>> u.dimension_of(f)
    PhysicalType('length')
    """

    name: str
    size: int
    dimension: apyu.PhysicalType = dataclasses.field(
        default=apyu.get_physical_type("dimensionless")
    )

    def __post_init__(self) -> None:
        # Coerce string / other inputs to PhysicalType via unxt.dimension().
        object.__setattr__(self, "dimension", dimension(self.dimension))

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
