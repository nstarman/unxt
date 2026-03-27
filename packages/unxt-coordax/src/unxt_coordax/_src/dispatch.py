"""Plum dispatches extending :func:`unxt.dimension_of` for coordax types, and
arithmetic restrictions for :class:`~coordax.Field` objects backed by
:class:`~unxt_coordax.DimensionAxis` coordinates.

Importing this module (which happens automatically when ``unxt_coordax`` is
imported) does two things:

1. Registers two new overloads of :func:`unxt.dimension_of`:

   * ``dimension_of(DimensionAxis)`` – returns the physical dimension stored on
     the coordinate itself.
   * ``dimension_of(coordax.Field)`` – returns the physical dimension of the
     **first** :class:`~unxt_coordax.DimensionAxis` found among the field's
     axes, or ``None`` if no such axis is present.

2. Patches :class:`coordax.Field` arithmetic to enforce dimension safety:

   * **Multiply / divide** raise :class:`~unxt_coordax.DimensionOperationError`
     when the *other* operand is a :class:`~coordax.Field` that contains a
     :class:`~unxt_coordax.DimensionAxis`.  Scaling by a plain number or a
     field *without* a ``DimensionAxis`` is allowed.
   * **Power** always raises :class:`~unxt_coordax.DimensionOperationError`
     when the base field contains a :class:`~unxt_coordax.DimensionAxis`.
"""

from __future__ import annotations

__all__: list[str] = []

from coordax import Field
from plum import dispatch

from unxt.dims import AbstractDimension, dimension_of
from unxt_coordax._src.dimension_axis import DimensionAxis, DimensionOperationError


# ---------------------------------------------------------------------------
# Plum dispatch overloads for unxt.dimension_of


@dispatch
def dimension_of(coord: DimensionAxis, /) -> AbstractDimension:
    """Return the physical dimension stored on a :class:`DimensionAxis`.

    Parameters
    ----------
    coord:
        A ``DimensionAxis`` coordinate.

    Returns
    -------
    :class:`astropy.units.PhysicalType`
        The physical dimension attached to *coord*.

    Examples
    --------
    >>> import unxt as u
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 5, "length")
    >>> u.dimension_of(x)
    PhysicalType('length')
    """
    return coord.dimension


@dispatch
def dimension_of(field: Field, /) -> AbstractDimension | None:
    """Return the physical dimension of the first ``DimensionAxis`` in *field*.

    Iterates over the axes of *field* and returns the
    :class:`astropy.units.PhysicalType` of the first
    :class:`~unxt_coordax.DimensionAxis` found, or ``None`` if no such axis is
    present.

    Parameters
    ----------
    field:
        A :class:`coordax.Field` whose axes are inspected.

    Returns
    -------
    :class:`astropy.units.PhysicalType` or ``None``

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt as u
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 3, "length")
    >>> f = cx.field(np.ones(3), x)
    >>> u.dimension_of(f)
    PhysicalType('length')

    Fields with no ``DimensionAxis`` return ``None``:

    >>> plain = cx.SizedAxis("x", 3)
    >>> f_plain = cx.field(np.ones(3), plain)
    >>> u.dimension_of(f_plain) is None
    True
    """
    for coord in field.axes.values():
        if isinstance(coord, DimensionAxis):
            return coord.dimension
    return None


# ---------------------------------------------------------------------------
# Field arithmetic restrictions
#
# coordax.Field arithmetic does not propagate DimensionAxis dimension labels.
# We patch Field.__mul__, __truediv__, __pow__, and __rtruediv__ to raise
# DimensionOperationError when the operation would silently lose dimension info.


def _has_dimension_axis(obj: object) -> bool:
    """Return True if *obj* is a Field that contains at least one DimensionAxis."""
    return isinstance(obj, Field) and any(
        isinstance(c, DimensionAxis) for c in obj.axes.values()
    )


_ORIG_MUL = Field.__mul__
_ORIG_TRUEDIV = Field.__truediv__
_ORIG_POW = Field.__pow__
_ORIG_RTRUEDIV = Field.__rtruediv__


def _checked_mul(self: Field, other: object) -> Field:
    if _has_dimension_axis(other):
        raise DimensionOperationError(
            "Cannot multiply a field by another field that has physical "
            "dimensions: the resulting dimension cannot be automatically "
            "propagated.  Compute the result dimension explicitly with "
            "u.dimension_of(f_a) * u.dimension_of(f_b)."
        )
    return _ORIG_MUL(self, other)


def _checked_truediv(self: Field, other: object) -> Field:
    if _has_dimension_axis(other):
        raise DimensionOperationError(
            "Cannot divide a field by another field that has physical "
            "dimensions: the resulting dimension cannot be automatically "
            "propagated.  Compute the result dimension explicitly with "
            "u.dimension_of(f_a) / u.dimension_of(f_b)."
        )
    return _ORIG_TRUEDIV(self, other)


def _checked_pow(self: Field, other: object) -> Field:
    if _has_dimension_axis(self):
        raise DimensionOperationError(
            "Cannot raise a field with physical dimensions to a power: "
            "the resulting dimension cannot be automatically propagated.  "
            "Compute the result dimension explicitly with "
            "u.dimension_of(f) ** n."
        )
    return _ORIG_POW(self, other)


def _checked_rtruediv(self: Field, other: object) -> Field:
    # self.__rtruediv__(other) handles `other / self`.
    # If self has a DimensionAxis the result dimension (1/dim) is not tracked.
    if _has_dimension_axis(self):
        raise DimensionOperationError(
            "Cannot divide by a field that has physical dimensions: "
            "the resulting dimension cannot be automatically propagated."
        )
    return _ORIG_RTRUEDIV(self, other)


Field.__mul__ = _checked_mul
Field.__truediv__ = _checked_truediv
Field.__pow__ = _checked_pow
Field.__rtruediv__ = _checked_rtruediv

