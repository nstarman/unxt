"""Plum dispatches extending :func:`unxt.dimension_of` for coordax types.

Importing this module (which happens automatically when ``unxt_coordax`` is
imported) registers two new overloads of :func:`unxt.dimension_of`:

* ``dimension_of(DimensionAxis)`` – returns the physical dimension stored on the
  coordinate itself.
* ``dimension_of(coordax.Field)`` – returns the physical dimension of the
  **first** :class:`~unxt_coordax.DimensionAxis` found among the field's axes,
  or ``None`` if no such axis is present.

.. note:: Arithmetic behaviour

   Standard operators (``+``, ``-``, ``*``, ``/``, ``**``) work directly on
   :class:`coordax.Field` objects.  Addition and subtraction are
   dimension-safe: coordax enforces that operands sharing the same axis name
   must carry **identical** coordinate objects, so adding two fields whose
   ``DimensionAxis`` carry different physical dimensions raises a
   ``ValueError`` from coordax.

   **Multiplication, division, and powers do not update the dimension label**
   on the resulting :class:`~unxt_coordax.DimensionAxis`.  The result field
   retains the coordinate objects of the first operand unchanged.  This means
   that, for example, squaring a *length* field yields a field whose axis is
   still labelled *length* even though the values are now an *area*, and
   multiplying *length* by *time* silently produces a field still labelled
   *length*.  If you need the result dimension, compute it explicitly::

       import unxt as u

       dim_result = u.dimension_of(f_a) * u.dimension_of(f_b)

   and create a new :class:`~unxt_coordax.DimensionAxis` with that dimension.

   .. admonition:: Future improvement

      This limitation is a consequence of the current coordax arithmetic
      model, which does not provide a hook for coordinate objects to transform
      their metadata during binary operations.  The coordax maintainers are
      being approached to add first-class support for propagating coordinate
      metadata through multiplication, division, and power operations.  Once
      that support lands, ``DimensionAxis`` will be updated to propagate
      dimensions automatically.
"""

from __future__ import annotations

__all__: list[str] = []

from coordax import Field
from plum import dispatch

from unxt.dims import AbstractDimension, dimension_of
from unxt_coordax._src.dimension_axis import DimensionAxis


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

