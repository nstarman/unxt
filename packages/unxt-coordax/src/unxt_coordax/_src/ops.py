"""Dimension-aware arithmetic operations for coordax Fields."""

from __future__ import annotations

__all__ = ["dadd", "ddiv", "dmul", "dpow", "dsub", "get_dimension"]

from typing import TYPE_CHECKING

import astropy.units as apyu

# Ensure unxt quantities are registered with coordax on import.
from unxt_coordax._src import register  # noqa: F401
from unxt_coordax._src.dimension_axis import DimensionAxis, DimensionMismatchError

if TYPE_CHECKING:
    import coordax


def get_dimension(
    field: "coordax.Field",
    axis_name: str | None = None,
) -> apyu.PhysicalType | None:
    """Return the physical dimension of a field from its ``DimensionAxis``.

    Parameters
    ----------
    field:
        A :class:`coordax.Field` whose axes are inspected.
    axis_name:
        If given, look up *that* axis specifically.  When ``None`` (default)
        the function returns the dimension of the **first** ``DimensionAxis``
        found among the field's axes.

    Returns
    -------
    :class:`astropy.units.PhysicalType` or ``None``
        The physical dimension, or ``None`` if no ``DimensionAxis`` is present
        (or if the named axis is not a ``DimensionAxis``).

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 3, "length")
    >>> f = cx.field(np.ones(3), x)
    >>> ucx.get_dimension(f)
    PhysicalType('length')
    >>> ucx.get_dimension(f, "x")
    PhysicalType('length')
    >>> ucx.get_dimension(f, "y") is None
    True
    """
    axes = field.axes
    if axis_name is not None:
        coord = axes.get(axis_name)
        if isinstance(coord, DimensionAxis):
            return coord.dimension
        return None
    # Return the dimension of the first DimensionAxis found.
    for coord in axes.values():
        if isinstance(coord, DimensionAxis):
            return coord.dimension
    return None


def _require_same_dim(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> None:
    """Raise :class:`DimensionMismatchError` if the fields differ in dimension.

    Only ``DimensionAxis`` coordinates are considered.  Axes without a
    ``DimensionAxis`` (or fields that share no ``DimensionAxis`` axes) are
    silently ignored.
    """
    for name, coord1 in f1.axes.items():
        if not isinstance(coord1, DimensionAxis):
            continue
        coord2 = f2.axes.get(name)
        if coord2 is None:
            continue
        if not isinstance(coord2, DimensionAxis):
            continue
        if coord1.dimension != coord2.dimension:
            raise DimensionMismatchError(
                f"Cannot add/subtract fields: axis {name!r} has dimension "
                f"{coord1.dimension!r} in the first field but "
                f"{coord2.dimension!r} in the second field."
            )


def _compute_mul_axes(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> dict[str, DimensionAxis]:
    """Compute new ``DimensionAxis`` coordinates after multiplying two fields.

    For each axis that appears in *both* fields with a ``DimensionAxis``, the
    resulting dimension is the **product** of the two input dimensions.  Axes
    that appear in only one field are left unchanged.
    """
    new_axes: dict[str, DimensionAxis] = {}
    all_names = set(f1.axes) | set(f2.axes)
    for name in all_names:
        c1 = f1.axes.get(name)
        c2 = f2.axes.get(name)
        if isinstance(c1, DimensionAxis) and isinstance(c2, DimensionAxis):
            new_dim = c1.dimension * c2.dimension
            new_axes[name] = DimensionAxis(name, c1.size, new_dim)
        elif isinstance(c1, DimensionAxis):
            new_axes[name] = c1
        elif isinstance(c2, DimensionAxis):
            new_axes[name] = c2
    return new_axes


def _compute_div_axes(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> dict[str, DimensionAxis]:
    """Compute new ``DimensionAxis`` coordinates after dividing two fields."""
    new_axes: dict[str, DimensionAxis] = {}
    all_names = set(f1.axes) | set(f2.axes)
    for name in all_names:
        c1 = f1.axes.get(name)
        c2 = f2.axes.get(name)
        if isinstance(c1, DimensionAxis) and isinstance(c2, DimensionAxis):
            new_dim = c1.dimension / c2.dimension
            new_axes[name] = DimensionAxis(name, c1.size, new_dim)
        elif isinstance(c1, DimensionAxis):
            new_axes[name] = c1
        elif isinstance(c2, DimensionAxis):
            new_axes[name] = c2
    return new_axes


def _replace_dimension_axes(
    field: "coordax.Field",
    new_axes: dict[str, DimensionAxis],
) -> "coordax.Field":
    """Return a field with its ``DimensionAxis`` coordinates replaced.

    Uses coordax's ``untag`` / ``tag`` round-trip to swap out one or more
    ``DimensionAxis`` coordinates on *field*.
    """
    import coordax as cx

    result = field
    for name, new_coord in new_axes.items():
        if name not in result.axes:
            continue
        if not isinstance(result.axes[name], DimensionAxis):
            continue
        # Remove the named axis, then reattach with the new DimensionAxis.
        result = cx.tag(cx.untag(result, name), new_coord)
    return result


# ---------------------------------------------------------------------------
# Public API


def dadd(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> "coordax.Field":
    """Add two fields, raising if their physical dimensions are incompatible.

    Parameters
    ----------
    f1, f2:
        Fields to add.  Any ``DimensionAxis`` axes present in both fields must
        carry the same physical dimension.

    Returns
    -------
    :class:`coordax.Field`
        The element-wise sum.

    Raises
    ------
    DimensionMismatchError
        If the fields have incompatible physical dimensions on any shared
        ``DimensionAxis``.

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 3, "length")
    >>> f = cx.field(np.array([1.0, 2.0, 3.0]), x)
    >>> ucx.dadd(f, f)
    <Field dims=('x',) shape=(3,) axes={'x': DimensionAxis} >
    """
    _require_same_dim(f1, f2)
    return f1 + f2


def dsub(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> "coordax.Field":
    """Subtract two fields, raising if their physical dimensions are incompatible.

    Parameters
    ----------
    f1, f2:
        Fields to subtract.  Shared ``DimensionAxis`` axes must carry the same
        physical dimension.

    Returns
    -------
    :class:`coordax.Field`
        The element-wise difference.

    Raises
    ------
    DimensionMismatchError
        If the fields have incompatible physical dimensions on any shared
        ``DimensionAxis``.

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 3, "length")
    >>> f = cx.field(np.array([1.0, 2.0, 3.0]), x)
    >>> ucx.dsub(f, f)
    <Field dims=('x',) shape=(3,) axes={'x': DimensionAxis} >
    """
    _require_same_dim(f1, f2)
    return f1 - f2


def dmul(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> "coordax.Field":
    """Multiply two fields, computing the resulting physical dimension.

    For each ``DimensionAxis`` that appears in both fields, the result axis
    gets the **product** of the two input dimensions.

    Parameters
    ----------
    f1, f2:
        Fields to multiply.

    Returns
    -------
    :class:`coordax.Field`
        The element-wise product with updated ``DimensionAxis`` coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x_len = ucx.DimensionAxis("x", 3, "length")
    >>> f_len = cx.field(np.array([1.0, 2.0, 3.0]), x_len)
    >>> result = ucx.dmul(f_len, f_len)
    >>> ucx.get_dimension(result)
    PhysicalType('area')
    """
    new_axes = _compute_mul_axes(f1, f2)
    result = f1 * f2
    return _replace_dimension_axes(result, new_axes)


def ddiv(
    f1: "coordax.Field",
    f2: "coordax.Field",
) -> "coordax.Field":
    """Divide two fields, computing the resulting physical dimension.

    For each ``DimensionAxis`` appearing in both fields, the result axis gets
    the **quotient** of the two input dimensions.

    Parameters
    ----------
    f1, f2:
        Fields to divide (*f1* / *f2*).

    Returns
    -------
    :class:`coordax.Field`
        The element-wise quotient with updated ``DimensionAxis`` coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x_len = ucx.DimensionAxis("x", 3, "length")
    >>> f_len = cx.field(np.array([2.0, 4.0, 6.0]), x_len)
    >>> f_len2 = cx.field(np.array([1.0, 2.0, 3.0]), x_len)
    >>> result = ucx.ddiv(f_len, f_len2)
    >>> ucx.get_dimension(result)
    PhysicalType('dimensionless')
    """
    new_axes = _compute_div_axes(f1, f2)
    result = f1 / f2
    return _replace_dimension_axes(result, new_axes)


def dpow(
    field: "coordax.Field",
    exponent: int | float,
) -> "coordax.Field":
    """Raise a field to a power, computing the resulting physical dimension.

    Each ``DimensionAxis`` in the field is updated so that its dimension is
    raised to *exponent*.

    Parameters
    ----------
    field:
        The field to exponentiate.
    exponent:
        The power to raise the field to.

    Returns
    -------
    :class:`coordax.Field`
        The field raised to *exponent*, with updated ``DimensionAxis``
        coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import coordax as cx
    >>> import unxt_coordax as ucx

    >>> x = ucx.DimensionAxis("x", 3, "length")
    >>> f = cx.field(np.array([1.0, 2.0, 3.0]), x)
    >>> result = ucx.dpow(f, 2)
    >>> ucx.get_dimension(result)
    PhysicalType('area')
    """
    import coordax as cx  # local import to avoid circular dependency at module level

    new_axes: dict[str, DimensionAxis] = {}
    for name, coord in field.axes.items():
        if isinstance(coord, DimensionAxis):
            new_dim = coord.dimension**exponent
            new_axes[name] = DimensionAxis(name, coord.size, new_dim)

    result = field**exponent
    return _replace_dimension_axes(result, new_axes)
