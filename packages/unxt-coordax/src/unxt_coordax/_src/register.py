"""Register unxt types with coordax so they can be used as field data."""

from __future__ import annotations

__all__: list[str] = []

from coordax import NDArray

# Register unxt's AbstractQuantity as a coordax NDArray so that Quantity
# instances can be used as the underlying array in a coordax Field.
# AbstractQuantity is a JAX pytree (via quax/equinox), and it already supports
# the shape/ndim/size/__getitem__ interface required for use inside a Field.
try:
    from unxt._src.quantity.base import AbstractQuantity

    NDArray.register(AbstractQuantity)
except ImportError:  # pragma: no cover
    pass
