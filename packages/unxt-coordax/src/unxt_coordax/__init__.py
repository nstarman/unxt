"""unxt-coordax: interoperability between unxt and coordax."""

# Register unxt.AbstractQuantity with coordax.NDArray and extend
# unxt.dimension_of for coordax types.
from unxt_coordax._src import dispatch, register  # noqa: F401

from unxt_coordax._src.dimension_axis import DimensionAxis

__all__ = [
    "DimensionAxis",
]
