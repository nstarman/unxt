__all__ = ["value_converter"]

import warnings
from typing import Any, NoReturn

import quax
from jaxtyping import Array, ArrayLike
from plum import dispatch

import quaxed.numpy as jnp

from .base import AbstractQuantity


@dispatch.abstract
def value_converter(obj: Any, /) -> Any:
    """Convert for the value field of an `AbstractQuantity` subclass."""
    raise NotImplementedError  # pragma: no cover


@dispatch
def value_converter(obj: quax.ArrayValue, /) -> Any:
    """Convert a `quax.ArrayValue` for the value field."""
    warnings.warn(
        f"'quax.ArrayValue' subclass {type(obj).__name__!r} does not have a registered "
        "converter. Returning the object as is.",
        category=UserWarning,
        stacklevel=2,
    )
    return obj


@dispatch
def value_converter(obj: ArrayLike | list[Any] | tuple[Any, ...], /) -> Array:
    """Convert an array-like object to a `jax.numpy.ndarray`.

    >>> import jax.numpy as jnp
    >>> from unxt.quantity import value_converter

    >>> value_converter([1, 2, 3])
    Array([1, 2, 3], dtype=int32)

    """
    return jnp.asarray(obj)


@dispatch
def value_converter(obj: AbstractQuantity, /) -> NoReturn:
    """Disallow conversion of `AbstractQuantity` to a value.

    >>> import unxt as u
    >>> from unxt.quantity import value_converter

    >>> try:
    ...     value_converter(u.Quantity(1, "m"))
    ... except TypeError as e:
    ...     print(e)
    Cannot convert 'Quantity[PhysicalType('length')]' to a value.
    For a Quantity, use the `.from_` constructor instead.

    """
    msg = (
        f"Cannot convert '{type(obj).__name__}' to a value. "
        "For a Quantity, use the `.from_` constructor instead."
    )
    raise TypeError(msg)
