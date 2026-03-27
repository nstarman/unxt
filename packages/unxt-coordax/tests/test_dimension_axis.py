"""Tests for unxt-coordax.

Tests cover:
- DimensionAxis creation and attribute access
- Field creation with NumPy, JAX, and unxt Quantity array data
- unxt.dimension_of dispatches for DimensionAxis and coordax.Field
- Standard arithmetic operators on fields (add/sub/mul/div/pow)
- Dimension mismatch enforcement via coordax coordinate identity check
- Dimension label non-propagation for mul/div/pow (documented limitation)
- JAX JIT compatibility
"""

from __future__ import annotations

import numpy as np
import pytest

import astropy.units as apyu
import coordax as cx
import jax
import jax.numpy as jnp
import unxt as u
import unxt_coordax as ucx
from unxt.dims import dimension_of

from unxt_coordax import DimensionAxis


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture()
def dim_length() -> apyu.PhysicalType:
    return apyu.get_physical_type("length")


@pytest.fixture()
def dim_time() -> apyu.PhysicalType:
    return apyu.get_physical_type("time")


@pytest.fixture()
def axis_length(dim_length: apyu.PhysicalType) -> DimensionAxis:
    return DimensionAxis("x", 5, dim_length)


@pytest.fixture()
def axis_time(dim_time: apyu.PhysicalType) -> DimensionAxis:
    return DimensionAxis("x", 5, dim_time)


# ---------------------------------------------------------------------------
# DimensionAxis creation


class TestDimensionAxisCreation:
    def test_from_physical_type(self, dim_length: apyu.PhysicalType) -> None:
        ax = DimensionAxis("x", 5, dim_length)
        assert ax.name == "x"
        assert ax.size == 5
        assert ax.dimension == dim_length

    def test_from_string(self, dim_length: apyu.PhysicalType) -> None:
        ax = DimensionAxis("x", 5, "length")
        assert ax.dimension == dim_length

    def test_dims_property(self, axis_length: DimensionAxis) -> None:
        assert axis_length.dims == ("x",)

    def test_shape_property(self, axis_length: DimensionAxis) -> None:
        assert axis_length.shape == (5,)

    def test_repr(self, axis_length: DimensionAxis) -> None:
        r = repr(axis_length)
        assert "DimensionAxis" in r
        assert "'x'" in r
        assert "5" in r
        assert "length" in r

    def test_equality(self, dim_length: apyu.PhysicalType) -> None:
        ax1 = DimensionAxis("x", 5, dim_length)
        ax2 = DimensionAxis("x", 5, dim_length)
        ax3 = DimensionAxis("y", 5, dim_length)
        assert ax1 == ax2
        assert ax1 != ax3

    def test_hashable(self, axis_length: DimensionAxis) -> None:
        # Must be hashable for JAX static pytree registration.
        h = hash(axis_length)
        assert isinstance(h, int)

    def test_default_dimension(self) -> None:
        ax = DimensionAxis("x", 3)
        assert ax.dimension == apyu.get_physical_type("dimensionless")


# ---------------------------------------------------------------------------
# Field creation with different array backends


class TestFieldCreation:
    def test_numpy_array(self, axis_length: DimensionAxis) -> None:
        data = np.ones(5)
        f = cx.field(data, axis_length)
        assert isinstance(f, cx.Field)
        assert f.dims == ("x",)
        assert f.shape == (5,)

    def test_jax_array(self, axis_length: DimensionAxis) -> None:
        data = jnp.ones(5)
        f = cx.field(data, axis_length)
        assert isinstance(f, cx.Field)
        assert f.dims == ("x",)

    def test_unxt_quantity(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.ones(5), "m")
        f = cx.field(data, axis_length)
        assert isinstance(f, cx.Field)
        assert f.dims == ("x",)
        assert f.shape == (5,)

    def test_unxt_quantity_jax_backend(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(jnp.ones(5), "m")
        f = cx.field(data, axis_length)
        assert isinstance(f, cx.Field)


# ---------------------------------------------------------------------------
# dimension_of dispatches


class TestDimensionOf:
    def test_dimension_of_axis(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        assert dimension_of(axis_length) == dim_length

    def test_dimension_of_field(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        f = cx.field(np.ones(5), axis_length)
        assert dimension_of(f) == dim_length

    def test_dimension_of_field_no_dimension_axis(self) -> None:
        plain = cx.SizedAxis("x", 5)
        f = cx.field(np.ones(5), plain)
        assert dimension_of(f) is None

    def test_dimension_of_via_unxt(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        # Verify the dispatch is reachable via the top-level unxt.dimension_of.
        f = cx.field(np.ones(5), axis_length)
        assert u.dimension_of(f) == dim_length


# ---------------------------------------------------------------------------
# Standard arithmetic operators
#
# Add / subtract:
#   coordax enforces that same-named axes carry identical coordinate objects.
#   Two DimensionAxis instances with different ``dimension`` attributes are
#   unequal, so adding/subtracting fields with incompatible dimensions raises
#   ValueError from coordax.
#
# Multiply / divide / power:
#   Operations succeed on the field *values*, but the DimensionAxis label on
#   the result is NOT updated (it retains the coordinate of the left operand).
#   This is a known limitation of the current coordax arithmetic model.
#   See the DimensionAxis docstring for details and how to compute the correct
#   result dimension manually.


class TestAddSubtract:
    def test_add_same_dimension_numpy(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = f + f
        np.testing.assert_allclose(np.asarray(result.data), 2 * np.arange(1, 6, dtype=float))

    def test_add_same_dimension_jax(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)
        result = f + f
        np.testing.assert_allclose(np.asarray(result.data), np.full(5, 2.0))

    def test_add_same_dimension_quantity(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.ones(5), "m")
        f = cx.field(data, axis_length)
        result = f + f
        assert isinstance(result, cx.Field)

    def test_add_incompatible_dimensions_raises(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
    ) -> None:
        # coordax enforces coordinate identity: same axis name but different
        # DimensionAxis objects (different dimensions) → ValueError.
        f1 = cx.field(np.ones(5), axis_length)
        f2 = cx.field(np.ones(5), axis_time)
        with pytest.raises(ValueError, match="Coordinates"):
            f1 + f2

    def test_sub_same_dimension(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([3.0, 6.0, 9.0, 12.0, 15.0]), axis_length)
        result = f - f
        np.testing.assert_allclose(np.asarray(result.data), np.zeros(5))

    def test_sub_incompatible_dimensions_raises(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
    ) -> None:
        f1 = cx.field(np.ones(5), axis_length)
        f2 = cx.field(np.ones(5), axis_time)
        with pytest.raises(ValueError, match="Coordinates"):
            f1 - f2


class TestMultiplyDivide:
    """Multiply/divide operate on values; dimension label is NOT propagated.

    This is a documented limitation of the current coordax arithmetic model.
    The result field retains the coordinate of the left-hand operand.
    """

    def test_mul_same_axis_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = f * f
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([4.0, 9.0, 16.0, 25.0, 36.0])
        )

    def test_mul_retains_left_coord(self, axis_length: DimensionAxis) -> None:
        # The result retains the coordinate of the left operand unchanged.
        # NOTE: the dimension label is NOT updated to reflect area.
        f = cx.field(np.ones(5), axis_length)
        result = f * f
        assert result.axes["x"] == axis_length

    def test_mul_by_scalar(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = f * 2.0
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        )

    def test_scalar_times_field(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = 2.0 * f
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        )

    def test_div_same_axis_values_numpy(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([4.0, 9.0, 16.0, 25.0, 36.0]), axis_length)
        divisor = cx.field(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = f / divisor
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_div_same_axis_values_jax(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([4.0, 9.0, 16.0, 25.0, 36.0]), axis_length)
        divisor = cx.field(jnp.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = f / divisor
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_div_by_scalar(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), axis_length)
        result = f / 2.0
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_mul_quantity_values(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "m")
        f = cx.field(data, axis_length)
        result = f * f
        assert isinstance(result, cx.Field)

    def test_div_quantity_values(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), "m")
        f = cx.field(data, axis_length)
        result = f / f
        assert isinstance(result, cx.Field)


class TestPower:
    """Power operates on values; dimension label is NOT propagated.

    This is a documented limitation of the current coordax arithmetic model.
    The result field retains the coordinate of the base operand unchanged.
    """

    def test_pow_int(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = f**2
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        )

    def test_pow_retains_coord(self, axis_length: DimensionAxis) -> None:
        # NOTE: the dimension label is NOT updated (result axis is still
        # "length", not "length²").  This is the documented limitation.
        f = cx.field(np.ones(5), axis_length)
        result = f**2
        assert result.axes["x"] == axis_length

    def test_pow_quantity_values(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "m")
        f = cx.field(data, axis_length)
        result = f**2
        assert isinstance(result, cx.Field)

    def test_pow_jax(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = f**2
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        )


# ---------------------------------------------------------------------------
# JAX JIT compatibility


class TestJaxJIT:
    def test_add_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)

        @jax.jit
        def add_twice(x: cx.Field) -> cx.Field:
            return x + x

        result = add_twice(f)
        np.testing.assert_allclose(np.asarray(result.data), np.full(5, 2.0))

    def test_mul_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)

        @jax.jit
        def square(x: cx.Field) -> cx.Field:
            return x * x

        result = square(f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([1.0, 4.0, 9.0, 16.0, 25.0]),
        )

    def test_mul_by_scalar_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)

        @jax.jit
        def double(x: cx.Field) -> cx.Field:
            return x * 2.0

        result = double(f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )

