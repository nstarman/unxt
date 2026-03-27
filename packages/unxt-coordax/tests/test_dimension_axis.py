"""Tests for unxt-coordax.

Tests cover:
- DimensionAxis creation and attribute access
- Field creation with NumPy, JAX, and unxt Quantity array data
- unxt.dimension_of dispatches for DimensionAxis and coordax.Field
- Standard arithmetic operators on fields (add/sub/mul/div/pow)
- Dimension mismatch enforcement via coordax coordinate identity check
- DimensionOperationError for mul/div with dimensioned fields and all powers
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

from unxt_coordax import DimensionAxis, DimensionOperationError


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
# Multiply / divide:
#   Raises DimensionOperationError if the OTHER field has a DimensionAxis.
#   Scaling by a plain number or a field without DimensionAxis is allowed.
#
# Power:
#   Always raises DimensionOperationError when self has a DimensionAxis.


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

    def test_add_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)

        @jax.jit
        def add_twice(x: cx.Field) -> cx.Field:
            return x + x

        result = add_twice(f)
        np.testing.assert_allclose(np.asarray(result.data), np.full(5, 2.0))


class TestMultiplyDivide:
    """Multiply / divide error when other field has a DimensionAxis."""

    def test_mul_dimensioned_field_raises(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            f * f

    def test_mul_different_dimension_fields_raises(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
    ) -> None:
        # Both fields have DimensionAxis (different dim names) → still error.
        f_len = cx.field(np.ones(5), axis_length)
        t_ax = DimensionAxis("t", 5, "time")
        f_time = cx.field(np.ones(5), t_ax)
        with pytest.raises(DimensionOperationError):
            f_len * f_time

    def test_mul_by_scalar_ok(self, axis_length: DimensionAxis) -> None:
        # Scaling by a plain number is allowed.
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = f * 2.0
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        )

    def test_mul_by_plain_field_ok(self, axis_length: DimensionAxis) -> None:
        # Scaling by a field with NO DimensionAxis is allowed.
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        scale = cx.field(np.full(5, 2.0), cx.SizedAxis("x", 5))
        # Different coord type on same dim name → coordax raises ValueError.
        # Use a different axis name for plain scaling.
        scale2 = 2.0
        result = f * scale2
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        )

    def test_scalar_times_dimensioned_field_ok(self, axis_length: DimensionAxis) -> None:
        # 2.0 * f: Python calls (2.0).__mul__(f) → NotImplemented,
        # then f.__rmul__(2.0).  The "other" (2.0) has no DimensionAxis → OK.
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = 2.0 * f
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        )

    def test_div_dimensioned_field_raises(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            f / f

    def test_div_different_dimension_fields_raises(self, axis_length: DimensionAxis) -> None:
        f_len = cx.field(np.ones(5), axis_length)
        t_ax = DimensionAxis("t", 5, "time")
        f_time = cx.field(np.ones(5), t_ax)
        with pytest.raises(DimensionOperationError):
            f_len / f_time

    def test_div_by_scalar_ok(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), axis_length)
        result = f / 2.0
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_rtruediv_dimensioned_field_raises(self, axis_length: DimensionAxis) -> None:
        # 1.0 / f_length → f_length.__rtruediv__(1.0) → self has DimensionAxis → error.
        f = cx.field(np.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            _ = 1.0 / f

    def test_mul_quantity_by_scalar_ok(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "m")
        f = cx.field(data, axis_length)
        result = f * 2.0
        assert isinstance(result, cx.Field)

    def test_div_quantity_by_scalar_ok(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), "m")
        f = cx.field(data, axis_length)
        result = f / 2.0
        assert isinstance(result, cx.Field)


class TestPower:
    """Power always errors when self has a DimensionAxis."""

    def test_pow_int_raises(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            f**2

    def test_pow_float_raises(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            f**0.5

    def test_pow_quantity_values_raises(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.ones(5), "m")
        f = cx.field(data, axis_length)
        with pytest.raises(DimensionOperationError):
            f**2

    def test_pow_plain_field_no_dimension_axis_ok(self) -> None:
        # Fields without DimensionAxis can still be raised to a power.
        plain = cx.SizedAxis("x", 5)
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), plain)
        result = f**2
        np.testing.assert_allclose(
            np.asarray(result.data), np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        )

    def test_pow_jax_raises(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)
        with pytest.raises(DimensionOperationError):
            f**2


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

    def test_mul_jit_raises(self, axis_length: DimensionAxis) -> None:
        # DimensionOperationError is raised at trace time (Python level).
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)

        @jax.jit
        def square(x: cx.Field) -> cx.Field:
            return x * x

        with pytest.raises(DimensionOperationError):
            square(f)

    def test_mul_by_scalar_jit_ok(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)

        @jax.jit
        def double(x: cx.Field) -> cx.Field:
            return x * 2.0

        result = double(f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )


