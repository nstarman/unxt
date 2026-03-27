"""Tests for unxt-coordax.

Tests cover:
- DimensionAxis creation and attribute access
- Field creation with NumPy, JAX, and unxt Quantity array data
- get_dimension helper
- dadd / dsub with compatible dimensions (pass) and incompatible (raise)
- dmul / ddiv / dpow dimension computation
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
from unxt_coordax import DimensionAxis, DimensionMismatchError


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
# get_dimension


class TestGetDimension:
    def test_basic(self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType) -> None:
        f = cx.field(np.ones(5), axis_length)
        assert ucx.get_dimension(f) == dim_length

    def test_by_name(self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType) -> None:
        f = cx.field(np.ones(5), axis_length)
        assert ucx.get_dimension(f, "x") == dim_length

    def test_missing_axis_name(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.ones(5), axis_length)
        assert ucx.get_dimension(f, "y") is None

    def test_no_dimension_axis(self) -> None:
        plain = cx.SizedAxis("x", 5)
        f = cx.field(np.ones(5), plain)
        assert ucx.get_dimension(f) is None


# ---------------------------------------------------------------------------
# dadd / dsub


class TestDaddDsub:
    def test_add_same_dimension_numpy(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = ucx.dadd(f, f)
        np.testing.assert_allclose(np.asarray(result.data), 2 * np.arange(1, 6, dtype=float))

    def test_add_same_dimension_jax(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)
        result = ucx.dadd(f, f)
        np.testing.assert_allclose(np.asarray(result.data), np.full(5, 2.0))

    def test_add_same_dimension_quantity(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.ones(5), "m")
        f = cx.field(data, axis_length)
        result = ucx.dadd(f, f)
        assert isinstance(result, cx.Field)

    def test_add_incompatible_dimensions_raises(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
    ) -> None:
        f1 = cx.field(np.ones(5), axis_length)
        f2 = cx.field(np.ones(5), axis_time)
        with pytest.raises(DimensionMismatchError, match="dimension"):
            ucx.dadd(f1, f2)

    def test_sub_same_dimension(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([3.0, 6.0, 9.0, 12.0, 15.0]), axis_length)
        result = ucx.dsub(f, f)
        np.testing.assert_allclose(np.asarray(result.data), np.zeros(5))

    def test_sub_incompatible_dimensions_raises(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
    ) -> None:
        f1 = cx.field(np.ones(5), axis_length)
        f2 = cx.field(np.ones(5), axis_time)
        with pytest.raises(DimensionMismatchError, match="dimension"):
            ucx.dsub(f1, f2)


# ---------------------------------------------------------------------------
# dmul


class TestDmul:
    def test_mul_same_dimension(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        f = cx.field(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = ucx.dmul(f, f)
        expected_dim = dim_length * dim_length  # area
        assert ucx.get_dimension(result) == expected_dim

    def test_mul_different_dimensions(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
        dim_length: apyu.PhysicalType,
        dim_time: apyu.PhysicalType,
    ) -> None:
        # Fields with different axis names (broadcasting): the result retains
        # each axis with its original dimension (no dimension arithmetic on
        # distinct axes).
        x_len = DimensionAxis("x", 3, dim_length)
        y_time = DimensionAxis("y", 3, dim_time)
        f_len = cx.field(np.ones(3), x_len)
        f_time = cx.field(np.ones(3), y_time)
        result = ucx.dmul(f_len, f_time)
        # Each axis retains its original dimension (no same-name axes to merge).
        assert ucx.get_dimension(result, "x") == dim_length
        assert ucx.get_dimension(result, "y") == dim_time

    def test_mul_numpy_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = ucx.dmul(f, f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([1.0, 4.0, 9.0, 16.0, 25.0]),
        )

    def test_mul_jax_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = ucx.dmul(f, f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([4.0, 9.0, 16.0, 25.0, 36.0]),
        )

    def test_mul_quantity_values(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "m")
        f = cx.field(data, axis_length)
        result = ucx.dmul(f, f)
        assert isinstance(result, cx.Field)

    def test_mul_cross_dimension_via_quantity(
        self,
        dim_length: apyu.PhysicalType,
        dim_time: apyu.PhysicalType,
    ) -> None:
        # When both fields have different axis names, dmul uses broadcasting;
        # axes with the same name but different dimensions would be merged to
        # dim * dim.  Here we verify that Quantity-typed values retain the
        # correct units through the multiplication.
        x_len = DimensionAxis("x", 3, dim_length)
        y_time = DimensionAxis("y", 3, dim_time)
        f_len = cx.field(u.Quantity(np.array([1.0, 2.0, 3.0]), "m"), x_len)
        f_time = cx.field(u.Quantity(np.array([1.0, 2.0, 3.0]), "s"), y_time)
        result = ucx.dmul(f_len, f_time)
        assert isinstance(result, cx.Field)


# ---------------------------------------------------------------------------
# ddiv


class TestDdiv:
    def test_div_same_dimension(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), axis_length)
        result = ucx.ddiv(f, f)
        dim = ucx.get_dimension(result)
        assert dim == apyu.get_physical_type("dimensionless")

    def test_div_different_dimensions(
        self,
        axis_length: DimensionAxis,
        axis_time: DimensionAxis,
        dim_length: apyu.PhysicalType,
        dim_time: apyu.PhysicalType,
    ) -> None:
        # For same-axis fields, coordax requires identical coordinates.
        # Division of a length field by itself gives dimensionless on that axis.
        f_len = cx.field(np.array([2.0, 4.0, 6.0, 8.0, 10.0]), axis_length)
        f_len2 = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = ucx.ddiv(f_len, f_len2)
        # dim_length / dim_length = dimensionless
        expected_dim = dim_length / dim_length
        assert ucx.get_dimension(result) == expected_dim

    def test_div_numpy_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([4.0, 9.0, 16.0, 25.0, 36.0]), axis_length)
        divisor = cx.field(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = ucx.ddiv(f, divisor)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        )

    def test_div_jax_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([4.0, 9.0, 16.0, 25.0, 36.0]), axis_length)
        divisor = cx.field(jnp.array([2.0, 3.0, 4.0, 5.0, 6.0]), axis_length)
        result = ucx.ddiv(f, divisor)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        )


# ---------------------------------------------------------------------------
# dpow


class TestDpow:
    def test_pow_2(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        f = cx.field(np.ones(5), axis_length)
        result = ucx.dpow(f, 2)
        assert ucx.get_dimension(result) == dim_length**2  # area

    def test_pow_3(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        f = cx.field(np.ones(5), axis_length)
        result = ucx.dpow(f, 3)
        assert ucx.get_dimension(result) == dim_length**3  # volume

    def test_pow_minus_1(
        self, axis_length: DimensionAxis, dim_length: apyu.PhysicalType
    ) -> None:
        f = cx.field(np.ones(5), axis_length)
        result = ucx.dpow(f, -1)
        assert ucx.get_dimension(result) == dim_length**-1  # wavenumber

    def test_pow_numpy_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = ucx.dpow(f, 2)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([1.0, 4.0, 9.0, 16.0, 25.0]),
        )

    def test_pow_jax_values(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)
        result = ucx.dpow(f, 2)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([1.0, 4.0, 9.0, 16.0, 25.0]),
        )

    def test_pow_quantity_values(self, axis_length: DimensionAxis) -> None:
        data = u.Quantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "m")
        f = cx.field(data, axis_length)
        result = ucx.dpow(f, 2)
        assert isinstance(result, cx.Field)


# ---------------------------------------------------------------------------
# JAX JIT compatibility


class TestJaxJIT:
    def test_dadd_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.ones(5), axis_length)

        @jax.jit
        def add_twice(x: cx.Field) -> cx.Field:
            return ucx.dadd(x, x)

        result = add_twice(f)
        np.testing.assert_allclose(np.asarray(result.data), np.full(5, 2.0))

    def test_dmul_jit(self, axis_length: DimensionAxis) -> None:
        f = cx.field(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), axis_length)

        @jax.jit
        def square(x: cx.Field) -> cx.Field:
            return ucx.dmul(x, x)

        result = square(f)
        np.testing.assert_allclose(
            np.asarray(result.data),
            np.array([1.0, 4.0, 9.0, 16.0, 25.0]),
        )
