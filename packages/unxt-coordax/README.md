# unxt-coordax

Interoperability between [unxt](https://github.com/GalacticDynamics/unxt) and
[coordax](https://coordax.readthedocs.io/en/latest/index.html).

## Overview

`unxt-coordax` provides coordinate types for use with `coordax` that are
dimension-aware via `unxt`.  The primary addition is `DimensionAxis`: a
one-dimensional `coordax.Coordinate` that carries an
`astropy.units.PhysicalType` label so that the physical dimensions of field
values are tracked.

## Features

- `DimensionAxis` – a `coordax.Coordinate` subclass that stores a physical
  dimension label (e.g. `"length"`, `"time"`, `"mass"`).
- An overload of `unxt.dimension_of` that works with `DimensionAxis` and
  `coordax.Field` objects.
- Registration of `unxt.AbstractQuantity` as a `coordax` NDArray so that
  Quantity arrays can be used as field data alongside plain NumPy and JAX
  arrays.

## Installation

```bash
pip install unxt-coordax
```

## Quick start

```python
import numpy as np
import jax.numpy as jnp
import coordax as cx
import unxt as u
import unxt_coordax as ucx

# Create a dimension-annotated axis
x = ucx.DimensionAxis("x", 5, "length")

# Field backed by a NumPy array
f_np = cx.field(np.ones(5), x)

# Field backed by a JAX array
f_jax = cx.field(jnp.ones(5), x)

# Field backed by a unxt Quantity
f_q = cx.field(u.Quantity(np.ones(5), "m"), x)

# Query the physical dimension of a field via unxt.dimension_of
u.dimension_of(x)   # PhysicalType('length')
u.dimension_of(f_np)  # PhysicalType('length')

# Standard arithmetic operators work directly on fields
f_sum = f_np + f_np  # addition (dimension must match on shared axes)
f_sq  = f_np ** 2    # squaring (values updated, dimension label unchanged)
```

## Arithmetic

Standard `coordax.Field` operators (`+`, `-`, `*`, `/`, `**`) work directly
without any wrappers.

**Addition and subtraction** are dimension-safe by construction: coordax
enforces that operands sharing the same axis name must carry **identical**
coordinate objects.  Because two `DimensionAxis` instances with different
`dimension` attributes are not equal, adding or subtracting fields whose
shared axes have different physical dimensions raises a `ValueError` from
coordax automatically.

**Multiplication, division, and powers** work on the values but **do not
update the dimension label** on the resulting `DimensionAxis`.  The result
field retains the coordinate objects of the first operand unchanged.  If you
need the resulting dimension, compute it explicitly:

```python
import astropy.units as apyu

dim_result = u.dimension_of(f_a) * u.dimension_of(f_b)
```

and attach a new `DimensionAxis` with that dimension to the result field.

## Dimension arithmetic with `unxt.dimension_of`

Where `x` is a `DimensionAxis` and `f` is a `coordax.Field` with a `DimensionAxis`:

| Expression             | Result |
|------------------------|--------|
| `u.dimension_of(x)`    | physical dimension stored on `x` |
| `u.dimension_of(f)`    | physical dimension of the first `DimensionAxis` in `f` |
| `dim_a * dim_b`        | product physical type (e.g. `length * time` → `absement`) |
| `dim_a / dim_b`        | quotient physical type (e.g. `length / time` → `velocity`) |
| `dim_a ** n`           | power physical type (e.g. `length ** 2` → `area`) |
