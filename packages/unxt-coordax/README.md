# unxt-coordax

Interoperability between [unxt](https://github.com/GalacticDynamics/unxt) and
[coordax](https://coordax.readthedocs.io/en/latest/index.html).

## Overview

`unxt-coordax` provides coordinate types for use with `coordax` that are
dimension-aware via `unxt`.  The primary addition is `DimensionAxis`: a
one-dimensional `coordax.Coordinate` that carries an
`astropy.units.PhysicalType` label so that the physical dimensions of field
values are tracked and enforced.

## Features

- `DimensionAxis` – a `coordax.Coordinate` subclass that stores a physical
  dimension label (e.g. `"length"`, `"time"`, `"mass"`).
- Dimension-checked arithmetic helpers (`dadd`, `dsub`, `dmul`, `ddiv`,
  `dpow`) that raise `DimensionMismatchError` when operands have incompatible
  physical dimensions, and compute the correct result dimension for
  multiplication, division, and powers.
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
import astropy.units as apyu
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

# Dimension-checked addition (raises if dimensions differ)
f_sum = ucx.dadd(f_np, f_np)

# Dimension-aware multiplication (result gets the combined dimension)
t = ucx.DimensionAxis("t", 5, "time")
f_time = cx.field(np.ones(5), t)
f_absement = ucx.dmul(f_np, f_time)  # result axes: x (length), t (time)

# Power (dimension of result is length**2)
f_sq = ucx.dpow(f_np, 2)
```

## Dimension arithmetic

| Operation     | Input dims    | Result dim             |
|---------------|---------------|------------------------|
| `dadd(a, b)`  | `D`, `D`      | `D`  (must match)      |
| `dsub(a, b)`  | `D`, `D`      | `D`  (must match)      |
| `dmul(a, b)`  | `D1`, `D2`    | `D1 * D2`              |
| `ddiv(a, b)`  | `D1`, `D2`    | `D1 / D2`              |
| `dpow(a, n)`  | `D`           | `D ** n`               |
