# 📊 Matplotlib

If `matplotlib` is installed, `unxt` will automatically detect this and will
register parsers with `matplotlib` to enable plotting `Quantity` objects.

To ensure that a compatible version of `matplotlib` is installed, you can
install `unxt` with the `interop-mpl` extra:

::::{tab-set}

:::{tab-item} uv

```bash
uv add "unxt[interop-mpl]"
```

:::

:::{tab-item} pip

```bash
pip install unxt[interop-mpl]
```

::::

Once installed, you can plot `Quantity` objects directly with `matplotlib`:

```{code-block} python

import matplotlib.pyplot as plt
import jax.numpy as jnp
from unxt import Quantity

x = Quantity(jnp.linspace(0, 2 * jnp.pi, 100), "rad")
y = Quantity(jnp.sin(x.value), "")

plt.plot(x, y)
```