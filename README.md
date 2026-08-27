# bvp_lrs

<p align="center">
<img src="figures/bvp_4_anim.gif" width="250"
</p>

This repository builds learning-rate schedules from boundary-value problems (BVPs). The idea is to shape a parameterized profile over the interval r in [0, 1] and then convert it into a smooth learning-rate window via a transform such as a linear or raised-cosine envelope.

The core building blocks live in:

- `bvp_lrs_np.py` for NumPy-based implementation
- `bvp_lrs_torch.py` for PyTorch-based implementations

## BVP families

The library includes several families:

#### Simpler BVPs

- `x2(r)`: 2-point BVP
- `x3(r, m)`: 3-point BVP
- `x4(r, m, e)`: 4-point BVP

#### Complex 8-point BVPs

- `x8M(r, m1, m2, m3, e1, e2, e3, c)`: M-shaped 8-point BVP
- `x8Dome(r, m1, m2, m3, e1, e2, e3, c)`: Dome-shaped 8-point BVP

Each bvp profile transforms into a learning-rate schedule with:

- p-th root linear window: `glin(x, p=1) = (1 - x)^(1/p)`
- p-th root raised cosine window: `grcos(x, p=1) = cos(0.5 * pi * x)^(2/p)`

They provide templates for learning-rate schedule design to be used when building optimizers.

## NumPy/PyTorch example

```python
import numpy as np
from bvp_lrs_np import x4, glin, grcos
# from bvp_lrs_torch import x4, glin, grcos

# normalized horizon of learning iterations
r = np.linspace(0.0, 1.0, 1000)

# 10% warmup, 20% constant
bvp_profile = x4(r, 0.1, 0.2)

# Dirichlet-energy minimizing window
schedule = glin(bvp_profile, p=1)

# Regularized Dirichlet-energy minimizing window
reg_schedule = grcos(bvp_profile, p=1)

```

## NumPy example

```python
import numpy as np
from bvp_lrs_np import x8M, x8Dome, grcos
# from bvp_lrs_torch import x8M, x8Dome, grcos 

r = np.linspace(0.0, 1.0, 1000)

profile = x8M(r, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.85)
schedule = glin(profile, p=1)


profile = x8Dome(r, 0.05, 0.2, 0.45, 0.0, 0.05, 0.05, 0.85)
schedule = glin(profile, p=1)
```

<p align="center">
<img src="figures/bvp8_plt.png" width="250"
</p>

## Visualization

Run the static or animated BVP exploration script:

```bash
python bvp4lrs_anim.py
```

The animation sweeps `m` and `e` together, making the transition between shape regimes easier to inspect.

---

```bash
python bvp8lrs_anim.py
```

The animation demonstrates the complicated shape construction capabilites of the `x8M` and `x8Dome` families.


<p align="center">
<img src="figures/bvp_8_anim.gif" width="250"
</p>

## References

Cite Preprint: https://somefunagba.github.io/assets/pdf/vantr_lrschedule.pdf

The paper develops a tust-region framework for learning-rate schedule construction in neural-network optimizers.
