# ArchivePrior

NumPyro/JAX Dirichlet Process Gaussian Mixture Model (DP-GMM) for building an interactive empirical prior from NASA Exoplanet Archive data with asymmetric measurement errors.

## What this implements

- Truncated stick-breaking DP-GMM in NumPyro.
- Latent population model: `x_true ~ G` where `G` is a Gaussian mixture.
- Observation model with asymmetric uncertainties via split-normal likelihood:
  - `x_observed ~ SplitNormal(x_true, sigma_minus, sigma_plus)`
- JAX-native arrays and vectorized inference.
- Public class API:
  - `fit(data, errors)`
  - `score_density(values)`
  - `condition(given_dict)`
  - `sample_conditional(given_dict, target_columns, n_samples)`
  - `marginalize(keep_columns)`
- `metadata` dictionary for reproducibility.
- The installable package lives under `src/archiveprior`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Input shapes

- `data`: `(N, D)` JAX array of observed values.
- `errors`: `(N, D, 2)` JAX array of asymmetric errors:
  - `errors[..., 0] = sigma_minus`
  - `errors[..., 1] = sigma_plus`

All error entries must be strictly positive.

## Quick usage

```python
import jax.numpy as jnp
from archiveprior import ArchiveConditionalPrior

prior = ArchiveConditionalPrior(
    columns=["planet_mass", "planet_radius"],
    n_components=12,
    learning_rate=1e-2,
    svi_steps=3000,
    seed=0,
)

prior.fit(data, errors)
logp = prior.score_density(jnp.array([[1.0, 0.5]]))
cond = prior.condition({"planet_mass": 1.0})
samples = prior.sample_conditional({"planet_mass": 1.0}, ["planet_radius"], 256)
marg = prior.marginalize(["planet_mass", "planet_radius"])
```

For archive-backed workflows, use `archiveprior.ExoPrior` with a `VariableRegistry` and `ArchiveClient`.

## Demo

```bash
python examples/demo_fit_and_condition.py
```

## Notes

- The mixture uses diagonal component covariance for speed and stability.
- Conditioning updates component weights using the observed dimensions and keeps unknown-dimension Gaussian parameters analytically consistent with the diagonal model.
- The learned smooth prior is a mixture model, not a KDE.
