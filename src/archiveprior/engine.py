from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam


Array = jax.Array


def _stick_breaking(v: Array) -> Array:
    """Convert stick-breaking variables to simplex weights."""
    if v.ndim != 1:
        raise ValueError("v must be a 1D array")
    if v.shape[0] == 1:
        return jnp.ones((1,), dtype=v.dtype)

    head = v[:-1]
    prefix = jnp.concatenate([jnp.ones((1,), dtype=v.dtype), jnp.cumprod(1.0 - head[:-1])])
    w_head = head * prefix
    w_last = jnp.cumprod(1.0 - head)[-1]
    return jnp.concatenate([w_head, w_last[None]])


class SplitNormal(dist.Distribution):
    """Independent split-normal likelihood for asymmetric measurement uncertainty."""

    arg_constraints = {
        "loc": constraints.real,
        "sigma_left": constraints.positive,
        "sigma_right": constraints.positive,
    }
    support = constraints.real
    reparametrized_params = ["loc", "sigma_left", "sigma_right"]

    def __init__(self, loc: Array, sigma_left: Array, sigma_right: Array, validate_args: bool | None = None):
        batch_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(sigma_left), jnp.shape(sigma_right))
        super().__init__(batch_shape=batch_shape, event_shape=(), validate_args=validate_args)
        self.loc = loc
        self.sigma_left = sigma_left
        self.sigma_right = sigma_right

    def log_prob(self, value: Array) -> Array:
        loc = self.loc
        sig_l = self.sigma_left
        sig_r = self.sigma_right
        sigma = jnp.where(value < loc, sig_l, sig_r)
        log_norm = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(sig_l + sig_r)
        quad = -0.5 * jnp.square((value - loc) / sigma)
        return log_norm + quad


@dataclass
class ConditionalMixture:
    component_weights: Array
    component_locs: Array
    component_scales: Array
    given_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


class ArchiveConditionalPrior:
    """
    Dirichlet-process-like Gaussian mixture with truncated stick-breaking variational inference.

    The model is:
      - latent population: x_true ~ G, where G is a Gaussian mixture (stick-breaking weights)
      - observation model: x_obs ~ SplitNormal(x_true, sigma_minus, sigma_plus)
    """

    def __init__(
        self,
        columns: list[str],
        n_components: int = 12,
        learning_rate: float = 1e-2,
        svi_steps: int = 3_000,
        seed: int = 0,
    ) -> None:
        if n_components < 2:
            raise ValueError("n_components must be >= 2")
        self.columns = list(columns)
        self.n_components = int(n_components)
        self.learning_rate = float(learning_rate)
        self.svi_steps = int(svi_steps)
        self.seed = int(seed)

        self._col_to_idx = {name: i for i, name in enumerate(self.columns)}
        self._fitted = False
        self._weights: Array | None = None
        self._locs: Array | None = None
        self._scales: Array | None = None
        self._svi_params: dict[str, Array] | None = None
        self.metadata: dict[str, Any] = {
            "pipeline_state": "initialized",
            "execution_timestamp": None,
            "data_size": None,
            "dimensions": len(self.columns),
            "n_components": self.n_components,
            "inference": "svi_autonormal",
            "observation_likelihood": "split_normal",
        }

    def _model(self, data: Array, errors: Array) -> None:
        n, d = data.shape

        alpha = numpyro.sample("alpha", dist.Gamma(2.0, 1.0))
        with numpyro.plate("sticks", self.n_components - 1):
            v = numpyro.sample("v", dist.Beta(1.0, alpha))
        with numpyro.plate("components", self.n_components):
            locs = numpyro.sample("locs", dist.Normal(jnp.zeros((d,)), 3.0 * jnp.ones((d,))).to_event(1))
            scales = numpyro.sample("scales", dist.LogNormal(jnp.zeros((d,)), 0.5 * jnp.ones((d,))).to_event(1))

        weights = _stick_breaking(jnp.concatenate([v, jnp.array([1.0], dtype=v.dtype)]))
        numpyro.deterministic("weights", weights)

        mix = dist.Categorical(probs=weights)
        comp = dist.Independent(dist.Normal(locs, scales), 1)

        with numpyro.plate("data", n):
            x_true = numpyro.sample("x_true", dist.MixtureSameFamily(mix, comp))
            # Combine per-dimension split-normal terms into one multivariate observation factor.
            obs_ll = jnp.sum(
                SplitNormal(
                    loc=x_true,
                    sigma_left=errors[..., 0],
                    sigma_right=errors[..., 1],
                ).log_prob(data),
                axis=-1,
            )
            numpyro.factor("obs_factor", obs_ll)

    def _require_fitted(self) -> None:
        if not self._fitted or self._weights is None or self._locs is None or self._scales is None:
            raise RuntimeError("Model is not fitted. Call fit(data, errors) first.")

    def fit(self, data: Array, errors: Array) -> "ArchiveConditionalPrior":
        """
        Fit the DP-GMM empirical prior with SVI.

        Args:
            data: (N, D) observed catalog values.
            errors: (N, D, 2) asymmetric uncertainties [sigma_minus, sigma_plus].
        """
        data = jnp.asarray(data)
        errors = jnp.asarray(errors)

        if data.ndim != 2:
            raise ValueError("data must have shape (N, D)")
        if errors.ndim != 3 or errors.shape[:2] != data.shape or errors.shape[2] != 2:
            raise ValueError("errors must have shape (N, D, 2)")
        if data.shape[1] != len(self.columns):
            raise ValueError("data second dimension must match len(columns)")
        if jnp.any(errors <= 0.0):
            raise ValueError("All uncertainty values in errors must be > 0")

        guide = AutoNormal(self._model)
        optimizer = Adam(self.learning_rate)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())

        rng_key = random.PRNGKey(self.seed)
        svi_state = svi.init(rng_key, data=data, errors=errors)

        @jax.jit
        def step(state: Any, x: Array, e: Array) -> tuple[Any, Array]:
            return svi.update(state, data=x, errors=e)

        losses = []
        for _ in range(self.svi_steps):
            svi_state, loss = step(svi_state, data, errors)
            losses.append(loss)

        params = svi.get_params(svi_state)
        med = guide.median(params)

        v_med = med["v"]
        self._weights = _stick_breaking(jnp.concatenate([v_med, jnp.array([1.0], dtype=v_med.dtype)]))
        self._locs = med["locs"]
        self._scales = med["scales"]
        self._svi_params = params
        self._fitted = True

        self.metadata.update(
            {
                "pipeline_state": "fitted",
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_size": int(data.shape[0]),
                "dimensions": int(data.shape[1]),
                "final_elbo": float(losses[-1]),
                "svi_steps": self.svi_steps,
            }
        )
        return self

    def _log_component_density(self, values: Array) -> Array:
        """Return shape (N, K) component log-densities under latent mixture."""
        self._require_fitted()
        assert self._weights is not None and self._locs is not None and self._scales is not None

        x = jnp.asarray(values)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != len(self.columns):
            raise ValueError("values must have trailing dimension D=len(columns)")

        x_e = x[:, None, :]  # (N,1,D)
        mu = self._locs[None, :, :]  # (1,K,D)
        sig = self._scales[None, :, :]  # (1,K,D)

        logp_dims = dist.Normal(mu, sig).log_prob(x_e)
        return jnp.sum(logp_dims, axis=-1)

    def score_density(self, values: Array) -> Array:
        """
        Evaluate exact latent mixture log-density log p(values) under learned G.

        Args:
            values: (D,) or (N, D) array.

        Returns:
            (N,) log-densities; for input (D,), returns shape (1,).
        """
        self._require_fitted()
        assert self._weights is not None

        comp_logp = self._log_component_density(values)  # (N,K)
        logw = jnp.log(self._weights)[None, :]
        return jax.scipy.special.logsumexp(logw + comp_logp, axis=-1)

    def _parse_given(self, given_dict: dict[str, float]) -> tuple[Array, Array, Array]:
        if not given_dict:
            given_idx = jnp.array([], dtype=jnp.int32)
            given_vals = jnp.array([], dtype=jnp.float32)
        else:
            missing = [k for k in given_dict if k not in self._col_to_idx]
            if missing:
                raise KeyError(f"Unknown columns in given_dict: {missing}")
            given_names = list(given_dict.keys())
            given_idx = jnp.array([self._col_to_idx[k] for k in given_names], dtype=jnp.int32)
            given_vals = jnp.array([given_dict[k] for k in given_names], dtype=jnp.float32)

        all_idx = jnp.arange(len(self.columns))
        if given_idx.size == 0:
            target_idx = all_idx
        else:
            mask = jnp.ones((len(self.columns),), dtype=bool).at[given_idx].set(False)
            target_idx = all_idx[mask]
        return given_idx, given_vals, target_idx

    def condition(self, given_dict: dict[str, float]) -> ConditionalMixture:
        """
        Return conditional mixture for unobserved dimensions given known values.

        For diagonal-covariance components this reweights component probabilities by p(given|k),
        while keeping per-component means/scales for unknown dimensions unchanged.
        """
        self._require_fitted()
        assert self._weights is not None and self._locs is not None and self._scales is not None

        given_idx, given_vals, target_idx = self._parse_given(given_dict)

        if given_idx.size == 0:
            cond_w = self._weights
        else:
            mu_g = self._locs[:, given_idx]  # (K,G)
            sig_g = self._scales[:, given_idx]  # (K,G)
            logp_given_k = jnp.sum(dist.Normal(mu_g, sig_g).log_prob(given_vals[None, :]), axis=-1)
            log_unnorm = jnp.log(self._weights) + logp_given_k
            cond_w = jax.nn.softmax(log_unnorm)

        target_locs = self._locs[:, target_idx]
        target_scales = self._scales[:, target_idx]
        given_cols = tuple(given_dict.keys())
        target_cols = tuple(self.columns[i] for i in target_idx.tolist())

        return ConditionalMixture(
            component_weights=cond_w,
            component_locs=target_locs,
            component_scales=target_scales,
            given_columns=given_cols,
            target_columns=target_cols,
        )

    def sample_conditional(
        self,
        given_dict: dict[str, float],
        target_columns: list[str],
        n_samples: int,
        seed: int | None = None,
    ) -> Array:
        """Draw samples from p(target_columns | given_dict)."""
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")

        cond = self.condition(given_dict)

        unknown_target = [c for c in target_columns if c not in cond.target_columns]
        if unknown_target:
            raise KeyError(
                "target_columns must be unobserved columns not present in given_dict. "
                f"Invalid targets: {unknown_target}"
            )

        # Align requested order with conditional target parameter order.
        idx = jnp.array([cond.target_columns.index(c) for c in target_columns], dtype=jnp.int32)
        locs = cond.component_locs[:, idx]
        scales = cond.component_scales[:, idx]

        key = random.PRNGKey(self.seed if seed is None else seed)
        key_k, key_x = random.split(key)
        k_samples = dist.Categorical(probs=cond.component_weights).sample(key_k, sample_shape=(n_samples,))
        loc_sel = locs[k_samples, :]
        scale_sel = scales[k_samples, :]
        eps = random.normal(key_x, shape=loc_sel.shape)
        return loc_sel + eps * scale_sel

    def marginalize(self, keep_columns: list[str]) -> ConditionalMixture:
        """
        Analytically marginalize nuisance dimensions in the latent Gaussian mixture.

        Returns a lower-dimensional mixture over keep_columns.
        """
        self._require_fitted()
        assert self._weights is not None and self._locs is not None and self._scales is not None

        missing = [c for c in keep_columns if c not in self._col_to_idx]
        if missing:
            raise KeyError(f"Unknown columns in keep_columns: {missing}")

        idx = jnp.array([self._col_to_idx[c] for c in keep_columns], dtype=jnp.int32)
        return ConditionalMixture(
            component_weights=self._weights,
            component_locs=self._locs[:, idx],
            component_scales=self._scales[:, idx],
            given_columns=tuple(),
            target_columns=tuple(keep_columns),
        )