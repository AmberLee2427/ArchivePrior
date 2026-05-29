import jax.numpy as jnp
from jax import random

from archiveprior import ArchiveConditionalPrior


def make_synthetic_catalog(n=600, seed=13):
    key = random.PRNGKey(seed)
    k1, k2, k3, k4 = random.split(key, 4)

    # Two latent sub-populations in log-mass/log-radius space.
    mix = random.bernoulli(k1, p=0.55, shape=(n,))
    means = jnp.where(mix[:, None], jnp.array([0.4, 0.1]), jnp.array([1.2, 0.6]))
    scales = jnp.where(mix[:, None], jnp.array([0.22, 0.18]), jnp.array([0.16, 0.12]))
    x_true = means + random.normal(k2, shape=(n, 2)) * scales

    # Asymmetric catalog uncertainties per datum and per dimension.
    sigma_minus = 0.03 + 0.05 * random.uniform(k3, shape=(n, 2))
    sigma_plus = 0.04 + 0.06 * random.uniform(k4, shape=(n, 2))

    # Simple asymmetric perturbation for synthetic observed data.
    eps = random.normal(k2, shape=(n, 2))
    sigma_sel = jnp.where(eps < 0, sigma_minus, sigma_plus)
    x_obs = x_true + eps * sigma_sel

    errors = jnp.stack([sigma_minus, sigma_plus], axis=-1)
    return x_obs, errors


if __name__ == "__main__":
    data, errors = make_synthetic_catalog()

    prior = ArchiveConditionalPrior(
        columns=["planet_mass", "planet_radius"],
        n_components=10,
        learning_rate=2e-2,
        svi_steps=2500,
        seed=7,
    )
    prior.fit(data, errors)

    print("Metadata:", prior.metadata)

    pts = jnp.array([[0.8, 0.3], [1.0, 0.5]])
    print("Log density:", prior.score_density(pts))

    cond = prior.condition({"planet_mass": 0.95})
    print("Conditional target columns:", cond.target_columns)
    print("Conditional component weights:", cond.component_weights)

    s = prior.sample_conditional(
        given_dict={"planet_mass": 0.95},
        target_columns=["planet_radius"],
        n_samples=5,
        seed=123,
    )
    print("Conditional samples:", s)

    marg = prior.marginalize(["planet_mass", "planet_radius"])
    print("Marginal target columns:", marg.target_columns)
