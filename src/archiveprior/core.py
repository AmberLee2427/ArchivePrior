from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import pandas as pd

from .client import ArchiveClient
from .engine import ArchiveConditionalPrior
from .registry import VariableRegistry


class ArchivePrior:
    """User-facing wrapper that fetches archive data, transforms variables, and fits the engine."""

    def __init__(
        self,
        variables: Iterable[str | Mapping[str, Any]] | None,
        source: str = "nea",
        cache_dir: str | None = None,
        learning_rate: float = 1e-2,
        svi_steps: int = 3_000,
        seed: int = 0,
    ) -> None:
        if source != "nea":
            raise NotImplementedError("Only source='nea' is currently supported.")
        self.source = source
        self.registry = VariableRegistry(variables=variables, source=source)
        self.client = ArchiveClient(cache_dir=cache_dir)
        self.learning_rate = float(learning_rate)
        self.svi_steps = int(svi_steps)
        self.seed = int(seed)
        self.engine: ArchiveConditionalPrior | None = None
        self.provenance: dict[str, Any] | None = None
        self.raw_frame: pd.DataFrame | None = None
        self.training_frame: pd.DataFrame | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if self.engine is None:
            return {
                "pipeline_state": "initialized",
                "source": self.source,
                "variables": self.registry.variables,
            }
        return self.engine.metadata

    def _require_engine(self) -> ArchiveConditionalPrior:
        if self.engine is None:
            raise RuntimeError("Model has not been built yet. Call build() first.")
        return self.engine

    def build(self, n_components: int = 12, refresh: bool = False) -> "ExoPrior":
        """Fetch archive data, transform requested variables, and fit the NumPyro engine."""
        if not self.registry.specs:
            raise ValueError("No variables have been registered.")

        frame, provenance = self.client.fetch_pscomppars(refresh=refresh)
        data, errors, cleaned = self.registry.compile_dataframe(frame)

        self.engine = ArchiveConditionalPrior(
            columns=list(self.registry.variables),
            n_components=n_components,
            learning_rate=self.learning_rate,
            svi_steps=self.svi_steps,
            seed=self.seed,
        )
        self.engine.fit(jnp.asarray(data), jnp.asarray(errors))
        self.raw_frame = frame
        self.training_frame = cleaned
        self.provenance = {
            **provenance,
            "training_row_count": int(len(cleaned)),
            "requested_variables": list(self.registry.variables),
            "model_columns": list(self.registry.variables),
            "n_components": int(n_components),
        }
        return self

    def score_density(self, values: Any) -> jax.Array:
        return self._require_engine().score_density(values)

    def condition(self, given_dict: dict[str, float]):
        return self._require_engine().condition(given_dict)

    def sample_conditional(self, given_dict: dict[str, float], target_columns: list[str], n_samples: int, seed: int | None = None):
        return self._require_engine().sample_conditional(given_dict, target_columns, n_samples, seed=seed)

    def marginalize(self, keep_columns: list[str]):
        return self._require_engine().marginalize(keep_columns)

    def compare_solutions(self, solutions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Score solution dictionaries against the archive prior and return relative log probabilities."""
        engine = self._require_engine()
        if not solutions:
            return []

        frame = pd.DataFrame(list(solutions))
        transformed, mask = self.registry.transform_values(frame)
        if not bool(mask.all()):
            invalid_rows = frame.index[~mask].tolist()
            raise ValueError(f"Solutions missing required variables or invalid after transformation: {invalid_rows}")

        archive_log_prob = engine.score_density(jnp.asarray(transformed.to_numpy(dtype=float)))

        external = None
        for candidate in ("external_log_likelihood", "log_likelihood", "likelihood"):
            if candidate in frame.columns:
                external = pd.to_numeric(frame[candidate], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                break
        if external is None:
            external = jnp.zeros((len(frame),), dtype=jnp.float32)

        combined = archive_log_prob + jnp.asarray(external)
        relative = combined - jax.scipy.special.logsumexp(combined)

        results: list[dict[str, Any]] = []
        for index, (_, row) in enumerate(frame.iterrows()):
            item = dict(row)
            item["archive_log_prob"] = float(archive_log_prob[index])
            item["external_log_likelihood"] = float(external[index])
            item["archive_weighted_log_prob"] = float(combined[index])
            item["relative_log_prob"] = float(relative[index])
            item["relative_probability"] = float(jnp.exp(relative[index]))
            results.append(item)
        return results

    def __getattr__(self, name: str):
        if self.engine is None:
            raise AttributeError(name)
        if hasattr(self.engine, name):
            return getattr(self.engine, name)
        raise AttributeError(name)