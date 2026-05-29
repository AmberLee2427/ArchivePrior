from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariableSpec:
    name: str
    value_column: str
    error_minus_column: str
    error_plus_column: str
    transform: str = "identity"


DEFAULT_NEA_VARIABLES: dict[str, VariableSpec] = {
    "planet_mass": VariableSpec("planet_mass", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2"),
    "planet_radius": VariableSpec("planet_radius", "pl_rade", "pl_radeerr1", "pl_radeerr2"),
    "stellar_mass": VariableSpec("stellar_mass", "st_mass", "st_masserr1", "st_masserr2"),
    "stellar_radius": VariableSpec("stellar_radius", "st_rad", "st_raderr1", "st_raderr2"),
    "orbital_period": VariableSpec("orbital_period", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2"),
    "semi_major_axis": VariableSpec("semi_major_axis", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2"),
    "equilibrium_temp": VariableSpec("equilibrium_temp", "pl_eqt", "pl_eqterr1", "pl_eqterr2"),
    "insolation_flux": VariableSpec("insolation_flux", "pl_insol", "pl_insolerr1", "pl_insolerr2"),
}


def _to_spec(variable: str | Mapping[str, Any] | VariableSpec) -> VariableSpec:
    if isinstance(variable, VariableSpec):
        return variable
    if isinstance(variable, str):
        if variable not in DEFAULT_NEA_VARIABLES:
            raise KeyError(f"Unknown variable '{variable}'. Provide an explicit VariableSpec.")
        return DEFAULT_NEA_VARIABLES[variable]
    data = dict(variable)
    required = {"name", "value_column", "error_minus_column", "error_plus_column"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"VariableSpec mapping is missing keys: {sorted(missing)}")
    return VariableSpec(
        name=str(data["name"]),
        value_column=str(data["value_column"]),
        error_minus_column=str(data["error_minus_column"]),
        error_plus_column=str(data["error_plus_column"]),
        transform=str(data.get("transform", "identity")),
    )


class VariableRegistry:
    """Map logical archive variables to engine-ready values and asymmetric errors."""

    def __init__(
        self,
        variables: Iterable[str | Mapping[str, Any] | VariableSpec] | None = None,
        source: str = "nea",
    ) -> None:
        self.source = source
        self._specs: list[VariableSpec] = []
        if variables is not None:
            for variable in variables:
                self.register(variable)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs)

    @property
    def specs(self) -> tuple[VariableSpec, ...]:
        return tuple(self._specs)

    def register(self, variable: str | Mapping[str, Any] | VariableSpec) -> VariableSpec:
        spec = _to_spec(variable)
        if any(existing.name == spec.name for existing in self._specs):
            raise ValueError(f"Variable '{spec.name}' is already registered.")
        self._specs.append(spec)
        return spec

    def _resolve_specs(self, variables: Sequence[str] | None = None) -> list[VariableSpec]:
        if variables is None:
            return list(self._specs)
        wanted = list(variables)
        spec_map = {spec.name: spec for spec in self._specs}
        missing = [name for name in wanted if name not in spec_map]
        if missing:
            raise KeyError(f"Unknown registered variables: {missing}")
        return [spec_map[name] for name in wanted]

    @staticmethod
    def _series(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
        for column in candidates:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce")
        raise KeyError(f"None of the candidate columns are present: {list(candidates)}")

    @staticmethod
    def _apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
        if transform == "identity":
            return values
        if transform == "log10":
            return np.log10(values)
        raise ValueError(f"Unsupported transform '{transform}'.")

    @staticmethod
    def _propagate_errors(values: np.ndarray, err_minus: np.ndarray, err_plus: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
        if transform == "identity":
            return err_minus, err_plus
        if transform == "log10":
            scale = values * np.log(10.0)
            return np.abs(err_minus / scale), np.abs(err_plus / scale)
        raise ValueError(f"Unsupported transform '{transform}'.")

    def transform_values(self, frame: pd.DataFrame, variables: Sequence[str] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Transform a DataFrame into logical variable values and return the row mask used."""
        specs = self._resolve_specs(variables)
        if not specs:
            return pd.DataFrame(index=frame.index.copy()), np.ones(len(frame), dtype=bool)

        mask = np.ones(len(frame), dtype=bool)
        transformed: dict[str, np.ndarray] = {}

        for spec in specs:
            values = self._series(frame, (spec.name, spec.value_column)).to_numpy(dtype=float)
            valid = np.isfinite(values)
            if spec.transform == "log10":
                valid &= values > 0.0
            mask &= valid
            transformed[spec.name] = self._apply_transform(values, spec.transform)

        data = pd.DataFrame({name: values[mask] for name, values in transformed.items()}).reset_index(drop=True)
        return data, mask

    def compile_dataframe(self, frame: pd.DataFrame, variables: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Compile a DataFrame into `(N, D)` values and `(N, D, 2)` asymmetric errors."""
        specs = self._resolve_specs(variables)
        if not specs:
            raise ValueError("No variables registered.")

        mask = np.ones(len(frame), dtype=bool)
        values_by_name: dict[str, np.ndarray] = {}
        errors_by_name: dict[str, np.ndarray] = {}

        for spec in specs:
            values = self._series(frame, (spec.name, spec.value_column)).to_numpy(dtype=float)
            err_minus = self._series(frame, (f"{spec.name}_err_minus", spec.error_minus_column)).to_numpy(dtype=float)
            err_plus = self._series(frame, (f"{spec.name}_err_plus", spec.error_plus_column)).to_numpy(dtype=float)

            valid = np.isfinite(values) & np.isfinite(err_minus) & np.isfinite(err_plus) & (err_minus > 0.0) & (err_plus > 0.0)
            if spec.transform == "log10":
                valid &= values > 0.0
            mask &= valid

            values_by_name[spec.name] = self._apply_transform(values, spec.transform)
            errors_by_name[spec.name] = np.stack(self._propagate_errors(values, err_minus, err_plus, spec.transform), axis=-1)

        filtered = frame.loc[mask].reset_index(drop=True)
        data = np.column_stack([values_by_name[spec.name][mask] for spec in specs])
        errors = np.stack([errors_by_name[spec.name][mask] for spec in specs], axis=1)
        return data, errors, filtered