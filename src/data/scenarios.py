"""Demand-scenario generation for the UKCI 2026 robust allocation problem.

Phase B of the pipeline: convert the per-region quantile forecasts produced
by ``forecasting.composite_loss.mc_dropout_quantiles`` into a discrete set
of demand scenarios consumed by the MILP / metaheuristics in
``optimization.allocate``.

Per ``02_METHODOLOGY.md`` §2.1, the canonical scenario set is:

==========  ================================  =================
Scenario s  Demand :math:`d_{i,h}^s`           Weight :math:`\\pi_s`
==========  ================================  =================
Low         :math:`q_{i,h}^{0.1}`             0.20
Median      :math:`q_{i,h}^{0.5}`             0.60
High        :math:`q_{i,h}^{0.9}`             0.20
==========  ================================  =================

For Experiment E5 (worst-case planning) an optional tail scenario at
:math:`q^{0.95}` with weight 0.05 can be appended and the weights renormalised.

This module is deliberately small — the heavy lifting lives in the
forecasting and optimisation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical 3-point scenario set
# ---------------------------------------------------------------------------

CANONICAL_QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
CANONICAL_NAMES: tuple[str, str, str] = ("low", "median", "high")
CANONICAL_WEIGHTS: tuple[float, float, float] = (0.20, 0.60, 0.20)

TAIL_QUANTILE: float = 0.95
TAIL_NAME: str = "tail"
TAIL_WEIGHT: float = 0.05


@dataclass
class ScenarioSet:
    """A set of discrete demand scenarios over ``(region, horizon)``.

    Attributes:
        demand: ``DataFrame`` indexed by ``(region, horizon, scenario)`` with
            a single column ``d`` carrying the demand value.
        weights: ``Series`` indexed by ``scenario`` summing to 1.0.
        meta: Free-form metadata dict (provenance, dates, model versions).
    """

    demand: pd.DataFrame
    weights: pd.Series
    meta: dict = field(default_factory=dict)

    def scenarios(self) -> list[str]:
        """Return the scenario names in canonical order."""
        return list(self.weights.index)

    def at(self, region: str, horizon: int, scenario: str) -> float:
        """Convenience accessor for a single demand entry."""
        return float(self.demand.loc[(region, horizon, scenario), "d"])

    def to_global_matrix(self) -> tuple[np.ndarray, list[str], list[int], list[str]]:
        """Return a 3-D demand array indexed by (region, horizon, scenario).

        Returns:
            ``(matrix, regions, horizons, scenarios)`` where ``matrix`` has
            shape ``(|R|, |H|, |S|)`` and the three lists give the axis labels.
        """
        regions = sorted(self.demand.index.get_level_values("region").unique())
        horizons = sorted(self.demand.index.get_level_values("horizon").unique())
        scenarios = self.scenarios()
        matrix = np.zeros((len(regions), len(horizons), len(scenarios)))
        for i, r in enumerate(regions):
            for j, h in enumerate(horizons):
                for k, s in enumerate(scenarios):
                    matrix[i, j, k] = self.demand.loc[(r, h, s), "d"]
        return matrix, regions, horizons, scenarios


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _validate_quantile_frame(quantile_frame: pd.DataFrame) -> None:
    required = {"q_low", "q_mid", "q_hi"}
    if not required.issubset(quantile_frame.columns):
        raise ValueError(
            f"quantile_frame must contain columns {required}; "
            f"got {set(quantile_frame.columns)}"
        )
    if not set(quantile_frame.index.names) >= {"region", "horizon"}:
        raise ValueError(
            "quantile_frame must be indexed by at least (region, horizon)."
        )


def build_scenarios(
    quantile_frame: pd.DataFrame,
    *,
    include_tail: bool = False,
    tail_column: str | None = None,
) -> ScenarioSet:
    """Build the canonical 3-point (optionally 4-point) scenario set.

    Args:
        quantile_frame: ``DataFrame`` indexed by ``(region, horizon)`` with
            at minimum columns ``q_low, q_mid, q_hi`` corresponding to the
            10/50/90 quantiles of the forecast. Optionally a fourth column
            (named via ``tail_column``) gives the 95% quantile for tail
            scenarios.
        include_tail: If ``True``, append a 4th tail scenario at
            ``tail_column`` (default name ``"q_tail"``) and renormalise.
        tail_column: Name of the tail-quantile column. Defaults to
            ``"q_tail"`` when ``include_tail`` is True.

    Returns:
        A ``ScenarioSet``.
    """
    _validate_quantile_frame(quantile_frame)

    names: list[str] = list(CANONICAL_NAMES)
    weights: list[float] = list(CANONICAL_WEIGHTS)
    columns: dict[str, str] = {
        "low": "q_low",
        "median": "q_mid",
        "high": "q_hi",
    }

    if include_tail:
        tcol = tail_column or "q_tail"
        if tcol not in quantile_frame.columns:
            raise ValueError(
                f"include_tail=True but column {tcol!r} not in quantile_frame."
            )
        names.append(TAIL_NAME)
        weights.append(TAIL_WEIGHT)
        columns[TAIL_NAME] = tcol

    # Renormalise weights to sum to 1 (only meaningful when tail added).
    total = sum(weights)
    weights = [w / total for w in weights]

    records: list[dict] = []
    for (region, horizon), row in quantile_frame.iterrows():
        for name in names:
            records.append(
                {
                    "region": region,
                    "horizon": int(horizon),
                    "scenario": name,
                    "d": float(row[columns[name]]),
                }
            )
    demand = pd.DataFrame.from_records(records).set_index(
        ["region", "horizon", "scenario"]
    )
    weights_series = pd.Series(weights, index=names, name="weight")
    return ScenarioSet(
        demand=demand,
        weights=weights_series,
        meta={"include_tail": include_tail, "quantile_columns": columns},
    )


def from_mc_dropout(
    quantiles: Mapping[float, pd.DataFrame],
    *,
    include_tail: bool = False,
    tail_quantile: float = TAIL_QUANTILE,
) -> ScenarioSet:
    """Convenience builder from the output of
    ``forecasting.composite_loss.mc_dropout_quantiles``.

    Args:
        quantiles: Mapping ``p -> DataFrame`` where each DataFrame is indexed
            by ``(region, horizon)`` and has one column with the quantile
            point forecast at probability ``p``. Must contain at least
            ``0.1``, ``0.5``, ``0.9``. If ``include_tail`` is True, must also
            contain ``tail_quantile``.
        include_tail: Whether to append a tail scenario at
            ``tail_quantile``.
        tail_quantile: Quantile probability for the tail scenario.

    Returns:
        A ``ScenarioSet``.
    """
    required = {0.1, 0.5, 0.9}
    if not required.issubset(quantiles.keys()):
        raise ValueError(
            f"quantiles must contain keys {required}; got {set(quantiles.keys())}"
        )

    frames = []
    for p, name in [(0.1, "q_low"), (0.5, "q_mid"), (0.9, "q_hi")]:
        df = quantiles[p].copy()
        # Accept single-column DataFrames whose column name varies.
        df.columns = [name]
        frames.append(df)
    qf = pd.concat(frames, axis=1)

    if include_tail:
        if tail_quantile not in quantiles:
            raise ValueError(
                f"include_tail=True but quantiles[{tail_quantile}] is missing."
            )
        tail = quantiles[tail_quantile].copy()
        tail.columns = ["q_tail"]
        qf = pd.concat([qf, tail], axis=1)

    return build_scenarios(qf, include_tail=include_tail, tail_column="q_tail")

