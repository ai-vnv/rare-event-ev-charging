"""Load the Surabaya case study (or any case stored in the same four files)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

VEHICLES = ("motorcycle", "car")


def default_data_dir() -> Path:
    """Folder that holds the bundled Surabaya data (``<repo>/data``)."""
    return Path(__file__).resolve().parents[2] / "data"


@dataclass
class Case:
    """One planning instance.

    Notation follows the paper: demand nodes ``I``, candidate stations ``J`` and
    vehicle types ``K``.
    """

    I: List[str]
    J: List[str]
    K: List[str]
    w: Dict[Tuple[str, str], int]            # w[i, k]   vehicles of type k at node i per day
    d: Dict[Tuple[str, str, str], float]     # d[i, j, k] travel time in minutes
    h: Dict[str, float]                      # h[j]      daily station cost (IDR)
    mu: Dict[str, float]                     # masked electricity demand: mean
    sigma: Dict[str, float]                  #                            standard deviation
    z: Dict[str, float]                      # outage threshold
    params: dict
    stations: pd.DataFrame
    demand: pd.DataFrame

    # -- model parameters (Table II of the paper) ---------------------------------
    @property
    def r(self) -> float:
        return float(self.params["electricity_price_idr_per_kwh"])

    @property
    def s(self) -> float:
        return float(self.params["penalty_idr_per_minute_per_vehicle"])

    @property
    def e(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.params["charge_energy_kwh"].items()}

    @property
    def t(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.params["charge_time_min"].items()}

    @property
    def g(self) -> float:
        return float(self.params["connector_daily_cost_idr"])

    @property
    def c(self) -> float:
        return float(self.params["connector_minutes_per_day"])

    @property
    def q(self) -> int:
        return int(self.params["max_connectors_per_station"])

    @property
    def N(self) -> int:
        return int(self.params["max_stations"])

    @property
    def dmax(self) -> float:
        return float(self.params["max_travel_time_min"])

    @property
    def pbar(self) -> float:
        return float(self.params["min_service_level"])

    @property
    def existing(self) -> List[str]:
        """Stations that already operate and therefore stay open in every design."""
        return list(self.params.get("existing_stations", []))

    # -- reliability under the masked Gaussian demand model (Table I) -------------
    @property
    def kappa(self) -> Dict[str, float]:
        """Standardised outage threshold ``(z_j - mu_j) / sigma_j``."""
        return {j: (self.z[j] - self.mu[j]) / self.sigma[j] for j in self.J}

    @property
    def reliability(self) -> Dict[str, float]:
        """Exact station reliability ``p_j = P(Z_j <= z_j)``."""
        return {j: float(norm.cdf(k)) for j, k in self.kappa.items()}

    def total_demand(self) -> Dict[str, int]:
        return {k: sum(self.w[i, k] for i in self.I) for k in self.K}


def load_case(data_dir: Optional[Path] = None) -> Case:
    """Read ``stations.csv``, ``demand.csv``, ``travel_time.csv`` and ``params.json``."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    if not (data_dir / "stations.csv").exists():
        raise FileNotFoundError(f"no case data in {data_dir}; clone the repository or pass data_dir")
    stations = pd.read_csv(data_dir / "stations.csv")
    demand = pd.read_csv(data_dir / "demand.csv")
    travel = pd.read_csv(data_dir / "travel_time.csv")
    with open(data_dir / "params.json") as f:
        params = json.load(f)

    I = list(demand["subdistrict"])
    J = list(stations["station"])
    K = list(VEHICLES)

    w = {(row.subdistrict, k): int(getattr(row, f"demand_{k}")) for row in demand.itertuples() for k in K}
    d = {(row.subdistrict, row.station, k): float(getattr(row, f"minutes_{k}"))
         for row in travel.itertuples() for k in K}
    missing = [(i, j) for i in I for j in J if (i, j, K[0]) not in d]
    if missing:
        raise ValueError(f"travel_time.csv has no row for {missing[:3]} ...")

    by = stations.set_index("station")
    return Case(
        I=I, J=J, K=K, w=w, d=d,
        h=by["daily_cost_idr"].astype(float).to_dict(),
        mu=by["demand_mean"].astype(float).to_dict(),
        sigma=by["demand_sd"].astype(float).to_dict(),
        z=by["outage_threshold"].astype(float).to_dict(),
        params=params, stations=stations, demand=demand,
    )


def scale_outage(p: Dict[str, float], factor: float) -> Dict[str, float]:
    """Reliabilities whose outage probability ``1 - p_j`` is multiplied by ``factor``.

    ``factor > 1`` underestimates reliability, ``factor < 1`` overestimates it.  The
    result always stays a valid probability, unlike scaling ``p_j`` itself.
    """
    return {j: float(np.clip(1.0 - factor * (1.0 - pj), 0.0, 1.0)) for j, pj in p.items()}
