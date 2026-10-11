"""Baseline model of expected daily energy use (spec R2) — hybrid two-part estimator.

    [1] base load sub-model   : kwh_offprod_d ~ cdd + hdd + day-type effects
        (off-production intervals only: nights, weekends, holidays, zero-volume days)
    [2] production sub-model  : kwh_prod_d ~ units_A + units_B + ... + cdd + hdd + saturday
        (production surplus = whole-day kWh minus off-production kWh)

    predicted_kwh_d = max(0, base_pred_d) + max(0, prod_pred_d)

Why this shape:
  * Daily volume is close to constant in plant data; a single pooled regression then
    confounds base load, production intensity and weather (quarter co-movement).
    Splitting the target by operating state fixes identification: base load is
    estimated where volume is zero, production intensity where volume varies.
  * Per-product intensities need within-period mix variation to be identified; the
    fit uses each day's mix shares. Products unseen in training fall back to the
    mean intensity (flagged as a limitation).
  * Robust fitting (Huber RLM + residual trimming) keeps anomalous consumption —
    exactly what the waste term looks for separately — from contaminating the model.
  * Weather enters BOTH parts, so the decomposition's weather factor shifts
    degree-days through both sub-models coherently.

Counterfactual interface:
    predict(vol, shares, cdd, hdd, dow, holiday) -> daily kWh
  shares may be scalar (quarterly mix) or per-day Series; calendar is taken from the
  modelled period (counterfactuals don't refit it).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analytics.ops_calendar import daytype_of, off_production_mask, production_calendar_from_production_df


# --------------------------------------------------------------------------- #
# daily frame
# --------------------------------------------------------------------------- #

def build_daily_frame(
    meter_df: pd.DataFrame,
    production_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    meter_id: str = "MAIN",
) -> pd.DataFrame:
    """Daily frame for one meter: full/offprod/prod kWh targets + drivers + calendar."""
    m = meter_df[meter_df["meter_id"] == meter_id].copy()
    m["ts"] = pd.to_datetime(m["ts"], utc=True).dt.floor("15min")
    grid = pd.DatetimeIndex(sorted(m["ts"].unique()))
    kwh = m.groupby("ts")["kwh"].sum(min_count=1).reindex(grid)

    _, hour_windows, _ = production_calendar_from_production_df(production_df)
    offmask = off_production_mask(grid, hour_windows)
    date = grid.normalize()
    full = kwh.groupby(lambda t: t.normalize()).sum(min_count=1).rename("kwh")
    off = kwh[offmask].groupby(lambda t: pd.Timestamp(t).normalize()).sum(min_count=1).rename("kwh_off")

    df = pd.DataFrame(index=pd.DatetimeIndex(sorted(pd.Index(set(date))), name="date"))
    df = df.join(full).join(off)
    df["kwh_prod"] = (df["kwh"] - df["kwh_off"]).clip(lower=0.0)

    if production_df is not None and not production_df.empty:
        p = production_df.copy()
        p["date"] = pd.to_datetime(p["date"], utc=True).dt.normalize()
        units = p.pivot_table(index="date", columns="product_id", values="volume", aggfunc="sum")
        units.columns = [f"units_{c}" for c in units.columns]
        df = df.join(units, how="left")
        df = df.join(p.groupby("date")["is_holiday"].agg("max").rename("is_holiday"), how="left")
    df["is_holiday"] = df.get("is_holiday", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if weather_df is not None and not weather_df.empty:
        w = weather_df.copy()
        w["date"] = pd.to_datetime(w["date"], utc=True).dt.normalize()
        df = df.join(w.set_index("date")[["cdd", "hdd", "temperature_c"]], how="left")
    for c in ("cdd", "hdd"):
        if c not in df:
            df[c] = 0.0

    df["dow"] = df.index.dayofweek
    df["daytype"] = [daytype_of(d, bool(h)) for d, h in zip(df.index, df["is_holiday"])]
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #

@dataclass
class LinearSpec:
    products: list[str]
    unseen_coef: float                    # fallback intensity for products unseen in training
    base: dict[str, float]                # {"intercept","cdd","hdd","weekend","holiday","weekday"}
    prod: dict[str, float]                # {"intercept","units_p",...,"cdd","hdd","sat"}
    rmse_base: float = 0.0
    rmse_prod: float = 0.0

    def intensity_series(self, vol: pd.Series, shares: dict[str, object]) -> pd.Series:
        """kWh per unit as a per-day series given daily/quarterly mix shares."""
        intensity = pd.Series(0.0, index=vol.index, dtype=float)
        for pname, s in shares.items():
            coef = self.prod.get(f"units_{pname}", self.unseen_coef)
            if np.isscalar(s):
                intensity = intensity + coef * float(s)
            else:
                intensity = intensity + coef * s.reindex(vol.index).fillna(0.0).astype(float)
        return intensity

    def predict(self, vol: pd.Series, shares: dict[str, object], cdd: pd.Series, hdd: pd.Series,
                dow: pd.Series, holiday: pd.Series) -> pd.Series:
        idx = vol.index
        vol = vol.reindex(idx).fillna(0.0).astype(float)
        cdd = cdd.reindex(idx).fillna(0.0).astype(float)
        hdd = hdd.reindex(idx).fillna(0.0).astype(float)
        hol = holiday.reindex(idx).fillna(0).astype(float)

        dts = pd.Series(np.where(hol.values == 1, "holiday",
                                 np.where(dow.reindex(idx).values >= 5, "weekend", "weekday")), index=idx)
        base_pred = (pd.Series(self.base["intercept"], index=idx)
                     + self.base.get("cdd", 0.0) * cdd + self.base.get("hdd", 0.0) * hdd
                     + dts.map({k: v for k, v in self.base.items()
                                if k in ("weekday", "weekend", "holiday")}).fillna(0.0))

        sat = dts.ne("weekday").astype(float) * float(self.prod.get("sat", 0.0))
        prod_pred = (self.prod.get("intercept", 0.0) + self.prod.get("cdd", 0.0) * cdd
                     + self.prod.get("hdd", 0.0) * hdd + sat)
        prod_var = self.intensity_series(vol, shares) * vol
        return (base_pred.clip(lower=0.0) + (prod_pred + prod_var).clip(lower=0.0))


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def _units_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if str(c).startswith("units_")]


def _daily_shares(frame: pd.DataFrame, seen: list[str]) -> dict[str, pd.Series]:
    vol = frame[[f"units_{p}" for p in seen]].fillna(0.0).sum(axis=1) if seen else pd.Series(0.0, index=frame.index)
    tot = vol.replace(0.0, np.nan)
    return {p: frame[f"units_{p}"].fillna(0.0) / tot for p in seen}


def _predict_frame(frame: pd.DataFrame, spec: LinearSpec, seen: list[str]) -> pd.Series:
    vol = frame[[f"units_{p}" for p in seen]].fillna(0.0).sum(axis=1) if seen else pd.Series(0.0, index=frame.index)
    return spec.predict(vol, _daily_shares(frame, seen), frame["cdd"], frame["hdd"], frame["dow"], frame["is_holiday"])


def _robust_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Huber RLM + one round of residual trimming. Returns (params, resid_kept, n_trimmed)."""
    r1 = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(maxiter=200)
    resid = np.asarray(y) - np.asarray(X) @ np.asarray(r1.params)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    keep = np.abs(resid) <= max(3.0 * rmse, 1e-9)
    n_trim = int((~keep).sum())
    if keep.sum() >= max(20, int(0.5 * len(y))) and n_trim > 0:
        r2 = sm.RLM(y[keep], X[keep], M=sm.robust.norms.HuberT()).fit(maxiter=200)
        resid = np.asarray(y)[keep] - np.asarray(X)[keep] @ np.asarray(r2.params)
        return np.asarray(r2.params), resid, n_trim
    return np.asarray(r1.params), resid, 0


def fit_baseline(daily_pooled: pd.DataFrame, products: list[str] | None = None) -> tuple[LinearSpec, dict]:
    """Fit the hybrid model on the pooled daily frame; return spec + quality stats."""
    d = daily_pooled.copy()
    all_products = [c.replace("units_", "") for c in _units_cols(d)]
    products = (products or all_products)
    seen = [p for p in products if d.get(f"units_{p}", pd.Series(0, index=d.index)).fillna(0).sum() > 0]
    if not seen:
        raise ValueError("no product volumes available to train the baseline")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # --- [1] base load sub-model ---
        Xb = pd.DataFrame({
            "const": 1.0,
            "cdd": d["cdd"].fillna(0.0), "hdd": d["hdd"].fillna(0.0),
            "weekend": (d["daytype"] == "weekend").astype(float),
            "holiday": (d["daytype"] == "holiday").astype(float),
        }, index=d.index)
        yb = d["kwh_off"].fillna(0.0)
        okb = Xb.notna().all(axis=1) & yb.notna()
        pb, resid_b, trim_b = _robust_fit(Xb[okb].values, yb[okb].values)
        base_params = dict(zip(["const", "cdd", "hdd", "weekend", "holiday"], pb))
        base_params["weekday"] = 0.0

        # --- [2] production sub-model ---
        Xp = pd.DataFrame({f"units_{p}": d[f"units_{p}"].fillna(0.0) for p in seen},
                          index=d.index)
        Xp["cdd"] = d["cdd"].fillna(0.0)
        Xp["hdd"] = d["hdd"].fillna(0.0)
        Xp["sat"] = (d["dow"] == 5).astype(float)
        Xp.insert(0, "const", 1.0)
        yp = d["kwh_prod"].fillna(0.0)
        okp = Xp.notna().all(axis=1) & yp.notna()
        pp, resid_p, trim_p = _robust_fit(Xp[okp].values, yp[okp].values)
        prod_params = dict(zip(["const"] + [f"units_{p}" for p in seen] + ["cdd", "hdd", "sat"], pp))

    unseen_coef = float(np.mean([prod_params[f"units_{p}"] for p in seen]))
    spec = LinearSpec(
        products=list(seen), unseen_coef=unseen_coef,
        base={"intercept": float(base_params["const"]), "cdd": float(base_params["cdd"]),
              "hdd": float(base_params["hdd"]), "weekend": float(base_params.get("weekend", 0.0)),
              "holiday": float(base_params.get("holiday", 0.0)), "weekday": 0.0},
        prod={"intercept": float(prod_params["const"]),
              **{f"units_{p}": float(prod_params[f"units_{p}"]) for p in seen},
              "cdd": float(prod_params["cdd"]), "hdd": float(prod_params["hdd"]),
              "sat": float(prod_params["sat"])},
        rmse_base=float(np.sqrt(np.mean(resid_b ** 2))),
        rmse_prod=float(np.sqrt(np.mean(resid_p ** 2))),
    )

    # quality stats (informational, time-blocked inside the pooled sample)
    pred_all = _predict_frame(d, spec, seen)
    act = d["kwh"]
    ss_res = float(((act - pred_all).dropna() ** 2).sum())
    ss_tot = float(((act - act.mean()).dropna() ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(d.dropna(subset=["kwh"]))
    mape = None
    if n >= 40:
        cut = d.index[int(n * 0.75)]
        te = d[d.index > cut]
        act_te = te["kwh"]
        pred_te = _predict_frame(te, spec, seen)
        mask = (act_te != 0) & act_te.notna() & pred_te.notna()
        if mask.sum() >= 5:
            mape = float(np.mean(np.abs((act_te[mask] - pred_te[mask]) / act_te[mask])) * 100)

    stats = {
        "r2": round(r2, 4),
        "rmse": round(spec.rmse_prod, 1),
        "holdout_mape_pct": None if mape is None else round(mape, 2),
        "n_train_days": int(n),
        "robust": "RLM(Huber)+trimming",
        "trimmed_days": trim_b + trim_p,
        "base_model": {k: round(float(v), 2) for k, v in spec.base.items() if not np.isnan(v)},
        "kwh_per_unit": {p: round(float(spec.prod[f"units_{p}"]), 3) for p in seen},
        "prod_cdd_coef": round(float(spec.prod["cdd"]), 1),
        "base_cdd_coef": round(float(spec.base["cdd"]), 1),
        "products": list(seen),
    }
    return spec, stats
