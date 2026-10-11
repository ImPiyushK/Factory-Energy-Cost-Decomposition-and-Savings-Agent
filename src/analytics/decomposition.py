"""Bill decomposition (spec R3, R4) — the 25-point core.

Method: counterfactual bill simulation + Shapley attribution (Config A)
        or sequential chained counterfactuals (Config B).

Counterfactuals are LEVEL TRANSLATIONS applied to the current period's calendar
(quarters have different dates, so per-day mapping is impossible — the standard
index-decomposition approach):
    * volume block : current daily volumes scaled by total-units ratio (base/cur)
    * mix block    : quarterly product mix shares (base or cur)
    * weather block: current CDD/HDD series shifted so its mean equals the other
                     quarter's mean ("as mild as last quarter, same calendar")
    * tariff block : exact per-interval price lookup; the old regime is projected
                     forward over the current grid ("what if the change never
                     happened") — mid-period changes handled exactly (R4)

    phi_f (Shapley) over {P, V, M, W}  ->  explained = sum(phi_f)
    waste = (B1 - B0) - explained
    => the five causes add up to the observed change BY CONSTRUCTION and all four
       drivers are accounted for BEFORE anything is called waste.

Uncertainty: weekly block bootstrap. Each draw resamples the baseline and current
weeks with replacement; the baseline model is REFIT per draw; blocks (volume ratio,
mix shares, weather shifts) are re-derived from the weighted resample; actual-bill
draws reweight the exact per-week actual bills. Week weights enter the bill
function, so draws stay consistent with the full-period point estimates.

Disclosed: the 'waste' term absorbs baseline model error and noise; waste is a range
and only becomes an asset-level claim when corroborated by meter evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from src.analytics import baseline as bl
from src.analytics.billing import bucket_series, build_price_schedule_ordered, legacy_schedule
from src.analytics.periods import quarter_bounds

BRIDGE_CAUSES = ["tariff", "volume", "product_mix", "weather", "waste"]
F_FACTORS = ["tariff", "volume", "mix", "weather"]


def _grid_for(meter_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    m = meter_df[meter_df["meter_id"] == "MAIN"]
    ts = pd.to_datetime(m["ts"], utc=True)
    m = m[(ts >= start) & (ts <= end)]
    return pd.DatetimeIndex(sorted(pd.to_datetime(m["ts"].unique(), utc=True))).floor("15min")


def kwh_series(meter_df: pd.DataFrame, meter_id: str, grid: pd.DatetimeIndex) -> pd.Series:
    m = meter_df[meter_df["meter_id"] == meter_id]
    s = pd.Series(m["kwh"].astype(float).values,
                  index=pd.DatetimeIndex(pd.to_datetime(m["ts"], utc=True)))
    return s[~s.index.duplicated()].reindex(grid)


@dataclass
class Blocks:
    """Level translations of the four drivers, evaluated on one quarter's calendar."""

    vol_ratio: float                       # total-units ratio base/cur
    mix_base: dict[str, float]
    mix_cur: dict[str, float]
    cdd_shift: float
    hdd_shift: float
    products: list[str]

    @classmethod
    def from_frames(cls, daily_base: pd.DataFrame, daily_cur: pd.DataFrame,
                    products: list[str],
                    w_base: pd.Series | None = None, w_cur: pd.Series | None = None) -> "Blocks":
        wb = (pd.Series(1.0, index=daily_base.index) if w_base is None
              else w_base.reindex(daily_base.index).fillna(0.0))
        wc = (pd.Series(1.0, index=daily_cur.index) if w_cur is None
              else w_cur.reindex(daily_cur.index).fillna(0.0))

        def wunits(frame: pd.DataFrame, weight: pd.Series) -> dict[str, float]:
            return {p: float((frame.get(f"units_{p}", pd.Series(0.0, index=frame.index))
                              .fillna(0.0) * weight).sum()) for p in products}

        base_units, cur_units = wunits(daily_base, wb), wunits(daily_cur, wc)
        base_tot, cur_tot = sum(base_units.values()), sum(cur_units.values())
        vol_ratio = float(base_tot / cur_tot) if cur_tot > 0 else 1.0
        mix_base = {p: (v / base_tot if base_tot > 0 else 0.0) for p, v in base_units.items()}
        mix_cur = {p: (v / cur_tot if cur_tot > 0 else 0.0) for p, v in cur_units.items()}

        def wmean(frame: pd.DataFrame, col: str, weight: pd.Series) -> float:
            v = frame[col].fillna(0.0)
            m = float(weight.sum())
            return float((v * weight).sum() / m) if m > 0 else 0.0

        cdd_b, cdd_c = wmean(daily_base, "cdd", wb), wmean(daily_cur, "cdd", wc)
        return cls(
            vol_ratio=vol_ratio,
            mix_base=mix_base, mix_cur=mix_cur,
            cdd_shift=cdd_b - cdd_c,
            hdd_shift=wmean(daily_base, "hdd", wb) - wmean(daily_cur, "hdd", wc),
            products=list(products),
        )

    def weather_pairs(self, block: str, cdd_cur: pd.Series, hdd_cur: pd.Series):
        if block == "cur":
            return cdd_cur, hdd_cur
        return (cdd_cur + self.cdd_shift).clip(lower=0.0), (hdd_cur + self.hdd_shift).clip(lower=0.0)


def _alloc_shares(grid: pd.DatetimeIndex, buckets: pd.Series,
                  profile: dict[str, dict[str, float]],
                  days: pd.DataFrame) -> pd.Series:
    """Per-interval share of the DAILY energy, from baseline-observed profile shapes."""
    hol = set(days.index[days["is_holiday"] == 1].normalize())
    dowtype = pd.Series(np.where(grid.dayofweek >= 5, "weekend", "weekday"), index=grid)
    dowtype[dowtype.index.normalize().isin(hol)] = "holiday"
    bucket = buckets.astype(str)
    fallback = profile.get("weekday", {})
    share_ts = pd.Series(
        [profile.get(dowtype.iloc[i], fallback).get(bucket.iloc[i], 0.0) for i in range(len(grid))],
        index=grid,
    )
    return share_ts


def _product_list(daily_base: pd.DataFrame, daily_cur: pd.DataFrame) -> list[str]:
    cols = sorted({c.replace("units_", "") for c in list(daily_base.columns) + list(daily_cur.columns)
                   if str(c).startswith("units_")})
    out = []
    for p in cols:
        tot = (float(daily_base.get(f"units_{p}", pd.Series(0, index=daily_base.index)).fillna(0).sum())
               + float(daily_cur.get(f"units_{p}", pd.Series(0, index=daily_cur.index)).fillna(0).sum()))
        if tot > 0:
            out.append(p)
    return out


def _tou_profile(meter_df: pd.DataFrame, grid: pd.DatetimeIndex, buckets: pd.Series,
                 daily_base: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Observed per-INTERVAL energy share per bucket and day-type (baseline quarter).

    share_per_interval(t) = E(bucket, day) / n_intervals(bucket) / day_total, so that
    sum over one day's grid === 1 exactly (daily energy allocated once).
    """
    kwh = kwh_series(meter_df, "MAIN", grid)
    df = pd.DataFrame({"kwh": kwh, "bucket": buckets.astype(str), "day": grid.normalize()})
    df["dow"] = df["day"].dt.dayofweek
    hol = set(daily_base.index[daily_base["is_holiday"] == 1].normalize())
    df["dowtype"] = np.where(df["dow"] >= 5, "weekend", "weekday")
    df.loc[df["day"].isin(hol), "dowtype"] = "holiday"
    day_total = df.groupby("day")["kwh"].transform("sum")
    df["ival_kwh"] = df["kwh"] / day_total.replace(0, np.nan)
    prof = df.groupby(["dowtype", "bucket"])["ival_kwh"].mean().unstack(fill_value=np.nan)
    return {str(dt): {b: (float(v) if v == v else 0.0) for b, v in prof.loc[dt].items()}
            for dt in prof.index}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def decompose(
    meter_df: pd.DataFrame,
    tariff_df: pd.DataFrame,
    production_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    period: str,
    compare_to: str,
    method: str = "shapley",
    n_boot: int = 60,
) -> dict:
    """Verified bill bridge for period vs compare_to (amounts, pct_points, 10-90 ranges)."""
    s_cur, e_cur = quarter_bounds(period)
    s_base, e_base = quarter_bounds(compare_to)

    grid_cur, grid_base = _grid_for(meter_df, s_cur, e_cur), _grid_for(meter_df, s_base, e_base)
    if len(grid_cur) < 100 or len(grid_base) < 100:
        raise ValueError("meter data does not cover the requested quarters")

    price_cur = build_price_schedule_ordered(tariff_df, grid_cur)
    price_base_actual = build_price_schedule_ordered(tariff_df, grid_base)
    price_old_on_cur = legacy_schedule(tariff_df, grid_cur)
    cur_buckets = bucket_series(tariff_df, grid_cur)

    daily_all = bl.build_daily_frame(meter_df, production_df, weather_df, "MAIN")
    daily_base = daily_all[(daily_all.index >= s_base) & (daily_all.index <= e_base)]
    daily_cur = daily_all[(daily_all.index >= s_cur) & (daily_all.index <= e_cur)]
    if daily_base.empty or daily_cur.empty:
        raise ValueError("production/weather data does not cover the requested quarters")

    products = _product_list(daily_base, daily_cur)
    if not products:
        raise ValueError("no product volumes available to train the baseline")
    # Baseline fitted on the BASELINE quarter only (no leakage into the current period).
    spec, model_stats = bl.fit_baseline(daily_base, products)

    profile = _tou_profile(meter_df, grid_base, bucket_series(tariff_df, grid_base), daily_base)
    share_ts = _alloc_shares(grid_cur, cur_buckets, profile, daily_cur)

    kwh_cur = kwh_series(meter_df, "MAIN", grid_cur)
    kwh_base = kwh_series(meter_df, "MAIN", grid_base)
    b1_actual = float((kwh_cur * price_cur.series).sum(skipna=True))
    b0_actual = float((kwh_base * price_base_actual.series).sum(skipna=True))
    if b0_actual <= 0:
        raise ValueError("baseline bill is not positive")

    blk = Blocks.from_frames(daily_base, daily_cur, products)
    cdd_cur, hdd_cur = daily_cur["cdd"].fillna(0.0), daily_cur["hdd"].fillna(0.0)
    vol_cur_daily = daily_cur[bl._units_cols(daily_cur)].fillna(0.0).sum(axis=1)
    prices = {"base": price_old_on_cur.series, "cur": price_cur.series}

    def subset_bill(assign: frozenset[str], spec_use: bl.LinearSpec,
                    days: pd.DataFrame, shares_b: dict[str, float],
                    weight_day: pd.Series | None = None) -> float:
        stack_key = {f: ("cur" if f in assign else "base") for f in F_FACTORS}
        ratio = shares_b["vol_ratio"]
        vol = vol_cur_daily * (1.0 if stack_key["volume"] == "cur" else ratio)
        shares = shares_b[f"mix_{stack_key['mix']}"]
        cdd, hdd = shares_b["blk"].weather_pairs(stack_key["weather"], cdd_cur, hdd_cur)
        pred = spec_use.predict(vol, shares, cdd, hdd, days["dow"], days["is_holiday"])
        if weight_day is not None:
            pred = pred * weight_day.reindex(pred.index).fillna(0.0)
        day_map = pd.Series(pred.values, index=pred.index.normalize()).groupby(level=0).sum()
        per_ts = day_map.reindex(grid_cur.normalize()).values
        alloc = share_ts.values * per_ts
        return float(np.sum(alloc * prices[stack_key["tariff"]].values))

    def attribution(spec_use: bl.LinearSpec, days: pd.DataFrame, shares_b: dict[str, float],
                    weight_day: pd.Series | None = None) -> dict[str, float]:
        if method == "shapley":
            n = len(F_FACTORS)
            subsets = [frozenset(c) for r in range(n + 1) for c in combinations(F_FACTORS, r)]
            bills = {s: subset_bill(s, spec_use, days, shares_b, weight_day) for s in subsets}
            phi = {}
            for f in F_FACTORS:
                others = [x for x in F_FACTORS if x != f]
                total = 0.0
                for r in range(len(others) + 1):
                    for combo in combinations(others, r):
                        S = frozenset(combo)
                        w = factorial(r) * factorial(n - r - 1) / factorial(n)
                        total += w * (bills[S | {f}] - bills[S])
                phi[f] = total
            return phi
        phi, prev = {}, frozenset()
        for f in F_FACTORS:
            cur_set = prev | {f}
            phi[f] = subset_bill(cur_set, spec_use, days, shares_b, weight_day) \
                - subset_bill(prev, spec_use, days, shares_b, weight_day)
            prev = cur_set
        return phi

    phis = {"point": {"vol_ratio": blk.vol_ratio, "mix_base": blk.mix_base, "mix_cur": blk.mix_cur,
                      "blk": blk}}
    phi = attribution(spec, daily_cur, phis["point"])
    explained = sum(phi.values())
    b_model_cur = subset_bill(frozenset(F_FACTORS), spec, daily_cur, phis["point"])

    # ---- exact per-week actual bills (interval-level), reused as bootstrap draw weights
    def week_bills(kwh: pd.Series, price: pd.Series) -> dict[int, float]:
        df = pd.DataFrame({"kwh": kwh.fillna(0.0), "price": price})
        df["week"] = df.index.isocalendar()["week"].values
        return {int(w): float(g["kwh"].dot(g["price"])) for w, g in df.groupby("week")}

    wb_cur = week_bills(kwh_cur, price_cur.series)
    wb_base = week_bills(kwh_base, price_base_actual.series)
    b1_actual = float(sum(wb_cur.values()))
    b0_actual = float(sum(wb_base.values()))
    change = b1_actual - b0_actual

    def weeks_of(days: pd.DataFrame) -> dict[int, list[pd.Timestamp]]:
        wk = days.index.isocalendar()
        return {int(w): sorted(days.index[wk["week"] == w]) for w in sorted(set(wk["week"]))}

    wb, wc = weeks_of(daily_base), weeks_of(daily_cur)
    pos_base = {w: i for i, w in enumerate(sorted(wb))}
    pos_cur = {w: i for i, w in enumerate(sorted(wc))}
    base_week_of_day = {d: pos_base[int(d.isocalendar()[1])] for d in daily_base.index}
    week_of_day_cur = {d: pos_cur[int(d.isocalendar()[1])] for d in daily_cur.index}

    rng = np.random.default_rng(42)
    draws: list[dict[str, float]] = []
    n_failed = 0
    for _ in range(max(n_boot, 50)):
        kb, kc = list(wb), list(wc)
        counts_b = np.bincount(rng.integers(0, len(kb), size=len(kb)), minlength=len(kb))
        counts_c = np.bincount(rng.integers(0, len(kc), size=len(kc)), minlength=len(kc))
        # resampled baseline weeks (kept with multiplicity -> re-weights the robust fit)
        parts = []
        for i, cnt in enumerate(counts_b):
            for _ in range(int(cnt)):
                parts.append(daily_base.loc[wb[kb[i]]])
        resampled_base = pd.concat(parts) if parts else daily_base
        resampled_base = resampled_base[~resampled_base.index.duplicated(keep=False)]
        try:
            spec_b, _ = bl.fit_baseline(resampled_base, products)
        except Exception:
            n_failed += 1
            continue
        # weighted blocks + weight maps
        w_cur = pd.Series({d: float(counts_c[week_of_day_cur[d]]) for d in daily_cur.index})
        w_base = pd.Series({d: float(counts_b[base_week_of_day[d]]) for d in daily_base.index})
        try:
            blk_b = Blocks.from_frames(daily_base, daily_cur, products, w_base=w_base, w_cur=w_cur)
            phi_b = attribution(spec_b, daily_cur,
                                {"vol_ratio": blk_b.vol_ratio, "mix_base": blk_b.mix_base,
                                 "mix_cur": blk_b.mix_cur, "blk": blk_b},
                                weight_day=w_cur)
        except Exception:
            n_failed += 1
            continue
        explained_b = sum(phi_b.values())
        b1_draw = float(sum(wb_cur[kc[i]] * counts_c[i] for i in range(len(kc))))
        b0_draw = float(sum(wb_base[kb[i]] * counts_b[i] for i in range(len(kb))))
        waste_b = (b1_draw - b0_draw) - explained_b
        base_b = b0_draw if b0_draw > 0 else np.nan
        if base_b != base_b:
            n_failed += 1
            continue
        draws.append({
            "tariff": phi_b["tariff"] / base_b * 100,
            "volume": phi_b["volume"] / base_b * 100,
            "product_mix": phi_b["mix"] / base_b * 100,
            "weather": phi_b["weather"] / base_b * 100,
            "waste": waste_b / base_b * 100,
        })

    amounts = {"tariff": phi["tariff"], "volume": phi["volume"],
               "product_mix": phi["mix"], "weather": phi["weather"], "waste": change - explained}
    bridge = []
    for cause in BRIDGE_CAUSES:
        pp = amounts[cause] / b0_actual * 100.0
        if draws:
            lo, hi = np.percentile([d[cause] for d in draws], [10, 90])
            rng_lo, rng_hi = sorted([float(lo), float(hi)])
        else:
            rng_lo, rng_hi = pp, pp
        bridge.append({"cause": cause, "amount": round(float(amounts[cause]), 2),
                       "pct_points": round(float(pp), 2), "range": [round(rng_lo, 2), round(rng_hi, 2)]})

    model_error_pp = (subset_bill(frozenset(), spec, daily_cur, phis["point"]) - b0_actual) / b0_actual * 100.0
    assumptions = [
        "Baseline model: robust (Huber) two-part regression on daily kWh — base load estimated from "
        "off-production intervals with a degree-day response; production surplus regressed on per-product "
        "unit volumes, degree-days and calendar effects. Fitted on the baseline quarter only (no leakage).",
        "Counterfactuals are level translations on the current period's calendar: volume scaled by the "
        "total-units ratio, quarterly mix shares, and current weather shifted to the other quarter's mean.",
        "Daily modelled energy is allocated to tariff buckets with the baseline quarter's observed "
        "intraday shares (profile-shape stability assumption).",
        f"Attribution: {'Shapley (order-independent, exact)' if method == 'shapley' else 'sequential (order documented)'}; "
        "tariff prices looked up per 15-minute interval, so the mid-period tariff change is handled exactly.",
    ]
    limitations = [
        f"Baseline fit: train R2 {model_stats['r2']:.2f}"
        + (f", holdout MAPE {model_stats['holdout_mape_pct']:.1f}%" if model_stats["holdout_mape_pct"] is not None else "")
        + f"; the residual 'waste' term includes baseline model error of about "
        f"{model_error_pp:+.2f} pct points and estimation noise.",
        "The waste component is an unexplained residual in currency terms. It does NOT by itself "
        "identify a faulty asset — asset-level claims require corroborating meter evidence.",
    ]

    return {
        "bill": {"base": round(b0_actual, 2), "current": round(b1_actual, 2),
                 "change": round(change, 2), "change_pct": round(change / b0_actual * 100.0, 2)},
        "bridge": bridge,
        "model": model_stats,
        "method": method,
        "n_bootstrap_draws": len(draws),
        "n_failed_draws": n_failed,
        "model_expected_current_bill": round(float(b_model_cur), 2),
        "assumptions": assumptions,
        "limitations": limitations,
    }
