#!/usr/bin/env python3
"""
01_vrp_core.py -- Paper 1 core analysis: the crypto volatility risk premium.

Builds VRP_t = implied_var(DVOL) - forward realized variance for BTC & ETH over
the full Deribit window (2021-03 .. 2026-06, 12h bars), then evaluates the
short-variance-swap harvest. The time-series harvest runs off DVOL (risk-neutral
30d implied) vs OHLCV realized vol -- the option surface is a single end-of-sample
snapshot and cannot price the options leg historically (see paper1_data_profile.md).

NO LOOK-AHEAD: at each decision bar i we use only DVOL known at/before i (merge_asof
backward) and the *trailing* realized vol for gating; the swap payoff uses realized
variance over (i, i+30d], which is resolved later -- that is the trade, not leakage.

Outputs -> experiments/paper1_vrp_core.json  (+ a printed summary).
Reuses the cost-aware philosophy + rigor module (degenerate check, PSR, NW/non-overlap t).
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.eval import rigor  # noqa: E402

DERIBIT = "/apps/jepa-trader/data/raw_deribit"
H = 60                         # 30 days @ 12h bars
BARS_PER_YEAR = 365.25 * 2     # 730.5  (two 12h bars/day)
WIN_PER_YEAR = 365.25 / 30.0   # ~12.17 non-overlapping 30d windows / year
ANN = BARS_PER_YEAR / H        # annualization for a 30d realized-variance window
COSTS_VP = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # round-trip cost in annualized VOL POINTS
COINS = ["BTC", "ETH"]


def load(coin):
    px = pd.read_parquet(f"{DERIBIT}/price_{coin}.parquet").sort_values("time").reset_index(drop=True)
    dv = pd.read_parquet(f"{DERIBIT}/dvol_{coin}.parquet").sort_values("time").reset_index(drop=True)
    px["time"] = pd.to_datetime(px["time"]); dv["time"] = pd.to_datetime(dv["time"])
    df = pd.merge_asof(px[["time", "open", "high", "low", "close"]],
                       dv[["time", "dvol_c"]], on="time", direction="backward")
    return df


def build(df):
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    lr = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
    r2 = pd.Series(lr ** 2)
    fwd_sum = r2.rolling(H).sum().shift(-H)     # sum r2[i+1..i+H]  (forward, the swap horizon)
    trail_sum = r2.rolling(H).sum()             # sum r2[i-H+1..i]  (trailing, for the gate; no look-ahead)
    park = (1.0 / (4 * np.log(2))) * (np.log(high / low)) ** 2
    park_fwd = pd.Series(park).rolling(H).sum().shift(-H)
    closeS = pd.Series(close)
    out = pd.DataFrame(dict(
        time=df["time"],
        impl_vol=df["dvol_c"].to_numpy(float) / 100.0,
        fwd_rv_var=fwd_sum.to_numpy() * ANN,
        trail_rv_var=trail_sum.to_numpy() * ANN,
        park_fwd_var=park_fwd.to_numpy() * ANN,
        fwd_ret=np.log(closeS.shift(-H) / closeS).to_numpy(),
    ))
    out["impl_var"] = out["impl_vol"] ** 2
    out["vrp_var"] = out["impl_var"] - out["fwd_rv_var"]            # variance-swap fair PnL (short)
    out["vrp_vol"] = out["impl_vol"] - np.sqrt(out["fwd_rv_var"])   # vol-points (vol-swap) PnL (short)
    return out


def ann_sharpe(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 8 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(WIN_PER_YEAR))


def tail_stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 8:
        return {}
    k = max(1, int(round(0.05 * x.size)))
    cum = np.cumsum(x); dd = cum - np.maximum.accumulate(cum)
    return dict(mean=float(x.mean()), std=float(x.std()), sharpe=ann_sharpe(x),
                skew=float(sp_skew(x)), kurt=float(sp_kurt(x, fisher=False)),
                cvar5=float(np.sort(x)[:k].mean()), worst=float(x.min()),
                best=float(x.max()), maxdd=float(dd.min()), n=int(x.size))


def nw_t(x, lags):
    try:
        import statsmodels.api as sm
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 20:
            return float("nan")
        m = sm.OLS(x, np.ones(x.size)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(m.tvalues[0])
    except Exception as e:
        return f"err:{e}"


def analyze(coin):
    d = build(load(coin))
    res = {"coin": coin, "n_bars": int(len(d)),
           "span": [str(d["time"].iloc[0]), str(d["time"].iloc[-1])]}

    # ---- full overlapping series (for sign-test significance with all data) ----
    ov = d.dropna(subset=["vrp_var", "vrp_vol", "impl_vol"])
    res["sign"] = dict(
        mean_vrp_volpts=float(ov["vrp_vol"].mean() * 100),
        median_vrp_volpts=float(ov["vrp_vol"].median() * 100),
        pct_positive=float((ov["vrp_vol"] > 0).mean()),
        mean_vrp_var=float(ov["vrp_var"].mean()),
        nw_t_var=nw_t(ov["vrp_var"].to_numpy(), H),
        nonoverlap_t_var=rigor.nonoverlapping_tstat(ov["vrp_var"].to_numpy(), thin=H),
        mean_impl_volpts=float(ov["impl_vol"].mean() * 100),
        mean_realized_volpts=float(np.sqrt(ov["fwd_rv_var"]).mean() * 100),
    )
    # by-year sign
    ov = ov.assign(year=ov["time"].dt.year)
    res["by_year_sign"] = {int(y): dict(mean_volpts=float(g["vrp_vol"].mean() * 100),
                                        pct_pos=float((g["vrp_vol"] > 0).mean()), n=int(len(g)))
                           for y, g in ov.groupby("year")}

    # ---- NON-OVERLAPPING 30d harvest windows (phase 0) ----
    ent = d.iloc[::H].dropna(subset=["impl_vol", "fwd_rv_var", "trail_rv_var", "fwd_ret"]).copy()
    impl = ent["impl_vol"].to_numpy(); rv = ent["fwd_rv_var"].to_numpy()
    trail_rvol = np.sqrt(ent["trail_rv_var"].to_numpy())
    fwd_ret = ent["fwd_ret"].to_numpy()
    gate = impl > trail_rvol                                  # sell only when implied > trailing realized
    size = np.clip(np.nanmedian(impl) / impl, 0.25, 4.0)     # inverse-vol (constant-vega-ish) sizing

    def swap_pnl(cc):                                         # cc = cost in vol decimals
        return (np.maximum(impl - cc, 0.0)) ** 2 - rv         # short variance swap, net strike
    def vol_pnl(cc):
        return (impl - cc) - np.sqrt(rv)                      # short vol swap (vol points), net

    res["n_windows"] = int(len(ent))
    res["in_market_frac_gated"] = float(gate.mean())

    # headline strategies at a representative 2.0 vol-pt cost
    cc2 = 2.0 / 100
    strat = {
        "passive_gross_var": swap_pnl(0.0),
        "passive_net_var": swap_pnl(cc2),
        "gated_net_var": np.where(gate, swap_pnl(cc2), 0.0),
        "gated_vt_net_var": np.where(gate, size * swap_pnl(cc2), 0.0),
        "passive_net_vol": vol_pnl(cc2),
        "gated_net_vol": np.where(gate, vol_pnl(cc2), 0.0),
    }
    res["strategies"] = {k: tail_stats(v) for k, v in strat.items()}

    # cost sweep on the (preferred) gated + vol-targeted variance-swap harvest
    res["cost_sweep_gated_vt_var"] = {
        f"{c}vp": ann_sharpe(np.where(gate, size * swap_pnl(c / 100), 0.0)) for c in COSTS_VP}
    res["cost_sweep_gated_vol"] = {
        f"{c}vp": ann_sharpe(np.where(gate, vol_pnl(c / 100), 0.0)) for c in COSTS_VP}

    # beta-to-spot of the harvest (is it just short/long the coin?)
    base = np.where(gate, size * swap_pnl(cc2), 0.0)
    m = np.isfinite(base) & np.isfinite(fwd_ret)
    res["beta_to_spot"] = dict(
        corr=float(np.corrcoef(base[m], fwd_ret[m])[0, 1]) if m.sum() > 8 else float("nan"),
        corr_passive=float(np.corrcoef(swap_pnl(cc2)[m], fwd_ret[m])[0, 1]) if m.sum() > 8 else float("nan"))

    # decay: by-year + first-half/second-half Sharpe of the headline harvest
    ent = ent.assign(pnl=base, year=ent["time"].dt.year)
    res["decay_by_year_sharpe"] = {int(y): ann_sharpe(g["pnl"].to_numpy())
                                   for y, g in ent.groupby("year")}
    half = ent["time"] < pd.Timestamp("2024-01-01", tz="UTC")
    res["decay_split"] = dict(
        sharpe_2021_2023=ann_sharpe(ent.loc[half, "pnl"].to_numpy()),
        sharpe_2024_2026=ann_sharpe(ent.loc[~half, "pnl"].to_numpy()),
        mean_2021_2023=float(ent.loc[half, "pnl"].mean()),
        mean_2024_2026=float(ent.loc[~half, "pnl"].mean()))

    # light rigor
    deg_sig = rigor.degenerate_signal_check(ov["vrp_vol"].to_numpy())
    deg_pnl = rigor.degenerate_signal_check(base)
    res["rigor"] = dict(
        degenerate_vrp=dict(flagged=deg_sig.is_degenerate, reasons=deg_sig.reasons),
        degenerate_harvest=dict(flagged=deg_pnl.is_degenerate, reasons=deg_pnl.reasons),
        psr_harvest=rigor.probabilistic_sharpe_ratio(base[np.isfinite(base)], 0.0),
        nonoverlap_t_harvest=rigor.nonoverlapping_tstat(base[np.isfinite(base)], thin=1),
    )
    return res


def main():
    out = {"meta": dict(H_bars=H, days=30, bars_per_year=BARS_PER_YEAR,
                        windows_per_year=WIN_PER_YEAR, costs_vol_points=COSTS_VP)}
    for coin in COINS:
        out[coin] = analyze(coin)

    os.makedirs(f"{ROOT}/experiments", exist_ok=True)
    with open(f"{ROOT}/experiments/paper1_vrp_core.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ---- printed summary ----
    for coin in COINS:
        r = out[coin]; s = r["sign"]
        print(f"\n===== {coin}  ({r['span'][0][:10]} .. {r['span'][1][:10]}, "
              f"{r['n_windows']} non-overlap 30d windows) =====")
        print(f"  SIGN: mean VRP = {s['mean_vrp_volpts']:+.2f} vol pts "
              f"(impl {s['mean_impl_volpts']:.1f} - realized {s['mean_realized_volpts']:.1f}), "
              f"{s['pct_positive']*100:.0f}% positive")
        print(f"        NW t(var) = {s['nw_t_var']}, non-overlap t(var) = {s['nonoverlap_t_var']['t']:.2f} "
              f"(n={s['nonoverlap_t_var']['n']})")
        print(f"  HARVEST Sharpe (annualized, net @2vp): "
              f"passive={r['strategies']['passive_net_var']['sharpe']:.2f}  "
              f"gated={r['strategies']['gated_net_var']['sharpe']:.2f}  "
              f"gated+voltgt={r['strategies']['gated_vt_net_var']['sharpe']:.2f}  "
              f"(gross passive={r['strategies']['passive_gross_var']['sharpe']:.2f}; "
              f"in-market {r['in_market_frac_gated']*100:.0f}%)")
        print(f"  VOL-POINTS Sharpe net@2vp: passive={r['strategies']['passive_net_vol']['sharpe']:.2f} "
              f"gated={r['strategies']['gated_net_vol']['sharpe']:.2f}")
        gv = r['strategies']['gated_vt_net_var']
        print(f"  TAIL (gated+voltgt net@2vp): skew={gv['skew']:.2f} kurt={gv['kurt']:.1f} "
              f"CVaR5%={gv['cvar5']:.4f} worst={gv['worst']:.4f} maxDD={gv['maxdd']:.4f}")
        print(f"  COST SWEEP (gated+voltgt var Sharpe): "
              + "  ".join(f"{c}={v:.2f}" for c, v in r['cost_sweep_gated_vt_var'].items()))
        print(f"  DECAY split Sharpe: 2021-23={r['decay_split']['sharpe_2021_2023']:.2f}  "
              f"2024-26={r['decay_split']['sharpe_2024_2026']:.2f}  "
              f"| by year: " + " ".join(f"{y}:{v:.1f}" for y, v in r['decay_by_year_sharpe'].items()))
        print(f"  BETA-to-spot corr (gated+voltgt)={r['beta_to_spot']['corr']:+.2f} "
              f"(passive={r['beta_to_spot']['corr_passive']:+.2f})")
        print(f"  RIGOR: degenerate(vrp)={r['rigor']['degenerate_vrp']['flagged']} "
              f"degenerate(harvest)={r['rigor']['degenerate_harvest']['flagged']} "
              f"PSR(harvest)={r['rigor']['psr_harvest']:.2f} "
              f"non-overlap t(harvest)={r['rigor']['nonoverlap_t_harvest']['t']:.2f}")
    print(f"\nWrote experiments/paper1_vrp_core.json")


if __name__ == "__main__":
    main()
