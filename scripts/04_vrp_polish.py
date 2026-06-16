#!/usr/bin/env python3
"""
04_vrp_polish.py -- Paper 1 robustness polish.

A) HIGH-FREQUENCY (1h) realized-vol VRP for the 2025+ regime -- the most realistic RV
   estimator in the window that actually matters (where the premium has decayed). 1h
   close-to-close captures more of the path than 12h, so it should shrink the premium
   further, reinforcing the "harvest-relevant RV is higher" argument.
B) OPTIONS-EXECUTION COST calibrated from the LIVE Deribit ATM bid-ask -> vol points,
   to anchor the cost sweep with a real number instead of an assumed haircut.

Results -> experiments/paper1_vrp_polish.json
"""
import os, sys, json, re, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import vrp  # noqa: E402

BARS1H = "/apps/jepa-trader/data/raw_crypto/bars_1h.csv"
API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={}&kind=option"


# ---------- A) high-frequency realized-vol VRP, recent regime ----------
def hf_vrp(coin):
    d = pd.read_csv(BARS1H)
    d = d[d["symbol"].astype(str).str.upper() == f"{coin}-USD"].copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.sort_values("time").reset_index(drop=True)
    H1, ANN1 = 720, 24 * 365.25                      # 30d @ 1h ; bars/yr
    close = d["close"].to_numpy(float)
    r2 = pd.Series(np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]) ** 2)
    d["fwd_rv_var_1h"] = r2.rolling(H1).sum().shift(-H1).to_numpy() * (ANN1 / H1)
    d["trail_rv_var_1h"] = r2.rolling(H1).sum().to_numpy() * (ANN1 / H1)

    dv = vrp.load_coin(coin)[["time", "dvol_c"]].copy()
    dv["time"] = pd.to_datetime(dv["time"], utc=True).dt.as_unit("ns")
    left = d[["time", "fwd_rv_var_1h", "trail_rv_var_1h"]].copy()
    left["time"] = left["time"].dt.as_unit("ns")
    m = pd.merge_asof(left, dv.sort_values("time"), on="time", direction="backward")
    m["impl_vol"] = m["dvol_c"] / 100.0
    m["vrp_vol_1h"] = m["impl_vol"] - np.sqrt(m["fwd_rv_var_1h"])
    rec = m.dropna(subset=["vrp_vol_1h", "impl_vol"])
    start = rec["time"].min()

    # 12h estimators over the SAME recent window, for apples-to-apples
    p = vrp.build(coin, "c2c"); p = p[p["time"] >= start]
    pk = vrp.build(coin, "parkinson"); pk = pk[pk["time"] >= start]

    # harvest Sharpe under 1h RV, non-overlapping 30d windows, net 2vp + gated
    ent = rec.iloc[::H1]
    impl = ent["impl_vol"].to_numpy(); rvf = ent["fwd_rv_var_1h"].to_numpy()
    gate = impl > np.sqrt(ent["trail_rv_var_1h"].to_numpy())
    pnl_1h = np.where(gate, (np.maximum(impl - 0.02, 0.0)) ** 2 - rvf, 0.0)

    # 12h c2c harvest over same recent window (gated+voltgt net 2vp)
    ent12 = vrp.harvest_pnl(vrp.build(coin, "c2c")[lambda x: x["time"] >= start],
                            phase=0, gate=True, voltgt=True, cost_vp=2.0)
    return dict(
        window=[str(start), str(rec["time"].max())], n_1h_obs=int(len(rec)),
        mean_vrp_1h_volpts=float(rec["vrp_vol_1h"].mean() * 100),
        mean_vrp_12h_c2c_volpts=float(p["vrp_vol"].mean() * 100),
        mean_vrp_12h_parkinson_volpts=float(pk["vrp_vol"].mean() * 100),
        pct_pos_1h=float((rec["vrp_vol_1h"] > 0).mean()),
        harvest_sharpe_1h_net2vp=vrp.ann_sharpe(pnl_1h), n_windows_1h=int((~np.isnan(pnl_1h)).sum()),
        harvest_sharpe_12h_c2c_net2vp_recent=vrp.ann_sharpe(ent12["pnl"].to_numpy()))


# ---------- B) live ATM bid-ask -> vol-point execution cost ----------
def atm_cost(cur):
    with urllib.request.urlopen(API.format(cur), timeout=30) as r:
        df = pd.DataFrame(json.load(r)["result"])

    def parse(n):
        m = re.match(r"[A-Z]+-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-[CP]", str(n))
        return pd.Series(m.groups() if m else (None, None))
    df[["exp", "strike"]] = df["instrument_name"].apply(parse)
    df = df.dropna(subset=["strike"]).copy()
    df["strike"] = df["strike"].astype(float)
    df["exp_dt"] = pd.to_datetime(df["exp"], format="%d%b%y", utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    df["tenor_d"] = (df["exp_dt"] - now).dt.total_seconds() / 86400
    U = float(df["underlying_price"].median())
    df["moneyness"] = df["strike"] / U
    atm = df[(df["tenor_d"].between(3, 21)) & (df["moneyness"].between(0.95, 1.05))
             & (df["bid_price"] > 0) & (df["ask_price"] > 0) & (df["mark_price"] > 0)].copy()
    hs_frac = ((atm["ask_price"] - atm["bid_price"]) / (2 * atm["mark_price"])).to_numpy()
    iv = atm["mark_iv"].to_numpy()                    # vol points
    cost_ow = hs_frac * iv                            # cost in vol points (one way) = halfspread% * IV
    cost_ow = cost_ow[np.isfinite(cost_ow)]
    return dict(n_atm=int(len(atm)), median_iv_volpts=float(np.median(iv)),
                median_halfspread_pct_of_premium=float(np.median(hs_frac) * 100),
                median_cost_oneway_volpts=float(np.median(cost_ow)),
                median_cost_roundtrip_volpts=float(2 * np.median(cost_ow)), underlying=U)


def main():
    out = {"A_hf_rv": {}, "B_atm_cost": {}}
    for c in ["BTC", "ETH"]:
        out["A_hf_rv"][c] = hf_vrp(c)
        try:
            out["B_atm_cost"][c] = atm_cost(c)
        except Exception as e:
            out["B_atm_cost"][c] = {"error": str(e)}
    json.dump(out, open(f"{ROOT}/experiments/paper1_vrp_polish.json", "w"), indent=2, default=str)

    print("=== A) HIGH-FREQUENCY (1h) RV, recent regime ===")
    for c in ["BTC", "ETH"]:
        a = out["A_hf_rv"][c]
        print(f" {c} [{a['window'][0][:10]}..{a['window'][1][:10]}]: "
              f"mean VRP  1h={a['mean_vrp_1h_volpts']:+.2f}  "
              f"12h-c2c={a['mean_vrp_12h_c2c_volpts']:+.2f}  "
              f"12h-park={a['mean_vrp_12h_parkinson_volpts']:+.2f} vp  ({a['pct_pos_1h']*100:.0f}% pos)")
        print(f"     harvest Sharpe recent: 1h-RV={a['harvest_sharpe_1h_net2vp']:.2f} "
              f"(n={a['n_windows_1h']})  12h-c2c={a['harvest_sharpe_12h_c2c_net2vp_recent']:.2f}")
    print("=== B) LIVE ATM bid-ask execution cost (vol points) ===")
    for c in ["BTC", "ETH"]:
        b = out["B_atm_cost"][c]
        if "error" in b:
            print(f" {c}: ERROR {b['error']}"); continue
        print(f" {c}: ATM IV {b['median_iv_volpts']:.0f}vp, half-spread {b['median_halfspread_pct_of_premium']:.1f}% of premium "
              f"-> cost {b['median_cost_oneway_volpts']:.2f}vp one-way, {b['median_cost_roundtrip_volpts']:.2f}vp round-trip "
              f"(n={b['n_atm']} ATM opts)")
    print("\nWrote experiments/paper1_vrp_polish.json")


if __name__ == "__main__":
    main()
