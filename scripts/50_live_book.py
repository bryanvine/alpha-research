#!/usr/bin/env python3
"""
50_live_book.py -- LIVE paper-traded diversified book (Paper 7, taken forward).

Tracks the diversified risk-premium book as a paper portfolio, marked daily, net of costs.
It trades the daily-priceable subset of the Paper-7 survivors (the sleeves a non-broker
participant can actually mark every day from free data):
  * crypto funding carry   (cross-sectional, Binance perps)
  * crypto trend           (cross-sectional 30d momentum, top-30)
  * FX carry               (G10 rate-differential sort, marked on daily spot)
  * FX value               (G10 REER sort, marked on daily spot)
Each sleeve return is already net of its own trading cost; the book applies inverse-trailing-
vol risk parity (lagged), scales to a 10%/yr vol target, and charges a small overlay
rebalancing cost. (Equity quality / reversal are part of the full Paper-7 book but update
monthly from Ken French and need a broker/ETF to trade daily — tracked separately, not here.)

Idempotent: recomputes the full NAV from the current data each run, so a daily cron that first
refreshes the crypto/FX caches extends the track forward. Inception 2024-01-01.

Outputs -> data/live_book/nav.csv (date, sleeve rets, weights, combined, nav) + state.json.
"""
import os, sys, json, datetime, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import carry as C        # noqa: E402
from alpha_research.factors import equity_zoo as EZ  # noqa: E402
from alpha_research.factors import macro_carry as MC  # noqa: E402

OUT = f"{ROOT}/data/live_book"
INCEPTION = "2024-01-01"
TARGET_VOL_D = 0.10 / np.sqrt(252)     # 10%/yr daily target
OVERLAY_BP = 3.0                       # overlay rebalancing cost (bps of weight turnover)


def _naive(s):
    s = s.copy()
    idx = pd.to_datetime(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)).normalize()
    return s.groupby(level=0).last()


def daily_ls(sig, ret, n=3, cost_bps=2.0):
    """Daily cross-sectional long-top-n / short-bottom-n (lagged signal), dollar-neutral."""
    sig = sig.reindex(ret.index).shift(1)
    sl = (sig.rank(axis=1, ascending=False) <= n).astype(float)
    ss = (sig.rank(axis=1, ascending=True) <= n).astype(float)
    W = sl.div(sl.sum(1), axis=0).fillna(0) * 0.5 - ss.div(ss.sum(1), axis=0).fillna(0) * 0.5
    pnl = (W * ret.fillna(0)).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0)
    return pnl - turn * cost_bps / 1e4


def sleeves():
    # crypto (already daily, net of cost)
    F, Rc = C.load_daily_funding(), C.load_spot_returns()
    cc = _naive(C.cross_sectional_factor(F, Rc, 7, 0.33, 6.0, 5))
    cmom = np.log1p(Rc.fillna(0)).rolling(30).sum()
    ct = _naive(EZ.ls_backtest(cmom, Rc, q=0.33, rebal=7, cost_bps=10.0)[0])
    # FX daily (mark the carry/value baskets on daily spot)
    carry_d, spot, reer = MC.load_fx()
    spot = spot.sort_index()
    exret = (-np.log(spot / spot.shift(1))) + (carry_d.reindex(spot.index, method="ffill") / 100 / 252).reindex(columns=spot.columns)
    fxc = daily_ls(carry_d.reindex(spot.index, method="ffill"), exret, n=3)
    rsig = reer.reindex(spot.index, method="ffill"); vsig = -(rsig.sub(rsig.mean(axis=1), axis=0))
    fxv = daily_ls(vsig[spot.columns.intersection(vsig.columns)], exret, n=3)
    df = pd.DataFrame({"crypto_carry": cc, "crypto_trend": ct,
                       "fx_carry": _naive(fxc), "fx_value": _naive(fxv)})
    return df[df.index >= pd.Timestamp(INCEPTION)].sort_index()


def build():
    S = sleeves().dropna(how="all")
    vol = S.rolling(90, min_periods=20).std()
    w = (1.0 / vol); w = w.div(w.sum(axis=1), axis=0).shift(1)          # risk parity, lagged
    gross = (w * S).sum(axis=1)
    overlay = w.diff().abs().sum(axis=1).fillna(0) * OVERLAY_BP / 1e4
    pvol = gross.rolling(90, min_periods=20).std().shift(1)
    scale = (TARGET_VOL_D / pvol).clip(upper=3.0)
    combined = (scale * gross - overlay).dropna()
    nav = (1.0 + combined).cumprod()
    led = S.reindex(combined.index).copy()
    led["weight_" ] = None
    led = led.assign(combined_ret=combined, nav=nav)
    for c in S.columns:
        led[f"w_{c}"] = w[c].reindex(combined.index)
    return S, w, combined, nav, led


def stats(combined):
    x = combined.values
    cum = (1 + combined).cumprod(); dd = (cum / cum.cummax() - 1)
    return dict(ann_return_pct=round(float((cum.iloc[-1] ** (252 / len(x)) - 1) * 100), 1),
                ann_vol_pct=round(float(np.std(x) * np.sqrt(252) * 100), 1),
                sharpe=round(float(np.mean(x) / np.std(x) * np.sqrt(252)), 2) if np.std(x) > 0 else None,
                maxdd_pct=round(float(dd.min() * 100), 1), total_return_pct=round(float((cum.iloc[-1] - 1) * 100), 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    S, w, combined, nav, led = build()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    led.drop(columns=["weight_"]).to_csv(f"{OUT}/nav.csv")
    cur_w = w.iloc[-1].round(3).to_dict()
    st = dict(updated=now, inception=INCEPTION, last_data_date=str(combined.index[-1].date()),
              n_days=int(len(combined)), latest_nav=round(float(nav.iloc[-1]), 4),
              current_target_weights=cur_w, since_inception=stats(combined),
              sleeve_sharpe={c: round(float(S[c].mean() / S[c].std() * np.sqrt(252)), 2)
                            for c in S.columns if S[c].std() > 0})
    json.dump(st, open(f"{OUT}/state.json", "w"), indent=2, default=str)

    print(f"=== LIVE diversified book (paper-traded) — updated {now} ===")
    print(f"  inception {INCEPTION} · last data {st['last_data_date']} · {st['n_days']} days")
    print(f"  NAV {st['latest_nav']}  (since inception: {st['since_inception']['total_return_pct']}% total, "
          f"{st['since_inception']['ann_return_pct']}%/yr, vol {st['since_inception']['ann_vol_pct']}%, "
          f"Sharpe {st['since_inception']['sharpe']}, maxDD {st['since_inception']['maxdd_pct']}%)")
    print(f"  current target weights: {cur_w}")
    print(f"  sleeve Sharpes (since inception): {st['sleeve_sharpe']}")
    print(f"\nWrote {OUT}/nav.csv + state.json")
    print("NOTE: paper-traded; daily-tradable sleeves only (crypto carry/trend + FX carry/value).")
    print("Equity quality/reversal are in the full Paper-7 book but tracked monthly (French) — not here.")


if __name__ == "__main__":
    main()
