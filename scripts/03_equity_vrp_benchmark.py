#!/usr/bin/env python3
"""
03_equity_vrp_benchmark.py -- Paper 1 EQUITY cross-asset benchmark (the control).

Quantifies the EQUITY volatility risk premium (VIX^2 - forward realized variance of
SPX) and tests whether *harvesting* it via index option-writing (Cboe PUT PutWrite /
BXM BuyWrite total-return indices) is dead net-of-cost since ~2010. This is the
control that contextualizes the crypto VRP result in 01_vrp_core.py.

Methodology mirrors 01_vrp_core.py for comparability:
  * VRP in variance units AND vol points; non-overlapping t-stat (thin=21);
    rigor.degenerate_signal_check / probabilistic_sharpe_ratio on the harvest.
  * Equity conventions: 252 trading days/yr; forward window H=21 trading days (~1m),
    matching VIX's 30-calendar-day horizon (21 trading days). Annualize variance x252.

NO LOOK-AHEAD for the VRP sign series: VIX_t is known at close t; the realized leg
uses returns over (t, t+21] -- resolved later, that *is* the variance-swap payoff, not
leakage. The option-writing indices are themselves traded total-return series, so their
returns are already net of the strategy's mechanics (the only cost question is whether
the index's embedded assumptions are realistic -- see notes; we add no extra cost).

DATA (data/equity_vol/):
  * VIX:  vixcls.csv  (FRED VIXCLS, 1990-)         [implied 30d, annualized vol %]
  * SPX:  spx_yahoo.csv (Yahoo ^GSPC daily close, 1990-2026)  -- LONGEST CLEAN SERIES
          fred_SP500.csv (FRED, 2016-) used only as a cross-check (agrees to 1e-4).
          (Stooq was blocked by a JavaScript proof-of-work anti-bot wall -> Access denied;
           Yahoo's v8 chart JSON is the clean full-history substitute. See research md.)
  * PUT:  cboe_PUT.csv  (PutWrite TR index, 1991-)
  * BXM:  cboe_BXM.csv  (BuyWrite TR index, 2002-)
  * also BXMD / WPUT / PPUT for breadth.
  * RF:   fred_DGS3MO.csv (3M T-bill, 1981-) for excess-return Sharpes.

Outputs -> experiments/paper1_equity_benchmark.json  (+ a printed summary).
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew, kurtosis as sp_kurt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.eval import rigor    # noqa: E402
from alpha_research.eval import metrics  # noqa: E402

DATA = os.path.join(ROOT, "data", "equity_vol")
TD = 252                 # trading days / year
H = 21                   # forward realized-variance window (~1 month, matches VIX 30cd)
MAX_GAP_DAYS = 7         # drop any "return" spanning a >7-calendar-day stitching gap.
                         # Cboe PUT has 7 sparse backfill anchors (1991-2004) before it
                         # goes daily in 2007; without this filter their multi-YEAR gaps
                         # become single spurious ~50% "daily" returns. No-op for the
                         # genuinely-daily indices (BXM 2002+, BXMD/PPUT 1986+, WPUT 2006+).
ANN = TD / H             # annualization for an H-day realized-variance window
WIN_PER_YEAR = TD / H    # ~12 non-overlapping 21-day windows / year (mirrors crypto WIN_PER_YEAR)

# Sub-period boundaries for the "is it dead net?" test.
P_2010 = pd.Timestamp("2010-01-01")
P_2020 = pd.Timestamp("2020-01-01")
DECADES = [("1990s", "1990-01-01", "2000-01-01"),
           ("2000s", "2000-01-01", "2010-01-01"),
           ("2010s", "2010-01-01", "2020-01-01"),
           ("2020s", "2020-01-01", "2030-01-01")]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_fred(fname, col):
    df = pd.read_csv(os.path.join(DATA, fname), na_values=["."])
    df.columns = ["Date", col]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.dropna(subset=[col]).sort_values("Date").reset_index(drop=True)


def load_cboe_close(fname, sym):
    """Cboe CSV: DATE + (OHLC) or DATE+<SYM>. Use CLOSE if present else the lone col."""
    df = pd.read_csv(os.path.join(DATA, fname))
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    val = "CLOSE" if "CLOSE" in df.columns else [c for c in df.columns if c != "DATE"][0]
    out = df[["DATE", val]].rename(columns={"DATE": "Date", val: sym})
    out[sym] = pd.to_numeric(out[sym], errors="coerce")
    return out.dropna().sort_values("Date").reset_index(drop=True)


def load_spx():
    y = pd.read_csv(os.path.join(DATA, "spx_yahoo.csv"), parse_dates=["Date"])
    y = y.rename(columns={"Close": "spx"}).sort_values("Date").reset_index(drop=True)
    return y[["Date", "spx"]]


# ---------------------------------------------------------------------------
# Stats helpers (mirror 01_vrp_core.py)
# ---------------------------------------------------------------------------
def ann_sharpe_window(x):
    """Annualized Sharpe of a per-WINDOW (non-overlapping 21d) return series."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 8 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(WIN_PER_YEAR))


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


def maxdd_from_levels(levels):
    """Max drawdown from a level series (decimal, negative)."""
    lv = np.asarray(levels, float); lv = lv[np.isfinite(lv)]
    if lv.size < 2:
        return float("nan")
    peak = np.maximum.accumulate(lv)
    return float((lv / peak - 1.0).min())


def perf_from_daily(daily_ret, rf_daily=None):
    """Annualized return/vol/Sharpe/maxDD from a DAILY simple-return series.

    If rf_daily aligned series given, Sharpe is on EXCESS returns; ann_return is the
    geometric total return (gross). Drawdown from the gross cumulative level.
    """
    r = pd.Series(daily_ret).dropna()
    if len(r) < 30:
        return dict(ann_return=float("nan"), ann_vol=float("nan"), sharpe=float("nan"),
                    sharpe_excess=float("nan"), maxdd=float("nan"), n=int(len(r)))
    ann_ret = float((1.0 + r).prod() ** (TD / len(r)) - 1.0)
    ann_vol = float(r.std() * np.sqrt(TD))
    sharpe = float(r.mean() / r.std() * np.sqrt(TD)) if r.std() > 0 else float("nan")
    if rf_daily is not None:
        ex = (r - pd.Series(rf_daily).reindex(r.index)).dropna()
        sh_ex = float(ex.mean() / ex.std() * np.sqrt(TD)) if len(ex) > 30 and ex.std() > 0 else float("nan")
    else:
        sh_ex = float("nan")
    lv = (1.0 + r).cumprod()
    return dict(ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sharpe_excess=sh_ex, maxdd=maxdd_from_levels(lv.to_numpy()), n=int(len(r)))


# ===========================================================================
# 1. EQUITY VRP
# ===========================================================================
def build_vrp():
    """VRP_t = (VIX_t/100)^2 - forward 21-trading-day realized variance (annualized x252)."""
    spx = load_spx()
    vix = load_fred("vixcls.csv", "vix")
    df = spx.merge(vix, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    close = df["spx"].to_numpy(float)
    lr = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
    r2 = pd.Series(lr ** 2)
    # forward sum of squared close-to-close returns over (t, t+H]  (the realized leg)
    fwd_sum = r2.rolling(H).sum().shift(-H)
    trail_sum = r2.rolling(H).sum()            # trailing (no look-ahead) -- for the harvest gate

    df["impl_vol"] = df["vix"] / 100.0
    df["impl_var"] = df["impl_vol"] ** 2
    df["fwd_rv_var"] = fwd_sum.to_numpy() * ANN
    df["trail_rv_var"] = trail_sum.to_numpy() * ANN
    df["vrp_var"] = df["impl_var"] - df["fwd_rv_var"]
    df["vrp_vol"] = df["impl_vol"] - np.sqrt(df["fwd_rv_var"])
    df["fwd_ret"] = np.log(pd.Series(close).shift(-H) / pd.Series(close)).to_numpy()
    return df


def analyze_vrp(df):
    ov = df.dropna(subset=["vrp_var", "vrp_vol", "impl_vol"]).copy()
    res = {
        "span": [str(ov["Date"].iloc[0].date()), str(ov["Date"].iloc[-1].date())],
        "n_days": int(len(ov)),
        "sign": dict(
            mean_vrp_volpts=float(ov["vrp_vol"].mean() * 100),
            median_vrp_volpts=float(ov["vrp_vol"].median() * 100),
            pct_positive=float((ov["vrp_vol"] > 0).mean()),
            mean_vrp_var=float(ov["vrp_var"].mean()),
            nw_t_var=nw_t(ov["vrp_var"].to_numpy(), H),
            nonoverlap_t_var=rigor.nonoverlapping_tstat(ov["vrp_var"].to_numpy(), thin=H),
            nonoverlap_t_vol=rigor.nonoverlapping_tstat(ov["vrp_vol"].to_numpy(), thin=H),
            mean_impl_volpts=float(ov["impl_vol"].mean() * 100),
            mean_realized_volpts=float(np.sqrt(ov["fwd_rv_var"]).mean() * 100),
        ),
    }
    # by decade
    bydec = {}
    for name, lo, hi in DECADES:
        g = ov[(ov["Date"] >= pd.Timestamp(lo)) & (ov["Date"] < pd.Timestamp(hi))]
        if len(g) == 0:
            continue
        bydec[name] = dict(
            mean_volpts=float(g["vrp_vol"].mean() * 100),
            median_volpts=float(g["vrp_vol"].median() * 100),
            pct_pos=float((g["vrp_vol"] > 0).mean()),
            mean_var=float(g["vrp_var"].mean()),
            nonoverlap_t_var=rigor.nonoverlapping_tstat(g["vrp_var"].to_numpy(), thin=H)["t"],
            mean_impl_volpts=float(g["impl_vol"].mean() * 100),
            mean_realized_volpts=float(np.sqrt(g["fwd_rv_var"]).mean() * 100),
            n=int(len(g)),
        )
    res["by_decade"] = bydec

    # pre / post 2010
    pre = ov[ov["Date"] < P_2010]; post = ov[ov["Date"] >= P_2010]
    res["pre_post_2010"] = dict(
        pre2010=dict(mean_volpts=float(pre["vrp_vol"].mean() * 100),
                     pct_pos=float((pre["vrp_vol"] > 0).mean()),
                     mean_var=float(pre["vrp_var"].mean()),
                     nonoverlap_t_var=rigor.nonoverlapping_tstat(pre["vrp_var"].to_numpy(), thin=H)["t"],
                     n=int(len(pre))),
        post2010=dict(mean_volpts=float(post["vrp_vol"].mean() * 100),
                      pct_pos=float((post["vrp_vol"] > 0).mean()),
                      mean_var=float(post["vrp_var"].mean()),
                      nonoverlap_t_var=rigor.nonoverlapping_tstat(post["vrp_var"].to_numpy(), thin=H)["t"],
                      n=int(len(post))),
    )
    # rigor on the VRP signal itself
    deg = rigor.degenerate_signal_check(ov["vrp_vol"].to_numpy())
    res["rigor"] = dict(degenerate_vrp=dict(flagged=deg.is_degenerate, reasons=deg.reasons))
    return res, ov


# ===========================================================================
# 2. OPTION-WRITING HARVEST  (the real "is it dead net?" test)
# ===========================================================================
def analyze_writer(name, idx, spx, rf):
    """idx: DataFrame[Date, <name>] TR-index levels. Returns sub-period perf + vs-SPX."""
    s = idx.rename(columns={name: "lvl"}).set_index("Date")["lvl"].sort_index()
    sp = spx.set_index("Date")["spx"].sort_index()
    rf_s = rf.set_index("Date")["rf"].sort_index() if rf is not None else None

    # daily simple returns, with stitching-gap filter (see MAX_GAP_DAYS).
    gap = s.index.to_series().diff().dt.days
    r = s.pct_change().mask(gap > MAX_GAP_DAYS)         # drop returns spanning a backfill gap
    n_dropped = int((gap > MAX_GAP_DAYS).sum())
    # first date after which the series is continuously daily (the honest start)
    big = gap[gap > MAX_GAP_DAYS]
    dense_start = (big.index.max() if len(big) else s.index.min())
    rsp = sp.pct_change()
    # daily risk-free: DGS3MO is an annualized % yield -> per-trading-day simple
    rf_daily = (rf_s / 100.0 / TD) if rf_s is not None else None

    res = {"span": [str(s.index.min().date()), str(s.index.max().date())],
           "dense_daily_start": str(dense_start.date()),
           "n_returns_dropped_gap": n_dropped,
           "n_days": int(len(s))}

    def block(lo, hi, label):
        msk = (r.index >= pd.Timestamp(lo)) & (r.index < pd.Timestamp(hi))
        rr = r[msk].dropna()
        if len(rr) < 30:
            return None
        rfd = rf_daily.reindex(rr.index) if rf_daily is not None else None
        wp = perf_from_daily(rr, rfd)
        # SPX over the same realized dates (use the writer's own trading dates)
        rsp_b = rsp.reindex(rr.index).dropna()
        rfd_sp = rf_daily.reindex(rsp_b.index) if rf_daily is not None else None
        spp = perf_from_daily(rsp_b, rfd_sp)
        # excess of writer over SPX (annualized geometric return diff)
        return dict(label=label,
                    realized_span=[str(rr.index.min().date()), str(rr.index.max().date())],
                    writer=wp, spx=spp,
                    excess_ann_return=wp["ann_return"] - spp["ann_return"],
                    sharpe_diff=wp["sharpe"] - spp["sharpe"])

    res["full_sample"] = block("1900-01-01", "2100-01-01", "full")
    res["sub_periods"] = {}
    for lo, hi, label in [("1900-01-01", "2010-01-01", "pre2010"),
                          ("2010-01-01", "2020-01-01", "2010_2020"),
                          ("2020-01-01", "2030-01-01", "2020_2026")]:
        b = block(lo, hi, label)
        if b is not None:
            res["sub_periods"][label] = b

    # rigor on the FULL-SAMPLE daily writer returns
    rr_full = r.dropna()
    deg = rigor.degenerate_signal_check(rr_full.to_numpy())
    res["rigor"] = dict(
        degenerate=dict(flagged=deg.is_degenerate, reasons=deg.reasons),
        psr_daily=rigor.probabilistic_sharpe_ratio(rr_full.to_numpy(), 0.0),
    )
    # PSR by sub-period on daily returns (P(true SR>0))
    res["psr_by_period"] = {}
    for lo, hi, label in [("1900-01-01", "2010-01-01", "pre2010"),
                          ("2010-01-01", "2020-01-01", "2010_2020"),
                          ("2020-01-01", "2030-01-01", "2020_2026")]:
        rr = r[(r.index >= pd.Timestamp(lo)) & (r.index < pd.Timestamp(hi))].dropna()
        res["psr_by_period"][label] = rigor.probabilistic_sharpe_ratio(rr.to_numpy(), 0.0)
    return res


# ===========================================================================
# 3. CRYPTO vs EQUITY comparison
# ===========================================================================
def crypto_vs_equity(vrp_res):
    try:
        cj = json.load(open(os.path.join(ROOT, "experiments", "paper1_vrp_core.json")))
    except Exception as e:
        return {"error": f"could not load paper1_vrp_core.json: {e}"}
    out = {"equity": dict(
        mean_vrp_volpts=vrp_res["sign"]["mean_vrp_volpts"],
        pct_positive=vrp_res["sign"]["pct_positive"],
        nonoverlap_t_var=vrp_res["sign"]["nonoverlap_t_var"]["t"],
        pre2010_volpts=vrp_res["pre_post_2010"]["pre2010"]["mean_volpts"],
        post2010_volpts=vrp_res["pre_post_2010"]["post2010"]["mean_volpts"],
        decay_ratio_post_over_pre=(vrp_res["pre_post_2010"]["post2010"]["mean_volpts"] /
                                   vrp_res["pre_post_2010"]["pre2010"]["mean_volpts"]
                                   if vrp_res["pre_post_2010"]["pre2010"]["mean_volpts"] else float("nan")),
    )}
    for coin in ("BTC", "ETH"):
        if coin not in cj:
            continue
        c = cj[coin]; sign = c["sign"]; dec = c.get("by_year_sign", {})
        early = np.mean([dec[y]["mean_volpts"] for y in dec if int(y) <= 2023]) if dec else float("nan")
        late = np.mean([dec[y]["mean_volpts"] for y in dec if int(y) >= 2024]) if dec else float("nan")
        out[coin] = dict(
            mean_vrp_volpts=sign["mean_vrp_volpts"],
            pct_positive=sign["pct_positive"],
            nonoverlap_t_var=sign["nonoverlap_t_var"]["t"],
            early_21_23_volpts=float(early),
            late_24_26_volpts=float(late),
            decay_split_sharpe_early=c.get("decay_split", {}).get("sharpe_2021_2023"),
            decay_split_sharpe_late=c.get("decay_split", {}).get("sharpe_2024_2026"),
            harvest_sharpe_gated_net2vp=c.get("strategies", {}).get("gated_net_var", {}).get("sharpe"),
        )
    return out


# ===========================================================================
def main():
    spx = load_spx()
    rf = load_fred("fred_DGS3MO.csv", "rf")

    out = {"meta": dict(
        trading_days_per_year=TD, fwd_window_days=H, ann_factor=ANN,
        windows_per_year=WIN_PER_YEAR,
        spx_source="Yahoo ^GSPC daily close (spx_yahoo.csv), 1990-2026; longest clean series",
        vix_source="FRED VIXCLS (vixcls.csv), 1990-",
        rf_source="FRED DGS3MO 3M T-bill (fred_DGS3MO.csv), annualized %, /252 to daily",
        cost_assumption=("Cboe PUT/BXM are TRADED total-return indices -> their daily "
                         "returns already embed the option-writing mechanics; NO extra "
                         "trading cost is layered on. Excess-over-SPX and excess-over-RF "
                         "Sharpes are reported to make the 'dead net' question concrete. "
                         "(The crypto study layers explicit vol-point costs because it has "
                         "no traded harvest index; here the index IS the net experience.)"),
        data_notes=("Stooq blocked by JS proof-of-work anti-bot wall (Access denied even "
                    "after solving PoW); FRED SP500 only spans 2016-. Yahoo v8 chart JSON "
                    "used for full 1990+ SPX (agrees with FRED SP500 to ~1e-4, log-ret "
                    "corr 0.99999 over 2016-2026 overlap). Yahoo close ends 2026-06-08."),
    )}

    # ---- 1. EQUITY VRP ----
    vrp_df = build_vrp()
    vrp_res, ov = analyze_vrp(vrp_df)
    out["equity_vrp"] = vrp_res

    # ---- 2. OPTION-WRITING HARVEST ----
    writers = {}
    writer_files = {"PUT": "cboe_PUT.csv", "BXM": "cboe_BXM.csv",
                    "BXMD": "cboe_BXMD.csv", "WPUT": "cboe_WPUT.csv", "PPUT": "cboe_PPUT.csv"}
    for sym, fn in writer_files.items():
        idx = load_cboe_close(fn, sym)
        writers[sym] = analyze_writer(sym, idx, spx, rf)
    out["option_writing"] = writers

    # ---- 3. CRYPTO vs EQUITY ----
    out["crypto_vs_equity"] = crypto_vs_equity(vrp_res)

    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "paper1_equity_benchmark.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ---------------------------- printed summary ----------------------------
    v = out["equity_vrp"]; s = v["sign"]; pp = v["pre_post_2010"]
    print("\n================ EQUITY VRP (control for crypto) ================")
    print(f"  span {v['span'][0]} .. {v['span'][1]}  ({v['n_days']} days)")
    print(f"  SIGN: mean VRP = {s['mean_vrp_volpts']:+.2f} vol pts "
          f"(impl {s['mean_impl_volpts']:.1f} - realized {s['mean_realized_volpts']:.1f}), "
          f"{s['pct_positive']*100:.0f}% positive")
    print(f"        non-overlap t(var) = {s['nonoverlap_t_var']['t']:.2f} (n={s['nonoverlap_t_var']['n']}); "
          f"NW t(var) = {s['nw_t_var']}")
    print("  BY DECADE (mean vol pts / %pos / non-overlap t):")
    for k, d in v["by_decade"].items():
        print(f"    {k}: {d['mean_volpts']:+.2f} vp  {d['pct_pos']*100:.0f}%  t={d['nonoverlap_t_var']:.2f}  (n={d['n']})")
    print(f"  PRE-2010:  {pp['pre2010']['mean_volpts']:+.2f} vp  ({pp['pre2010']['pct_pos']*100:.0f}% pos, "
          f"t={pp['pre2010']['nonoverlap_t_var']:.2f}, n={pp['pre2010']['n']})")
    print(f"  POST-2010: {pp['post2010']['mean_volpts']:+.2f} vp  ({pp['post2010']['pct_pos']*100:.0f}% pos, "
          f"t={pp['post2010']['nonoverlap_t_var']:.2f}, n={pp['post2010']['n']})")

    print("\n================ OPTION-WRITING HARVEST: is it dead net? ================")
    print("  (annualized; Sharpe = total-return SR; ex-SPX = ann return diff vs SPX same dates)")
    for sym in ["PUT", "BXM", "BXMD", "WPUT", "PPUT"]:
        w = out["option_writing"][sym]
        f = w["full_sample"]
        gapnote = (f"  [dense daily from {w['dense_daily_start']}, {w['n_returns_dropped_gap']} backfill-gap returns dropped]"
                   if w["n_returns_dropped_gap"] else "")
        print(f"\n  --- {sym}  ({w['span'][0]} .. {w['span'][1]}){gapnote} ---")
        print(f"   FULL:  ret={f['writer']['ann_return']*100:+.2f}%  vol={f['writer']['ann_vol']*100:.1f}%  "
              f"SR={f['writer']['sharpe']:.2f} (excess-RF SR={f['writer']['sharpe_excess']:.2f})  "
              f"maxDD={f['writer']['maxdd']*100:.1f}%")
        print(f"          SPX: ret={f['spx']['ann_return']*100:+.2f}%  SR={f['spx']['sharpe']:.2f}  "
              f"|  {sym}-SPX: ann ret {f['excess_ann_return']*100:+.2f}%  SR diff {f['sharpe_diff']:+.2f}")
        for lab in ["pre2010", "2010_2020", "2020_2026"]:
            b = w["sub_periods"].get(lab)
            if b is None:
                continue
            rs = b["realized_span"]
            print(f"   {lab:10s}[{rs[0]}..{rs[1]}]: SR={b['writer']['sharpe']:.2f} "
                  f"(exRF={b['writer']['sharpe_excess']:.2f})  ret={b['writer']['ann_return']*100:+.2f}%  "
                  f"vs SPX SR={b['spx']['sharpe']:.2f} ret={b['spx']['ann_return']*100:+.2f}%  "
                  f"-> ex-SPX ann {b['excess_ann_return']*100:+.2f}% (SRdiff {b['sharpe_diff']:+.2f})")
        rg = w["rigor"]
        print(f"   RIGOR: degenerate={rg['degenerate']['flagged']}  PSR(daily,full)={rg['psr_daily']:.3f}  "
              f"PSR by period: " + " ".join(f"{k}={vv:.2f}" for k, vv in w["psr_by_period"].items()))

    print("\n================ CRYPTO vs EQUITY VRP ================")
    ce = out["crypto_vs_equity"]
    eq = ce["equity"]
    print(f"  EQUITY: mean {eq['mean_vrp_volpts']:+.2f} vp ({eq['pct_positive']*100:.0f}% pos, t={eq['nonoverlap_t_var']:.2f})  "
          f"pre2010 {eq['pre2010_volpts']:+.2f} -> post2010 {eq['post2010_volpts']:+.2f} vp "
          f"(post/pre={eq['decay_ratio_post_over_pre']:.2f})")
    for coin in ("BTC", "ETH"):
        if coin in ce:
            c = ce[coin]
            print(f"  {coin}:    mean {c['mean_vrp_volpts']:+.2f} vp ({c['pct_positive']*100:.0f}% pos, t={c['nonoverlap_t_var']:.2f})  "
                  f"early '21-23 {c['early_21_23_volpts']:+.2f} -> late '24-26 {c['late_24_26_volpts']:+.2f} vp  "
                  f"harvest SR {c['decay_split_sharpe_early']:.2f}->{c['decay_split_sharpe_late']:.2f}")
    print("\nWrote experiments/paper1_equity_benchmark.json")
    return out


if __name__ == "__main__":
    main()
