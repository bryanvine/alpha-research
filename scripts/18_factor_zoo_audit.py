#!/usr/bin/env python3
"""
18_factor_zoo_audit.py -- Paper 4, Part A: the authoritative replication audit.

Open-Source Asset Pricing (Chen-Zimmermann) 212 published predictors + Ken French factors:
  * post-publication decay (McLean-Pontiff): in-sample vs out-of-sample mean return
  * t>3 survival under multiple testing (Harvey-Liu-Zhu) vs the lax t>2 bar
  * recent-decade (2015-2024) reality: how many anomalies are still alive
  * net-of-cost haircut: EW-decile long-shorts are high-turnover -> flat cost sensitivity
  * French factors recent-decade performance (which canonical factors survive)

Results -> experiments/paper4_zoo_audit.json
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EF = f"{ROOT}/data/equity_factors"


def ann_sharpe_m(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(12)) if x.size >= 12 and x.std() > 0 else np.nan


def tstat_m(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std() * np.sqrt(x.size)) if x.size >= 12 and x.std() > 0 else np.nan


def main():
    R = pd.read_parquet(f"{EF}/oap_ls_returns.parquet")
    R.index = pd.to_datetime(R.index)
    doc = pd.read_csv(f"{EF}/oap_signaldoc.csv")
    # locate acronym + sample-window + t columns robustly
    acol = next(c for c in doc.columns if c.lower() in ("acronym", "signalname", "predictor"))
    scol = next((c for c in doc.columns if "samplestart" in c.lower().replace("_", "")), None)
    ecol = next((c for c in doc.columns if "sampleend" in c.lower().replace("_", "")), None)
    doc = doc.set_index(acol)

    rows = []
    for col in R.columns:
        if col not in doc.index:
            continue
        r = R[col].dropna()
        if len(r) < 36:
            continue
        try:
            e = int(doc.loc[col, ecol]); s = int(doc.loc[col, scol])
        except Exception:
            continue
        is_r = r[(r.index.year >= s) & (r.index.year <= e)]
        oos_r = r[r.index.year > e]
        rec_r = r[r.index.year >= 2015]
        if len(is_r) < 24 or len(oos_r) < 24:
            continue
        rows.append(dict(pred=col, is_mean=is_r.mean(), oos_mean=oos_r.mean(),
                         is_t=tstat_m(is_r), oos_t=tstat_m(oos_r), full_t=tstat_m(r),
                         rec_mean=rec_r.mean(), rec_sharpe=ann_sharpe_m(rec_r), rec_t=tstat_m(rec_r),
                         oos_sharpe=ann_sharpe_m(oos_r)))
    D = pd.DataFrame(rows)
    n = len(D)

    # sign sanity: OSAP LS should be pre-signed (mostly positive in-sample)
    sign_ok = float((D["is_mean"] > 0).mean())

    # McLean-Pontiff decay
    decay = 1 - D["oos_mean"].mean() / D["is_mean"].mean()
    # cost haircut on OOS (EW-decile LS are high-turnover): flat monthly cost sensitivity
    cost = {f"{int(c*1e4)}bp_mo": round(float(((D["oos_mean"] - c) > 0).mean()), 3)
            for c in [0.0, 0.003, 0.006]}

    out = dict(
        n_predictors=n, sign_check_frac_is_positive=round(sign_ok, 3),
        decay=dict(mean_is_monthly_pct=round(D["is_mean"].mean() * 100, 3),
                   mean_oos_monthly_pct=round(D["oos_mean"].mean() * 100, 3),
                   mclean_pontiff_decay_pct=round(decay * 100, 1),
                   frac_oos_lower_than_is=round(float((D["oos_mean"] < D["is_mean"]).mean()), 3),
                   frac_oos_positive=round(float((D["oos_mean"] > 0).mean()), 3),
                   frac_oos_t_gt2=round(float((D["oos_t"] > 2).mean()), 3),
                   frac_oos_t_gt3=round(float((D["oos_t"] > 3).mean()), 3)),
        t_hurdle=dict(frac_full_t_gt2=round(float((D["full_t"] > 2).mean()), 3),
                      frac_full_t_gt3=round(float((D["full_t"] > 3).mean()), 3),
                      median_full_t=round(float(D["full_t"].median()), 2)),
        recent_decade=dict(frac_positive=round(float((D["rec_mean"] > 0).mean()), 3),
                           frac_t_gt3=round(float((D["rec_t"] > 3).mean()), 3),
                           median_sharpe=round(float(D["rec_sharpe"].median()), 2),
                           frac_sharpe_gt_floor=round(float((D["rec_sharpe"] > 0.3).mean()), 3)),
        oos_net_of_cost_frac_positive=cost,
    )

    # Ken French recent-decade
    F = pd.read_parquet(f"{EF}/famafrench_monthly.parquet"); F.index = pd.to_datetime(F.index)
    rec = F[F.index.year >= 2015]
    ff = {}
    for c in [x for x in F.columns if x.upper() != "RF"]:
        ff[c] = dict(sharpe_2015_25=round(ann_sharpe_m(rec[c].values), 2), t_2015_25=round(tstat_m(rec[c].values), 2),
                     sharpe_full=round(ann_sharpe_m(F[c].dropna().values), 2))
    out["french_recent"] = ff
    out["french_span"] = [str(F.index.min())[:7], str(F.index.max())[:7]]

    json.dump(out, open(f"{ROOT}/experiments/paper4_zoo_audit.json", "w"), indent=2, default=str)

    print(f"OSAP replication audit: {n} predictors with valid IS+OOS windows (sign-positive IS: {sign_ok*100:.0f}%)")
    d = out["decay"]
    print(f"\nPOST-PUBLICATION DECAY (McLean-Pontiff):")
    print(f"  mean monthly LS return: in-sample {d['mean_is_monthly_pct']}%  ->  out-of-sample {d['mean_oos_monthly_pct']}%  "
          f"(decay {d['mclean_pontiff_decay_pct']}%)")
    print(f"  {d['frac_oos_positive']*100:.0f}% still positive OOS; {d['frac_oos_t_gt2']*100:.0f}% keep |t|>2, "
          f"{d['frac_oos_t_gt3']*100:.0f}% keep |t|>3")
    t = out["t_hurdle"]
    print(f"\nMULTIPLE TESTING (full-sample): {t['frac_full_t_gt2']*100:.0f}% clear |t|>2, only "
          f"{t['frac_full_t_gt3']*100:.0f}% clear |t|>3 (median |t|={t['median_full_t']})")
    rc = out["recent_decade"]
    print(f"\nRECENT DECADE 2015-2024: {rc['frac_positive']*100:.0f}% positive, {rc['frac_t_gt3']*100:.0f}% |t|>3, "
          f"median Sharpe {rc['median_sharpe']}, {rc['frac_sharpe_gt_floor']*100:.0f}% beat +0.3 floor")
    print(f"\nNET-OF-COST (OOS, flat monthly cost) frac still positive: {cost}")
    print(f"\nKEN FRENCH factors (Sharpe 2015-2025 | t | full-sample Sharpe), span {out['french_span']}:")
    for c, v in ff.items():
        print(f"  {c:8s}: {v['sharpe_2015_25']:>6} | t={v['t_2015_25']:>6} | full {v['sharpe_full']}")
    print("\nWrote experiments/paper4_zoo_audit.json")


if __name__ == "__main__":
    main()
