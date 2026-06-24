#!/usr/bin/env python3
"""Paper 9 — analysis + publication figures for the two LLM fundamental-valuation bots.

Reads the LIVE SQLite stores the bots serve from:
  Vinebot (Damodaran DCF, prices everything):     /apps/dbot/data/dbot.db
  Buffybot (Buffett quality + owner-earnings):     /apps/buffybot/data/buffy.db

Takes the latest valuation per ticker from each (coverage accumulates over runs), then:
  * tabulates each bot's rating / tier distribution and breadth,
  * measures CROSS-BOT agreement (Spearman rank corr + sign crosstab) on shared names —
    two independent LLM-driven valuation methods triangulating the same universe,
  * surfaces the value/sector tilt the bots are implicitly taking,
  * lists the two-method-agreement longs (the fat-pitch overlap).

Writes docs/figures/p9_fig1_breadth.png, p9_fig2_crossbot.png, p9_fig3_sector_tilt.png
and experiments/paper9_stats.json (the auditable numbers the paper cites).
Prints a full AUDIT block to stdout so every number in the paper is reproducible.
"""
from __future__ import annotations
import os, json, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/docs/figures"; os.makedirs(FIG, exist_ok=True)
EXP = f"{ROOT}/experiments"; os.makedirs(EXP, exist_ok=True)
DBOT_DB = "/apps/dbot/data/dbot.db"
BUFFY_DB = "/apps/buffybot/data/buffy.db"

plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
ACC, ACC2, NEG, MUT, POS = "#1f6feb", "#e8a33d", "#b3261e", "#5b6470", "#137a3e"
RATING_ORDER = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
RATING_COLOR = {"STRONG BUY": "#0b7a33", "BUY": "#5fbf7e", "HOLD": "#b8bdc4",
                "SELL": "#e08a7a", "STRONG SELL": "#b3261e"}


def latest_per_ticker(db: str) -> pd.DataFrame:
    """Latest valuation row per ticker (coverage accumulates across run_dates)."""
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM recommendation", con)
    con.close()
    df = df.sort_values("run_date").groupby("ticker", as_index=False).tail(1)
    return df.reset_index(drop=True)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    d = latest_per_ticker(DBOT_DB)
    b = latest_per_ticker(BUFFY_DB)
    dmax = d["run_date"].max(); bmax = b["run_date"].max()

    # ---- per-bot breadth ----
    d_rating = d[d["margin_of_safety"].notna()]
    d_counts = d_rating["rating"].value_counts().reindex(RATING_ORDER).fillna(0).astype(int)
    d_buy = int(d_counts.get("STRONG BUY", 0) + d_counts.get("BUY", 0))
    d_sell = int(d_counts.get("SELL", 0) + d_counts.get("STRONG SELL", 0))
    d_frac_cheap = float((d_rating["margin_of_safety"] >= 0).mean())

    b_pass = b[b["passed"] == 1]
    b_tier = b["tier"].value_counts()
    b_counts = b_pass[b_pass["margin_of_safety"].notna()]["rating"].value_counts().reindex(RATING_ORDER).fillna(0).astype(int)
    b_buy = int(b_counts.get("STRONG BUY", 0) + b_counts.get("BUY", 0))

    # ---- cross-bot agreement on shared names (both have a numeric MOS) ----
    m = pd.merge(d[["ticker", "name", "sector", "margin_of_safety", "rating"]],
                 b[["ticker", "margin_of_safety", "rating", "passed", "tier", "quality_score"]],
                 on="ticker", suffixes=("_d", "_b"))
    m = m[m["margin_of_safety_d"].notna() & m["margin_of_safety_b"].notna()].copy()
    sh = len(m)
    both_cheap = int(((m.margin_of_safety_d >= 0) & (m.margin_of_safety_b >= 0)).sum())
    both_rich = int(((m.margin_of_safety_d < 0) & (m.margin_of_safety_b < 0)).sum())
    disagree = int(((m.margin_of_safety_d >= 0) != (m.margin_of_safety_b >= 0)).sum())
    sign_agree = (both_cheap + both_rich) / sh
    rho = spearman(m.margin_of_safety_d.to_numpy(), m.margin_of_safety_b.to_numpy())
    pear = float(np.corrcoef(np.clip(m.margin_of_safety_d, -1.5, 1.5),
                             np.clip(m.margin_of_safety_b, -1.5, 1.5))[0, 1])

    buys = {"STRONG BUY", "BUY"}
    fat = m[(m.rating_d.isin(buys)) & (m.rating_b.isin(buys)) & (m.passed == 1)]
    fat = fat.sort_values("margin_of_safety_d", ascending=False)

    # ---- sector tilt (Vinebot, full index) ----
    sect = (d_rating.groupby("sector")["margin_of_safety"]
            .agg(["median", "count"]).query("count >= 4").sort_values("median"))

    stats = {
        "as_of": {"vinebot_run": dmax, "buffybot_run": bmax},
        "vinebot": {"covered": int(len(d)), "valued": int(len(d_rating)),
                    "rating_counts": d_counts.to_dict(), "n_buy": d_buy, "n_sell": d_sell,
                    "frac_cheap": round(d_frac_cheap, 4),
                    "mean_mos": round(float(d_rating.margin_of_safety.mean()), 4)},
        "buffybot": {"scanned": int(len(b)), "watchlist": int(len(b_pass)),
                     "tier_counts": b_tier.to_dict(),
                     "watchlist_rating_counts": b_counts.to_dict(), "n_buy": b_buy,
                     "frac_pass": round(float((b["passed"] == 1).mean()), 4)},
        "crossbot": {"shared": sh, "both_cheap": both_cheap, "both_rich": both_rich,
                     "disagree": disagree, "sign_agreement": round(sign_agree, 4),
                     "spearman": round(rho, 4), "pearson_winsor": round(pear, 4),
                     "fat_pitch_overlap": int(len(fat)),
                     "fat_pitch_names": fat[["ticker", "name", "margin_of_safety_d",
                                             "margin_of_safety_b", "quality_score"]].to_dict("records")},
        "sector_tilt": {s: {"median_mos": round(float(r["median"]), 4), "n": int(r["count"])}
                        for s, r in sect.iterrows()},
    }
    json.dump(stats, open(f"{EXP}/paper9_stats.json", "w"), indent=2)

    # ===================== AUDIT =====================
    print("=" * 66)
    print(f"PAPER 9 AUDIT  ·  Vinebot {dmax} · Buffybot {bmax}")
    print("=" * 66)
    print(f"\nVINEBOT (prices everything): {len(d)} covered, {len(d_rating)} valued")
    for r in RATING_ORDER:
        print(f"   {r:12s} {d_counts[r]:4d}")
    print(f"   BUY+ : {d_buy}  ({d_buy/len(d_rating):.1%})   SELL+ : {d_sell}  ({d_sell/len(d_rating):.1%})")
    print(f"   fraction trading below DCF value (cheap): {d_frac_cheap:.1%}   mean MOS {d_rating.margin_of_safety.mean():+.3f}")
    print(f"\nBUFFYBOT (wonderful businesses only): {len(b)} scanned, {len(b_pass)} on watchlist ({(b['passed']==1).mean():.1%})")
    for t in ["wonderful", "good", "fair", "too hard"]:
        if t in b_tier:
            print(f"   {t:10s} {b_tier[t]:4d}")
    print("   watchlist ratings:", {r: int(b_counts[r]) for r in RATING_ORDER})
    print(f"   actual BUYs among wonderful businesses: {b_buy}  ({b_buy/len(b):.1%} of universe)")
    print(f"\nCROSS-BOT (shared {sh} names, both valued):")
    print(f"   both cheap {both_cheap} · both rich {both_rich} · disagree {disagree}")
    print(f"   sign agreement {sign_agree:.1%}   Spearman rho {rho:.3f}   Pearson(winsor) {pear:.3f}")
    print(f"   two-method-agreement longs (fat-pitch overlap): {len(fat)}")
    for _, row in fat.head(12).iterrows():
        print(f"      {row.ticker:6s} {str(row['name'])[:26]:26s}  Vine {row.margin_of_safety_d:+.2f}  Buffy {row.margin_of_safety_b:+.2f}  Q{row.quality_score:.0f}")
    print("\nSECTOR TILT (Vinebot median MOS, sectors with n>=4):")
    for s, r in sect.iterrows():
        print(f"   {s:20s} {r['median']:+.3f}  (n={int(r['count'])})")
    print("=" * 66)

    # ===================== FIGURES =====================
    fig1_breadth(d_counts, b_counts, d_frac_cheap, len(d_rating), len(b_pass), len(b), b_buy)
    fig2_crossbot(m, rho, sign_agree, fat)
    fig3_sector_tilt(sect)
    print("wrote figures + experiments/paper9_stats.json")


def fig1_breadth(d_counts, b_counts, d_frac_cheap, n_d, n_bwatch, n_b, b_buy):
    """Both bots lean bearish: rating breadth, Vinebot (all 503) vs Buffybot (watchlist)."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.7))
    for ax, counts, title, sub in [
        (axA, d_counts, f"Vinebot · entire S&P 500 ({n_d})",
         f"only {d_frac_cheap:.0%} trade below DCF value"),
        (axB, b_counts, f"Buffybot · wonderful watchlist ({n_bwatch})",
         f"just {b_buy} buys in the {n_b}-name universe")]:
        vals = [counts[r] for r in RATING_ORDER]
        cols = [RATING_COLOR[r] for r in RATING_ORDER]
        y = np.arange(len(RATING_ORDER))[::-1]
        ax.barh(y, vals, color=cols)
        ax.set_xlim(0, max(vals) * 1.16)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, f"{v}", va="center", fontsize=10, color="#333")
        ax.set_yticks(y); ax.set_yticklabels(RATING_ORDER, fontsize=9.5)
        ax.set_xlabel("number of names"); ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
        ax.text(0.97, 0.80, sub, transform=ax.transAxes, ha="right", fontsize=9.8,
                style="italic", color=NEG)
        ax.grid(axis="y", alpha=0)
    fig.suptitle("Two independent valuation engines, one verdict: mid-2026 is richly priced",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{FIG}/p9_fig1_breadth.png"); plt.close(fig)


def fig2_crossbot(m, rho, sign_agree, fat):
    """Cross-bot scatter: do a DCF and an owner-earnings engine agree on cheapness?"""
    x = np.clip(m.margin_of_safety_d, -1.25, 1.25)
    y = np.clip(m.margin_of_safety_b, -1.25, 1.25)
    cheap_x = m.margin_of_safety_d >= 0; cheap_y = m.margin_of_safety_b >= 0
    col = np.where(cheap_x & cheap_y, POS, np.where(~cheap_x & ~cheap_y, NEG, ACC2))
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.fill([0, 1.3, 1.3, 0], [0, 0, 1.3, 1.3], color="#eef7f0", zorder=0)   # both-cheap
    ax.fill([-1.3, 0, 0, -1.3], [-1.3, -1.3, 0, 0], color="#fbeeec", zorder=0)  # both-rich
    ax.scatter(x, y, s=22, c=col, alpha=0.72, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color="#888", lw=1.0); ax.axvline(0, color="#888", lw=1.0)
    lim = 1.3; ax.plot([-lim, lim], [-lim, lim], color=MUT, ls="--", lw=1.0, zorder=2)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Vinebot margin of safety  (DCF — value/price − 1)")
    ax.set_ylabel("Buffybot margin of safety  (owner-earnings)")
    # quadrant labels
    ax.text(0.62, 0.62, "both say CHEAP", color=POS, fontsize=10, fontweight="bold", ha="center")
    ax.text(-0.62, -0.62, "both say RICH", color=NEG, fontsize=10, fontweight="bold", ha="center")
    ax.text(0.62, -0.62, "disagree", color=ACC2, fontsize=9.5, ha="center")
    ax.text(-0.62, 0.62, "disagree", color=ACC2, fontsize=9.5, ha="center")
    # name the two-method-agreement longs in a clean corner note (not per-point, to avoid overlap)
    names = ", ".join(fat["ticker"].head(8).tolist())
    ax.text(-1.25, 1.18, f"{len(fat)} two-method longs:\n{names}…", fontsize=8.4,
            color=POS, va="top", ha="left")
    ax.set_title(f"Two methods, one signal: Spearman ρ = {rho:.2f}, "
                 f"sign agreement {sign_agree:.0%}\non the {len(m)} names both value",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/p9_fig2_crossbot.png"); plt.close(fig)


def fig3_sector_tilt(sect):
    """Where the cheapness is: Vinebot median margin of safety by sector."""
    labels = [s.replace("_", " ") for s in sect.index]
    vals = sect["median"].to_numpy()
    cols = [POS if v >= 0 else NEG for v in vals]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.barh(y, vals, color=cols, alpha=0.9)
    for yi, v, n in zip(y, vals, sect["count"]):
        lab = "≈0%" if abs(v) < 0.005 else f"{v:+.0%}"
        ax.text(v + (0.012 if v >= 0 else -0.012), yi, f"{lab} (n={int(n)})",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.6, color="#333")
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("median margin of safety (value/price − 1)")
    ax.set_title("The implicit bet: nothing screens cheap in mid-2026 —\n"
                 "yield / balance-sheet sectors least-rich, growth the richest",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0)
    pad = max(0.05, float(np.abs(vals).max()) * 0.25)
    ax.set_xlim(vals.min() - pad, vals.max() + pad)
    fig.tight_layout(); fig.savefig(f"{FIG}/p9_fig3_sector_tilt.png"); plt.close(fig)


if __name__ == "__main__":
    main()
