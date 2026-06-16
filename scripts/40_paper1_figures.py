#!/usr/bin/env python3
"""
40_paper1_figures.py -- publication figures for Paper 1 (the volatility risk premium).

Reads experiments/paper1_vrp_core.json + paper1_vrp_rigor.json (crypto) and uses the
equity numbers from research/paper1_equity_benchmark.md (stable, cited inline). Recomputes
the implied/realized time series + cumulative harvest PnL via alpha_research.factors.vrp.
Writes PNGs to docs/figures/ (and an OG social card to docs/og-vrp.png).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import vrp  # noqa: E402

FIG = f"{ROOT}/docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
ACC, ACC2, NEG, MUT = "#1f6feb", "#e8a33d", "#b3261e", "#5b6470"

core = json.load(open(f"{ROOT}/experiments/paper1_vrp_core.json"))
rig = json.load(open(f"{ROOT}/experiments/paper1_vrp_rigor.json"))

# Equity numbers (from research/paper1_equity_benchmark.md / paper1_equity_benchmark.json)
EQ = dict(vrp_full=4.09, vrp_pre2010=4.40, vrp_post2010=3.73, t=4.22,
          ex_spx_2020_26=dict(PUT=-4.53, BXM=-6.19, WPUT=-10.12, BXMD=-2.92, PPUT=-1.05))


def fig1_implied_vs_realized():
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    for ax, coin in zip(axs, ["BTC", "ETH"]):
        p = vrp.build(coin, "c2c").dropna(subset=["impl_vol", "fwd_rv_var"])
        t = p["time"]
        ax.plot(t, p["impl_vol"] * 100, color=ACC, lw=1.3, label="Implied (DVOL, 30d)")
        ax.plot(t, np.sqrt(p["fwd_rv_var"]) * 100, color=NEG, lw=1.0, alpha=0.8,
                label="Realized (fwd 30d, close-to-close)")
        ax.fill_between(t, np.sqrt(p["fwd_rv_var"]) * 100, p["impl_vol"] * 100,
                        where=(p["impl_vol"] * 100 >= np.sqrt(p["fwd_rv_var"]) * 100),
                        color=ACC, alpha=0.10)
        ax.set_title(f"{coin}: implied vs realized volatility", loc="left", fontsize=12)
        ax.set_ylabel("annualized vol (%)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    axs[-1].set_xlabel("")
    fig.suptitle("The crypto volatility risk premium: implied (DVOL) runs above realized",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"{FIG}/fig1_implied_vs_realized.png"); plt.close(fig)


def fig2_rv_fragility():
    methods = ["c2c", "parkinson", "gk"]; labels = ["Close-to-close", "Parkinson", "Garman-Klass"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    x = np.arange(len(methods)); w = 0.36
    for i, coin in enumerate(["BTC", "ETH"]):
        vrps = [rig[coin]["rv_estimator_robustness"][m]["mean_vrp_volpts"] for m in methods]
        shs = [rig[coin]["rv_estimator_robustness"][m]["harvest_sharpe"] for m in methods]
        a1.bar(x + (i - 0.5) * w, vrps, w, color=[ACC, ACC2][i], label=coin)
        a2.bar(x + (i - 0.5) * w, shs, w, color=[ACC, ACC2][i], label=coin)
    for ax, ttl, yl in [(a1, "VRP magnitude (vol points)", "mean VRP (vp)"),
                        (a2, "Harvest Sharpe (gated+vol-tgt, net 2vp)", "annualized Sharpe")]:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(ttl, loc="left", fontsize=11); ax.set_ylabel(yl); ax.legend(fontsize=9)
    fig.suptitle("Fragility to the realized-vol estimator: the premium halves (BTC) or vanishes (ETH)",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{FIG}/fig2_rv_estimator_fragility.png"); plt.close(fig)


def fig3_decay_cumpnl():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for coin, col in [("BTC", ACC), ("ETH", ACC2)]:
        p = vrp.build(coin, "c2c")
        ent = vrp.harvest_pnl(p, phase=0, gate=True, voltgt=True, cost_vp=2.0)
        ax.plot(ent["time"], np.cumsum(ent["pnl"].to_numpy()), color=col, lw=1.8, label=coin)
    ax.axvspan(np.datetime64("2024-01-01"), np.datetime64("2026-06-16"), color="#b3261e", alpha=0.06)
    ax.text(np.datetime64("2024-09-01"), ax.get_ylim()[0] * 0.0 + 0.05, "live regime\n(Sharpe → ~0)",
            color=NEG, fontsize=9, va="bottom")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Cumulative harvest PnL: the edge is concentrated in 2021–2023",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("cumulative PnL (variance units)"); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_decay_cumulative_pnl.png"); plt.close(fig)


def fig4_crypto_vs_equity():
    # early vs late VRP magnitude (vol points). Crypto early/late from the equity-benchmark
    # comparison (BTC 13.2/3.4, ETH 11.4/0.2); equity pre/post-2010 (4.40/3.73).
    cats = ["BTC", "ETH", "Equity (SPX)"]
    early = [13.2, 11.4, EQ["vrp_pre2010"]]; late = [3.4, 0.2, EQ["vrp_post2010"]]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, early, w, color=ACC, label="early (crypto 2021–22 / equity pre-2010)")
    ax.bar(x + w / 2, late, w, color=NEG, label="late (crypto 2025–26 / equity post-2010)")
    for i, (e, l) in enumerate(zip(early, late)):
        ax.text(i - w / 2, e + 0.2, f"{e:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, l + 0.2, f"{l:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("mean VRP (vol points)")
    ax.set_title("Crypto VRP collapses toward the small, sticky level equities sustained for 35 years",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_crypto_vs_equity.png"); plt.close(fig)


def fig5_equity_dead_net():
    idx = ["PUT", "BXM", "WPUT", "BXMD", "PPUT"]; vals = [EQ["ex_spx_2020_26"][k] for k in idx]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(idx, vals, color=[NEG if v < -2 else MUT for v in vals])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color="#222")
    ax.axhline(0, color="#444", lw=1.0); ax.set_ylim(-11, 0.9)
    ax.set_ylabel("excess ann. return vs S&P 500 (%)")
    ax.set_title("Equity option-writing: no excess over the S&P 500 (2020–2026)",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_equity_dead_net.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.78, "The Volatility Risk Premium,\nCross-Asset", color="white",
            fontsize=40, fontweight="bold", va="top", linespacing=1.1)
    ax.text(0.06, 0.40, "Is it harvestable net of costs?  A real, robust premium —\n"
            "but the crypto edge is collapsing and equity writing beats nothing since 2010.",
            color="#c7d0e0", fontsize=18, va="top", linespacing=1.3)
    ax.text(0.06, 0.12, "Alpha Research · Paper 1", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-vrp.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_implied_vs_realized(); fig2_rv_fragility(); fig3_decay_cumpnl()
    fig4_crypto_vs_equity(); fig5_equity_dead_net(); og_card()
    print("Wrote figures to docs/figures/ :")
    for f in sorted(os.listdir(FIG)):
        print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("  docs/og-vrp.png")
