#!/usr/bin/env python3
"""Publication figures for Paper 2 (crypto funding-rate carry). Reads the carry panels
+ experiments/paper2_carry_{core,rigor}.json. Writes docs/figures/p2_*.png + docs/og-carry.png."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import carry as C  # noqa: E402

FIG = f"{ROOT}/docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
ACC, ACC2, NEG, MUT, POS = "#1f6feb", "#e8a33d", "#b3261e", "#5b6470", "#137a3e"
core = json.load(open(f"{ROOT}/experiments/paper2_carry_core.json"))
rig = json.load(open(f"{ROOT}/experiments/paper2_carry_rigor.json"))
F = C.load_daily_funding(); R = C.load_spot_returns()


def fig1_funding_decay():
    yrs = [2023, 2024, 2025, 2026]
    med = []
    for y in yrs:
        sub = F[F.index.year == y]
        med.append(float(np.nanmedian(sub.mean(axis=0) * 365 * 100)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar([str(y) for y in yrs], med, color=[ACC, ACC, ACC2, NEG])
    for b, v in zip(bars, med):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("median annualized funding (%)")
    ax.set_title("The funding LEVEL decayed: the 2024 carry boom is gone", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_fig1_funding_decay.png"); plt.close(fig)


def fig2_factor_cumpnl():
    f = C.cross_sectional_factor(F, R, 7, 0.33, 6.0, 5).dropna()
    cum = np.cumsum(f.values) * 100
    dd = cum - np.maximum.accumulate(cum)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(f.index, cum, color=ACC, lw=1.8)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvspan(np.datetime64("2025-10-01"), np.datetime64("2025-11-15"), color="#888", alpha=0.10)
    ax.text(np.datetime64("2025-10-05"), cum.min() * 0.5, "Oct-2025\ncascade\n(+1.7%)", fontsize=8.5, color=MUT)
    ax.set_ylabel("cumulative net PnL (%)")
    ax.set_title("Cross-sectional funding carry: cumulative net PnL (net of 6 bp/side)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_fig2_factor_cumpnl.png"); plt.close(fig)


def fig3_rigor_scorecard():
    labels = ["In-sample", "Best config\n(of 36)", "Walk-forward\nOOS"]
    vals = [rig["default_ann_sharpe"], rig["best_ann_sharpe"], rig["walkforward_oos_sharpe"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bars = ax.bar(labels, vals, color=[ACC, MUT, POS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=11)
    ax.axhline(0.3, color=NEG, ls="--", lw=1.3)
    ax.text(2.4, 0.32, "+0.3 mean-reversion floor", color=NEG, fontsize=9, ha="right")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Modest but real: OOS clears the floor; can't pick the 'best' config",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.text(0.0, -0.30, f"Deflated Sharpe {rig['deflated_sharpe_default']}  ·  PBO {rig['pbo']} "
            f"(36/36 configs positive, 0.58–1.5 → indistinguishable)",
            fontsize=9, color=MUT)
    ax.set_ylim(-0.4, max(vals) * 1.15)
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_fig3_rigor.png"); plt.close(fig)


def fig4_cashcarry_decay():
    cc = C.cash_and_carry(F, 7, 0.0, 18.0)
    yrs = sorted(set(cc.index.year))
    ret = [float(cc[cc.index.year == y].mean() * 365 * 100) for y in yrs]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar([str(y) for y in yrs], ret, color=[POS if v > 0 else NEG for v in ret])
    for b, v in zip(bars, ret):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.2 if v >= 0 else -0.5), f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_ylabel("net annualized return (%)")
    ax.set_title("Cash-and-carry net return: decays below cash, turns negative",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p2_fig4_cashcarry.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "Crypto Carry:\nThe Funding-Rate Cross-Section", color="white",
            fontsize=38, fontweight="bold", va="top", linespacing=1.1)
    ax.text(0.06, 0.40, "The funding LEVEL is a decayed, artifact-Sharpe trade — but the\n"
            "cross-sectional carry FACTOR is the program's first real net-of-cost,\n"
            "out-of-sample-positive edge (OOS Sharpe ~0.4).", color="#c7d0e0", fontsize=17,
            va="top", linespacing=1.3)
    ax.text(0.06, 0.10, "Alpha Research · Paper 2", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-carry.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_funding_decay(); fig2_factor_cumpnl(); fig3_rigor_scorecard(); fig4_cashcarry_decay(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p2_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-carry.png")
