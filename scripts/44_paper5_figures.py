#!/usr/bin/env python3
"""Publication figures for Paper 5 (liquidity provision). Reads experiments/paper5_microstructure.json.
Writes docs/figures/p5_*.png + docs/og-liquidity.png."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
ACC, ACC2, NEG, MUT, POS = "#1f6feb", "#e8a33d", "#b3261e", "#5b6470", "#137a3e"
A = json.load(open(f"{ROOT}/experiments/paper5_microstructure.json"))


def fig1_ofi():
    h2 = A["H2_ofi"]
    labels = ["contemporaneous"] + [k.replace("bar", "") for k in h2["r2_predictive_by_horizon"]]
    bars_lbl = ["now\n(same tick)", "+0.1s", "+1s", "+5s", "+10s", "+30s"]
    vals = [h2["r2_contemporaneous"]] + list(h2["r2_predictive_by_horizon"].values())
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    cols = [ACC] + [MUT] * (len(vals) - 1)
    bars = ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, (f"{v:.2f}" if v >= 0.01 else f"{v:.4f}"), ha="center", fontsize=9.5)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(bars_lbl, fontsize=9)
    ax.set_ylabel("R²  (order-flow imbalance → mid move)")
    ax.set_title("Order-flow imbalance explains the present, not the future",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p5_fig1_ofi.png"); plt.close(fig)


def fig2_overlay():
    h3 = A["H3_execution_overlay"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    vals = [h3["cost_naive_bps"], h3["cost_overlay_bps"]]
    bars = ax.bar(["Immediate\nmarket order", "Micro-price\noverlay"], vals, color=[MUT, POS], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f} bp", ha="center", fontsize=11)
    ax.set_ylabel("effective execution cost (bps)")
    ax.set_title(f"Execution overlay: {h3['slippage_reduction_pct']:.0f}% less slippage (the real use)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p5_fig2_overlay.png"); plt.close(fig)


def fig3_reversal_vix():
    h4 = A["H4_reversal_vs_vix"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    vals = [h4["sharpe_low_vix"], h4["sharpe_high_vix"]]
    bars = ax.bar([f"Low VIX\n(< {h4['vix_median']})", f"High VIX\n(> {h4['vix_median']})"], vals,
                  color=[ACC2, POS], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.axhline(0.3, color=NEG, ls="--", lw=1.2); ax.text(1.45, 0.33, "+0.3 floor", color=NEG, fontsize=9, ha="right")
    ax.set_ylabel("short-term reversal Sharpe")
    ax.set_title("The liquidity-provision premium rises with volatility",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p5_fig3_reversal_vix.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "Liquidity Provision as Alpha", color="white", fontsize=42, fontweight="bold", va="top")
    ax.text(0.06, 0.54, "Standalone HFT/market-making alpha is structurally unavailable (a sub-0.1bp\n"
            "game). But microstructure pays twice: execution-cost reduction (~13% less\n"
            "slippage) and the vol-conditional liquidity-provision premium (reversal).",
            color="#c7d0e0", fontsize=16, va="top", linespacing=1.35)
    ax.text(0.06, 0.12, "Alpha Research · Paper 5", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-liquidity.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_ofi(); fig2_overlay(); fig3_reversal_vix(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p5_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-liquidity.png")
