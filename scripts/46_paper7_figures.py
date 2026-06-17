#!/usr/bin/env python3
"""Publication figures for Paper 7 (synthesis / portfolio construction). Reads
experiments/paper7_portfolio.json. Writes docs/figures/p7_*.png + docs/og-portfolio.png."""
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
A = json.load(open(f"{ROOT}/experiments/paper7_portfolio.json"))
L = A["long_history_book"]
QSPIX, FLOOR = A["benchmarks"]["qspix_live"], A["benchmarks"]["floor"]
NAMES = {"fx_carry": "FX carry", "fx_value": "FX value", "quality": "Equity quality", "reversal": "Reversal/LP"}


def fig1_corr():
    cols = L["sleeves"]; cm = np.array([[A["long_history_book"]["corr"][c][r] for c in cols] for r in cols])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    labs = [NAMES[c] for c in cols]
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(labs, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(cm[i, j]) > 0.6 else "#222", fontsize=10)
    ax.set_title("The sleeves are near-uncorrelated\n(the raw material of diversification)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/p7_fig1_corr.png"); plt.close(fig)


def fig2_diversification():
    cols = L["sleeves"]
    labs = [NAMES[c] for c in cols] + ["COMBINED\nbook"]
    vals = [L["individual_sharpe"][c] for c in cols] + [L["combined_sharpe"]]
    cols_c = [ACC] * len(cols) + [POS]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(range(len(vals)), vals, color=cols_c)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(QSPIX, color=ACC2, ls="--", lw=1.3); ax.text(len(vals) - 0.4, QSPIX + 0.01, f"QSPIX {QSPIX}", color="#9a6a14", fontsize=9, ha="right")
    ax.axhline(FLOOR, color=NEG, ls=":", lw=1.2); ax.text(0.0, FLOOR + 0.01, f"+{FLOOR} floor", color=NEG, fontsize=9)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("Sharpe (long-history, net of cost)")
    ax.set_title("Four ~0.4 edges → a combined 0.53, above the QSPIX ceiling",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p7_fig2_diversification.png"); plt.close(fig)


def fig3_decay():
    bd = {int(k): v for k, v in L["by_decade_sharpe"].items() if isinstance(v, (int, float)) and v == v}
    decs = sorted(bd); vals = [bd[d] for d in decs]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bars = ax.bar([f"{d}s" for d in decs], vals, color=[POS if v > QSPIX else (MUT if v > 0 else NEG) for v in vals])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(QSPIX, color=ACC2, ls="--", lw=1.2); ax.text(len(decs) - 0.4, QSPIX + 0.02, f"QSPIX {QSPIX}", color="#9a6a14", fontsize=9, ha="right")
    ax.axhline(FLOOR, color=NEG, ls=":", lw=1.0)
    ax.set_ylabel("combined book Sharpe")
    ax.set_title("Diversification works — but the premia are decaying (2020s ≈ 0.1)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p7_fig3_decay.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "Do the Modest Edges Add Up?", color="white", fontsize=42, fontweight="bold", va="top")
    ax.text(0.06, 0.54, "Four near-uncorrelated ~0.4-Sharpe premia (FX carry & value, equity quality,\n"
            "liquidity provision) combine to 0.53 net — above the QSPIX 0.41 ceiling, a 1.33×\n"
            "diversification gain. The free lunch is real — but the premia are decaying.",
            color="#c7d0e0", fontsize=16, va="top", linespacing=1.35)
    ax.text(0.06, 0.12, "Alpha Research · Paper 7 · Synthesis", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-portfolio.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_corr(); fig2_diversification(); fig3_decay(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p7_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-portfolio.png")
