#!/usr/bin/env python3
"""Publication figures for Paper 6 (FX carry & commodity roll yield) + a program-wide capstone.
Reads experiments/paper6_macro.json. Writes docs/figures/p6_*.png + docs/og-carry-macro.png."""
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
A = json.load(open(f"{ROOT}/experiments/paper6_macro.json"))


def fig1_fx():
    fx = A["fx"]; periods = list(fx["carry"]["by_period"].keys())
    carry = [fx["carry"]["by_period"][p] for p in periods]
    value = [fx["value"]["by_period"][p] for p in periods]
    x = np.arange(len(periods)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(x - w / 2, carry, w, color=ACC, label=f"FX carry (full {fx['carry']['sharpe']})")
    ax.bar(x + w / 2, value, w, color=POS, label=f"FX value (full {fx['value']['sharpe']})")
    ax.axhline(0.3, color=NEG, ls="--", lw=1.2); ax.text(len(periods) - 0.5, 0.32, "+0.3 floor", color=NEG, fontsize=9, ha="right")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["pre-2008", "2008-15", "post-2015"])
    ax.set_ylabel("Sharpe"); ax.legend(fontsize=9)
    ax.set_title("FX carry decays (and crashes); FX value is steadier",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p6_fig1_fx.png"); plt.close(fig)


def fig2_commodity():
    co = A["commodity"]
    cats = ["Roll-yield\ncarry", "Basis-\nmomentum"]
    pre = [co["roll_yield_carry"]["by_period"]["pre_2010"], co["basis_momentum"]["by_period"]["pre_2010"]]
    post = [co["roll_yield_carry"]["by_period"]["post_2010"], co["basis_momentum"]["by_period"]["post_2010"]]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x - w / 2, pre, w, color=ACC2, label="pre-2010")
    ax.bar(x + w / 2, post, w, color=NEG, label="post-2010 (financialized)")
    ax.axhline(0.3, color=ACC, ls="--", lw=1.0); ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Sharpe"); ax.legend(fontsize=9)
    ax.set_title(f"Energy roll yield: no net edge ({co['n_commodities']} commodities)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p6_fig2_commodity.png"); plt.close(fig)


def fig3_capstone():
    # Net-of-cost Sharpe of every edge tested across the 6-paper program (real numbers from each paper).
    edges = [
        ("Crypto momentum / trend (P4)", 0.53),
        ("FX value (P6)", 0.46),
        ("FX carry, full-sample (P6)", 0.44),
        ("Reversal / liquidity provision, calm (P5)", 0.42),
        ("Crypto funding carry, OOS (P2)", 0.39),
        ("Equity quality / RMW, recent (P4)", 0.39),
        ("Crypto VRP harvest, recent (P1)", 0.27),
        ("Equity momentum (P4)", -0.01),
        ("Commodity energy roll yield (P6)", -0.22),
        ("Daily LLM sentiment, contrarian (P4)", -0.46),
        ("Crypto stat-arb, daily liquid (P3)", -0.58),
    ]
    edges.sort(key=lambda e: e[1])
    labels = [e[0] for e in edges]; vals = [e[1] for e in edges]
    cols = [POS if v >= 0.3 else (MUT if v >= 0 else NEG) for v in vals]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.axvline(0.3, color=ACC, ls="--", lw=1.3); ax.text(0.31, len(labels) - 0.4, "+0.3 floor", color=ACC, fontsize=9)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5); ax.set_xlim(-0.8, 0.75)
    ax.set_xlabel("net-of-cost Sharpe ratio")
    ax.set_title("What actually survived: the real edges are modest risk premia (~0.4)",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p6_fig3_capstone.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "FX Carry & Commodity Roll Yield", color="white", fontsize=37, fontweight="bold", va="top")
    ax.text(0.06, 0.52, "Finance's most-documented anomalies, on newly-sourced data: FX carry is\n"
            "modest and decaying (0.64→0.2, crash-prone); FX value steadier (~0.46);\n"
            "energy roll yield has no net edge. Modest risk premia, not standalone alpha.",
            color="#c7d0e0", fontsize=16, va="top", linespacing=1.35)
    ax.text(0.06, 0.12, "Alpha Research · Paper 6 (finale)", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-carry-macro.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_fx(); fig2_commodity(); fig3_capstone(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p6_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-carry-macro.png")
