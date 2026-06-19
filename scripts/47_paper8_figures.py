#!/usr/bin/env python3
"""Publication figures for Paper 8 (autonomous search; the stablecoin-flow lead that
died out-of-window). Reads experiments/paper8_inwindow.json + paper8_extended.json
(copied from the auto-researcher study). Writes docs/figures/p8_*.png + docs/og-onchain.png."""
import os, json
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

IN = json.load(open(f"{ROOT}/experiments/paper8_inwindow.json"))
EX = json.load(open(f"{ROOT}/experiments/paper8_extended.json"))


def fig1_mechanism():
    """Spread-timing t by horizon — in-window vs extended (the kill chart)."""
    hz = ["h1d", "h5d", "h21d"]; labs = ["1-day", "5-day", "21-day"]
    tin = [IN["mechanism"][h]["timing_t"] for h in hz]
    tex = [EX["mechanism"][h]["timing_t"] for h in hz]
    x = np.arange(len(hz)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(x - w / 2, tin, w, color=ACC, label="In-window (2023–26)")
    ax.bar(x + w / 2, tex, w, color=NEG, label="Extended (2020–26)")
    for xi, v in zip(x - w / 2, tin):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=10, color=ACC)
    for xi, v in zip(x + w / 2, tex):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=10, color=NEG)
    ax.axhline(3.0, color="#444", ls="--", lw=1.1); ax.text(2.45, 3.05, "t = 3 (real-claim bar)", fontsize=9, color="#444", ha="right")
    ax.axhline(2.0, color=MUT, ls=":", lw=1.0); ax.text(2.45, 2.05, "t = 2", fontsize=9, color=MUT, ha="right")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("spread-timing t-statistic")
    ax.set_title("Does stablecoin flow predict the trend−carry spread?\nThe 5-day mechanism (t=3.22) collapses to noise (t=0.48) out-of-window",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/p8_fig1_mechanism.png"); plt.close(fig)


def fig2_overlay_vs_ew():
    """Overlay vs equal-weight OOS Sharpe — in-window vs extended."""
    groups = ["In-window\n(2023–26)", "Extended\n(2020–26)"]
    overlay = [IN["best_oos_sharpe"], EX["best_oos_sharpe"]]
    ew = [IN["equal_weight_oos_sharpe"], EX["equal_weight_oos_sharpe"]]
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - w / 2, overlay, w, color=POS, label="Stablecoin overlay")
    ax.bar(x + w / 2, ew, w, color=MUT, label="Equal-weight (50/50)")
    for xi, v in zip(x - w / 2, overlay):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=POS)
    for xi, v in zip(x + w / 2, ew):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color=MUT)
    ax.annotate("≈2× edge", (x[0], 1.05), ha="center", color=POS, fontsize=10, fontweight="bold")
    ax.annotate("edge gone", (x[1], 1.05), ha="center", color=NEG, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("out-of-sample Sharpe (net)")
    ax.set_title("The overlay's edge over equal-weight evaporates with more data",
                 loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/p8_fig2_overlay_vs_ew.png"); plt.close(fig)


def fig3_ic_by_year():
    """Mechanism IC by year over the extended window — flips sign, no durable edge."""
    d = {int(k): v for k, v in EX["mechanism_ic_by_year"].items() if isinstance(v, (int, float))}
    yrs = sorted(d); vals = [d[y] for y in yrs]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bars = ax.bar([str(y) for y in yrs], vals, color=[POS if v > 0 else NEG for v in vals])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.003 if v >= 0 else -0.006),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5)
    ax.axhline(0, color="#444", lw=0.9)
    ax.set_ylabel("stablecoin → spread rank-IC (1-day)")
    ax.set_title("Year by year, the signal has no durable sign — it is noise",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p8_fig3_ic_by_year.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.82, "On-Chain Liquidity & the Crypto Premia", color="white", fontsize=38, fontweight="bold", va="top")
    ax.text(0.06, 0.55, "An autonomous LLM research loop searched on-chain conditioning overlays and\n"
            "surfaced one promising market-neutral lead (OOS 1.18, 5-day mechanism t=3.22).\n"
            "Extending the window to 2020–2026 killed it: a fit, not an edge. Rigor is the alpha.",
            color="#c7d0e0", fontsize=15.5, va="top", linespacing=1.4)
    ax.text(0.06, 0.12, "Alpha Research · Paper 8 · The Autonomous Researcher", color=ACC, fontsize=17, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-onchain.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_mechanism(); fig2_overlay_vs_ew(); fig3_ic_by_year(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p8_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f)) // 1024} KB)")
    print("   docs/og-onchain.png")
