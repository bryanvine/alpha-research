#!/usr/bin/env python3
"""Publication figures for Paper 3 (crypto statistical arbitrage). Reads
experiments/paper3_statarb_{core,rigor}.json + recomputes the daily PnL. Writes docs/figures/p3_*.png + docs/og-statarb.png."""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from alpha_research.factors import statarb as SA  # noqa: E402

FIG = f"{ROOT}/docs/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 140, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
ACC, ACC2, NEG, MUT, POS = "#1f6feb", "#e8a33d", "#b3261e", "#5b6470", "#137a3e"
rig = json.load(open(f"{ROOT}/experiments/paper3_statarb_rigor.json"))


def fig1_daily_dead():
    Rd = SA.load_daily_panel()
    net = SA.pca_sscore_pnl(Rd, 60, 3, 1.25, 5.0, 1)
    gross = SA.pca_sscore_pnl(Rd, 60, 3, 1.25, 0.0, 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(net.index, np.cumsum(net.values) * 100, color=NEG, lw=1.8, label="net (5 bp/side)")
    ax.plot(gross.index, np.cumsum(gross.values) * 100, color=MUT, lw=1.2, ls="--", label="gross (no cost)")
    ax.axhline(0, color="#888", lw=0.8); ax.legend(fontsize=9)
    ax.set_ylabel("cumulative PnL (%)")
    ax.set_title("Daily stat-arb on liquid coins is dead: no edge even gross, negative net",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p3_fig1_daily_dead.png"); plt.close(fig)


def fig2_tercile_cost():
    sw = rig["tercile_cost_sweep_net_sharpe"]
    costs = [5, 20, 50, 80]; xs = np.arange(len(costs)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (name, col, lab) in enumerate([("high_liq", ACC, "high liquidity"),
                                          ("mid_liq", ACC2, "mid liquidity"),
                                          ("low_liq", NEG, "low liquidity (illiquid alts)")]):
        vals = [sw[name][f"{c}bp"] for c in costs]
        ax.bar(xs + (i - 1) * w, vals, w, color=col, label=lab)
    ax.axhline(0, color="#444", lw=1.0)
    ax.axvspan(1.5, 3.5, color="#b3261e", alpha=0.06)
    ax.set_xticks(xs); ax.set_xticklabels([f"{c} bp/side" for c in costs])
    ax.set_ylabel("net Sharpe"); ax.legend(fontsize=9, loc="upper right")
    ax.set_title("The stat-arb 'edge' is a stale-price artifact: it dies at realistic costs",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p3_fig2_tercile_cost.png"); plt.close(fig)


def fig3_coint_collapse():
    cp = rig["coint_pairs_daily"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(["OOS gross", "OOS net (5 bp)"], [cp["oos_net_sharpe_0cost"], cp["oos_net_sharpe"]],
                  color=[MUT, NEG], width=0.5)
    for b, v in zip(bars, [cp["oos_net_sharpe_0cost"], cp["oos_net_sharpe"]]):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.05, f"{v:.2f}", ha="center", va="top", fontsize=11)
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_ylabel("annualized Sharpe (out-of-sample)")
    ax.set_title(f"Cointegration pairs: {cp['is_selected_pairs_total']} in-sample winners lose out-of-sample",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.text(0.0, 0.08, f"{cp['is_selected_pairs_total']} pairs passed the in-sample test (p<0.05)",
            fontsize=9, color=MUT)
    ax.set_ylim(min(cp["oos_net_sharpe"], cp["oos_net_sharpe_0cost"]) - 0.25, 0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/p3_fig3_coint_collapse.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "Statistical Arbitrage in Crypto", color="white",
            fontsize=40, fontweight="bold", va="top")
    ax.text(0.06, 0.52, "There is no net-of-cost stat-arb edge. The apparent hourly signal is a\n"
            "stale-price artifact in illiquid coins (dies at realistic spreads); cointegration\n"
            "pairs are a multiple-testing mirage that loses out-of-sample.", color="#c7d0e0",
            fontsize=17, va="top", linespacing=1.35)
    ax.text(0.06, 0.12, "Alpha Research · Paper 3", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-statarb.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_daily_dead(); fig2_tercile_cost(); fig3_coint_collapse(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p3_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-statarb.png")
