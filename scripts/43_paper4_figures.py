#!/usr/bin/env python3
"""Publication figures for Paper 4 (the directional anomaly zoo). Reads
experiments/paper4_zoo_audit.json + paper4_ourdata.json. Writes docs/figures/p4_*.png + docs/og-zoo.png."""
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
A = json.load(open(f"{ROOT}/experiments/paper4_zoo_audit.json"))


def fig1_decay():
    d = A["decay"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    vals = [d["mean_is_monthly_pct"], d["mean_oos_monthly_pct"]]
    bars = ax.bar(["In-sample\n(original study)", "Out-of-sample\n(post-publication)"], vals,
                  color=[ACC, NEG], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}%", ha="center", fontsize=11)
    ax.set_ylabel("mean long-short return (%/month)")
    ax.set_title(f"Published anomalies decay {d['mclean_pontiff_decay_pct']:.0f}% out-of-sample",
                 loc="left", fontsize=12, fontweight="bold")
    ax.text(0.5, max(vals) * 0.55, f"McLean–Pontiff\n(212 OSAP predictors)", ha="center", color=MUT, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_fig1_decay.png"); plt.close(fig)


def fig2_survival():
    d, t, rc = A["decay"], A["t_hurdle"], A["recent_decade"]
    items = [("Sign persists out-of-sample", d["frac_oos_positive"] * 100, MUT),
             ("Still positive in last decade", rc["frac_positive"] * 100, MUT),
             ("Full-sample |t| > 3", t["frac_full_t_gt3"] * 100, ACC2),
             ("Survive 30 bp/mo cost (OOS+)", A["oos_net_of_cost_frac_positive"]["30bp_mo"] * 100, ACC2),
             ("Beat +0.3 Sharpe (last decade)", rc["frac_sharpe_gt_floor"] * 100, ACC2),
             ("Out-of-sample |t| > 3", d["frac_oos_t_gt3"] * 100, NEG),
             ("|t| > 3 in last decade", rc["frac_t_gt3"] * 100, NEG)]
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]; vals = [i[1] for i in items]; cols = [i[2] for i in items]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("% of 212 published anomalies"); ax.set_xlim(0, 100)
    ax.set_title("The zoo collapses under stricter, recent, and cost-aware tests",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_fig2_survival.png"); plt.close(fig)


def fig3_cost():
    c = A["oos_net_of_cost_frac_positive"]
    costs = [0, 30, 60]; vals = [c["0bp_mo"] * 100, c["30bp_mo"] * 100, c["60bp_mo"] * 100]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(costs, vals, "-o", color=NEG, lw=2, ms=8)
    for x, v in zip(costs, vals):
        ax.text(x, v + 2.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_xlabel("transaction cost (bp/month of turnover)")
    ax.set_ylabel("% of anomalies positive (OOS)"); ax.set_ylim(0, 100)
    ax.set_title("High turnover: realistic costs erase most survivors",
                 loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_fig3_cost.png"); plt.close(fig)


def fig4_french():
    ff = A["french_recent"]
    order = ["Mkt-RF", "RMW", "Mom", "ST_Rev", "HML", "CMA", "SMB", "LT_Rev"]
    order = [c for c in order if c in ff]
    vals = [ff[c]["sharpe_2015_25"] for c in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bars = ax.bar(order, vals, color=[POS if v > 0.3 else (MUT if v > 0 else NEG) for v in vals])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.03), f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5)
    ax.axhline(0.3, color=ACC, ls="--", lw=1.2); ax.text(len(order) - 0.5, 0.33, "+0.3 floor", color=ACC, fontsize=9, ha="right")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Sharpe, 2015–2025")
    ax.set_title("Canonical factors, last decade: only market & quality (RMW) survive",
                 loc="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{FIG}/p4_fig4_french.png"); plt.close(fig)


def og_card():
    fig = plt.figure(figsize=(12, 6.3)); fig.patch.set_facecolor("#0f1320")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.06, 0.80, "The Cost of Direction", color="white", fontsize=44, fontweight="bold", va="top")
    ax.text(0.06, 0.56, "An honest net-of-cost audit of the directional anomaly zoo. 212 published\n"
            "predictors decay 53% out-of-sample; in the last decade the median has a\n"
            "Sharpe of 0.22 and only 3% are statistically strong. Only low-turnover quality survives.",
            color="#c7d0e0", fontsize=16, va="top", linespacing=1.35)
    ax.text(0.06, 0.12, "Alpha Research · Paper 4", color=ACC, fontsize=18, fontweight="bold")
    fig.savefig(f"{ROOT}/docs/og-zoo.png", facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    fig1_decay(); fig2_survival(); fig3_cost(); fig4_french(); og_card()
    print("wrote:")
    for f in sorted(os.listdir(FIG)):
        if f.startswith("p4_"):
            print("  ", f, f"({os.path.getsize(os.path.join(FIG, f))//1024} KB)")
    print("   docs/og-zoo.png")
