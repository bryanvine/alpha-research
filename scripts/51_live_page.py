#!/usr/bin/env python3
"""
51_live_page.py -- render the public live-NAV page (alpha.vineai.tech) from the paper-traded book.

Reads data/live_book/state.json + nav.csv (written by 50_live_book.py), renders a NAV equity-curve
+ drawdown chart and a self-contained index.html, into the live-site repo dir (default /apps/alpha-live).
The daily cron runs 50_live_book.py then this, then commits/pushes the live-site repo so the page stays current.
"""
import os, sys, json, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/apps/alpha-live"
ACC, MUT, POS, NEG = "#1f6feb", "#5b6470", "#137a3e", "#b3261e"
NICE = {"crypto_carry": "Crypto funding carry", "crypto_trend": "Crypto trend",
        "fx_carry": "FX carry", "fx_value": "FX value"}


def chart(nav_csv, path):
    df = pd.read_csv(nav_csv, parse_dates=[0], index_col=0)
    nav = df["nav"]; dd = nav / nav.cummax() - 1
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 4.8), height_ratios=[3, 1], sharex=True)
    a.plot(nav.index, nav.values, color=ACC, lw=1.8); a.axhline(1.0, color="#bbb", lw=0.8)
    a.fill_between(nav.index, 1.0, nav.values, where=nav.values >= 1.0, color=ACC, alpha=0.08)
    a.set_ylabel("NAV (start = 1.00)"); a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)
    b.fill_between(dd.index, dd.values * 100, 0, color=NEG, alpha=0.5)
    b.set_ylabel("drawdown %"); b.grid(alpha=0.25); b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    st = json.load(open(f"{ROOT}/data/live_book/state.json"))
    os.makedirs(OUTDIR, exist_ok=True)
    chart(f"{ROOT}/data/live_book/nav.csv", f"{OUTDIR}/nav.png")
    si = st["since_inception"]; w = st["current_target_weights"]; ss = st["sleeve_sharpe"]
    sign = lambda v: "pos" if v >= 0 else "neg"
    cards = [("NAV", f"{st['latest_nav']:.3f}", "pos"),
             ("Total return", f"{si['total_return_pct']:+.1f}%", sign(si['total_return_pct'])),
             ("Annualized", f"{si['ann_return_pct']:+.1f}%", sign(si['ann_return_pct'])),
             ("Sharpe", f"{si['sharpe']:.2f}", sign(si['sharpe'])),
             ("Volatility", f"{si['ann_vol_pct']:.1f}%", "mut"),
             ("Max drawdown", f"{si['maxdd_pct']:.1f}%", "neg")]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v {c}">{v}</div></div>' for k, v, c in cards)
    wrows = "".join(
        f'<tr><td>{NICE.get(k, k)}</td><td>{v*100:.1f}%</td>'
        f'<td class="{sign(ss.get(k,0))}">{ss.get(k, float("nan")):+.2f}</td></tr>'
        for k, v in sorted(w.items(), key=lambda x: -x[1]))
    upd = st["updated"].replace("T", " ").replace("Z", " UTC")

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpha Research — Live Book</title>
<meta name="description" content="Live paper-traded NAV of the Alpha Research diversified risk-premium book: crypto carry/trend + FX carry/value, risk-parity, vol-targeted, net of costs. Updated daily.">
<meta property="og:title" content="Alpha Research — Live Book">
<meta property="og:description" content="Paper-traded NAV of the diversified risk-premium book, updated daily. Sharpe {si['sharpe']}, NAV {st['latest_nav']:.3f} since {st['inception']}.">
<meta property="og:type" content="website"><meta property="og:url" content="https://alpha.vineai.tech/">
<meta property="og:image" content="https://alpha.vineai.tech/nav.png">
<meta property="og:image:width" content="1204"><meta property="og:image:height" content="672">
<meta name="twitter:card" content="summary_large_image">
<style>
:root{{--ink:#1a1a1a;--muted:#5b6470;--rule:#e3e6ea;--accent:#1f6feb;--bg:#fdfdfc;--card:#fff;}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
.wrap{{max-width:840px;margin:0 auto;padding:44px 22px 90px}}
header{{border-bottom:3px solid var(--ink);padding-bottom:20px}}
h1{{font-size:29px;margin:0 0 8px;letter-spacing:-.01em}}
.tag{{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
color:#fff;background:var(--accent);padding:2px 9px;border-radius:99px;margin-bottom:12px}}
.sub{{color:var(--muted);font-size:14.5px;margin:4px 0 0}}.sub a{{color:var(--accent);text-decoration:none}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:26px 0}}
.card{{flex:1 1 120px;background:var(--card);border:1px solid var(--rule);border-radius:11px;padding:14px 16px}}
.card .k{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.card .v{{font-size:25px;font-weight:680;margin-top:4px;font-variant-numeric:tabular-nums}}
.pos{{color:#137a3e}}.neg{{color:#b3261e}}.mut{{color:#33373d}}
figure{{margin:24px 0;text-align:center}}figure img{{max-width:100%;border:1px solid var(--rule);border-radius:10px}}
h2{{font-size:19px;margin:40px 0 10px}}
table{{border-collapse:collapse;width:100%;font-size:15px;margin:12px 0}}
th,td{{border-bottom:1px solid var(--rule);padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
thead th{{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted)}}
td{{font-variant-numeric:tabular-nums}}
.note{{background:#fff8e6;border:1px solid #f2e2af;border-radius:9px;padding:14px 18px;font-size:14.5px;margin:22px 0}}
p,li{{color:#23262b}}a{{color:var(--accent)}}
footer{{margin-top:46px;padding-top:16px;border-top:1px solid var(--rule);color:var(--muted);font-size:13.5px}}
</style></head><body><div class="wrap">
<header>
  <div class="tag">Alpha Research · Live Book</div>
  <h1>The diversified risk-premium book, paper-traded live</h1>
  <p class="sub">Inception {st['inception']} · data through {st['last_data_date']} · updated {upd}<br>
  The capstone of the <a href="https://bryanvine.github.io/alpha-research/">Alpha Research</a> series
  (<a href="https://bryanvine.github.io/alpha-research/paper7.html">Paper 7</a>), tracked forward as a paper portfolio.</p>
</header>

<div class="cards">{card_html}</div>

<figure><img src="nav.png" alt="Live NAV equity curve and drawdown"></figure>

<h2>Current target weights &amp; sleeve performance</h2>
<table><thead><tr><th>Sleeve</th><th>Weight</th><th>Sharpe (since inception)</th></tr></thead>
<tbody>{wrows}</tbody></table>

<div class="note"><strong>What this is.</strong> A risk-parity, 10%/yr-vol-targeted combination of the
daily-tradable survivors from the research program — crypto funding carry, crypto trend, and G10 FX
carry &amp; value — each net of its own trading cost, recomputed daily and marked to current data. It is the
live, out-of-sample test of the program's thesis: that a handful of modest, near-uncorrelated risk premia
diversify into a respectable book. The premia are decaying, so this is the honest forward experiment, not a
backtest.</div>

<h2>Caveats</h2>
<ul>
<li><strong>Paper-traded, not live capital</strong> — no broker, no real fills; capacity and borrow are
largely internalized in the sleeve returns but not separately stress-modeled.</li>
<li><strong>Daily-tradable subset</strong> — equity quality &amp; reversal are part of the full Paper-7 book
but update monthly (Ken French) and need a broker/ETFs to trade daily, so they are tracked separately, not here.</li>
<li>Realized volatility runs below the 10% target due to a leverage cap. <strong>Not investment advice.</strong></li>
</ul>

<h2>The nine papers</h2>
<p>1 <a href="https://bryanvine.github.io/alpha-research/paper1.html">Volatility Risk Premium</a> ·
2 <a href="https://bryanvine.github.io/alpha-research/paper2.html">Crypto Carry</a> ·
3 <a href="https://bryanvine.github.io/alpha-research/paper3.html">Crypto Stat-Arb</a> ·
4 <a href="https://bryanvine.github.io/alpha-research/paper4.html">The Cost of Direction</a> ·
5 <a href="https://bryanvine.github.io/alpha-research/paper5.html">Liquidity Provision</a> ·
6 <a href="https://bryanvine.github.io/alpha-research/paper6.html">FX &amp; Commodity Carry</a> ·
7 <a href="https://bryanvine.github.io/alpha-research/paper7.html">Synthesis</a> ·
8 <a href="https://bryanvine.github.io/alpha-research/paper8.html">Autonomous Researcher</a> ·
9 <a href="https://bryanvine.github.io/alpha-research/paper9.html">Machine Analyst</a></p>

<footer>Paper-traded; updated daily from free data. Code &amp; full research log:
<a href="https://github.com/bryanvine/alpha-research">github.com/bryanvine/alpha-research</a>. © {datetime.datetime.now(datetime.timezone.utc).year} Bryan Vine.</footer>
</div></body></html>"""
    open(f"{OUTDIR}/index.html", "w").write(html)
    open(f"{OUTDIR}/CNAME", "w").write("alpha.vineai.tech\n")
    open(f"{OUTDIR}/.nojekyll", "w").write("")
    print(f"wrote {OUTDIR}/index.html ({len(html)} bytes) + nav.png + CNAME + .nojekyll")
    print(f"  NAV {st['latest_nav']} · Sharpe {si['sharpe']} · updated {upd}")


if __name__ == "__main__":
    main()
