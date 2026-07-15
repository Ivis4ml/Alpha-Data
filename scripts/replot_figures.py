"""出版级统一重绘：从已保存的分析产物重绘论文全部图表（pub_style）。

集中所有最终图的绘制代码，避免为改样式重跑分析管线。数据来源：
``data/cn_futures/analysis/deep/*.parquet``、期货库、登记表；
f5 活动时钟直接用 DuckDB 对逐笔做小时聚合（单查询）。

改动要点（相对首版图）：ICML/Nature 风格 rcParams（pub_style.setup）、
多面板 (a)(b)(c) 标号、f1 由双轴改为上下两栏共享时间轴、统一尺寸
（单栏 3.5in / 双栏 7in）、300 dpi。

用法::

    .venv/bin/python scripts/replot_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402
from analyze_cn_futures_polymarket import SIGNAL_END  # noqa: E402
from deep_analysis_cn_polymarket import (  # noqa: E402
    DEEP_DIR,
    FIG_DIR,
    HEADLINE_PAIRS,
    apply_admit_mask,
    futures_fine_returns,
)
from intl_benchmark_analysis import (  # noqa: E402
    load_benchmark_beijing,
    window_benchmark_returns,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store
from alpha_data.polymarket import store as poly_store  # noqa: E402

P = pub_style.PALETTE
DIVERGING = LinearSegmentedColormap.from_list("dv", [P["blue"], "#f0efe9", P["red"]])
FIG_TOP = ROOT / "docs" / "figures"
FINE_WINDOWS = ["gap_pm", "night", "gap_am", "day"]


def heat(ax, mat, vmin, vmax):
    im = ax.imshow(mat, cmap=DIVERGING, vmin=vmin, vmax=vmax, aspect="auto")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_f1():
    ts = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    ms = ts[(ts.theme == "mideast_conflict") & (ts["product"] == "SC")]
    ms = ms[ms.trade_date <= SIGNAL_END].sort_values("trade_date")
    cum = ms[["s_night", "s_gap", "s_day"]].sum(axis=1, skipna=True).cumsum()
    sc = fut_store.read_daily("SC")
    sc = sc[sc.trade_date.isin(ms.trade_date)].sort_values("trade_date")
    x = pd.to_datetime(ms.trade_date)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True,
                             height_ratios=[1, 1])
    axes[0].plot(x, cum.to_numpy(), color=P["orange"], lw=1.2)
    axes[0].set_ylabel("累计方向加权\nlogit innovation")
    pub_style.panel(axes[0], "a")
    axes[1].plot(pd.to_datetime(sc.trade_date), sc.day_close, color=P["blue"], lw=1.2)
    axes[1].set_ylabel("SC 收盘价\n（元/桶）")
    pub_style.panel(axes[1], "b")
    for ax in axes:
        pub_style.soft_grid(ax)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(FIG_TOP / "f1_mideast_vs_sc.png")
    plt.close(fig)


def fig_f5():
    reg = pd.read_parquet(poly_store.FEATURES_DIR / "cn_registry.parquet")
    ids = ",".join(f"'{c}'" for c in reg.condition_id.unique())
    con = poly_store.connect()
    glob = poly_store.daily_aligned_glob().replace("'", "''")
    prof = con.execute(f"""
        SELECT hour(timezone('Asia/Shanghai', to_timestamp(block_timestamp))) AS h,
               sum(usdc_amount) / 1e6 AS usdc_m
        FROM read_parquet('{glob}')
        WHERE condition_id IN ({ids}) GROUP BY h ORDER BY h
    """).fetch_df()
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.bar(prof.h, prof.usdc_m, color=P["blue"], width=0.8)
    ax.axvspan(8.5, 15, alpha=0.10, color=P["aqua"])
    ax.axvspan(20.5, 23.5, alpha=0.10, color=P["orange"])
    ax.axvspan(-0.5, 2.5, alpha=0.10, color=P["orange"])
    ax.text(11.5, ax.get_ylim()[1] * 0.92, "国内日盘", fontsize=7,
            ha="center", color=P["ink2"])
    ax.text(22, ax.get_ylim()[1] * 0.92, "国内夜盘", fontsize=7,
            ha="center", color=P["ink2"])
    ax.set_xlabel("北京时间（小时）")
    ax.set_ylabel("登记市场成交额\n（百万 USDC）")
    ax.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    fig.savefig(FIG_TOP / "f5_activity_clock.png")
    plt.close(fig)


def fig_a():
    tab = pd.read_parquet(DEEP_DIR / "fine_absorption.parquet")
    sub = tab[(tab.subset == "all")
              & tab.set_index(["theme", "product"]).index.isin(HEADLINE_PAIRS)]
    piv = sub.pivot_table(index=["theme", "product"], columns="window",
                          values="pearson").reindex(columns=FINE_WINDOWS)
    npiv = sub.pivot_table(index=["theme", "product"], columns="window",
                           values="n").reindex(columns=FINE_WINDOWS)
    piv.index = [f"{t} × {p}" for t, p in piv.index]
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(piv) + 1.3))
    im = heat(ax, piv.to_numpy(dtype=float), -0.8, 0.8)
    ax.set_xticks(range(4), ["傍晚闭市\n15-21", "夜盘\n21-夜收",
                             "凌晨闭市\n夜收-09", "日盘\n09-15"])
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, n = piv.iloc[i, j], npiv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}\n(n={int(n)})", ha="center",
                        va="center", fontsize=6.5)
    fig.colorbar(im, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_fine_absorption.png")
    plt.close(fig)

    key = tab[tab.set_index(["theme", "product"]).index.isin(
        [("oil_price", "SC"), ("mideast_conflict", "SC"), ("metal_price", "AU")])]
    key = key[key.subset.isin(["weekday", "monday"]) & (key.window != "day")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for k, (ax, (pair, g)) in enumerate(zip(
            axes, key.groupby(key.theme + " × " + key["product"]), strict=False)):
        piv2 = (g.pivot_table(index="window", columns="subset", values="pearson")
                .reindex(index=FINE_WINDOWS[:3], columns=["weekday", "monday"]))
        x = np.arange(len(piv2))
        ax.bar(x - 0.18, piv2["weekday"].fillna(0), width=0.34, color=P["blue"],
               label="平日")
        ax.bar(x + 0.18, piv2["monday"].fillna(0), width=0.34, color=P["orange"],
               label="周一(含周末)")
        ax.set_xticks(x, ["傍晚", "夜盘", "凌晨"])
        ax.set_title(pair, fontsize=7.5)
        ax.axhline(0, color=P["ink2"], lw=0.6)
        pub_style.panel(ax, "abc"[k])
    axes[0].set_ylabel("Pearson r")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_weekday_vs_monday.png")
    plt.close(fig)


def fig_b():
    from analyze_cn_futures_polymarket import nw_regression
    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    rets = futures_fine_returns(sorted(theme_fine["product"].unique()))
    merged = theme_fine.merge(rets, on=["product", "trade_date"])
    merged = merged[(merged.trade_date <= SIGNAL_END)
                    & ~merged.roll.fillna(False)]
    panels = [
        ("oil_price", "SC", "s_gap", "gap_total", "油价 gap × SC 闭市"),
        ("oil_price", "SC", "s_night", "r_night", "油价夜盘 × SC 夜盘"),
        ("mideast_conflict", "SC", "s_night", "r_night", "中东夜盘 × SC 夜盘"),
        ("mideast_conflict", "SC", "s_gap", "gap_total", "中东 gap × SC 闭市"),
        ("metal_price", "AU", "s_day", "r_day", "金价日盘 × AU 日盘"),
        ("us_china_trade", "M", "s_gap", "gap_total", "中美贸易 gap × M 闭市"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for k, (ax, (theme, product, sc_col, rc, title)) in enumerate(
            zip(axes.flat, panels, strict=True)):
        g = merged[(merged.theme == theme) & (merged["product"] == product)]
        if rc == "gap_total":
            r = (g.r_gap_pm.fillna(0) + g.r_night.fillna(0) + g.r_gap_am.fillna(0))
            r[g[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
        else:
            r = g[rc]
        s = g[sc_col]
        ok = ~(s.isna() | r.isna())
        s, r = s[ok].to_numpy(), r[ok].to_numpy() * 1e4
        ax.scatter(s, r, s=9, alpha=0.65, color=P["blue"], edgecolor="white",
                   linewidth=0.3)
        if len(s) > 10:
            beta, t_hac, _, _ = nw_regression(r / 1e4, s, 1)
            xs = np.linspace(s.min(), s.max(), 40)
            a0 = r.mean() / 1e4 - beta * s.mean()
            ax.plot(xs, (a0 + beta * xs) * 1e4, color=P["red"], lw=1.1)
            ax.set_title(f"{title}\nr={np.corrcoef(s, r)[0, 1]:+.2f}, "
                         f"t={t_hac:+.1f}, n={len(s)}", fontsize=7)
        ax.set_xlabel("信号（Δlogit）", fontsize=7)
        if k % 3 == 0:
            ax.set_ylabel("收益（bp）", fontsize=7)
        pub_style.panel(ax, "abcdef"[k])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b_scatter_grid.png")
    plt.close(fig)


def fig_c():
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    registry = pd.read_parquet(poly_store.FEATURES_DIR / "cn_registry.parquet")
    market_sig = pd.read_parquet(DEEP_DIR / "market_signals_fine.parquet")
    sig_cols = ["s_night", "s_gap", "s_day"]
    df = apply_admit_mask(market_sig.copy(), registry, sig_cols)
    arr = df[sig_cols].to_numpy(dtype="float64")
    df["s_total"] = np.where(np.isnan(arr).all(axis=1), np.nan,
                             np.nansum(arr, axis=1)) * df["orientation"]
    df = df[df.trade_date <= SIGNAL_END]
    top = registry.drop_duplicates("condition_id").nlargest(120, "usdc_win")[
        "condition_id"]
    wide = (df[df.condition_id.isin(top)]
            .drop_duplicates(["condition_id", "trade_date"])
            .pivot(index="trade_date", columns="condition_id", values="s_total"))
    wide = wide.loc[:, wide.notna().sum() >= 25]
    corr = wide.corr(min_periods=15).fillna(0)
    mat = corr.to_numpy(copy=True)
    np.fill_diagonal(mat, 1.0)
    corr = pd.DataFrame(mat, index=corr.index, columns=corr.columns)
    link = hierarchy.linkage(
        squareform((1 - mat).clip(0, 2), checks=False), method="average")
    order = hierarchy.leaves_list(link)
    ordered = corr.iloc[order, order]
    slug_map = registry.drop_duplicates("condition_id").set_index("condition_id")
    labels = [slug_map.loc[c, "slug"][:34] for c in ordered.columns]
    themes = [slug_map.loc[c, "theme"] for c in ordered.columns]
    theme_colors = dict(zip(sorted(set(themes)),
                            [P["blue"], P["aqua"], P["yellow"], P["green"],
                             P["violet"], P["red"], P["magenta"], P["orange"]],
                            strict=False))
    fig, ax = plt.subplots(figsize=(7.0, 0.185 * len(labels) + 1.2))
    im = heat(ax, ordered.to_numpy(), -1, 1)
    ax.set_xticks([])
    ax.set_yticks(range(len(labels)), labels, fontsize=5.2)
    for i, t in enumerate(themes):
        ax.plot(-1.8, i, "s", color=theme_colors[t], ms=3, clip_on=False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=t)
               for t, c in theme_colors.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.01, 0.0),
              fontsize=6)
    fig.colorbar(im, shrink=0.5, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_market_clustermap.png")
    plt.close(fig)

    tcorr = pd.read_parquet(DEEP_DIR / "theme_corr.parquet")
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = heat(ax, tcorr.to_numpy(), -1, 1)
    ax.set_xticks(range(len(tcorr)), tcorr.columns, rotation=45, ha="right",
                  fontsize=6.5)
    ax.set_yticks(range(len(tcorr)), tcorr.columns, fontsize=6.5)
    for i in range(len(tcorr)):
        for j in range(len(tcorr)):
            v = tcorr.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_theme_corr.png")
    plt.close(fig)


def fig_d():
    tab = pd.read_parquet(DEEP_DIR / "stacking.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for k, (ax, ((product, sig_col, ret_col), g)) in enumerate(zip(
            axes, tab.groupby(["product", "signal", "ret"]), strict=False)):
        x = np.arange(len(g))
        ax.bar(x - 0.18, g.r2_univ, width=0.34, color=P["blue"], label="单因子 $R^2$")
        ax.bar(x + 0.18, g.partial_r2, width=0.34, color=P["orange"],
               label="联合边际 $R^2$")
        ax.set_xticks(x, [t.replace("_", "\n") for t in g.theme], fontsize=6)
        ax.set_title(f"{product}: {sig_col}→{ret_col}\n联合 $R^2$="
                     f"{g.r2_joint.iloc[0]:.2f} (n={g.n.iloc[0]})", fontsize=7)
        ax.axhline(0, color=P["ink2"], lw=0.6)
        pub_style.panel(ax, "abc"[k])
    axes[0].set_ylabel("$R^2$")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d_stacking.png")
    plt.close(fig)


def fig_e():
    tab = pd.read_parquet(DEEP_DIR / "horizon_curves.parquet")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    colors = {"oil_price": P["orange"], "mideast_conflict": P["red"],
              "metal_price": P["yellow"]}
    for (theme, product), g in tab.groupby(["theme", "product"]):
        ax.plot(g.K_hours, g.pearson, "o-", color=colors[theme], lw=1.1,
                label=f"{theme}×{product}")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([3, 6, 12, 24, 48, 120], ["3h", "6h", "12h", "24h", "48h", "120h"])
    ax.set_xlabel("信号累积窗 K（截止 09:00）")
    ax.set_ylabel("与当日日盘收益的相关")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e_horizon_curve.png")
    plt.close(fig)

    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")
    rets = futures_fine_returns(["SC", "AG"])
    merged = theme_fine.merge(rets, on=["product", "trade_date"])
    merged = merged[(merged.trade_date <= SIGNAL_END)
                    & ~merged.roll.fillna(False)]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for theme, product, color in [("oil_price", "SC", P["orange"]),
                                  ("mideast_conflict", "SC", P["red"]),
                                  ("metal_price", "AG", P["yellow"])]:
        g = merged[(merged.theme == theme)
                   & (merged["product"] == product)].sort_values("trade_date")
        pairs = g[["s_night", "r_night"]].dropna()
        if len(pairs) < 40:
            continue
        roll_corr = pairs.s_night.rolling(30, min_periods=20).corr(pairs.r_night)
        ax.plot(pd.to_datetime(g.loc[pairs.index, "trade_date"]), roll_corr,
                lw=1.1, color=color, label=f"{theme}×{product}")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_ylabel("30 日滚动相关（夜盘）")
    ax.legend()
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e_rolling_corr.png")
    plt.close(fig)


def fig_f_reverse():
    tab = pd.read_parquet(DEEP_DIR / "reverse_direction.parquet")
    key = tab[tab.set_index(["theme", "product"]).index.isin(
        [("oil_price", "SC"), ("mideast_conflict", "SC"), ("metal_price", "AU")])]
    designs = ["PM夜盘 -> 期货凌晨gap", "PM夜盘+闭市 -> 期货日盘",
               "期货夜盘 -> PM凌晨gap", "期货日盘 -> PM当晚傍晚gap",
               "期货日盘 -> PM当晚夜盘"]
    piv = key.pivot_table(index=["theme", "product"], columns="design",
                          values="pearson").reindex(columns=designs)
    piv.index = [f"{t}×{p}" for t, p in piv.index]
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    im = heat(ax, piv.to_numpy(dtype=float), -0.5, 0.5)
    ax.set_xticks(range(len(designs)),
                  [d.replace(" -> ", "\n→") for d in designs], fontsize=6.5)
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_reverse_direction.png")
    plt.close(fig)


def fig_f4():
    tab = pd.read_parquet(DEEP_DIR.parent / "leadlag_sc.parquet")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for theme, color in [("mideast_conflict", P["blue"]),
                         ("oil_price", P["orange"])]:
        g = tab[tab.theme == theme]
        ax.plot(g.lag_minutes, g.corr_ if hasattr(g, "corr_") else g["corr"],
                "o-", color=color, lw=1.1, label=theme)
    ax.axvline(0, color=P["ink2"], lw=0.6)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xlabel("滞后（分钟，正 = 信号领先收益）")
    ax.set_ylabel("交叉相关")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_TOP / "f4_night_leadlag_sc.png")
    plt.close(fig)


def fig_g():
    tab = pd.read_parquet(DEEP_DIR / "event_study.parquet")
    colors = {"E1_price_jump": P["red"], "E2_volume_burst": P["blue"],
              "E3_market_creation": P["aqua"]}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True)
    for k, (ax, (etype, g)) in enumerate(zip(axes, tab.groupby("type"),
                                             strict=False)):
        ax.fill_between(g.minutes, g.lo, g.hi, alpha=0.18, color=colors[etype],
                        lw=0)
        ax.plot(g.minutes, g.car_bp, lw=1.2, color=colors[etype])
        ax.axhline(0, color=P["ink2"], lw=0.6)
        ax.set_title(f"{etype}\n(n={int(g.n_events.iloc[0])})", fontsize=7)
        ax.set_xlabel("事件后分钟")
        pub_style.panel(ax, "abc"[k])
    axes[0].set_ylabel("SC 符号化 CAR（bp）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "g_event_study.png")
    plt.close(fig)


def fig_h():
    tab = pd.read_parquet(DEEP_DIR / "irf.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True)
    for k, (ax, ((theme, product), g)) in enumerate(zip(
            axes, tab.groupby(["theme", "product"]), strict=False)):
        ax.axhline(0, color=P["ink2"], lw=0.6)
        ax.errorbar(g.h, g.beta_bp, yerr=1.645 * g.se_bp, fmt="o-", lw=1.1,
                    color=P["blue"], ecolor=P["grid"])
        ax.set_title(f"{theme} × {product}", fontsize=7.5)
        ax.set_xlabel("视界 h（交易日）")
        pub_style.panel(ax, "abc"[k])
    axes[0].set_ylabel("β_h（bp / 1σ, 90% CI）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h_irf.png")
    plt.close(fig)


def fig_i():
    tab = pd.read_parquet(DEEP_DIR / "intl_control.parquet").dropna(subset=["t_raw"])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    lab = tab.theme + "×" + tab["product"] + "·" + tab.window
    x = np.arange(len(tab))
    ax.bar(x - 0.18, tab.t_raw, width=0.34, color=P["blue"], label="无控制")
    ax.bar(x + 0.18, tab.t_ctl, width=0.34, color=P["orange"], label="控制基准后")
    for yy in (2, -2):
        ax.axhline(yy, color=P["ink2"], lw=0.5, ls="--")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks(x, lab, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("信号系数 HAC t")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i1_intl_control.png")
    plt.close(fig)

    irf = pd.read_parquet(DEEP_DIR / "intl_irf_decomp.parquet")
    irf_oil = irf[irf.theme == "oil_price"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharex=True, sharey=True)
    for k, (ax, (comp, g)) in enumerate(zip(
            axes, irf_oil.groupby("component", sort=False), strict=False)):
        ax.axhline(0, color=P["ink2"], lw=0.6)
        ax.errorbar(g.h, g.beta_bp, yerr=1.645 * g.se_bp, fmt="o-", lw=1.1,
                    color=P["blue"], ecolor=P["grid"])
        ax.set_title(comp, fontsize=7.5)
        ax.set_xlabel("视界 h")
        pub_style.panel(ax, "abc"[k])
    axes[0].set_ylabel("β_h（bp / 1σ, 90% CI）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i2_irf_decomposition.png")
    plt.close(fig)

    # i3 升水时间线（重算 b_cc）
    days = fut_store.read_daily("SC")["trade_date"].tolist()
    specs = fut_store.read_product_specs().set_index("product")
    ne = sessions.parse_night_end(specs.loc["SC", "night_end"])
    win = sessions.signal_windows(days, night_end=ne)
    bench = window_benchmark_returns(load_benchmark_beijing("USO"), win)
    sc = fut_store.read_daily("SC")[["trade_date", "r_cc"]]
    m = sc.merge(bench[["trade_date", "b_cc"]], on="trade_date")
    m = m[(m.trade_date >= "2026-02-16") & (m.trade_date <= "2026-03-31")]
    q = m.r_cc - m.b_cc
    x = pd.to_datetime(m.trade_date)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.plot(x, m.r_cc.fillna(0).cumsum() * 100, lw=1.2, color=P["blue"],
            label="SC 累计")
    ax.plot(x, m.b_cc.fillna(0).cumsum() * 100, lw=1.2, color=P["orange"],
            label="USO 累计（北京窗口对齐）")
    ax.plot(x, q.fillna(0).cumsum() * 100, lw=1.4, color=P["red"],
            label="价差（SC-USO）累计")
    ax.axvline(pd.Timestamp("2026-03-09"), color=P["ink2"], lw=0.6, ls=":")
    ax.set_ylabel("累计对数收益（%）")
    ax.legend()
    pub_style.soft_grid(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i3_premium_timeline.png")
    plt.close(fig)


def fig_j():
    tab = pd.read_parquet(DEEP_DIR / "ablation.parquet")
    order = ["基线", "桶宽 5min", "桶宽 30min", "p_age 60min", "p_age 240min",
             "p_age 无上限", "截断 [0.01,0.99]", "截断 [0.05,0.95]", "等权聚合",
             "关闭时点化准入", "剔除近结算"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for k, (ax, theme) in enumerate(zip(axes,
                                        ["oil_price", "mideast_conflict"],
                                        strict=True)):
        for win_name, color in [("gap", P["orange"]), ("night", P["blue"])]:
            g = tab[(tab.theme == theme) & (tab.window == win_name)]
            g = g.set_index("variant").reindex(order)
            ax.plot(g.pearson, np.arange(len(order)), "o", ms=4, color=color,
                    label=f"{win_name} 窗口")
            ax.axvline(g.loc["基线", "pearson"], color=color, lw=0.6, ls=":",
                       alpha=0.7)
        ax.set_yticks(np.arange(len(order)), order, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("同期吸收 Pearson r")
        ax.set_title(f"{theme} × SC", fontsize=8)
        ax.axvline(0, color=P["ink2"], lw=0.6)
        pub_style.panel(ax, "ab"[k])
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "j_ablation.png")
    plt.close(fig)


def main() -> int:
    pub_style.setup()
    for fn in [fig_f1, fig_f5, fig_a, fig_b, fig_c, fig_d, fig_e,
               fig_f_reverse, fig_f4, fig_g, fig_h, fig_i, fig_j]:
        fn()
        print(f"  {fn.__name__} 完成")
    print(f"图已重绘至 {FIG_TOP} 与 {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
