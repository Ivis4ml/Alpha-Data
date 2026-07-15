"""扩充实验：安慰剂、置换检验、分位数组合、不对称、PM↔USO 分钟剖面、波动率。

五组新实验（编号接续论文第 12 节，图 k1..k5）：

K1 安慰剂矩阵 + 置换检验：
    - 安慰剂：mideast / oil 信号对**机制上无关**品种（JD 鸡蛋、C 玉米、SM 锰硅、
      IF 股指）的同窗口"吸收"，应接近 0；与真实配对（SC）同图对比。
    - 循环置换：把信号序列整体循环移位 2,000 次（保留自相关结构），
      观察真实 |r| 在置换分布中的位置，给出经验 p 值。
K2 信号五分位组合：按 s_gap 五分位分组，各组对应窗口收益均值 ± 标准误——
    线性回归之外的单调性检验（组合排序法）。
K3 正负不对称：s>0 与 s<0 子样本分别估计吸收 β（HAC），检验升级与降级
    消息的定价是否对称。
K4 PM ↔ USO 分钟级双剖面：美国活跃时段（北京 21:00-05:00）内，主题 5 分钟
    innovation 与 USO 5 分钟收益的交叉相关，叠加与 SC 的剖面——判别
    "Polymarket 领先国际市场"还是"两者同步而国内滞后"（规范 §5.2 三变量设计）。
K5 波动率预测：|s_gap| 对当日日盘已实现波动率（5 分钟收益平方和的平方根）
    的回归，控制前一日 RV（HAR-lite）——事件风险即便不预测方向，
    是否预测波动。

产物：analysis/deep/extra_*.parquet 与 docs/figures/deep/k*.png。

用法::

    .venv/bin/python scripts/extra_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pub_style  # noqa: E402
from analyze_cn_futures_polymarket import (  # noqa: E402
    REGISTRY_PATH,
    SIGNAL_END,
    load_all_trades,
    nw_regression,
)
from deep_analysis_cn_polymarket import DEEP_DIR, FIG_DIR, futures_fine_returns  # noqa: E402
from intl_benchmark_analysis import load_benchmark_beijing  # noqa: E402

from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402

P = pub_style.PALETTE
PLACEBO_PRODUCTS = ["JD", "C", "SM", "IF"]  # 鸡蛋、玉米、锰硅、股指：机制上无关


def merged_pair(theme_fine: pd.DataFrame, rets: pd.DataFrame, theme: str,
                product: str) -> pd.DataFrame:
    g = theme_fine[(theme_fine["theme"] == theme)
                   & (theme_fine["product"] == product)]
    if g.empty:  # 安慰剂品种不在登记表：借用 SC 的信号序列
        g = theme_fine[(theme_fine["theme"] == theme)
                       & (theme_fine["product"] == "SC")].copy()
    r = rets[rets["product"] == product].copy()
    r = r[~r["roll"].fillna(False)]
    rr = (r["r_gap_pm"].fillna(0) + r["r_night"].fillna(0)
          + r["r_gap_am"].fillna(0))
    rr[r[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
    r = r.assign(r_gap_total=rr)
    m = g.merge(r[["trade_date", "r_gap_total", "r_night", "r_day"]],
                on="trade_date")
    return m[m["trade_date"] <= SIGNAL_END].reset_index(drop=True)


# ---------------------------------------------------------------- K1
def k1_placebo_permutation(theme_fine: pd.DataFrame) -> None:
    rets_all = futures_fine_returns(["SC", *PLACEBO_PRODUCTS])
    rows = []
    for theme in ["oil_price", "mideast_conflict"]:
        for product in ["SC", *PLACEBO_PRODUCTS]:
            m = merged_pair(theme_fine, rets_all, theme, product)
            s = m["s_gap"].to_numpy(dtype="float64")
            r = m["r_gap_total"].to_numpy(dtype="float64")
            ok = ~(np.isnan(s) | np.isnan(r))
            if ok.sum() < 20:
                continue
            corr = float(np.corrcoef(s[ok], r[ok])[0, 1])
            rows.append({"theme": theme, "product": product,
                         "kind": "real" if product == "SC" else "placebo",
                         "n": int(ok.sum()), "pearson": corr})
    plac = pd.DataFrame(rows)

    # 循环置换检验（保留两序列各自的自相关，破坏两者的同步）
    rng = np.random.default_rng(20260714)
    perm_rows = []
    for theme in ["oil_price", "mideast_conflict"]:
        m = merged_pair(theme_fine, rets_all, theme, "SC")
        s = m["s_gap"].to_numpy(dtype="float64")
        r = m["r_gap_total"].to_numpy(dtype="float64")
        ok = ~(np.isnan(s) | np.isnan(r))
        s, r = s[ok], r[ok]
        real = float(np.corrcoef(s, r)[0, 1])
        n = len(s)
        null = np.empty(2000)
        for i in range(2000):
            k = int(rng.integers(1, n - 1))
            null[i] = np.corrcoef(np.roll(s, k), r)[0, 1]
        pval = float((np.abs(null) >= abs(real)).mean())
        perm_rows.append({"theme": theme, "real_r": real, "n": n,
                          "perm_p": pval,
                          "null_q99": float(np.quantile(np.abs(null), 0.99))})
        if theme == "oil_price":
            null_oil, real_oil = null.copy(), real
    perm = pd.DataFrame(perm_rows)
    plac.to_parquet(DEEP_DIR / "extra_placebo.parquet", index=False)
    perm.to_parquet(DEEP_DIR / "extra_permutation.parquet", index=False)
    print(plac.round(3).to_string(index=False))
    print(perm.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for i, theme in enumerate(["oil_price", "mideast_conflict"]):
        g = plac[plac["theme"] == theme].set_index("product").reindex(
            ["SC", *PLACEBO_PRODUCTS])
        x = np.arange(len(g)) + (i - 0.5) * 0.36
        colors = [P["orange"] if p == "SC" else P["blue"] for p in g.index]
        ax.bar(x, g["pearson"], width=0.34,
               color=colors, alpha=1.0 if i == 0 else 0.55,
               label=theme)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks(np.arange(5), ["SC\n(真实)", "JD\n鸡蛋", "C\n玉米",
                                 "SM\n锰硅", "IF\n股指"])
    ax.set_ylabel("闭市 gap 吸收 Pearson r")
    ax.legend(loc="upper right")
    pub_style.panel(ax, "a")
    pub_style.soft_grid(ax)

    ax = axes[1]
    ax.hist(np.abs(null_oil), bins=40, color=P["blue"], alpha=0.75)
    ax.axvline(abs(real_oil), color=P["red"], lw=1.4)
    ax.annotate(f"真实 |r|={abs(real_oil):.2f}\n置换 p<{max(perm.perm_p.iloc[0], 5e-4):.4g}",
                xy=(abs(real_oil), ax.get_ylim()[1] * 0.55),
                xytext=(-8, 0), textcoords="offset points", ha="right",
                fontsize=7, color=P["ink"])
    ax.set_xlabel("|r|（循环置换零分布，oil×SC）")
    ax.set_ylabel("频数")
    pub_style.panel(ax, "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k1_placebo_permutation.png")
    plt.close(fig)


# ---------------------------------------------------------------- K2 / K3
def k2_k3_quantile_asymmetry(theme_fine: pd.DataFrame) -> None:
    rets = futures_fine_returns(["SC"])
    rows_q, rows_a = [], []
    for theme in ["oil_price", "mideast_conflict"]:
        m = merged_pair(theme_fine, rets, theme, "SC")
        s = m["s_gap"].to_numpy(dtype="float64")
        r = m["r_gap_total"].to_numpy(dtype="float64") * 1e4
        ok = ~(np.isnan(s) | np.isnan(r))
        s, r = s[ok], r[ok]
        q = pd.qcut(s, 5, labels=False, duplicates="drop")
        for b in np.unique(q):
            sel = q == b
            rows_q.append({"theme": theme, "bin": int(b) + 1,
                           "s_mean": float(s[sel].mean()),
                           "r_mean": float(r[sel].mean()),
                           "r_se": float(r[sel].std(ddof=1) / np.sqrt(sel.sum())),
                           "n": int(sel.sum())})
        for name, sel in [("s>0（升级）", s > 0), ("s<0（降级）", s < 0)]:
            beta, t, _, n = nw_regression(r[sel] / 1e4, s[sel], 1)
            rows_a.append({"theme": theme, "side": name, "n": n,
                           "beta_bp_per_unit": beta * 1e4, "t_hac": t})
    qt = pd.DataFrame(rows_q)
    asym = pd.DataFrame(rows_a)
    qt.to_parquet(DEEP_DIR / "extra_quantile.parquet", index=False)
    asym.to_parquet(DEEP_DIR / "extra_asymmetry.parquet", index=False)
    print(asym.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for theme, color in [("oil_price", P["orange"]),
                         ("mideast_conflict", P["red"])]:
        g = qt[qt["theme"] == theme]
        ax.errorbar(g["bin"], g["r_mean"], yerr=g["r_se"], fmt="o-",
                    color=color, label=theme, lw=1.1)
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("s_gap 五分位（1=最负，5=最正）")
    ax.set_ylabel("SC 闭市收益均值（bp）")
    ax.legend(loc="upper left")
    pub_style.panel(ax, "a")
    pub_style.soft_grid(ax)

    ax = axes[1]
    x = np.arange(2)
    for i, theme in enumerate(["oil_price", "mideast_conflict"]):
        g = asym[asym["theme"] == theme]
        ax.bar(x + (i - 0.5) * 0.36, g["t_hac"], width=0.34,
               color=[P["orange"], P["red"]][i], label=theme)
    ax.axhline(2, color=P["ink2"], lw=0.6, ls="--")
    ax.axhline(0, color=P["ink2"], lw=0.6)
    ax.set_xticks(x, ["s>0（升级日）", "s<0（降级日）"])
    ax.set_ylabel("吸收 β 的 HAC t")
    ax.legend()
    pub_style.panel(ax, "b")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k2_quantile_asymmetry.png")
    plt.close(fig)


# ---------------------------------------------------------------- K4
def k4_pm_uso_leadlag(registry: pd.DataFrame,
                      trades: dict[str, pd.DataFrame]) -> None:
    """美国活跃时段内 PM 主题 innovation 对 SC 与 USO 的 5min 交叉相关双剖面。"""
    step = 5
    # SC 夜盘收益（复用 night 口径）
    sc_min = fut_store.read_minute("SC")
    night = sc_min[sc_min["session"] == "night"].sort_values("ts")
    px_sc = night.set_index("ts")["close"]
    r_sc = np.log(px_sc / px_sc.shift(1))
    r_sc[px_sc.index.to_series().diff() > pd.Timedelta(minutes=2)] = np.nan
    r_sc = r_sc.resample(f"{step}min", label="right", closed="right").sum(min_count=1)

    # USO 北京时间 5 分钟收益（21:00-05:00 段）
    px_uso = load_benchmark_beijing("USO")
    r_uso = np.log(px_uso / px_uso.shift(1))
    r_uso[px_uso.index.to_series().diff() > pd.Timedelta(minutes=2)] = np.nan
    r_uso = r_uso.resample(f"{step}min", label="right", closed="right").sum(min_count=1)
    hours = r_uso.index.hour
    r_uso = r_uso[(hours >= 21) | (hours < 5)]

    rows = []
    for theme in ["oil_price", "mideast_conflict"]:
        sub = registry[(registry["theme"] == theme)
                       & (registry["product"] == "SC")].drop_duplicates("condition_id")
        frames = []
        for rec in sub.itertuples(index=False):
            tr = trades.get(rec.condition_id)
            if tr is None or tr.empty:
                continue
            agg = cn_features.aggregate_price(tr, agg_minutes=step)
            lo = pd.Series(cn_features.logit(agg["p_agg"].to_numpy()),
                           index=pd.to_datetime(agg["bucket_end"]))
            consecutive = lo.index.to_series().diff() == pd.Timedelta(minutes=step)
            frames.append(lo.diff().where(consecutive)
                          * rec.orientation * np.sqrt(rec.usdc_win))
        pool = pd.concat(frames, axis=1)
        n_live = (~pool.isna()).sum(axis=1).replace(0, np.nan)
        innov = (pool.sum(axis=1, skipna=True) / np.sqrt(n_live)).dropna()

        for target, r_t in [("SC", r_sc), ("USO", r_uso)]:
            aligned = pd.DataFrame({"sig": innov, "ret": r_t}).dropna()
            gap = aligned.index.to_series()
            for k in range(-12, 13):
                ok = gap.diff(k) == pd.Timedelta(minutes=k * step)
                paired = pd.DataFrame({"sig": aligned["sig"].shift(k),
                                       "ret": aligned["ret"]})[ok].dropna()
                if len(paired) < 50:
                    continue
                rows.append({"theme": theme, "target": target,
                             "lag_minutes": k * step,
                             "corr": float(paired["sig"].corr(paired["ret"])),
                             "n": len(paired)})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "extra_pm_uso_leadlag.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, theme, letter in zip(axes, ["oil_price", "mideast_conflict"],
                                 ["a", "b"], strict=True):
        for target, color in [("USO", P["orange"]), ("SC", P["blue"])]:
            g = tab[(tab["theme"] == theme) & (tab["target"] == target)]
            ax.plot(g["lag_minutes"], g["corr"], "o-", color=color, lw=1.1,
                    label=f"→ {target}")
        ax.axvline(0, color=P["ink2"], lw=0.6)
        ax.axhline(0, color=P["ink2"], lw=0.6)
        ax.set_title(theme)
        ax.set_xlabel("滞后（分钟，正= PM 领先）")
        ax.legend()
        pub_style.panel(ax, letter)
    axes[0].set_ylabel("交叉相关")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k4_pm_uso_leadlag.png")
    plt.close(fig)
    lead = tab[(tab.lag_minutes > 0) & (tab.lag_minutes <= 15)]
    print(lead.pivot_table(index="lag_minutes", columns=["theme", "target"],
                           values="corr").round(3).to_string())


# ---------------------------------------------------------------- K5
def k5_volatility(theme_fine: pd.DataFrame) -> None:
    """|s_gap| 对当日日盘已实现波动率的 HAR-lite 回归。"""
    import statsmodels.api as sm

    sc_min = fut_store.read_minute("SC")
    day = sc_min[sc_min["session"] == "day"].sort_values("ts")
    px = day.set_index("ts")["close"]
    r5 = (np.log(px / px.shift(1))
          .resample("5min", label="right", closed="right").sum(min_count=1))
    rv = (r5.pow(2).groupby(r5.index.normalize()).sum() ** 0.5) * 1e4
    rv.index = rv.index.strftime("%Y-%m-%d")
    rv = rv.rename("rv_day")

    rows = []
    for theme in ["oil_price", "mideast_conflict"]:
        g = theme_fine[(theme_fine["theme"] == theme)
                       & (theme_fine["product"] == "SC")].copy()
        g = g.merge(rv, left_on="trade_date", right_index=True)
        g = g[g["trade_date"] <= SIGNAL_END].sort_values("trade_date")
        g["rv_lag"] = g["rv_day"].shift(1)
        g["abs_s"] = g["s_gap"].abs()
        d = g[["rv_day", "abs_s", "rv_lag"]].dropna()
        X = sm.add_constant(d[["abs_s", "rv_lag"]])
        fit = sm.OLS(d["rv_day"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        rows.append({"theme": theme, "n": len(d),
                     "beta_abs_s": float(fit.params["abs_s"]),
                     "t_abs_s": float(fit.tvalues["abs_s"]),
                     "beta_rv_lag": float(fit.params["rv_lag"]),
                     "t_rv_lag": float(fit.tvalues["rv_lag"]),
                     "r2": float(fit.rsquared)})
        if theme == "oil_price":
            d_plot = d.copy()
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "extra_volatility.parquet", index=False)
    print(tab.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.scatter(d_plot["abs_s"], d_plot["rv_day"], s=12, color=P["blue"],
               alpha=0.7, edgecolor="white", linewidth=0.3)
    b = np.polyfit(d_plot["abs_s"], d_plot["rv_day"], 1)
    xs = np.linspace(0, d_plot["abs_s"].max(), 40)
    ax.plot(xs, b[1] + b[0] * xs, color=P["red"], lw=1.2)
    row = tab[tab.theme == "oil_price"].iloc[0]
    ax.annotate(f"t(|s|)={row.t_abs_s:.1f}  n={int(row.n)}",
                xy=(0.05, 0.92), xycoords="axes fraction", fontsize=7)
    ax.set_xlabel("|s_gap|（oil_price，隔夜信号强度）")
    ax.set_ylabel("SC 日盘已实现波动率（bp）")
    pub_style.panel(ax, "a")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k5_volatility.png")
    plt.close(fig)


def main() -> int:
    pub_style.setup()
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    theme_fine = pd.read_parquet(DEEP_DIR / "theme_signals_fine.parquet")

    print("K1 安慰剂 + 置换")
    k1_placebo_permutation(theme_fine)
    print("\nK2/K3 分位数 + 不对称")
    k2_k3_quantile_asymmetry(theme_fine)
    print("\nK5 波动率")
    k5_volatility(theme_fine)

    print("\nK4 PM↔USO 分钟剖面（需拉逐笔）")
    registry = pd.read_parquet(REGISTRY_PATH)
    reg_sc = registry[(registry["theme"].isin(["oil_price", "mideast_conflict"]))
                      & (registry["product"] == "SC")]
    trades = load_all_trades(sorted(reg_sc["condition_id"].unique()))
    k4_pm_uso_leadlag(reg_sc, trades)

    print(f"\n产物：{DEEP_DIR}/extra_*.parquet 与 {FIG_DIR}/k*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
