"""第二轮深化分析：细分时段、聚类 / 叠加、窗口敏感性、双向传导、事件研究、IRF。

对应第一轮评审提出的七个问题（docs/cn_futures_polymarket_report.md v2 §0）：

A. 细分时段吸收：闭市拆为 gap_pm(15-21) / night(21-夜盘收) / gap_am(夜盘收-09)，
   各配**恰好对应**的期货收益（r_gap_pm / r_night / r_gap_am / r_day）；
   周一（闭市含周末）与平日分开。
B. 头部配对散点图矩阵（含 OLS 拟合线与统计注记）。
C. 因子结构：市场级信号相关矩阵 + 层次聚类 + PC1 方差占比（量化主题同质化）；
   主题级相关矩阵。
D. 叠加效应：同品种多主题联合 HAC 回归，报告联合 R² 与各因子偏贡献。
E. 窗口敏感性：信号累积窗 K 小时的相关曲线；30 交易日滚动相关；前后子样本。
F. 反向传导：期货各时段收益对**其后** Polymarket 窗口 innovation 的预测（与正向
   对称的转移表）；分钟级剖面的末笔价变体（校验中位数聚合的机械滞后）。
G. 事件研究：三类事件（E1 价格跳、E2 量爆发、E3 新市场首笔）后 SC 的分钟 / 日级
   CAR 曲线（0..H），自助法置信带。
H. 局部投影 IRF：``r_{t+h} ~ s_all_t`` 的 β_h 路径（h=0..8），给出"吸收后反转"的
   精确判据（见报告公式）。

产物：``data/cn_futures/analysis/deep/*.parquet`` 与 ``docs/figures/deep/*.png``。

用法::

    .venv/bin/python scripts/deep_analysis_cn_polymarket.py
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

from analyze_cn_futures_polymarket import (  # noqa: E402
    REGISTRY_PATH,
    SIGNAL_END,
    load_all_trades,
    market_signals_by_grid,
    nw_regression,
)

from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402

DEEP_DIR = fut_store.DB_DIR / "analysis" / "deep"
FIG_DIR = ROOT / "docs" / "figures" / "deep"

#: 校验过的分类色序（dataviz 参考调色板，light 模式）。
C = {
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
    "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d8d6cf", "surface": "#fcfcfb",
}
#: 发散色带：蓝（负）- 中性灰 - 红（正），红涨蓝跌与国内惯例一致。
DIVERGING = LinearSegmentedColormap.from_list(
    "dv", [C["blue"], "#f0efe9", C["red"]]
)

FINE_WINDOWS = ["gap_pm", "night", "gap_am", "day"]
FINE_RETURNS = {"gap_pm": "r_gap_pm", "night": "r_night",
                "gap_am": "r_gap_am", "day": "r_day"}
HEADLINE_PAIRS = [
    ("oil_price", "SC"), ("mideast_conflict", "SC"), ("russia_ukraine", "SC"),
    ("metal_price", "AU"), ("metal_price", "AG"), ("fed_policy", "CU"),
    ("us_china_trade", "M"),
]


def style_ax(ax: plt.Axes) -> None:
    """统一的图轴样式：细网格、去顶右边框、表格数字。"""
    ax.grid(True, alpha=0.35, lw=0.5, color=C["grid"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["ink2"])
    ax.tick_params(colors=C["ink2"], labelsize=8)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Hiragino Sans GB", "PingFang SC",
                                "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": C["surface"],
            "axes.facecolor": C["surface"],
            "savefig.facecolor": C["surface"],
            "text.color": C["ink"],
            "axes.labelcolor": C["ink2"],
            "axes.titlesize": 10.5,
            "axes.titlecolor": C["ink"],
        }
    )


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """登记表、逐笔、市场级信号（含细分 gap 两段）。

    市场级信号已存在时直接复用（删除
    ``analysis/deep/market_signals_fine.parquet`` 可强制重算）。
    """
    registry = pd.read_parquet(REGISTRY_PATH)
    trades = load_all_trades(sorted(registry["condition_id"].unique()))
    cache = DEEP_DIR / "market_signals_fine.parquet"
    if cache.exists():
        market_sig = pd.read_parquet(cache)
        print(f"复用缓存 {cache}")
    else:
        specs = fut_store.read_product_specs()
        market_sig = market_signals_by_grid(registry, trades, specs, agg_minutes=15)
        market_sig.to_parquet(cache, index=False)
    return registry, trades, market_sig


def apply_admit_mask(df: pd.DataFrame, registry: pd.DataFrame,
                     sig_cols: list[str]) -> pd.DataFrame:
    """时点化准入：早于 admit_ts（北京日期）的信号置 NaN。"""
    meta = registry[["condition_id", "product", "theme", "orientation",
                     "usdc_win", "admit_ts"]].drop_duplicates(
        ["condition_id", "product"])
    df = df.merge(meta, on=["condition_id", "product"], how="inner")
    admit_date = (
        pd.to_datetime(df["admit_ts"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    )
    mask = df["trade_date"] < admit_date
    for col in sig_cols:
        df.loc[mask, col] = np.nan
    return df


def aggregate_fine(market_sig: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """主题 × 品种 × 交易日的五窗口方向加权信号（sqrt(usdc) 权重）。"""
    sig_cols = [f"s_{w}" for w in ["night", "gap", "day"]] + ["s_gap_pm", "s_gap_am"]
    df = apply_admit_mask(market_sig, registry, sig_cols)
    df["w"] = np.sqrt(df["usdc_win"]) * df["orientation"]

    def wavg(v: pd.Series, w: pd.Series) -> float:
        vv = v.to_numpy(dtype="float64")
        ww = w.to_numpy(dtype="float64")
        ok = ~np.isnan(vv)
        if not ok.any():
            return np.nan
        return float(np.sum(vv[ok] * ww[ok]) / np.sum(np.abs(ww[ok])))

    rows = []
    for (theme, product, day), g in df.groupby(["theme", "product", "trade_date"]):
        rows.append({
            "theme": theme, "product": product, "trade_date": day,
            **{col: wavg(g[col], g["w"]) for col in sig_cols},
        })
    return pd.DataFrame(rows).sort_values(
        ["theme", "product", "trade_date"]).reset_index(drop=True)


def futures_fine_returns(products: list[str]) -> pd.DataFrame:
    """细分时段收益表 + 周一标记。"""
    frames = []
    for p in products:
        d = fut_store.read_daily(p)
        d["product"] = p
        d["is_monday"] = pd.to_datetime(d["trade_date"]).dt.dayofweek == 0
        if d["r_night"].isna().all():
            d["r_gap_pm"] = d["r_gap_full"]  # 无夜盘品种整段闭市记入 gap_pm
            d["r_gap_am"] = np.nan
        frames.append(d[["product", "trade_date", "is_monday", "roll",
                         "r_gap_pm", "r_night", "r_gap_am", "r_day", "r_cc"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- A. 细分时段
def fine_absorption(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """五窗口 × (全样本 / 平日 / 周一) 的同期吸收表。"""
    from scipy import stats as sps

    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[(merged["trade_date"] <= SIGNAL_END)
                    & ~merged["roll"].fillna(False)]
    rows = []
    for (theme, product), g in merged.groupby(["theme", "product"]):
        for w in FINE_WINDOWS:
            sig_col = f"s_{w}" if w in ("night", "day") else f"s_{w}"
            ret_col = FINE_RETURNS[w]
            for subset, sub in (("all", g), ("weekday", g[~g["is_monday"]]),
                                ("monday", g[g["is_monday"]])):
                s = sub[sig_col].to_numpy(dtype="float64")
                r = sub[ret_col].to_numpy(dtype="float64")
                ok = ~(np.isnan(s) | np.isnan(r))
                n = int(ok.sum())
                if n < 15:
                    continue
                pear = sps.pearsonr(s[ok], r[ok])
                beta, t_hac, p_hac, _ = nw_regression(r, s, 1)
                rows.append({
                    "theme": theme, "product": product, "window": w,
                    "subset": subset, "n": n,
                    "pearson": float(pear.statistic), "t_hac": t_hac,
                    "p_hac": p_hac,
                })
    return pd.DataFrame(rows)


def fig_fine_absorption(tab: pd.DataFrame) -> None:
    sub = tab[(tab["subset"] == "all")
              & tab.set_index(["theme", "product"]).index.isin(HEADLINE_PAIRS)]
    piv = sub.pivot_table(index=["theme", "product"], columns="window",
                          values="pearson").reindex(columns=FINE_WINDOWS)
    piv.index = [f"{t} × {p}" for t, p in piv.index]
    npiv = sub.pivot_table(index=["theme", "product"], columns="window",
                           values="n").reindex(columns=FINE_WINDOWS)
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(piv) + 1.6))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap=DIVERGING, vmin=-0.8, vmax=0.8,
                   aspect="auto")
    ax.set_xticks(range(4), ["傍晚闭市\n15:00-21:00", "夜盘\n21:00-夜盘收",
                             "凌晨闭市\n夜盘收-09:00", "日盘\n09:00-15:00"],
                  fontsize=8.5)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            n = npiv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}\n(n={int(n)})", ha="center", va="center",
                        fontsize=7.5, color=C["ink"])
    ax.set_title("细分时段同期吸收：主题信号 × 恰好对应时段的期货收益（Pearson）")
    fig.colorbar(im, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_fine_absorption.png", dpi=150)
    plt.close(fig)

    # 周一 vs 平日（含周末闭市的差异）
    key = tab[tab.set_index(["theme", "product"]).index.isin(
        [("oil_price", "SC"), ("mideast_conflict", "SC"), ("metal_price", "AU")])]
    key = key[key["subset"].isin(["weekday", "monday"]) & (key["window"] != "day")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, (pair, g) in zip(
        axes, key.groupby(key["theme"] + " × " + key["product"]), strict=False
    ):
        piv2 = g.pivot_table(index="window", columns="subset",
                             values="pearson").reindex(FINE_WINDOWS[:3])
        x = np.arange(len(piv2))
        ax.bar(x - 0.18, piv2.get("weekday", pd.Series(index=piv2.index)),
               width=0.34, color=C["blue"], label="平日")
        ax.bar(x + 0.18, piv2.get("monday", pd.Series(index=piv2.index)),
               width=0.34, color=C["orange"], label="周一（含周末）")
        ax.set_xticks(x, ["傍晚", "夜盘", "凌晨"], fontsize=8.5)
        ax.set_title(pair, fontsize=9.5)
        ax.axhline(0, color=C["ink2"], lw=0.7)
        style_ax(ax)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("闭市各段吸收：平日 vs 周一（周末积累的信号经周一实现）", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_weekday_vs_monday.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- B. 散点矩阵
def fig_scatter_grid(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> None:
    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[(merged["trade_date"] <= SIGNAL_END)
                    & ~merged["roll"].fillna(False)]
    panels = [
        ("oil_price", "SC", "s_gap", "r_gap_pm", "油价市场 gap 信号 × SC 傍晚+隔夜"),
        ("oil_price", "SC", "s_night", "r_night", "油价市场夜盘信号 × SC 夜盘收益"),
        ("mideast_conflict", "SC", "s_night", "r_night", "中东冲突夜盘 × SC 夜盘"),
        ("mideast_conflict", "SC", "s_gap", "r_gap_pm", "中东冲突 gap × SC 隔夜"),
        ("metal_price", "AU", "s_day", "r_day", "金价市场日盘 × AU 日盘"),
        ("us_china_trade", "M", "s_gap", "r_gap_pm", "中美贸易 gap × M 隔夜"),
    ]
    # 修正：r_gap_pm 只覆盖傍晚段；与 s_gap 全闭市信号配对时应使用整段 gap 收益。
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7))
    for ax, (theme, product, sc, rc, title) in zip(axes.flat, panels, strict=True):
        g = merged[(merged["theme"] == theme) & (merged["product"] == product)]
        if rc == "r_gap_pm" and sc == "s_gap":
            r = (g["r_gap_pm"].fillna(0) + g["r_night"].fillna(0)
                 + g["r_gap_am"].fillna(0))
            r[g[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
        else:
            r = g[rc]
        s = g[sc]
        ok = ~(s.isna() | r.isna())
        s, r = s[ok].to_numpy(), r[ok].to_numpy() * 1e4
        ax.scatter(s, r, s=16, alpha=0.65, color=C["blue"], edgecolor="white",
                   linewidth=0.4)
        if len(s) > 10:
            beta, t_hac, _, _ = nw_regression(r / 1e4, s, 1)
            xs = np.linspace(s.min(), s.max(), 50)
            alpha_fit = r.mean() / 1e4 - beta * s.mean()
            ax.plot(xs, (alpha_fit + beta * xs) * 1e4, color=C["red"], lw=1.6)
            corr = np.corrcoef(s, r)[0, 1]
            ax.text(0.03, 0.95, f"r={corr:+.2f}  HAC t={t_hac:+.1f}  n={len(s)}",
                    transform=ax.transAxes, fontsize=8, va="top", color=C["ink"])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("信号（Δlogit）", fontsize=8)
        ax.set_ylabel("收益（bp）", fontsize=8)
        style_ax(ax)
    fig.suptitle("头部配对散点：窗口内信号 × 恰好对应窗口的期货收益", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b_scatter_grid.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- C. 聚类
def clustering(market_sig: pd.DataFrame, registry: pd.DataFrame,
               top_n: int = 60) -> pd.DataFrame:
    """市场级日频总 innovation 的相关结构：层次聚类 + PC1 占比。"""
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    sig_cols = ["s_night", "s_gap", "s_day"]
    df = apply_admit_mask(market_sig.copy(), registry, sig_cols)
    arr = df[sig_cols].to_numpy(dtype="float64")
    all_nan = np.isnan(arr).all(axis=1)
    df["s_total"] = np.where(all_nan, np.nan, np.nansum(arr, axis=1))
    df["s_total"] *= df["orientation"]
    df = df[df["trade_date"] <= SIGNAL_END]

    top = (registry.drop_duplicates("condition_id")
           .nlargest(top_n, "usdc_win")["condition_id"])
    wide = (df[df["condition_id"].isin(top)]
            .drop_duplicates(["condition_id", "trade_date"])
            .pivot(index="trade_date", columns="condition_id", values="s_total"))
    wide = wide.loc[:, wide.notna().sum() >= 40]
    corr = wide.corr(min_periods=20)
    corr = corr.dropna(how="all").dropna(how="all", axis=1)
    corr = corr.loc[corr.index, corr.index].fillna(0)
    mat = corr.to_numpy(copy=True)
    np.fill_diagonal(mat, 1.0)
    corr = pd.DataFrame(mat, index=corr.index, columns=corr.columns)

    dist = squareform((1 - corr.to_numpy()).clip(0, 2), checks=False)
    link = hierarchy.linkage(dist, method="average")
    order = hierarchy.leaves_list(link)
    ordered = corr.iloc[order, order]

    slug_map = registry.drop_duplicates("condition_id").set_index("condition_id")
    labels = [slug_map.loc[c, "slug"][:34] for c in ordered.columns]
    themes = [slug_map.loc[c, "theme"] for c in ordered.columns]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(ordered.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks(range(len(labels)), labels, fontsize=5.6)
    theme_colors = {t: c for t, c in zip(
        sorted(set(themes)),
        [C["blue"], C["aqua"], C["yellow"], C["green"],
         C["violet"], C["red"], C["magenta"], C["orange"]], strict=False)}
    for i, t in enumerate(themes):
        ax.plot(-1.6, i, "s", color=theme_colors[t], ms=4, clip_on=False)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=t)
               for t, c in theme_colors.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False,
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"市场级日频信号相关矩阵（前 {len(labels)} 大市场，层次聚类排序）")
    fig.colorbar(im, shrink=0.6, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_market_clustermap.png", dpi=150)
    plt.close(fig)

    # PC1 占比（市场级与主题级）
    filled = wide.loc[:, ordered.columns]
    z = (filled - filled.mean()) / filled.std()
    z = z.fillna(0)
    eigvals = np.linalg.eigvalsh(np.cov(z.to_numpy().T))
    pc1_share_markets = float(eigvals[-1] / eigvals.sum())

    theme_daily = (df.groupby(["theme", "trade_date"])["s_total"].mean()
                   .unstack(level=0))
    tcorr = theme_daily.corr(min_periods=25)
    tz = ((theme_daily - theme_daily.mean()) / theme_daily.std()).fillna(0)
    teig = np.linalg.eigvalsh(np.cov(tz.to_numpy().T))
    pc1_share_themes = float(teig[-1] / teig.sum())

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(tcorr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(tcorr)), tcorr.columns, rotation=45, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(tcorr)), tcorr.columns, fontsize=8)
    for i in range(len(tcorr)):
        for j in range(len(tcorr)):
            v = tcorr.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title(f"主题级日频信号相关（PC1 方差占比：主题 {pc1_share_themes:.0%}，"
                 f"市场 {pc1_share_markets:.0%}）")
    fig.colorbar(im, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_theme_corr.png", dpi=150)
    plt.close(fig)

    stats = pd.DataFrame(
        {"metric": ["pc1_share_markets", "pc1_share_themes", "n_markets"],
         "value": [pc1_share_markets, pc1_share_themes, float(len(labels))]}
    )
    stats.to_parquet(DEEP_DIR / "clustering_stats.parquet", index=False)
    tcorr.to_parquet(DEEP_DIR / "theme_corr.parquet")
    return stats


# ---------------------------------------------------------------- D. 叠加
def stacking(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """同品种多主题联合回归：联合 R² 与各因子边际贡献（HAC）。"""
    import statsmodels.api as sm

    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[(merged["trade_date"] <= SIGNAL_END)
                    & ~merged["roll"].fillna(False)]
    specs_list = [
        ("SC", ["oil_price", "mideast_conflict", "russia_ukraine"], "s_gap",
         "r_gap_total"),
        ("SC", ["oil_price", "mideast_conflict", "russia_ukraine"], "s_night",
         "r_night"),
        ("AU", ["metal_price", "mideast_conflict", "fed_policy", "us_shutdown"],
         "s_night", "r_night"),
    ]
    rows = []
    for product, themes, sig_col, ret_col in specs_list:
        sub = merged[merged["product"] == product]
        if ret_col == "r_gap_total":
            r = (sub["r_gap_pm"].fillna(0) + sub["r_night"].fillna(0)
                 + sub["r_gap_am"].fillna(0))
            r[sub[["r_gap_pm", "r_night", "r_gap_am"]].isna().all(axis=1)] = np.nan
            sub = sub.assign(_ret=r)
        else:
            sub = sub.assign(_ret=sub[ret_col])
        wide = sub.pivot_table(index="trade_date", values=sig_col,
                               columns="theme").reindex(columns=themes)
        y = sub.drop_duplicates("trade_date").set_index("trade_date")["_ret"]
        data = wide.join(y).dropna()
        if len(data) < 25:
            continue
        X = sm.add_constant(data[themes])
        joint = sm.OLS(data["_ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        for th in themes:
            uni = sm.OLS(data["_ret"], sm.add_constant(data[[th]])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 1})
            drop = sm.OLS(data["_ret"],
                          sm.add_constant(data[[t for t in themes if t != th]])
                          ).fit()
            rows.append({
                "product": product, "signal": sig_col, "ret": ret_col, "theme": th,
                "n": len(data), "beta_joint": float(joint.params[th]),
                "t_joint": float(joint.tvalues[th]),
                "r2_joint": float(joint.rsquared),
                "r2_univ": float(uni.rsquared),
                "partial_r2": float(joint.rsquared - drop.rsquared),
            })
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "stacking.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, ((product, sig_col, ret_col), g) in zip(
        axes, tab.groupby(["product", "signal", "ret"]), strict=False
    ):
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["r2_univ"], width=0.34, color=C["blue"], label="单因子 R²")
        ax.bar(x + 0.18, g["partial_r2"], width=0.34, color=C["orange"],
               label="联合中的边际 R²")
        ax.set_xticks(x, [t.replace("_", "\n") for t in g["theme"]], fontsize=7.5)
        ax.set_title(f"{product} {sig_col}→{ret_col}\n联合 R²={g['r2_joint'].iloc[0]:.2f}"
                     f" (n={g['n'].iloc[0]})", fontsize=9)
        ax.axhline(0, color=C["ink2"], lw=0.7)
        style_ax(ax)
    axes[0].set_ylabel("R²")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("叠加效应：单因子解释力 vs 联合回归中的边际贡献", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d_stacking.png", dpi=150)
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- E. 窗口敏感性
def horizon_curves(registry: pd.DataFrame, trades: dict[str, pd.DataFrame],
                   rets: pd.DataFrame) -> pd.DataFrame:
    """信号累积窗 K 小时（截止 09:00）对当日日盘收益的相关曲线 + 滚动相关。"""
    from scipy import stats as sps

    horizons = [3, 6, 12, 18, 24, 48, 72, 120]
    pairs = [("oil_price", "SC"), ("mideast_conflict", "SC"), ("metal_price", "AU")]
    rows = []
    day_open = {}
    for theme, product in pairs:
        sub = registry[(registry["theme"] == theme)
                       & (registry["product"] == product)]
        rsub = rets[rets["product"] == product]
        rsub = rsub[(rsub["trade_date"] <= SIGNAL_END) & ~rsub["roll"].fillna(False)]
        days = rsub["trade_date"].tolist()
        if product not in day_open:
            day_open[product] = pd.to_datetime([f"{d} 09:00:00" for d in days])
        opens = day_open[product]

        # 主题级 15min 桶 logit 序列（方向加权池化后按端点差取累积 innovation）
        frames = []
        for rec in sub.drop_duplicates("condition_id").itertuples(index=False):
            tr = trades.get(rec.condition_id)
            if tr is None or tr.empty:
                continue
            agg = cn_features.aggregate_price(tr, agg_minutes=15)
            lo = pd.Series(cn_features.logit(agg["p_agg"].to_numpy()),
                           index=pd.to_datetime(agg["bucket_end"]))
            consecutive = lo.index.to_series().diff() == pd.Timedelta(minutes=15)
            innov = lo.diff().where(consecutive)
            frames.append(innov * rec.orientation * np.sqrt(rec.usdc_win))
        pool = pd.concat(frames, axis=1)
        n_live = (~pool.isna()).sum(axis=1).replace(0, np.nan)
        innov_series = (pool.sum(axis=1, skipna=True) / np.sqrt(n_live)).dropna()
        cum = innov_series.cumsum()

        for K in horizons:
            # s_K(t) = 累积 innovation 在 (09:00 - K, 09:00] 的变化
            end_v = cum.reindex(opens, method="ffill").to_numpy()
            start_v = cum.reindex(opens - pd.Timedelta(hours=K),
                                  method="ffill").to_numpy()
            sK = end_v - start_v
            r = rsub["r_day"].to_numpy(dtype="float64")
            ok = ~(np.isnan(sK) | np.isnan(r))
            if ok.sum() < 25:
                continue
            pear = sps.pearsonr(sK[ok], r[ok])
            rows.append({"theme": theme, "product": product, "K_hours": K,
                         "n": int(ok.sum()), "pearson": float(pear.statistic),
                         "p": float(pear.pvalue)})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "horizon_curves.parquet", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    for (theme, product), g in tab.groupby(["theme", "product"]):
        color = {"oil_price": C["orange"], "mideast_conflict": C["red"],
                 "metal_price": C["yellow"]}[theme]
        ax.plot(g["K_hours"], g["pearson"], marker="o", ms=4, lw=1.6, color=color,
                label=f"{theme} × {product}")
    ax.axhline(0, color=C["ink2"], lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([3, 6, 12, 24, 48, 72, 120],
                  ["3h", "6h", "12h", "24h", "48h", "72h", "120h"])
    ax.set_xlabel("信号累积窗 K（截止当日 09:00，对数轴）")
    ax.set_ylabel("与当日日盘收益的 Pearson r")
    ax.set_title("窗口敏感性：信号累积长度 K 对 r_day 预测相关的影响")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e_horizon_curve.png", dpi=150)
    plt.close(fig)
    return tab


def rolling_and_split(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> None:
    """30 交易日滚动相关（时变稳定性）与前后子样本分割。"""
    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[(merged["trade_date"] <= SIGNAL_END)
                    & ~merged["roll"].fillna(False)]
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for (theme, product, sig_col, ret_col, color) in [
        ("oil_price", "SC", "s_night", "r_night", C["orange"]),
        ("mideast_conflict", "SC", "s_night", "r_night", C["red"]),
        ("metal_price", "AG", "s_night", "r_night", C["yellow"]),
    ]:
        g = merged[(merged["theme"] == theme)
                   & (merged["product"] == product)].sort_values("trade_date")
        pairs = g[[sig_col, ret_col]].dropna()
        if len(pairs) < 40:
            continue
        roll_corr = pairs[sig_col].rolling(30, min_periods=20).corr(pairs[ret_col])
        x = pd.to_datetime(g.loc[pairs.index, "trade_date"])
        ax.plot(x, roll_corr, lw=1.6, color=color, label=f"{theme} × {product}")
    ax.axhline(0, color=C["ink2"], lw=0.7)
    ax.set_ylabel("30 交易日滚动 Pearson r（夜盘窗口）")
    ax.set_title("吸收相关性的时间稳定性（滚动窗口）")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e_rolling_corr.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- F. 反向
def reverse_direction(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """期货时段收益 -> 其后 Polymarket 窗口 innovation（与正向对称）。"""
    from scipy import stats as sps

    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[(merged["trade_date"] <= SIGNAL_END)
                    & ~merged["roll"].fillna(False)]
    # 转移设计：行 = 先行窗口，列 = 后续窗口。
    # 正向（PM->期货）：s_w1(t) 对 r_w2(t)，w1 时序早于 w2。
    # 反向（期货->PM）：r_w1(t) 对 s_w2(t 或 t+1)，w1 时序早于 w2。
    designs = [
        ("PM夜盘 -> 期货凌晨gap", "s_night", "r_gap_am", 0, "fwd"),
        ("PM夜盘+闭市 -> 期货日盘", "s_pre", "r_day", 0, "fwd"),
        ("期货夜盘 -> PM凌晨gap", "r_night", "s_gap_am", 0, "rev"),
        ("期货日盘 -> PM当晚傍晚gap", "r_day", "s_gap_pm", 1, "rev"),
        ("期货日盘 -> PM当晚夜盘", "r_day", "s_night", 1, "rev"),
    ]
    rows = []
    for (theme, product), g in merged.groupby(["theme", "product"]):
        if (theme, product) not in HEADLINE_PAIRS:
            continue
        g = g.sort_values("trade_date").reset_index(drop=True)
        g["s_pre"] = g["s_night"].fillna(0) + g["s_gap"]
        for name, a_col, b_col, shift_b, direction in designs:
            a = g[a_col].to_numpy(dtype="float64")
            b = g[b_col].shift(-shift_b).to_numpy(dtype="float64")
            ok = ~(np.isnan(a) | np.isnan(b))
            n = int(ok.sum())
            if n < 25:
                continue
            pear = sps.pearsonr(a[ok], b[ok])
            rows.append({"theme": theme, "product": product, "design": name,
                         "direction": direction, "n": n,
                         "pearson": float(pear.statistic), "p": float(pear.pvalue)})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "reverse_direction.parquet", index=False)

    key = tab[tab.set_index(["theme", "product"]).index.isin(
        [("oil_price", "SC"), ("mideast_conflict", "SC"), ("metal_price", "AU")])]
    piv = key.pivot_table(index=["theme", "product"], columns="design",
                          values="pearson")
    piv = piv.reindex(columns=[d[0] for d in designs])
    piv.index = [f"{t}×{p}" for t, p in piv.index]
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap=DIVERGING, vmin=-0.5, vmax=0.5,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)),
                  [c.replace(" -> ", "\n→") for c in piv.columns], fontsize=7.5)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("双向传导：先行窗口 → 后续窗口的相关（前 2 列 PM→期货，后 3 列 期货→PM）")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_reverse_direction.png", dpi=150)
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- G. 事件研究
def event_study(registry: pd.DataFrame, trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """三类事件后 SC 的分钟级 CAR（夜盘内事件）与全事件的日级响应。

    事件定义（15min 桶 b，主题限 SC 相关：oil_price / mideast_conflict / russia_ukraine）：

    - E1 价格跳：|Δlogit_b| > 3 × MAD_48(Δlogit) 且桶内成交 >= $10k；符号 =
      orientation × sign(Δlogit)。
    - E2 量爆发：usdc_b > 10 × 过去 48 桶中位数 且笔数 >= 20；符号取同桶 E1 符号
      （无价格变动则按 Δlogit 符号，容许小幅）。
    - E3 新市场：登记市场的首笔成交时刻；符号 = orientation × sign(首桶 Δlogit 缺省 +1)。
    """
    minute = fut_store.read_minute("SC")
    night = minute[minute["session"] == "night"]
    px = night.set_index("ts")["close"].sort_index()
    logret = np.log(px / px.shift(1))
    gaps = px.index.to_series().diff() > pd.Timedelta(minutes=2)
    logret[gaps] = np.nan

    themes = ("oil_price", "mideast_conflict", "russia_ukraine")
    sub = registry[(registry["theme"].isin(themes)) & (registry["product"] == "SC")]
    events = []
    for rec in sub.drop_duplicates("condition_id").itertuples(index=False):
        tr = trades.get(rec.condition_id)
        if tr is None or tr.empty:
            continue
        agg = cn_features.aggregate_price(tr, agg_minutes=15)
        if len(agg) < 60:
            continue
        lo = pd.Series(cn_features.logit(agg["p_agg"].to_numpy()),
                       index=pd.to_datetime(agg["bucket_end"]))
        consecutive = lo.index.to_series().diff() == pd.Timedelta(minutes=15)
        dlo = lo.diff().where(consecutive)
        usdc = pd.Series(agg["usdc"].to_numpy(), index=lo.index)
        ntr = pd.Series(agg["n_trades"].to_numpy(), index=lo.index)
        mad = (dlo - dlo.rolling(48, min_periods=24).median()).abs().rolling(
            48, min_periods=24).median()
        e1 = (dlo.abs() > 3 * 1.4826 * mad) & (usdc >= 10_000)
        med_usdc = usdc.rolling(48, min_periods=24).median()
        e2 = (usdc > 10 * med_usdc) & (ntr >= 20)
        for ts in lo.index[e1.fillna(False)]:
            events.append({"type": "E1_price_jump", "ts": ts,
                           "sign": float(rec.orientation * np.sign(dlo[ts]))})
        for ts in lo.index[e2.fillna(False)]:
            sgn = np.sign(dlo[ts]) if pd.notna(dlo[ts]) and dlo[ts] != 0 else 0.0
            events.append({"type": "E2_volume_burst", "ts": ts,
                           "sign": float(rec.orientation * sgn)})
        events.append({"type": "E3_market_creation",
                       "ts": pd.to_datetime(rec.first_ts, unit="s", utc=True)
                       .tz_convert("Asia/Shanghai").tz_localize(None),
                       "sign": 1.0})
    ev = pd.DataFrame(events)
    ev = ev[ev["sign"] != 0]

    cum = logret.fillna(0).cumsum()
    rows = []
    rng = np.random.default_rng(20260713)
    for etype, g in ev.groupby("type"):
        # 只保留事件时刻处于 SC 夜盘内（其后 120 分钟仍在夜盘）的事件
        base_idx = px.index
        pos = base_idx.searchsorted(g["ts"].to_numpy())
        valid = (pos > 0) & (pos < len(base_idx) - 25)
        g2 = g[valid]
        pos = pos[valid]
        in_night = np.asarray(
            (base_idx[np.minimum(pos + 24, len(base_idx) - 1)] - base_idx[pos])
            <= pd.Timedelta(minutes=130)
        )
        g2, pos = g2[in_night], pos[in_night]
        if len(g2) < 8:
            continue
        car_matrix = []
        for p0, sgn in zip(pos, g2["sign"], strict=True):
            base = cum.iloc[p0 - 1]
            path = (cum.iloc[p0 - 1:p0 + 25].to_numpy() - base) * sgn
            car_matrix.append(path[:26] if len(path) >= 26 else None)
        car_matrix = np.array([c for c in car_matrix if c is not None])
        mean_car = car_matrix.mean(axis=0) * 1e4
        boot = np.array([
            car_matrix[rng.integers(0, len(car_matrix), len(car_matrix))].mean(axis=0)
            for _ in range(500)
        ]) * 1e4
        lo_b, hi_b = np.percentile(boot, [5, 95], axis=0)
        for i, h in enumerate(range(0, 26)):
            rows.append({"type": etype, "minutes": h * 5 - 5,
                         "car_bp": float(mean_car[i]),
                         "lo": float(lo_b[i]), "hi": float(hi_b[i]),
                         "n_events": len(car_matrix)})
    tab = pd.DataFrame(rows)
    tab = tab[tab["minutes"] >= 0]
    tab.to_parquet(DEEP_DIR / "event_study.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    colors = {"E1_price_jump": C["red"], "E2_volume_burst": C["blue"],
              "E3_market_creation": C["aqua"]}
    for ax, (etype, g) in zip(axes, tab.groupby("type"), strict=False):
        ax.fill_between(g["minutes"], g["lo"], g["hi"], alpha=0.2,
                        color=colors[etype])
        ax.plot(g["minutes"], g["car_bp"], lw=1.8, color=colors[etype])
        ax.axhline(0, color=C["ink2"], lw=0.7)
        ax.set_title(f"{etype}（n={int(g['n_events'].iloc[0])}）", fontsize=9)
        ax.set_xlabel("事件后分钟数")
        style_ax(ax)
    axes[0].set_ylabel("SC 符号化 CAR（bp）")
    fig.suptitle("事件研究：Polymarket 事件（夜盘内）后 SC 的累计收益（90% 自助置信带）",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "g_event_study.png", dpi=150)
    plt.close(fig)
    return tab


# ---------------------------------------------------------------- H. IRF
def local_projection_irf(theme_fine: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """局部投影 IRF：r_{t->t+h} = a_h + b_h * s_all_t，h = 0..8（HAC）。"""
    merged = theme_fine.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[merged["trade_date"] <= SIGNAL_END]
    pairs = [("oil_price", "SC"), ("mideast_conflict", "SC"),
             ("us_china_trade", "M")]
    rows = []
    for theme, product in pairs:
        g = merged[(merged["theme"] == theme)
                   & (merged["product"] == product)].sort_values("trade_date")
        g = g.reset_index(drop=True)
        arr = g[["s_night", "s_gap", "s_day"]].to_numpy(dtype="float64")
        all_nan = np.isnan(arr).all(axis=1)
        s = np.where(all_nan, np.nan, np.nansum(arr, axis=1))
        rcc = g["r_cc"].to_numpy(dtype="float64")
        for h in range(0, 9):
            if h == 0:
                y = rcc
            else:
                fwd = pd.Series(rcc).shift(-1).rolling(h, min_periods=h).sum()
                y = fwd.shift(-(h - 1)).to_numpy()
            beta, t_hac, p_hac, n = nw_regression(y, s, max(h, 1))
            se = beta / t_hac if t_hac not in (0.0,) and np.isfinite(t_hac) else np.nan
            rows.append({"theme": theme, "product": product, "h": h, "n": n,
                         "beta_bp": beta * 1e4 * np.nanstd(s),
                         "se_bp": abs(se) * 1e4 * np.nanstd(s)
                         if np.isfinite(se) else np.nan,
                         "t_hac": t_hac})
    tab = pd.DataFrame(rows)
    tab.to_parquet(DEEP_DIR / "irf.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    for ax, ((theme, product), g) in zip(axes, tab.groupby(["theme", "product"]),
                                         strict=False):
        ax.axhline(0, color=C["ink2"], lw=0.7)
        ax.errorbar(g["h"], g["beta_bp"], yerr=1.645 * g["se_bp"], fmt="o-",
                    ms=4, lw=1.5, capsize=3, color=C["blue"],
                    ecolor=C["grid"])
        ax.set_title(f"{theme} × {product}", fontsize=9)
        ax.set_xlabel("视界 h（交易日；h=0 为同期）")
        style_ax(ax)
    axes[0].set_ylabel("β_h（bp / 1σ 信号，90% CI）")
    fig.suptitle("局部投影 IRF：β_h 路径（h=0 同期吸收；h≥1 为其后 h 日累计）",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h_irf.png", dpi=150)
    plt.close(fig)
    return tab


def main() -> int:
    setup_matplotlib()
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    registry, trades, market_sig = load_inputs()
    theme_fine = aggregate_fine(market_sig, registry)
    theme_fine.to_parquet(DEEP_DIR / "theme_signals_fine.parquet", index=False)
    rets = futures_fine_returns(sorted(registry["product"].unique()))

    print("A. 细分时段吸收")
    tab_a = fine_absorption(theme_fine, rets)
    tab_a.to_parquet(DEEP_DIR / "fine_absorption.parquet", index=False)
    fig_fine_absorption(tab_a)

    print("B. 散点矩阵")
    fig_scatter_grid(theme_fine, rets)

    print("C. 聚类 / 因子结构")
    stats = clustering(market_sig, registry)
    print(stats.to_string(index=False))

    print("D. 叠加效应")
    tab_d = stacking(theme_fine, rets)
    print(tab_d.round(3).to_string(index=False))

    print("E. 窗口敏感性")
    tab_e = horizon_curves(registry, trades, rets)
    rolling_and_split(theme_fine, rets)

    print("F. 反向传导")
    tab_f = reverse_direction(theme_fine, rets)
    print(tab_f.round(3).to_string(index=False))

    print("G. 事件研究")
    tab_g = event_study(registry, trades)

    print("H. 局部投影 IRF")
    tab_h = local_projection_irf(theme_fine, rets)

    print(f"\n产物：{DEEP_DIR} 与 {FIG_DIR}")
    _ = (tab_e, tab_g, tab_h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
