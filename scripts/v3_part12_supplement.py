"""第一、二部分的答辩级补充统计与窗口级组合信号。

按答辩篇（第三部分）确立的标准补齐主报告第一、二部分缺失的统计基本功，
并按导师"尝试各种组合"的要求首次在窗口级构造组合信号族。全部计算复用
第一部分的既有协议（``analyze_cn_futures_polymarket`` 的窗口收益、
Newey-West HAC、BH-FDR），样本为扩展窗口 2026-01-05 .. 07-13（125 个
交易日，v3 统一 tape 口径），与第二部分 §17 的扩展样本复核一致。

产出（``data/cn_futures/analysis/v3/supp/``）：

1.  ``signal_stats_window.parquet``   窗口信号（主题层）覆盖率与取值统计
2.  ``fig_supp_signal_dist.png``      头部主题 x 窗口的分布直方图
3.  ``window_ic.parquet``             Pearson / RankIC / 月度 ICIR 总表
4.  ``absorption_subsample.parquet``  同期吸收的冲突期内外子样本
5.  ``combo_signals.parquet``         W1-W10 窗口级组合信号面板
6.  ``combo_ic.parquet``              组合检验总表（同协议 + 族内 FDR）
7.  ``event_cooccur.parquet``         离散事件共现（双主题同夜 E1 等）
8.  ``event_monthly.parquet``         E1/E2/E3 月度频数（扩展窗口）
9.  ``v3_signal_stats.parquet``       第二部分测量层信号统计
10. ``meta_supp.json``                样本口径、族规模、阈值登记

用法::

    .venv/bin/python scripts/v3_part12_supplement.py [--skip-events]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_cn_futures_polymarket import (  # noqa: E402
    futures_returns,
    nw_regression,
)

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402
from alpha_data.polymarket import store as poly_store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

V3_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
SUPP_DIR = V3_DIR / "supp"
FIG_DIR = ROOT / "docs" / "figures"
REGISTRY_V3 = poly_store.FEATURES_DIR / "cn_registry_v3.parquet"

#: 冲突期子样本边界（与第三部分 §5 的口径一致）。
HOT_START, HOT_END = "2026-02-01", "2026-03-31"
#: 展开 z 分数的最小样本数（交易日）。
Z_MIN_DAYS = 20
#: 月度 IC 的最小逐月样本数。
IC_MIN_MONTH = 8
#: 组合信号阈值（展开 z 分数口径）。
TH_ACTIVE = 1.0
TH_SURPRISE = 1.5

#: 双主题组合的经济学预登记配对（供给冲击 x 地缘 / 避险 x 商品属性等，
#: 避免流动性规则选出两个高度相关的冲突主题造成重复计数）。未列品种回退
#: 到流动性 top2。
ECON_PAIRS: dict[str, tuple[str, str]] = {
    "SC": ("mideast_conflict", "oil_price"),
    "AU": ("mideast_conflict", "metal_price"),
    "AG": ("metal_price", "fed_policy"),
    "CU": ("us_china_trade", "fed_policy"),
}

WINDOW_SIGS = ("s_night", "s_gap", "s_day", "s_pre")


# ------------------------------------------------------------------ 工具
def expanding_z(s: pd.Series, min_periods: int = Z_MIN_DAYS) -> pd.Series:
    """展开窗口 z 分数（含当日，仅用 <=t 的信息，无前视）。"""
    mu = s.expanding(min_periods).mean()
    sd = s.expanding(min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def expanding_rank(s: pd.Series, min_periods: int = Z_MIN_DAYS) -> pd.Series:
    """展开窗口分位秩（0..1，含当日）。"""
    def _rank(a: np.ndarray) -> float:
        v = a[~np.isnan(a)]
        if len(v) < min_periods:
            return np.nan
        return float((v <= v[-1]).mean())
    return s.expanding(min_periods).apply(lambda a: _rank(np.asarray(a)), raw=True)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q 值（与第一部分 corr_tests 的实现一致）。"""
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    q = np.full(m, np.nan)
    cummin = 1.0
    for rank_pos in range(m - 1, -1, -1):
        i = order[rank_pos]
        cummin = min(cummin, p[i] * m / (rank_pos + 1))
        q[i] = cummin
    return q


def test_pair(
    df: pd.DataFrame, sig: str, ret: str, maxlags: int
) -> dict | None:
    """单配对检验：Pearson / Spearman / HAC + 月度 IC 序列与 ICIR。"""
    from scipy import stats as sps

    s = df[sig].to_numpy(dtype="float64")
    r = df[ret].to_numpy(dtype="float64")
    ok = ~(np.isnan(s) | np.isnan(r))
    n = int(ok.sum())
    if n < 20 or np.nanstd(s[ok]) == 0:
        return None
    pear = sps.pearsonr(s[ok], r[ok])
    spear = sps.spearmanr(s[ok], r[ok])
    beta, t_hac, p_hac, _ = nw_regression(r, s, maxlags)
    sub = df.loc[ok, ["trade_date", sig, ret]].copy()
    sub["month"] = sub["trade_date"].str[:7]
    ics: list[float] = []
    for _, g in sub.groupby("month"):
        if len(g) >= IC_MIN_MONTH and g[sig].std() > 0:
            ics.append(float(sps.spearmanr(g[sig], g[ret]).statistic))
    icir = float(np.mean(ics) / np.std(ics, ddof=1)) if len(ics) >= 3 else np.nan
    return {
        "signal": sig, "ret": ret, "n": n,
        "pearson": float(pear.statistic), "p_pearson": float(pear.pvalue),
        "rank_ic": float(spear.statistic), "p_rank": float(spear.pvalue),
        "beta": beta, "t_hac": t_hac, "p_hac": p_hac,
        "bp_per_sd": beta * float(np.nanstd(s[ok])) * 1e4
        if np.isfinite(beta) else np.nan,
        "n_months": len(ics),
        "ic_mean": float(np.mean(ics)) if ics else np.nan,
        "ic_std": float(np.std(ics, ddof=1)) if len(ics) >= 3 else np.nan,
        "icir": icir,
        "n_pos_months": int(sum(x > 0 for x in ics)),
    }


# ------------------------------------------------- 1. 窗口信号统计与分布
def signal_stats(theme_tab: pd.DataFrame) -> pd.DataFrame:
    """主题层（去品种复制）窗口信号的覆盖率与取值统计。"""
    n_days = theme_tab["trade_date"].nunique()
    rows: list[dict] = []
    for theme, g in theme_tab.groupby("theme"):
        for sig in WINDOW_SIGS:
            v = g[sig].dropna()
            sd = float(v.std()) if len(v) > 1 else np.nan
            usdc_col = {"s_night": "usdc_night", "s_gap": "usdc_gap",
                        "s_day": "usdc_day"}.get(sig)
            u = g[usdc_col].dropna() if usdc_col else pd.Series(dtype=float)
            rows.append({
                "theme": theme, "signal": sig,
                "n_days": n_days, "n_valid": int(len(v)),
                "coverage": float(len(v) / n_days),
                "mean": float(v.mean()) if len(v) else np.nan,
                "std": sd,
                "q01": float(v.quantile(0.01)) if len(v) else np.nan,
                "q25": float(v.quantile(0.25)) if len(v) else np.nan,
                "q50": float(v.median()) if len(v) else np.nan,
                "q75": float(v.quantile(0.75)) if len(v) else np.nan,
                "q99": float(v.quantile(0.99)) if len(v) else np.nan,
                "skew": float(v.skew()) if len(v) > 2 else np.nan,
                "kurt": float(v.kurt()) if len(v) > 3 else np.nan,
                "share_zero": float((v == 0).mean()) if len(v) else np.nan,
                "extreme_rate": float((v.abs() > 3 * sd).mean())
                if len(v) and np.isfinite(sd) else np.nan,
                "usdc_q50": float(u.median()) if len(u) else np.nan,
                "usdc_q90": float(u.quantile(0.9)) if len(u) else np.nan,
            })
    return pd.DataFrame(rows)


def fig_signal_dist(theme_tab: pd.DataFrame) -> None:
    """头部主题 x 窗口信号的直方图网格。"""
    themes = ["mideast_conflict", "oil_price", "metal_price", "fed_policy"]
    cn = {"mideast_conflict": "中东冲突", "oil_price": "油价",
          "metal_price": "金属价格", "fed_policy": "美联储"}
    sigs = ("s_night", "s_gap", "s_day")
    fig, axes = plt.subplots(len(themes), len(sigs),
                             figsize=(11, 10), sharex="col")
    for i, th in enumerate(themes):
        g = theme_tab[theme_tab["theme"] == th]
        for j, sig in enumerate(sigs):
            ax = axes[i, j]
            v = g[sig].dropna()
            if len(v):
                ax.hist(v, bins=30, color="#4878a8", edgecolor="white")
                ax.axvline(0, color="#c44", lw=0.8)
                ax.set_title(
                    f"{cn[th]} {sig}  n={len(v)} "
                    f"μ={v.mean():+.3f} σ={v.std():.3f}",
                    fontsize=8,
                )
            ax.tick_params(labelsize=7)
    fig.suptitle("窗口级主题信号分布（扩展样本 125 交易日，主题层去品种复制）",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_supp_signal_dist.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------ 2. RankIC / ICIR
def window_ic(theme_sig: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """全部主题 x 品种配对的 Pearson / RankIC / 月度 ICIR 总表。"""
    merged = theme_sig.merge(rets, on=["product", "trade_date"], how="inner")
    rows: list[dict] = []
    for (theme, product), block in merged.groupby(["theme", "product"]):
        block = block.sort_values("trade_date")
        block = block[~block["roll"].fillna(False)]
        specs = [
            ("contemporaneous", "s_night", "r_night", 1),
            ("contemporaneous", "s_gap", "r_gap_total", 1),
            ("contemporaneous", "s_day", "r_day", 1),
            ("predictive", "s_pre", "r_day", 1),
        ]
        for kind, sig, ret, lags in specs:
            rec = test_pair(block, sig, ret, lags)
            if rec is not None:
                rows.append({"theme": theme, "product": product,
                             "kind": kind, **rec})
    return pd.DataFrame(rows)


# ------------------------------------------------- 3. 吸收子样本（冲突期）
def absorption_subsample(
    theme_sig: pd.DataFrame, rets: pd.DataFrame
) -> pd.DataFrame:
    """同期吸收在冲突期（2026-02-01..03-31）内外的分段检验。"""
    merged = theme_sig.merge(rets, on=["product", "trade_date"], how="inner")
    merged = merged[~merged["roll"].fillna(False)]
    hot = (merged["trade_date"] >= HOT_START) & (merged["trade_date"] <= HOT_END)
    rows: list[dict] = []
    for (theme, product), block in merged.groupby(["theme", "product"]):
        for subset, mask in (("all", np.ones(len(block), bool)),
                             ("hot", hot.loc[block.index].to_numpy()),
                             ("cold", ~hot.loc[block.index].to_numpy())):
            sub = block[mask].sort_values("trade_date")
            for sig, ret in (("s_night", "r_night"), ("s_gap", "r_gap_total"),
                             ("s_day", "r_day")):
                rec = test_pair(sub, sig, ret, 1)
                if rec is not None:
                    rows.append({"theme": theme, "product": product,
                                 "subset": subset, **rec})
    return pd.DataFrame(rows)


# ------------------------------------------------- 4. 窗口级组合信号 W1-W10
def build_combos(
    theme_sig: pd.DataFrame, rets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """W1-W10 组合信号面板与检验总表。

    设计三类（对应导师"尝试各种组合"的要求）：

    - 共现类：W1 双主题同向共现、W2 双主题分歧、W3 信念广度、W5 双 surprise
      共现。
    - 交互类：W6 信号 x 前日波动状态、W7 信号 x 当窗资金流、W8 信号 x 兄弟
      品种夜盘价格确认。
    - 结构类：W4 多主题复合信念、W9 五日未兑现缺口（第三部分 C8 的窗口级
      对应物）、W10 窗口内符号一致性。

    归一化：主题信号取展开窗口 z 分数（min 20 日，仅用 <=t 信息）；阈值
    |z|>1（活跃）、|z|>1.5（surprise）。检验协议与第一部分一致
    （Pearson/RankIC/HAC/月度 ICIR + 族内 BH-FDR）。
    """
    # 主题 x 品种映射与 top2 主题（按窗口累计成交额，声明为结构选择）
    pair_usdc = (
        theme_sig.assign(u=lambda d: d[["usdc_night", "usdc_gap", "usdc_day"]]
                         .sum(axis=1))
        .groupby(["product", "theme"])["u"].sum().reset_index()
    )
    top2: dict[str, list[str]] = {}
    pair_rule: dict[str, str] = {}
    for product, g in pair_usdc.groupby("product"):
        avail = set(g["theme"])
        econ = ECON_PAIRS.get(product)
        if econ is not None and set(econ) <= avail:
            top2[product] = list(econ)
            pair_rule[product] = "econ_registered"
        else:
            top2[product] = list(
                g.sort_values("u", ascending=False)["theme"].head(2))
            pair_rule[product] = "liquidity_top2"

    # 主题层信号宽表（品种间复制，取主题层）+ 展开 z
    theme_tab = theme_sig.drop_duplicates(["theme", "trade_date"]).copy()
    wide: dict[str, pd.DataFrame] = {}
    for theme, g in theme_tab.groupby("theme"):
        g = g.sort_values("trade_date").set_index("trade_date")
        z = pd.DataFrame(index=g.index)
        for sig in WINDOW_SIGS:
            z[f"z_{sig[2:]}"] = expanding_z(g[sig])
        z["usdc_gap"] = g["usdc_gap"]
        wide[theme] = z

    # W12 输入：逐品种夜盘成交额（元），索引 trade_date
    daily_money: dict[str, pd.Series] = {}
    for product in theme_sig["product"].unique():
        d = fut_store.read_daily(product).set_index("trade_date")
        daily_money[product] = d["night_money"]

    # W13 输入：主题内逐市场方向一致率 A = |Σ orient·sign(s_gap)| / n
    consensus: dict[tuple[str, str], pd.Series] = {}
    ms_path = V3_DIR.parent / "market_signals.parquet"
    reg_path = poly_store.FEATURES_DIR / "cn_registry.parquet"
    if ms_path.exists() and reg_path.exists():
        ms = pd.read_parquet(ms_path)
        reg = pd.read_parquet(reg_path)[
            ["condition_id", "product", "theme", "orientation"]
        ].drop_duplicates(["condition_id", "product"])
        ms = ms.merge(reg, on=["condition_id", "product"], how="inner")
        ms = ms[ms["s_gap"].notna() & (ms["s_gap"] != 0)]
        grp = ms.groupby(["theme", "product", "trade_date"]).apply(
            lambda g: float(abs((g["orientation"] * np.sign(g["s_gap"])).sum())
                            / len(g)), include_groups=False)
        for (theme, product), s in grp.groupby(level=[0, 1]):
            consensus[(theme, product)] = s.droplevel([0, 1])

    # 兄弟品种：同主题下成交额次大的有夜盘品种（用于 W8 的价格确认）
    night_products = set(
        fut_store.read_product_specs().dropna(subset=["night_end"])["product"]
    )
    sibling: dict[str, str | None] = {}
    for product in top2:
        main_theme = top2[product][0]
        cand = (pair_usdc[(pair_usdc["theme"] == main_theme)
                          & (pair_usdc["product"] != product)
                          & (pair_usdc["product"].isin(night_products))]
                .sort_values("u", ascending=False))
        sibling[product] = cand["product"].iloc[0] if len(cand) else None

    ret_wide = rets.pivot_table(index="trade_date", columns="product",
                                values="r_night")

    panels: list[pd.DataFrame] = []
    for product in sorted(theme_sig["product"].unique()):
        themes = top2.get(product, [])
        if not themes:
            continue
        a = wide[themes[0]]
        b = wide[themes[1]] if len(themes) > 1 else None
        r = (rets[rets["product"] == product]
             .sort_values("trade_date").set_index("trade_date"))
        idx = r.index
        p = pd.DataFrame(index=idx)
        p["product"] = product
        za_gap = a["z_gap"].reindex(idx)
        za_pre = a["z_pre"].reindex(idx)
        if b is not None:
            zb_gap = b["z_gap"].reindex(idx)
            zb_pre = b["z_pre"].reindex(idx)
            agree = np.sign(za_gap) == np.sign(zb_gap)
            p["W1"] = np.where(
                agree & za_gap.notna() & zb_gap.notna(),
                np.sign(za_gap) * np.minimum(za_gap.abs(), zb_gap.abs()), 0.0)
            p.loc[za_gap.isna() | zb_gap.isna(), "W1"] = np.nan
            p["W2"] = (za_gap - zb_gap).abs()
            both = za_pre.notna() & zb_pre.notna()
            hit = (za_pre.abs() > TH_SURPRISE) & (zb_pre.abs() > TH_SURPRISE)
            p["W5"] = np.where(both & hit, np.sign(za_pre + zb_pre), 0.0)
            p.loc[~both, "W5"] = np.nan
        # W3 广度 / W4 复合信念：全部映射主题
        mapped = list(pair_usdc[pair_usdc["product"] == product]["theme"])
        zmat = pd.DataFrame({t: wide[t]["z_gap"].reindex(idx) for t in mapped})
        umat = pd.DataFrame({t: wide[t]["usdc_gap"].reindex(idx) for t in mapped})
        p["W3"] = (zmat.abs() > TH_ACTIVE).sum(axis=1).where(
            zmat.notna().any(axis=1)) / zmat.notna().sum(axis=1)
        w = np.sqrt(umat.clip(lower=0.0))
        p["W4"] = (zmat * w).sum(axis=1) / w.where(zmat.notna()).sum(axis=1)
        # W6 波动状态交互 / W7 资金流交互
        rv_state = (r["r_cc"].abs().shift(1)
                    > r["r_cc"].abs().shift(1).expanding(Z_MIN_DAYS).median())
        p["W6"] = za_pre * rv_state.astype(float).where(
            r["r_cc"].shift(1).notna())
        p["W7"] = za_gap * expanding_rank(a["usdc_gap"].reindex(idx))
        # W8 兄弟品种夜盘价格确认
        sib = sibling.get(product)
        if sib is not None and sib in ret_wide.columns:
            sib_night = ret_wide[sib].reindex(idx)
            confirm = np.sign(sib_night) == np.sign(za_gap)
            p["W8"] = za_gap * confirm.astype(float).where(
                sib_night.notna() & za_gap.notna())
            # W8 的两个归因对照：兄弟品种夜盘收益单独、纯 PM 信号单独
            p["W8_ctrl_sib"] = sib_night
            p["W8_ctrl_pm"] = za_gap
        # W11 隔夜未兑现缺口：信念已动而本品种夜盘价格未动的部分（09:00 可知）
        if r["r_night"].notna().any():
            p["W11"] = za_pre - expanding_z(r["r_night"])
        # W12 期货量能交互：夜盘成交额高于展开 75 分位的窗口才计入信念创新
        money = daily_money.get(product)
        if money is not None:
            mz = money.reindex(idx)
            hi = mz > mz.expanding(Z_MIN_DAYS).quantile(0.75)
            p["W12"] = za_gap * hi.astype(float).where(mz.notna())
        # W13 市场级共识度：主主题内逐市场方向一致率加权
        cons_a = consensus.get((themes[0], product))
        if cons_a is not None:
            p["W13"] = za_gap * cons_a.reindex(idx)
        # W9 五日未兑现缺口（C8 的窗口级对应物）
        s_all = (a["z_night"].reindex(idx).fillna(0.0)
                 + a["z_gap"].reindex(idx).fillna(0.0)
                 + a["z_day"].reindex(idx).fillna(0.0))
        n_valid5 = (a[["z_night", "z_gap", "z_day"]].notna().any(axis=1)
                    .reindex(idx).fillna(False).rolling(5).sum())
        cum_s = s_all.rolling(5, min_periods=3).sum().where(n_valid5 >= 3)
        cum_r = r["r_cc"].fillna(0.0).rolling(5, min_periods=3).sum()
        p["W9"] = expanding_z(cum_s) - expanding_z(cum_r)
        # W10 窗口内符号一致性
        zs = a[["z_night", "z_gap", "z_day"]].reindex(idx)
        c1 = (np.sign(zs["z_night"]) == np.sign(zs["z_gap"])).astype(float)
        c2 = (np.sign(zs["z_gap"]) == np.sign(zs["z_day"])).astype(float)
        cons = np.where(zs["z_night"].notna(), (c1 + c2) / 2.0, c2)
        p["W10"] = cons * np.sign(zs["z_day"]) * zs.abs().mean(axis=1)
        p.loc[zs["z_gap"].isna() | zs["z_day"].isna(), "W10"] = np.nan
        # 目标收益
        for col in ("r_day", "r_cc", "roll"):
            p[col] = r[col]
        p["abs_r_day"] = r["r_day"].abs()
        p["r_cc_next"] = r["r_cc"].shift(-1)
        p["theme_a"] = themes[0]
        p["theme_b"] = themes[1] if len(themes) > 1 else ""
        panels.append(p.reset_index())

    combo_panel = pd.concat(panels, ignore_index=True)
    combo_panel = combo_panel[~combo_panel["roll"].fillna(False)]

    #: 组合 -> (检验目标, kind, 方向假设说明)
    combo_specs: dict[str, list[tuple[str, str]]] = {
        "W1": [("r_day", "predictive")],
        "W2": [("abs_r_day", "risk"), ("r_day", "predictive")],
        "W3": [("abs_r_day", "risk"), ("r_day", "predictive")],
        "W4": [("r_day", "predictive")],
        "W5": [("r_day", "predictive"), ("abs_r_day", "risk")],
        "W6": [("r_day", "predictive")],
        "W7": [("r_day", "predictive")],
        "W8": [("r_day", "predictive")],
        "W8_ctrl_sib": [("r_day", "control")],
        "W8_ctrl_pm": [("r_day", "control")],
        "W9": [("r_cc_next", "predictive")],
        "W10": [("r_cc_next", "predictive")],
        "W11": [("r_day", "predictive")],
        "W12": [("r_day", "predictive")],
        "W13": [("r_day", "predictive")],
    }
    rows: list[dict] = []
    for product, block in combo_panel.groupby("product"):
        block = block.sort_values("trade_date")
        for combo, specs in combo_specs.items():
            if combo not in block.columns:
                continue
            for ret, kind in specs:
                rec = test_pair(block, combo, ret, 1)
                if rec is not None:
                    rows.append({"combo": combo, "product": product,
                                 "kind": kind,
                                 "theme_a": block["theme_a"].iloc[0],
                                 "theme_b": block["theme_b"].iloc[0], **rec})
    combo_ic = pd.DataFrame(rows)
    if not combo_ic.empty:
        # FDR 族 = 预测与风险行；对照行只作归因参考，不进族。
        fam = combo_ic["kind"] != "control"
        q = np.full(len(combo_ic), np.nan)
        q[fam.to_numpy()] = bh_fdr(combo_ic.loc[fam, "p_hac"].to_numpy())
        combo_ic["q_bh"] = q
    meta = {
        "top2_themes": top2,
        "pair_rule": pair_rule,
        "sibling": {k: (v or "") for k, v in sibling.items()},
        "family_size": int((combo_ic["kind"] != "control").sum())
        if not combo_ic.empty else 0,
    }
    return combo_panel, combo_ic, meta


# ------------------------------------------- 5. 离散事件共现与事件月频
def detect_events(
    registry: pd.DataFrame, con: object
) -> pd.DataFrame:
    """扩展 tape 上重算 E1/E2/E3 事件时刻（与第一部分 §10 定义一致）。

    主题限 SC 相关（oil_price / mideast_conflict / russia_ukraine），阈值与
    ``deep_analysis_cn_polymarket.event_study`` 完全相同。
    """
    themes = ("oil_price", "mideast_conflict", "russia_ukraine")
    sub = registry[(registry["theme"].isin(themes))
                   & (registry["product"] == "SC")]
    events: list[dict] = []
    recs = list(sub.drop_duplicates("condition_id").itertuples(index=False))
    for i, rec in enumerate(recs, 1):
        tr = tape.fetch_trades_cn(con, rec.condition_id)
        if tr is None or tr.empty:
            continue
        first_ts = pd.to_datetime(
            tr["cn_ts"].iloc[0]) if "cn_ts" in tr else None
        agg = cn_features.aggregate_price(tr, agg_minutes=15)
        if len(agg) >= 60:
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
                events.append({"type": "E1", "theme": rec.theme, "ts": ts,
                               "sign": float(rec.orientation * np.sign(dlo[ts])),
                               "condition_id": rec.condition_id})
            for ts in lo.index[e2.fillna(False)]:
                sgn = np.sign(dlo[ts]) if pd.notna(dlo[ts]) and dlo[ts] != 0 else 0.0
                events.append({"type": "E2", "theme": rec.theme, "ts": ts,
                               "sign": float(rec.orientation * sgn),
                               "condition_id": rec.condition_id})
        if first_ts is not None:
            # E3 符号 = orientation × sign(首桶 Δlogit，缺省 +1)——修正第一部分
            # 代码中硬编码 +1 与声明口径不符的缺陷。
            e3_sgn = 1.0
            if len(agg) >= 2:
                lo0 = cn_features.logit(agg["p_agg"].to_numpy()[:2])
                d0 = float(lo0[1] - lo0[0])
                if d0 != 0 and np.isfinite(d0):
                    e3_sgn = float(np.sign(d0))
            events.append({"type": "E3", "theme": rec.theme, "ts": first_ts,
                           "sign": float(rec.orientation) * e3_sgn,
                           "condition_id": rec.condition_id})
        if i % 50 == 0:
            print(f"  事件检测 {i}/{len(recs)}")
    ev = pd.DataFrame(events)
    return ev[ev["sign"] != 0].reset_index(drop=True)


def event_cooccur(ev: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """离散事件共现的次日响应（导师"两个事件同时发生"的窗口级实现）。

    共现定义（事件归属交易日 = SC 三窗口边界所属交易日）：

    - V1 双主题同夜 E1 同向：oil 与 mideast 在同一夜盘窗口各有 >=1 个 E1
      且净符号同向。
    - V2 价量共振：同一市场同一 15min 桶内同时触发 E1 与 E2。
    - V3 单主题 E1（对照组）：仅一个主题有 E1。

    响应 = 事件归属交易日的 r_day（夜盘 / 缺口事件）符号化收益，与无条件
    基线对比；n 极小时如实报告，不做显著性声称。
    """
    specs = fut_store.read_product_specs().set_index("product")
    ne = sessions.parse_night_end(specs.loc["SC", "night_end"])
    days = list(fut_store.read_daily("SC")["trade_date"])
    win = sessions.signal_windows(days, night_end=ne)

    # 事件时刻 -> 归属交易日（夜盘或 gap 归当日，日盘内事件归次日避免同期）
    bounds = win.dropna(subset=["night_start"])
    ev = ev.copy()
    ev["trade_date"] = None
    for rec in bounds.itertuples(index=False):
        m = (ev["ts"] >= rec.night_start) & (ev["ts"] < rec.day_start)
        ev.loc[m, "trade_date"] = rec.trade_date
    ev = ev.dropna(subset=["trade_date"])

    e1 = ev[ev["type"] == "E1"]
    net = (e1.groupby(["trade_date", "theme"])["sign"].sum()
           .unstack(fill_value=0.0))
    for t in ("oil_price", "mideast_conflict"):
        if t not in net.columns:
            net[t] = 0.0
    both = (net["oil_price"] != 0) & (net["mideast_conflict"] != 0)
    same = both & (np.sign(net["oil_price"]) == np.sign(net["mideast_conflict"]))
    single = (net["oil_price"] != 0) ^ (net["mideast_conflict"] != 0)

    # V2 价量共振：同市场同桶 E1+E2
    key = ev.assign(bucket=ev["ts"].astype(str))
    e1k = set(map(tuple, key[key["type"] == "E1"]
                  [["condition_id", "bucket"]].to_numpy()))
    e2k = set(map(tuple, key[key["type"] == "E2"]
                  [["condition_id", "bucket"]].to_numpy()))
    resonance_dates = sorted(
        {key.loc[(key["condition_id"] == c) & (key["bucket"] == b),
                 "trade_date"].iloc[0]
         for (c, b) in (e1k & e2k)})

    # V4 新市场创建与同主题 E1 同一夜盘窗口共现
    e3 = ev[ev["type"] == "E3"]
    e3_keys = set(zip(e3["trade_date"], e3["theme"], strict=False))
    e1_keys = set(zip(e1["trade_date"], e1["theme"], strict=False))
    v4_pairs = e3_keys & e1_keys
    v4_dates = sorted({d for (d, _th) in v4_pairs})
    v4_sign = (e1[e1[["trade_date", "theme"]].apply(tuple, axis=1)
                  .isin(v4_pairs)]
               .groupby("trade_date")["sign"].sum())

    r_sc = (rets[rets["product"] == "SC"]
            .set_index("trade_date")["r_day"] * 1e4)
    base = r_sc.dropna()

    def sgn_ret(dates: pd.Index, signs: pd.Series | None) -> pd.Series:
        r = r_sc.reindex(dates)
        if signs is not None:
            r = r * np.sign(signs.reindex(dates))
        return r.dropna()

    rows: list[dict] = []
    for name, dates, signs in (
        ("V1_dual_theme_same_sign", net.index[same], net["oil_price"][same]),
        ("V3_single_theme", net.index[single],
         (net["oil_price"] + net["mideast_conflict"])[single]),
        ("V2_price_volume_resonance", pd.Index(resonance_dates), None),
        ("V4_e3_with_e1_same_theme", pd.Index(v4_dates), v4_sign),
        ("baseline_all_days", base.index, None),
    ):
        v = sgn_ret(dates, signs) if name != "baseline_all_days" else base
        if not len(v):
            rows.append({"group": name, "n": 0})
            continue
        rows.append({
            "group": name, "n": int(len(v)),
            "mean_bp": float(v.mean()), "median_bp": float(v.median()),
            "std_bp": float(v.std()) if len(v) > 1 else np.nan,
            "t_stat": float(v.mean() / v.std() * np.sqrt(len(v)))
            if len(v) > 2 and v.std() > 0 else np.nan,
            "share_pos": float((v > 0).mean()),
        })
    return pd.DataFrame(rows)


def event_monthly(ev: pd.DataFrame) -> pd.DataFrame:
    """E1/E2/E3 的月度频数（扩展窗口，自然月）。"""
    ev = ev.copy()
    ev["month"] = ev["ts"].dt.strftime("%Y-%m")
    out = (ev.groupby(["month", "type"]).size().rename("n").reset_index()
           .pivot_table(index="month", columns="type", values="n",
                        fill_value=0).reset_index())
    return out


# ------------------------------------------- 5b. 修正版事件研究（CAR）
def event_car(ev: pd.DataFrame) -> pd.DataFrame:
    """事件后 SC 夜盘 1 分钟 bar 的符号化 CAR（修正第一部分 §10 两处缺陷）。

    修正内容：(a) 视界按真实 1 分钟 bar 计（原实现取 25 根 1 分钟 bar 却标注
    0-120 分钟）；(b) E3 已按 orientation × sign(首桶 Δlogit) 符号化（原实现
    硬编码 +1）；(c) 增加无条件基线——漂移基线 μ1×h 与全部有效同长窗口的
    无条件 CAR 分布，报告超额。
    """
    minute = fut_store.read_minute("SC")
    night = minute[minute["session"] == "night"]
    px = night.set_index("ts")["close"].sort_index()
    logret = np.log(px / px.shift(1))
    gaps = px.index.to_series().diff() > pd.Timedelta(minutes=2)
    logret[gaps] = np.nan
    cum = logret.fillna(0).cumsum()
    base_idx = px.index
    horizons = (5, 15, 30, 60, 120)
    mu1 = float(logret.dropna().mean())

    # 无条件基线：全部"事件后 h 分钟完整落在夜盘内"的窗口 CAR 分布
    uncond: dict[int, float] = {}
    for h in horizons:
        pos_all = np.arange(len(base_idx) - h)
        ok = (base_idx[pos_all + h] - base_idx[pos_all]) <= pd.Timedelta(
            minutes=h + 10)
        car_all = (cum.to_numpy()[pos_all + h] - cum.to_numpy()[pos_all])[ok]
        uncond[h] = float(np.mean(car_all))

    rng = np.random.default_rng(20260717)
    rows: list[dict] = []
    for etype, g in ev.groupby("type"):
        pos = base_idx.searchsorted(g["ts"].to_numpy())
        for h in horizons:
            valid = (pos > 0) & (pos + h < len(base_idx))
            if not valid.any():
                continue
            p0 = pos[valid]
            in_night = np.asarray(
                (base_idx[p0 + h] - base_idx[p0])
                <= pd.Timedelta(minutes=h + 10))
            p0 = p0[in_night]
            sgn = g["sign"].to_numpy()[valid][in_night]
            if len(p0) < 5:
                continue
            car = (cum.to_numpy()[p0 + h] - cum.to_numpy()[p0]) * sgn
            boot = np.array([
                rng.choice(car, size=len(car), replace=True).mean()
                for _ in range(500)])
            share_pos = float((sgn > 0).mean())
            drift = (2 * share_pos - 1) * uncond[h]
            rows.append({
                "type": etype, "h_min": h, "n": int(len(car)),
                "car_bp": float(car.mean()) * 1e4,
                "lo": float(np.quantile(boot, 0.05)) * 1e4,
                "hi": float(np.quantile(boot, 0.95)) * 1e4,
                "drift_bp": mu1 * h * 1e4,
                "uncond_bp": uncond[h] * 1e4,
                "share_pos_sign": share_pos,
                "excess_bp": (float(car.mean()) - drift) * 1e4,
            })
    return pd.DataFrame(rows)


# ------------------------------------------- 6. 期货侧统计与涨跌停代理
def futures_window_stats(products: list[str]) -> pd.DataFrame:
    """期货窗口收益（回归因变量）的分布统计与换月剔除计数。"""
    rows: list[dict] = []
    for product in products:
        d = fut_store.read_daily(product)
        d["r_gap_total"] = d["r_gap_pm"] + d["r_gap_am"]
        if d["r_night"].isna().all():
            d["r_gap_total"] = d["r_gap_full"]
        n_roll = int(d["roll"].fillna(False).sum())
        keep = d[~d["roll"].fillna(False)]
        for col in ("r_night", "r_gap_total", "r_day", "r_cc"):
            v = keep[col].dropna() * 1e4
            if not len(v):
                continue
            rows.append({
                "product": product, "window": col, "n": int(len(v)),
                "n_roll_excl": n_roll,
                "mean_bp": float(v.mean()), "std_bp": float(v.std()),
                "q01_bp": float(v.quantile(0.01)),
                "q50_bp": float(v.median()),
                "q99_bp": float(v.quantile(0.99)),
                "skew": float(v.skew()),
            })
    return pd.DataFrame(rows)


def locked_proxy(products: list[str]) -> pd.DataFrame:
    """涨跌停代理：日盘 / 夜盘收尾 15 分钟内 high==low 锁死分钟占比。

    主力连续无涨跌停价表，按第三部分 F4 的口径用锁死分钟作代理。
    """
    rows: list[dict] = []
    for product in products:
        m = fut_store.read_minute(product)
        m = m.assign(locked=(m["high"] == m["low"]))
        for (td, sess), g in m.groupby(["trade_date", "session"]):
            tail = g.sort_values("ts").tail(15)
            rows.append({
                "product": product, "trade_date": td, "session": sess,
                "locked_share_tail15": float(tail["locked"].mean()),
                "locked_share_all": float(g["locked"].mean()),
            })
    return pd.DataFrame(rows)


# ------------------------------------- 7. 双主题交互回归与二维分位表
def interaction_tests(
    theme_sig: pd.DataFrame, rets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """乘积交互回归 r ~ z_a + z_b + z_a·z_b（HAC）与 3×3 分位表。"""
    import statsmodels.api as sm

    theme_tab = theme_sig.drop_duplicates(["theme", "trade_date"])
    wide: dict[str, pd.DataFrame] = {}
    for theme, g in theme_tab.groupby("theme"):
        g = g.sort_values("trade_date").set_index("trade_date")
        wide[theme] = pd.DataFrame({
            "z_night": expanding_z(g["s_night"]),
            "z_gap": expanding_z(g["s_gap"]),
        })
    pairs = [("mideast_conflict", "oil_price", "SC"),
             ("mideast_conflict", "metal_price", "AU")]
    reg_rows: list[dict] = []
    for ta, tb, product in pairs:
        r = (rets[rets["product"] == product]
             .sort_values("trade_date").set_index("trade_date"))
        r = r[~r["roll"].fillna(False)]
        for zcol, rcol in (("z_night", "r_night"), ("z_gap", "r_gap_total")):
            za = wide[ta][zcol].reindex(r.index)
            zb = wide[tb][zcol].reindex(r.index)
            y = r[rcol]
            ok = za.notna() & zb.notna() & y.notna()
            n = int(ok.sum())
            if n < 25:
                reg_rows.append({"theme_a": ta, "theme_b": tb,
                                 "product": product, "window": rcol,
                                 "n": n, "note": "n<25 不估计"})
                continue
            x = np.column_stack([za[ok], zb[ok], (za * zb)[ok]])
            res = sm.OLS(y[ok].to_numpy(), sm.add_constant(x)).fit(
                cov_type="HAC", cov_kwds={"maxlags": 1})
            reg_rows.append({
                "theme_a": ta, "theme_b": tb, "product": product,
                "window": rcol, "n": n,
                "b_a": float(res.params[1]), "t_a": float(res.tvalues[1]),
                "b_b": float(res.params[2]), "t_b": float(res.tvalues[2]),
                "b_int": float(res.params[3]), "t_int": float(res.tvalues[3]),
                "r2": float(res.rsquared), "note": "",
            })
    # 3×3 分位表：mideast × oil 的 s_gap 三分位交叉 → SC r_gap_total
    r_sc = (rets[rets["product"] == "SC"]
            .sort_values("trade_date").set_index("trade_date"))
    r_sc = r_sc[~r_sc["roll"].fillna(False)]
    za = wide["mideast_conflict"]["z_gap"].reindex(r_sc.index)
    zb = wide["oil_price"]["z_gap"].reindex(r_sc.index)
    y = r_sc["r_gap_total"] * 1e4
    ok = za.notna() & zb.notna() & y.notna()
    qa = pd.qcut(za[ok], 3, labels=["低", "中", "高"])
    qb = pd.qcut(zb[ok], 3, labels=["低", "中", "高"])
    q_rows: list[dict] = []
    for (ga, gb), g in y[ok].groupby([qa, qb], observed=False):
        q_rows.append({
            "mideast_tercile": str(ga), "oil_tercile": str(gb),
            "n": int(len(g)),
            "mean_bp": float(g.mean()) if len(g) else np.nan,
            "se_bp": float(g.std() / np.sqrt(len(g))) if len(g) > 1 else np.nan,
        })
    return pd.DataFrame(reg_rows), pd.DataFrame(q_rows)


# ------------------------------------- 8. 第二部分组合：tension 门控与
#                                          episode 共现
def tension_gate(rets: pd.DataFrame) -> pd.DataFrame:
    """条件吸收：族内张力高于展开 75 分位时信念创新的吸收是否不同。"""
    import statsmodels.api as sm

    v3 = pd.read_parquet(V3_DIR / "v3_theme_signals.parquet")
    rows: list[dict] = []
    for theme, product in (("mideast_conflict", "SC"),
                           ("mideast_conflict", "AU"),
                           ("metal_price", "AU")):
        g = v3[(v3["theme"] == theme) & (v3["product"] == product)]
        if g.empty or "tension" not in g.columns:
            continue
        g = g.sort_values("trade_date").set_index("trade_date")
        r = (rets[rets["product"] == product]
             .sort_values("trade_date").set_index("trade_date"))
        r = r[~r["roll"].fillna(False)]
        z = expanding_z(g["s_gap"]).reindex(r.index)
        tn = g["tension"].reindex(r.index)
        gate = (tn > tn.expanding(Z_MIN_DAYS).quantile(0.75)).astype(float)
        y = r["r_gap_total"]
        ok = z.notna() & tn.notna() & y.notna()
        n = int(ok.sum())
        if n < 25:
            rows.append({"theme": theme, "product": product, "n": n,
                         "note": "n<25 不估计"})
            continue
        x = np.column_stack([z[ok], (z * gate)[ok], gate[ok]])
        res = sm.OLS(y[ok].to_numpy(), sm.add_constant(x)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1})
        rows.append({
            "theme": theme, "product": product, "n": n,
            "b_z": float(res.params[1]), "t_z": float(res.tvalues[1]),
            "b_zgate": float(res.params[2]), "t_zgate": float(res.tvalues[2]),
            "b_gate": float(res.params[3]), "t_gate": float(res.tvalues[3]),
            "note": "",
        })
    return pd.DataFrame(rows)


def episode_cooccur(rets: pd.DataFrame) -> pd.DataFrame:
    """episode 层的事件共现：同日多主题结算 vs 单主题日的响应对比。"""
    ep_path = V3_DIR / "v3_episodes.parquet"
    if not ep_path.exists():
        return pd.DataFrame()
    ep = pd.read_parquet(ep_path)
    date_col = "ev_date" if "ev_date" in ep.columns else "trade_date"
    rows: list[dict] = []
    for product, g in ep.groupby("product"):
        r = (rets[rets["product"] == product]
             .set_index("trade_date")["r_cc"] * 1e4)
        day = g.groupby(date_col).agg(
            n_themes=("theme", "nunique"),
            s_sum=("surprise", "sum"),
            same_sign=("surprise", lambda s: (np.sign(s) == np.sign(s.iloc[0]))
                       .all() if len(s) > 1 else True),
        )
        for name, mask in (
            ("co_same", (day["n_themes"] >= 2) & day["same_sign"]),
            ("co_opposite", (day["n_themes"] >= 2) & ~day["same_sign"]),
            ("single", day["n_themes"] == 1),
        ):
            sub = day[mask]
            v = r.reindex(sub.index).dropna()
            signed = (r.reindex(sub.index)
                      * np.sign(sub["s_sum"].replace(0, np.nan))).dropna()
            rows.append({
                "product": product, "group": name, "n_days": int(len(v)),
                "mean_abs_bp": float(v.abs().mean()) if len(v) else np.nan,
                "mean_signed_bp": float(signed.mean()) if len(signed) else np.nan,
                "share_pos": float((signed > 0).mean())
                if len(signed) else np.nan,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------- 9. RV 的样本外检验
def rv_oos(theme_sig: pd.DataFrame) -> pd.DataFrame:
    """波动通道 OOS：SC 日盘 RV，M0=HAR-lite 基线 vs M1=+|s_gap|。

    展开窗逐日重估、一步向前预测，Clark-West 单侧 t。回应审计：全文最强
    正结论（风险可测）此前从未做过样本外验证。
    """
    m = fut_store.read_minute("SC")
    day = m[m["session"] == "day"].copy()
    day["lr"] = np.log(day["close"] / day["close"].shift(1))
    day.loc[day["trade_date"] != day["trade_date"].shift(1), "lr"] = np.nan
    rv = day.groupby("trade_date")["lr"].apply(
        lambda s: float(np.sqrt(np.nansum(s ** 2)))) * 1e4
    sig = (theme_sig[(theme_sig["theme"] == "mideast_conflict")
                     & (theme_sig["product"] == "SC")]
           .drop_duplicates("trade_date").set_index("trade_date")["s_gap"])
    df = pd.DataFrame({"rv": rv})
    df["rv_lag"] = df["rv"].shift(1)
    df["abs_s"] = sig.abs().reindex(df.index).fillna(0.0)
    df = df.dropna(subset=["rv", "rv_lag"])
    n = len(df)
    if n < 60:
        return pd.DataFrame([{"n_oos": 0, "note": "样本不足"}])
    y = df["rv"].to_numpy()
    x0 = df["rv_lag"].to_numpy()
    x1 = df["abs_s"].to_numpy()
    pred0 = np.full(n, np.nan)
    pred1 = np.full(n, np.nan)
    for t in range(40, n):
        a0 = np.polyfit(x0[:t], y[:t], 1)
        pred0[t] = a0[0] * x0[t] + a0[1]
        big_x = np.column_stack([np.ones(t), x0[:t], x1[:t]])
        coef, *_ = np.linalg.lstsq(big_x, y[:t], rcond=None)
        pred1[t] = coef[0] + coef[1] * x0[t] + coef[2] * x1[t]
    ok = ~np.isnan(pred0)
    e0 = y[ok] - pred0[ok]
    e1 = y[ok] - pred1[ok]
    f = e0 ** 2 - (e1 ** 2 - (pred0[ok] - pred1[ok]) ** 2)
    cw_t = float(f.mean() / f.std(ddof=1) * np.sqrt(len(f)))
    ybar = y[ok].mean()
    r2_0 = 1 - float((e0 ** 2).sum() / ((y[ok] - ybar) ** 2).sum())
    r2_1 = 1 - float((e1 ** 2).sum() / ((y[ok] - ybar) ** 2).sum())
    return pd.DataFrame([{
        "n_oos": int(len(f)), "r2_oos_m0": r2_0, "r2_oos_m1": r2_1,
        "delta_r2": r2_1 - r2_0, "cw_t": cw_t, "note": "",
    }])


# ------------------------------------------- 10. 检验总量核算（两部分）
def test_count_audit() -> pd.DataFrame:
    """按产物行数盘点第一、二部分的检验单元总量。"""
    an = V3_DIR.parent
    inventory = [
        ("I", "吸收+预测总表（corr_table）", an / "corr_table.parquet", None),
        ("I", "细分时段吸收", an / "deep" / "fine_absorption.parquet", None),
        ("I", "国际基准控制", an / "deep" / "intl_control.parquet", None),
        ("I", "中介链系数", an / "deep" / "chain_mediation.parquet", None),
        ("I", "局部投影 IRF", an / "deep" / "irf.parquet", None),
        ("I", "安慰剂", an / "deep" / "extra_placebo.parquet", None),
        ("I", "循环置换", an / "deep" / "extra_permutation.parquet", None),
        ("I", "分位数组合", an / "deep" / "extra_quantile.parquet", None),
        ("I", "波动率回归", an / "deep" / "extra_volatility.parquet", None),
        ("I", "代理充分性", an / "deep" / "surrogate_sufficiency.parquet", None),
        ("I", "策略回测", an / "deep" / "backtest_metrics.parquet", None),
        ("II", "功效表（ex-ante）", V3_DIR / "v3_power_table_exante.parquet",
         None),
        ("II", "OOS 主设计", V3_DIR / "v3_oos_results.parquet", None),
        ("II", "OOS 时段拆分", V3_DIR / "v3_oos_sessions.parquet", None),
        ("II", "OOS 稳健性变体", V3_DIR / "v3_oos_robustness.parquet", None),
        ("II", "吸收复核", V3_DIR / "v3_absorption.parquet", None),
        ("II", "H4 tension-波动", V3_DIR / "v3_h4_tension_vol.parquet", None),
        ("supp", "窗口 IC 桥接", SUPP_DIR / "window_ic.parquet", None),
        ("supp", "吸收子样本", SUPP_DIR / "absorption_subsample.parquet", None),
        ("supp", "组合族（W1-W13，不含对照）", SUPP_DIR / "combo_ic.parquet",
         "noncontrol"),
        ("supp", "交互回归", SUPP_DIR / "interaction_reg.parquet", None),
        ("supp", "tension 门控", SUPP_DIR / "tension_gate.parquet", None),
        ("supp", "J 族跳变因子（IC 格）", V3_DIR / "jump" / "ic.parquet",
         None),
        ("supp", "J 族时段剖面", V3_DIR / "jump" / "session_response.parquet",
         None),
        ("supp", "J 族事件级折叠对照", V3_DIR / "jump" /
         "event_factor_ic.parquet", None),
        ("supp", "跨族复合 XF", V3_DIR / "jump" / "composite_ic.parquet",
         None),
    ]
    rows: list[dict] = []
    for part, family, path, flt in inventory:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        n = int((df["kind"] != "control").sum()) if flt == "noncontrol" \
            else int(len(df))
        rows.append({"part": part, "family": family, "n_units": n,
                     "source": path.name})
    return pd.DataFrame(rows)


# ------------------------------------------- 6. 第二部分测量层信号统计
def v3_signal_stats() -> pd.DataFrame:
    """v3 测量层信号（过滤 + 正交化后）的主题层统计。"""
    v3 = pd.read_parquet(V3_DIR / "v3_theme_signals.parquet")
    theme_tab = v3.drop_duplicates(["theme", "trade_date"])
    n_days = theme_tab["trade_date"].nunique()
    cols = [c for c in ("s_pre", "s_gap_am", "s_gap_pm", "s_day", "s_night",
                        "tension", "hz_day", "hz_gap_am")
            if c in theme_tab.columns]
    rows: list[dict] = []
    for theme, g in theme_tab.groupby("theme"):
        for c in cols:
            v = g[c].dropna()
            rows.append({
                "theme": theme, "signal": c, "n_days": n_days,
                "n_valid": int(len(v)),
                "coverage": float(len(v) / n_days),
                "mean": float(v.mean()) if len(v) else np.nan,
                "std": float(v.std()) if len(v) > 1 else np.nan,
                "q50": float(v.median()) if len(v) else np.nan,
                "q01": float(v.quantile(0.01)) if len(v) else np.nan,
                "q99": float(v.quantile(0.99)) if len(v) else np.nan,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-events", action="store_true",
                    help="跳过需要重扫 tape 的事件部分（调试用）")
    args = ap.parse_args()
    t0 = time.time()
    SUPP_DIR.mkdir(parents=True, exist_ok=True)

    theme_sig = pd.read_parquet(V3_DIR / "baseline_theme_signals.parquet")
    products = sorted(theme_sig["product"].unique())
    rets = futures_returns(products)
    rets = rets[rets["trade_date"].isin(theme_sig["trade_date"].unique())]
    theme_tab = theme_sig.drop_duplicates(["theme", "trade_date"])

    stats = signal_stats(theme_tab)
    stats.to_parquet(SUPP_DIR / "signal_stats_window.parquet", index=False)
    fig_signal_dist(theme_tab)
    print(f"[1/6] 信号统计 {len(stats)} 行，{time.time() - t0:.0f}s")

    ic = window_ic(theme_sig, rets)
    ic.to_parquet(SUPP_DIR / "window_ic.parquet", index=False)
    print(f"[2/6] 窗口 IC {len(ic)} 行，{time.time() - t0:.0f}s")

    absorb = absorption_subsample(theme_sig, rets)
    absorb.to_parquet(SUPP_DIR / "absorption_subsample.parquet", index=False)
    print(f"[3/6] 吸收子样本 {len(absorb)} 行，{time.time() - t0:.0f}s")

    combo_panel, combo_ic, combo_meta = build_combos(theme_sig, rets)
    combo_panel.to_parquet(SUPP_DIR / "combo_signals.parquet", index=False)
    combo_ic.to_parquet(SUPP_DIR / "combo_ic.parquet", index=False)
    print(f"[4/6] 组合 {len(combo_ic)} 检验行（族规模 "
          f"{combo_meta['family_size']}），{time.time() - t0:.0f}s")

    fw = futures_window_stats(products)
    fw.to_parquet(SUPP_DIR / "futures_window_stats.parquet", index=False)
    lk = locked_proxy(["SC", "AU", "AG", "CU", "M"])
    lk.to_parquet(SUPP_DIR / "locked_proxy.parquet", index=False)
    print(f"[5/10] 期货统计 {len(fw)} 行、涨跌停代理 {len(lk)} 行，"
          f"{time.time() - t0:.0f}s")

    ireg, iq = interaction_tests(theme_sig, rets)
    ireg.to_parquet(SUPP_DIR / "interaction_reg.parquet", index=False)
    iq.to_parquet(SUPP_DIR / "interaction_quantile.parquet", index=False)
    tg = tension_gate(rets)
    tg.to_parquet(SUPP_DIR / "tension_gate.parquet", index=False)
    epc = episode_cooccur(rets)
    epc.to_parquet(SUPP_DIR / "episode_cooccur.parquet", index=False)
    print(f"[6/10] 交互 {len(ireg)}、分位表 {len(iq)}、tension 门控 "
          f"{len(tg)}、episode 共现 {len(epc)} 行，{time.time() - t0:.0f}s")

    ro = rv_oos(theme_sig)
    ro.to_parquet(SUPP_DIR / "rv_oos.parquet", index=False)
    print(f"[7/10] RV OOS：{ro.to_dict('records')}，{time.time() - t0:.0f}s")

    ev_meta: dict = {}
    if not args.skip_events:
        registry = pd.read_parquet(REGISTRY_V3)
        con = poly_store.connect()
        ev = detect_events(registry, con)
        ev.to_parquet(SUPP_DIR / "events.parquet", index=False)
        co = event_cooccur(ev, rets)
        co.to_parquet(SUPP_DIR / "event_cooccur.parquet", index=False)
        em = event_monthly(ev)
        em.to_parquet(SUPP_DIR / "event_monthly.parquet", index=False)
        car = event_car(ev)
        car.to_parquet(SUPP_DIR / "event_car.parquet", index=False)
        ev_meta = {"n_events": int(len(ev)),
                   "by_type": ev["type"].value_counts().to_dict()}
        print(f"[8/10] 事件 {len(ev)} 个，共现 {len(co)} 行，"
              f"修正 CAR {len(car)} 行，{time.time() - t0:.0f}s")

    v3s = v3_signal_stats()
    v3s.to_parquet(SUPP_DIR / "v3_signal_stats.parquet", index=False)
    fs_path = V3_DIR / "v3_filter_states.parquet"
    nu_meta: dict = {}
    if fs_path.exists():
        nu = pd.read_parquet(fs_path, columns=["nu_std"])["nu_std"].dropna()
        nu_meta = {"n": int(len(nu)), "mean": float(nu.mean()),
                   "std": float(nu.std())}
    print(f"[9/10] v3 测量层统计 {len(v3s)} 行，{time.time() - t0:.0f}s")

    tc = test_count_audit()
    tc.to_parquet(SUPP_DIR / "test_count.parquet", index=False)
    print(f"[10/10] 检验总量核算 {len(tc)} 族，共 "
          f"{int(tc['n_units'].sum())} 单元，{time.time() - t0:.0f}s")

    meta = {
        "sample": {"start": str(theme_sig["trade_date"].min()),
                   "end": str(theme_sig["trade_date"].max()),
                   "n_days": int(theme_sig["trade_date"].nunique())},
        "hot_window": [HOT_START, HOT_END],
        "z_caliber": f"expanding z, min {Z_MIN_DAYS} days, info <= t",
        "thresholds": {"active": TH_ACTIVE, "surprise": TH_SURPRISE,
                       "ic_min_month": IC_MIN_MONTH},
        "combo": combo_meta,
        "events": ev_meta,
        "nu_std": nu_meta,
    }
    (SUPP_DIR / "meta_supp.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"完成，共 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
