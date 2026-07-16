"""v3 双门控（Layer 4）与增量预测评估（Layer 5）。

阶段：

1. **Power Gate**：结算 surprise、MDE 功效表与降级决策（框架 §9）。
2. **Layer 4**：功效通过组合的 category-level beta（训练窗 < 2026-05-15 冻结），
   其余 sign-only（框架 §10）。
3. **Layer 5**：嵌套模型展开窗口 OOS（框架 §12.8）：

       M0: 控制集（同窗口境外 ETF 收益 + 美元 + 自身滞后 + 平台公共因子）
       M1: M0 + 测量层信号（正交化主题创新、hazard、价格分布特征）
       M2: M1 + 门控影响信号

   预测目标 ``r_day(t)``，信号窗 ``pre = [前收盘 15:00, 当日 09:00)`` 严格早于
   目标窗（next-tradable 对齐）。同一评估以 v1.1 基线信号替换测量层作平行对照。
   指标：OOS 增量 R^2（相对 M0）、Clark–West 嵌套检验（HAC）、样本内 HAC 系数。
4. **同期吸收复核**：头部配对在扩展样本上的 ``r_gap ~ s_gap``（含 / 不含基准
   控制），基线与 v3 信号并列。
5. **H4 波动检验**：``|r_day| ~ tension + |s_pre| + RV 滞后``。

控制集 ETF 至 2026-06-29（美股库右端），其后的 OOS 行按缺控制处理（剔除并
如实报告）。信号缺测填 0（"无新信息"先验，两套信号同等处理）。

用法::

    .venv/bin/python scripts/v3_layer5.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import statsmodels.api as sm  # noqa: E402

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import gates, impact  # noqa: E402

OUT_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
REGISTRY_V3 = store.FEATURES_DIR / "cn_registry_v3.parquet"
EQUITY_MINUTE = ROOT / "data" / "equity" / "minute_db" / "minute"

TRAIN_END = "2026-05-15"
MIN_TRAIN_OBS = 40

#: 品种 -> 直连境外基准 ETF；SPY / UUP 为共同控制。
BENCH: dict[str, list[str]] = {
    "SC": ["USO"], "AU": ["GLD"], "AG": ["SLV"], "CU": ["CPER"],
    "M": ["SOYB"], "CF": ["BAL"], "I": ["XME"], "IF": ["ASHR"],
}
COMMON_CTRL = ["SPY", "UUP"]

#: Layer 4 资格主题（gates 由 families 表判定；此处为品种 -> 事件主题清单缓存）。


def load_etf_beijing(symbol: str) -> pd.Series | None:
    """ETF 分钟收盘（美东 -> 北京 tz-naive；口径同 intl_benchmark_analysis）。

    标的在样本期无数据（如已退市的 BAL）时返回 None，调用方跳过该控制变量。
    """
    frames = []
    for year in (2025, 2026):
        f = EQUITY_MINUTE / symbol / f"{year}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f, columns=["ts", "close"]))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    ts = (pd.to_datetime(df["ts"])
          .dt.tz_localize("America/New_York")
          .dt.tz_convert("Asia/Shanghai")
          .dt.tz_localize(None))
    px = pd.Series(df["close"].to_numpy(dtype="float64"), index=ts).sort_index()
    return px[~px.index.duplicated(keep="last")]


def etf_window_returns(px: pd.Series, win: pd.DataFrame) -> pd.DataFrame:
    """窗口边界 LOCF 对数收益（严格早于边界，同信号口径）。"""
    logp = np.log(px)
    idx = logp.index
    arr = logp.to_numpy()

    def at(bound: pd.Series) -> np.ndarray:
        t = pd.to_datetime(bound).to_numpy()
        out = np.full(len(t), np.nan)
        ok = ~pd.isna(t)
        pos = idx.searchsorted(t[ok], side="left") - 1
        valid = pos >= 0
        vals = np.full(int(ok.sum()), np.nan)
        vals[valid] = arr[pos[valid]]
        out[ok] = vals
        return out

    out = pd.DataFrame({"trade_date": win["trade_date"].to_numpy()})
    out["b_pre"] = at(win["day_start"]) - at(win["gap1_start"])
    out["b_night"] = at(win["night_end"]) - at(win["night_start"])
    out["b_day"] = at(win["day_end"]) - at(win["day_start"])
    out["b_gap_pm"] = at(win["gap1_end"]) - at(win["gap1_start"])
    out["b_gap_am"] = at(win["gap2_end"]) - at(win["gap2_start"])
    out["b_gap"] = np.where(
        win["night_start"].notna(),
        out["b_gap_pm"] + out["b_gap_am"],
        out["b_gap_pm"],
    )
    return out


def hac_ols(y: np.ndarray, X: np.ndarray, maxlags: int = 1):
    """OLS + HAC；返回 statsmodels 结果（调用方负责样本对齐）。"""
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})


def expanding_oos(
    y: np.ndarray, blocks: dict[str, np.ndarray], *, min_train: int = MIN_TRAIN_OBS
) -> pd.DataFrame:
    """嵌套模型的展开窗口一步 OOS 预测。

    Args:
        y: 目标（T,）。
        blocks: 模型名 -> 设计矩阵（T, k）；须嵌套（同名行样本）。
        min_train: 首个预测前的最小训练样本。

    Returns:
        ``[t, model, yhat, y]`` 长表（仅 y 与全部设计行均非 NaN 的 t）。
    """
    T = len(y)
    any_nan = np.isnan(y)
    for X in blocks.values():
        any_nan |= np.isnan(X).any(axis=1)
    valid = ~any_nan
    rows: list[dict] = []
    for t in range(T):
        if not valid[t]:
            continue
        hist = valid & (np.arange(T) < t)
        if hist.sum() < min_train:
            continue
        for name, X in blocks.items():
            Xh = sm.add_constant(X[hist], has_constant="add")
            beta, *_ = np.linalg.lstsq(Xh, y[hist], rcond=None)
            xt = np.concatenate([[1.0], X[t]])
            rows.append({"t": t, "model": name,
                         "yhat": float(xt @ beta), "y": float(y[t])})
    return pd.DataFrame(rows)


def clark_west(e0: np.ndarray, e1: np.ndarray, yhat0: np.ndarray,
               yhat1: np.ndarray) -> tuple[float, float]:
    """Clark–West 嵌套 MSPE 检验，返回 (统计量均值, HAC t)。"""
    f = e0**2 - e1**2 + (yhat0 - yhat1) ** 2
    if len(f) < 10:
        return np.nan, np.nan
    res = sm.OLS(f, np.ones((len(f), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2})
    return float(np.mean(f)), float(res.tvalues[0])


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q 值（NaN 保留）。"""
    p = np.asarray(pvals, dtype="float64")
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    ps = p[ok]
    m = len(ps)
    if m == 0:
        return q
    order = np.argsort(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def mbb_pvalue_mean_positive(
    x: np.ndarray, *, block: int = 10, n_boot: int = 2000, seed: int = 11
) -> float:
    """循环块自助：H1 为 mean(x) > 0 的单侧 p（自助均值 <= 0 的占比）。"""
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 20:
        return np.nan
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        if float(np.mean(x[idx[:T]])) <= 0:
            count += 1
    return (count + 1) / (n_boot + 1)


def map_theme_col(sub: pd.DataFrame, theme_name: str, col: str,
                  dates: pd.Series) -> np.ndarray:
    """(theme 子表, 列) -> 与 dates 对齐的向量。"""
    s = sub.loc[sub["theme"] == theme_name].set_index("trade_date")[col]
    return dates.map(s).to_numpy(dtype="float64")


def main() -> int:
    t0 = time.time()
    registry = pd.read_parquet(REGISTRY_V3)
    fam = pd.read_parquet(OUT_DIR / "v3_families.parquet")
    theme = pd.read_parquet(OUT_DIR / "v3_theme_signals.parquet")
    base = pd.read_parquet(OUT_DIR / "baseline_theme_signals.parquet")
    zpre = pd.read_parquet(OUT_DIR / "v3_zpre.parquet")
    outcomes = pd.read_parquet(OUT_DIR / "v3_outcomes.parquet")

    products = sorted(registry["product"].unique())
    specs = fut_store.read_product_specs().set_index("product")

    rets = {}
    for p in products:
        d = fut_store.read_daily(p)
        d = d.loc[(d["trade_date"] >= "2026-01-05") & (d["trade_date"] <= "2026-07-13")]
        d = d.reset_index(drop=True)
        d["r_gap_total"] = d["r_gap_pm"] + d["r_gap_am"]
        if d["r_night"].isna().all():
            d["r_gap_total"] = d["r_gap_full"]
        rets[p] = d
    returns_cc = pd.concat(
        [rets[p][["trade_date", "r_cc"]].assign(product=p) for p in products],
        ignore_index=True)

    # ---- Power Gate 与 Layer 4（episode 口径 + ex-ante / ex-post 分离）----
    reg_res = registry.copy()
    reg_res["resolved_at"] = pd.to_datetime(reg_res["resolved_at"], utc=True)
    surp = gates.surprises(reg_res, fam, zpre, outcomes)
    trade_days_all = rets[products[0]]["trade_date"].tolist()
    surp_ev = gates.attach_event_date(surp, trade_days_all)
    sigma_r = {
        p: float(rets[p].loc[rets[p]["trade_date"] < TRAIN_END, "r_cc"].std())
        for p in products
    }

    # ex-ante 门控：只用训练截止前已入收益窗的 episode（决定 OOS 模型结构）。
    surp_ante = surp_ev.loc[surp_ev["ev_date"] < TRAIN_END]
    power_ante = gates.power_table_episode(surp_ante, pd.Series(sigma_r))
    power_ante.to_parquet(OUT_DIR / "v3_power_table_exante.parquet", index=False)
    print(f"== Power Gate（episode 口径，ex-ante，< {TRAIN_END}）==")
    print(power_ante.to_string(index=False))

    # ex-post 状态：全样本，仅回答"当前数据量下哪些参数可研究"。
    power_post = gates.power_table_episode(surp_ev, pd.Series(sigma_r))
    power_post.to_parquet(OUT_DIR / "v3_power_table_expost.parquet", index=False)
    print("\n== Power Gate（episode 口径，ex-post 全样本）==")
    print(power_post.to_string(index=False))

    # 旧市场级口径（family × 周聚类，rho=0.5）保留作对比，标注偏乐观。
    power_old = gates.power_table(surp, pd.Series(sigma_r))
    power_old.to_parquet(OUT_DIR / "v3_power_table.parquet", index=False)

    ep = gates.episode_aggregate(surp_ev, agg="mean")
    betas = impact.fit_category_beta_episode(
        ep, returns_cc, power_ante, train_end=TRAIN_END)
    betas.to_parquet(OUT_DIR / "v3_impact_betas.parquet", index=False)
    print("\n== Layer 4 影响系数（episode 级，训练窗冻结，wild bootstrap p）==")
    print(betas.to_string(index=False))
    ep_sum = gates.episode_aggregate(surp_ev, agg="sum")
    betas_sum = impact.fit_category_beta_episode(
        ep_sum, returns_cc, power_ante, train_end=TRAIN_END)
    betas_sum.to_parquet(OUT_DIR / "v3_impact_betas_sum.parquet", index=False)
    surp_stats = ep.groupby(["theme", "product"])["surprise"].describe()
    surp_stats.to_parquet(OUT_DIR / "v3_surprise_support.parquet")
    ep.to_parquet(OUT_DIR / "v3_episodes.parquet", index=False)

    theme = impact.impact_signal(theme, betas, col="s_pre_orth")

    # ---- 控制集 ----
    trade_days = rets[products[0]]["trade_date"].tolist()
    etf_cache: dict[tuple[str, str], pd.DataFrame] = {}
    px_cache: dict[str, pd.Series] = {}

    def etf_returns(sym: str, product: str) -> pd.DataFrame | None:
        ne = specs.loc[product, "night_end"]
        key = (sym, ne if isinstance(ne, str) else "none")
        if key not in etf_cache:
            if sym not in px_cache:
                px_cache[sym] = load_etf_beijing(sym)
            if px_cache[sym] is None:
                etf_cache[key] = None
            else:
                win = sessions.signal_windows(
                    trade_days, night_end=sessions.parse_night_end(
                        ne if isinstance(ne, str) else None))
                etf_cache[key] = etf_window_returns(px_cache[sym], win)
        return etf_cache[key]

    # ---- Layer 5 面板与嵌套评估 ----
    eligible_themes = set(
        fam.loc[fam["layer4_eligible"], "theme"].unique())
    price_themes = set(fam.loc[~fam["layer4_eligible"], "theme"].unique())

    oos_rows: list[dict] = []
    insample_rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    for p in products:
        d = rets[p].copy()
        d = d.loc[~d["roll"].fillna(False)].reset_index(drop=True)
        y = d["r_day"].to_numpy(dtype="float64")
        dates = d["trade_date"]

        # 控制块。
        ctrl_cols = []
        ctrl_mat = []
        for sym in BENCH.get(p, []) + COMMON_CTRL:
            etf = etf_returns(sym, p)
            if etf is None:
                continue
            b = etf.set_index("trade_date")["b_pre"]
            ctrl_mat.append(dates.map(b).to_numpy(dtype="float64"))
            ctrl_cols.append(f"b_pre_{sym}")
        r_lag = d["r_cc"].shift(1).to_numpy(dtype="float64")
        ctrl_mat.append(r_lag)
        ctrl_cols.append("r_cc_lag")
        # 自身夜盘收益：09:00 时已实现（02:30 收盘），是"下一可交易时点已经
        # 吸收了多少"的直接控制（评审修订：信号窗内含本品种夜盘开市段）。
        if not d["r_night"].isna().all():
            ctrl_mat.append(np.nan_to_num(d["r_night"].to_numpy(dtype="float64")))
            ctrl_cols.append("r_night")
        sub_t = theme.loc[theme["product"] == p]
        f_pm = (sub_t.drop_duplicates("trade_date")
                .set_index("trade_date")["f_pm_pre"])
        # 因子缺测（展开期不足 / 截面 family 不足）按 0（中性）处理，避免整行丢失。
        ctrl_mat.append(np.nan_to_num(dates.map(f_pm).to_numpy(dtype="float64")))
        ctrl_cols.append("f_pm_pre")
        X0 = np.column_stack(ctrl_mat)

        # 测量块（v3）：正交化事件主题创新 + hazard + 价格分布特征。
        def theme_col(sub: pd.DataFrame, theme_name: str, col: str,
                      dates: pd.Series = dates) -> np.ndarray:
            s = (sub.loc[sub["theme"] == theme_name]
                 .set_index("trade_date")[col])
            return dates.map(s).to_numpy(dtype="float64")

        v3_cols, v3_mat = [], []
        slim_mat = []  # 与基线同维度的精简变体：只含主题级正交化创新
        for th in sorted(set(sub_t["theme"])):
            if th in eligible_themes:
                s_orth = np.nan_to_num(theme_col(sub_t, th, "s_pre_orth"))
                v3_mat.append(s_orth)
                slim_mat.append(s_orth)
                v3_cols.append(f"s_{th}")
                if "hz_pre" in sub_t.columns:
                    hz = theme_col(sub_t, th, "hz_pre")
                    if np.isfinite(hz).sum() >= 20:
                        v3_mat.append(np.nan_to_num(hz))
                        v3_cols.append(f"hz_{th}")
            elif th in price_themes:
                if "px_med_pre" in sub_t.columns:
                    px_v = np.nan_to_num(theme_col(sub_t, th, "px_med_pre"))
                    v3_mat.append(px_v)
                    v3_cols.append(f"px_{th}")
                # 精简变体用价格主题的正交化创新（与基线主题集合一致）。
                slim_mat.append(np.nan_to_num(theme_col(sub_t, th, "s_pre_orth")))
        X1_v3 = np.column_stack([X0] + v3_mat) if v3_mat else X0
        X1_slim = np.column_stack([X0] + slim_mat) if slim_mat else X0

        # 影响块（M2；sign-only 与测量块同源，仅在 category_beta 模式下新增信息）。
        imp_mat = []
        for th in sorted(set(sub_t["theme"]) & eligible_themes):
            blk = sub_t.loc[sub_t["theme"] == th]
            if (blk["impact_mode"] == "category_beta").any():
                s = blk.set_index("trade_date")["impact"]
                imp_mat.append(np.nan_to_num(dates.map(s).to_numpy(dtype="float64")))
        X2 = np.column_stack([X1_v3] + imp_mat) if imp_mat else X1_v3

        # 基线块（v1.1 信号，全部主题）。
        sub_b = base.loc[base["product"] == p]
        b_mat, b_cols = [], []
        for th in sorted(set(sub_b["theme"])):
            b_mat.append(np.nan_to_num(theme_col(sub_b, th, "s_pre")))
            b_cols.append(f"s_{th}")
        X1_base = np.column_stack([X0] + b_mat) if b_mat else X0

        blocks = {"M0": X0, "M1_v3": X1_v3, "M1_v3slim": X1_slim,
                  "M1_base": X1_base}
        if X2.shape[1] > X1_v3.shape[1]:
            blocks["M2"] = X2
        preds = expanding_oos(y, blocks)
        if preds.empty:
            print(f"{p}: OOS 样本不足，跳过")
            continue
        preds["product"] = p
        all_preds.append(preds)

        piv = preds.pivot_table(index="t", columns="model", values="yhat")
        yv = preds.pivot_table(index="t", columns="model", values="y").iloc[:, 0]
        res_row = {"product": p, "n_oos": len(piv)}
        e0 = (yv - piv["M0"]).to_numpy()
        mse0 = float(np.mean(e0**2))
        for m in [c for c in piv.columns if c != "M0"]:
            e1 = (yv - piv[m]).to_numpy()
            res_row[f"dR2_{m}"] = 1.0 - float(np.mean(e1**2)) / mse0
            _, cw_t = clark_west(e0, e1, piv["M0"].to_numpy(), piv[m].to_numpy())
            res_row[f"cw_t_{m}"] = cw_t
        oos_rows.append(res_row)

        # 样本内 HAC（全样本，透明起见与 OOS 并列报告）。
        for name, X, cols in (
            ("M1_v3", X1_v3, ctrl_cols + v3_cols),
            ("M1_base", X1_base, ctrl_cols + b_cols),
        ):
            ok = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            if ok.sum() < 30:
                continue
            fit = hac_ols(y[ok], X[ok])
            for i, cname in enumerate(cols, start=1):
                if cname.startswith(("s_", "hz_", "px_")):
                    insample_rows.append(
                        {"product": p, "model": name, "var": cname,
                         "beta": float(fit.params[i]),
                         "t": float(fit.tvalues[i]), "n": int(ok.sum())}
                    )

    oos = pd.DataFrame(oos_rows)
    preds_all = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    # ---- 多重检验与稳健性（评审修订：8 品种不是 1 个检验）----
    from scipy import stats as sps
    for m in ("M1_base", "M1_v3slim", "M1_v3"):
        col = f"cw_t_{m}"
        if col in oos.columns:
            p1 = 1.0 - sps.norm.cdf(oos[col].to_numpy(dtype="float64"))
            oos[f"p_cw_{m}"] = p1
            oos[f"q_cw_{m}"] = bh_fdr(p1)

    mbb_rows: list[dict] = []
    diff_pooled: list[np.ndarray] = []
    for p in oos["product"]:
        sub = preds_all.loc[preds_all["product"] == p]
        piv = sub.pivot_table(index="t", columns="model", values="yhat")
        yv = sub.pivot_table(index="t", columns="model", values="y").iloc[:, 0]
        if "M1_v3slim" not in piv.columns:
            continue
        e0 = (yv - piv["M0"]).to_numpy()
        e1 = (yv - piv["M1_v3slim"]).to_numpy()
        eb = (yv - piv["M1_base"]).to_numpy()
        f_cw = e0**2 - e1**2 + (piv["M0"] - piv["M1_v3slim"]).to_numpy() ** 2
        half = len(f_cw) // 2
        mse0 = float(np.mean(e0**2))
        row = {
            "product": p,
            "p_mbb_cw_v3slim": mbb_pvalue_mean_positive(f_cw),
            "dR2_v3slim_h1": 1 - float(np.mean(e1[:half] ** 2))
            / float(np.mean(e0[:half] ** 2)),
            "dR2_v3slim_h2": 1 - float(np.mean(e1[half:] ** 2))
            / float(np.mean(e0[half:] ** 2)),
        }
        row["dR2_v3slim"] = 1 - float(np.mean(e1**2)) / mse0
        mbb_rows.append(row)
        d = eb**2 - e1**2  # 基线损失 − v3 精简损失（>0 = v3 更好）
        sd = float(np.std(d, ddof=1))
        if sd > 0:
            diff_pooled.append(d / sd)
    mbb = pd.DataFrame(mbb_rows)
    pooled = np.concatenate(diff_pooled) if diff_pooled else np.array([])
    p_pooled = mbb_pvalue_mean_positive(pooled) if len(pooled) else np.nan
    mbb.to_parquet(OUT_DIR / "v3_oos_robustness.parquet", index=False)

    oos.to_parquet(OUT_DIR / "v3_oos_results.parquet", index=False)
    insample = pd.DataFrame(insample_rows)
    insample.to_parquet(OUT_DIR / "v3_insample_gamma.parquet", index=False)
    if not preds_all.empty:
        preds_all.to_parquet(OUT_DIR / "v3_oos_predictions.parquet", index=False)

    print("\n== Layer 5 OOS 增量 R^2（相对 M0）与 Clark-West t / BH-FDR q ==")
    show_cols = [c for c in oos.columns if not c.startswith("p_cw")]
    print(oos[show_cols].to_string(index=False))
    print("\n== OOS 稳健性（块自助 p、前后半段一致性）==")
    print(mbb.to_string(index=False))
    print(f"跨品种合并（基线损失 − v3 精简损失，标准化）块自助单侧 p = {p_pooled:.3f}")
    print("\n== 样本内 HAC gamma（信号项）==")
    if not insample.empty:
        print(insample.loc[insample["t"].abs() >= 1.5].to_string(index=False))

    # ---- 时段拆分（评审修订：夜盘是多数品种真正的下一可交易时点）----
    session_specs = [
        # 设计, 目标, 信号列, 基准列, 因子列, 额外控制
        ("close_to_night", "r_night", "s_gap_pm_orth", "b_gap_pm",
         "f_pm_gap_pm", []),
        ("postnight_to_day", "r_day", "s_gap_am_orth", "b_gap_am",
         "f_pm_gap_am", ["r_night"]),
    ]
    sess_rows: list[dict] = []
    for name, y_col, sig_col, bench_col, fpm_col, extra in session_specs:
        for p in products:
            d = rets[p].loc[~rets[p]["roll"].fillna(False)].reset_index(drop=True)
            if d[y_col].isna().all():
                continue
            dates = d["trade_date"]
            y = d[y_col].to_numpy(dtype="float64")
            ctrl = []
            for sym in BENCH.get(p, []) + COMMON_CTRL:
                etf = etf_returns(sym, p)
                if etf is None:
                    continue
                b = etf.set_index("trade_date")[bench_col]
                ctrl.append(dates.map(b).to_numpy(dtype="float64"))
            ctrl.append(d["r_cc"].shift(1).to_numpy(dtype="float64"))
            for c in extra:
                ctrl.append(np.nan_to_num(d[c].to_numpy(dtype="float64")))
            sub_t = theme.loc[theme["product"] == p]
            if fpm_col in sub_t.columns:
                f = (sub_t.drop_duplicates("trade_date")
                     .set_index("trade_date")[fpm_col])
                ctrl.append(np.nan_to_num(dates.map(f).to_numpy(dtype="float64")))
            X0 = np.column_stack(ctrl)
            if sig_col not in sub_t.columns:
                continue
            sig_mat = [
                np.nan_to_num(map_theme_col(sub_t, th, sig_col, dates))
                for th in sorted(set(sub_t["theme"]))
            ]
            X1 = np.column_stack([X0] + sig_mat)
            preds = expanding_oos(y, {"M0": X0, "M1": X1})
            if preds.empty:
                continue
            piv = preds.pivot_table(index="t", columns="model", values="yhat")
            yv = preds.pivot_table(index="t", columns="model", values="y").iloc[:, 0]
            e0 = (yv - piv["M0"]).to_numpy()
            e1 = (yv - piv["M1"]).to_numpy()
            _, cw_t = clark_west(e0, e1, piv["M0"].to_numpy(), piv["M1"].to_numpy())
            sess_rows.append(
                {"design": name, "product": p, "n_oos": len(piv),
                 "dR2": 1 - float(np.mean(e1**2)) / float(np.mean(e0**2)),
                 "cw_t": cw_t}
            )
    sessions_oos = pd.DataFrame(sess_rows)
    sessions_oos.to_parquet(OUT_DIR / "v3_oos_sessions.parquet", index=False)
    print("\n== 时段拆分 OOS（傍晚信号->夜盘；凌晨信号->日盘，控夜盘收益）==")
    print(sessions_oos.to_string(index=False))

    # ---- 同期吸收复核（扩展样本；含基准控制）----
    absorb_rows: list[dict] = []
    headline = [("oil_price", "SC"), ("mideast_conflict", "SC"),
                ("metal_price", "AU"), ("metal_price", "AG"),
                ("fed_policy", "AU"), ("russia_ukraine", "SC")]
    for th, p in headline:
        d = rets[p].loc[~rets[p]["roll"].fillna(False)]
        dates = d["trade_date"]
        for label, sig_tab, col in (
            ("v3", theme.loc[(theme["theme"] == th) & (theme["product"] == p)], "s_gap"),
            ("base", base.loc[(base["theme"] == th) & (base["product"] == p)], "s_gap"),
        ):
            s = dates.map(sig_tab.set_index("trade_date")[col]).to_numpy(dtype="float64")
            yg = d["r_gap_total"].to_numpy(dtype="float64")
            bench_sym = BENCH.get(p, [None])[0]
            etf_b = etf_returns(bench_sym, p) if bench_sym else None
            bg = (dates.map(etf_b.set_index("trade_date")["b_gap"])
                  .to_numpy(dtype="float64")) if etf_b is not None \
                else np.full(len(d), np.nan)
            ok = ~(np.isnan(s) | np.isnan(yg))
            if ok.sum() < 25:
                continue
            r_raw = float(np.corrcoef(s[ok], yg[ok])[0, 1])
            fit_raw = hac_ols(yg[ok], s[ok].reshape(-1, 1))
            row = {"theme": th, "product": p, "signal": label, "n": int(ok.sum()),
                   "corr": r_raw, "t_raw": float(fit_raw.tvalues[1])}
            ok2 = ok & ~np.isnan(bg)
            if ok2.sum() >= 25:
                fit_ctl = hac_ols(yg[ok2], np.column_stack([s[ok2], bg[ok2]]))
                row["t_ctl"] = float(fit_ctl.tvalues[1])
                row["n_ctl"] = int(ok2.sum())
            absorb_rows.append(row)
    absorb = pd.DataFrame(absorb_rows)
    absorb.to_parquet(OUT_DIR / "v3_absorption.parquet", index=False)
    print("\n== 同期吸收复核（扩展样本）==")
    print(absorb.to_string(index=False))

    # ---- H4：tension 对波动（评审修订：加活动度控制判别"注意力代理"）----
    h4_rows: list[dict] = []
    for p in products:
        sub_t = theme.loc[theme["product"] == p]
        if "tension" not in sub_t.columns:
            continue
        tn = sub_t.groupby("trade_date")["tension"].max()
        n_act = sub_t.groupby("trade_date")["n_active"].max()
        usdc = (
            base.loc[base["product"] == p]
            .groupby("trade_date")[["usdc_night", "usdc_gap", "usdc_day"]]
            .sum().sum(axis=1)
        )
        d = rets[p].loc[~rets[p]["roll"].fillna(False)]
        dates = d["trade_date"]
        yv = np.abs(d["r_day"].to_numpy(dtype="float64"))
        x_t = dates.map(tn).to_numpy(dtype="float64")
        a1 = dates.map(n_act).to_numpy(dtype="float64")
        a2 = np.log1p(dates.map(usdc).to_numpy(dtype="float64"))
        lag = pd.Series(yv).shift(1).to_numpy()
        ok = ~(np.isnan(yv) | np.isnan(x_t) | np.isnan(lag)
               | np.isnan(a1) | np.isnan(a2))
        if ok.sum() < 40:
            continue
        fit0 = hac_ols(yv[ok], np.column_stack([x_t[ok], lag[ok]]))
        fit1 = hac_ols(
            yv[ok], np.column_stack([x_t[ok], lag[ok], a1[ok], a2[ok]]))
        h4_rows.append(
            {"product": p, "n": int(ok.sum()),
             "t_tension": float(fit0.tvalues[1]),
             "t_tension_actrl": float(fit1.tvalues[1]),
             "t_n_active": float(fit1.tvalues[3]),
             "t_log_usdc": float(fit1.tvalues[4])}
        )
    h4 = pd.DataFrame(h4_rows)
    h4.to_parquet(OUT_DIR / "v3_h4_tension_vol.parquet", index=False)
    print("\n== H4：narrative tension 对 |r_day|（无 / 有活动度控制）==")
    print(h4.to_string(index=False))

    summary = {
        "train_end": TRAIN_END,
        "power_decisions_exante": power_ante.groupby(
            "decision")["theme"].count().to_dict(),
        "power_decisions_expost": power_post.groupby(
            "decision")["theme"].count().to_dict(),
        "p_pooled_base_vs_v3slim": None if np.isnan(p_pooled) else round(
            float(p_pooled), 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "v3_layer5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n完成，耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
