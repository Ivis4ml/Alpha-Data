"""答辩数据包：分钟级 Polymarket 信号库（30 个定义）、IC / ICIR 与事件研究。

按汇报答辩要求构建（docs/v3_signal_defense.html 的数据层）：

- **分钟面板**：国内期货 1 分钟 bar（bar 收盘戳）× Polymarket 主题分钟聚合
  （分钟桶 [T-60s, T) 标签取右端 T，与 bar 收盘戳对齐，信号严格早于前向收益）。
- **30 个信号**：N1-N10 数值信号、C1-C10 量价组合信号、K1-K10 离散事件
  复杂统计指标；全部输出显式公式（报告脚本渲染）、频率、取值统计与分布。
- **IC 引擎**：数值信号对 1/2/3/5/10/15 分钟前向收益的 Pearson / Spearman IC，
  逐交易日 IC 序列的 mean/std（ICIR）。
- **事件研究**：离散事件后 1..15 分钟与"直到下一信号"的价格变化，
  |Δp| > 0.1% 比例对全样本基线的比较。
- **全历史频率**：2022-11 至今按月的主题事件频率（规则匹配全历史市场）。

归一化约定（答辩要求）：z(x) = (x - 滚动均值) / 滚动标准差，窗口 4800 个
交易分钟（约 10 个 SC 交易日），最少 960 个观测；跨时段连续计算。

时段处理：前向收益只在**连续分钟段**内计算（相邻 bar 时间差 > 1 分钟即断开
——午间休市、小节休息、日夜盘边界自动处理），跨段置 NaN。

用法::

    .venv/bin/python scripts/v3_defense_build.py [--skip-history]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import select_polymarket_markets as spm  # noqa: E402

from alpha_data.cn_futures import store as fut_store  # noqa: E402
from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

OUT = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
REGISTRY = store.FEATURES_DIR / "cn_registry_v3.parquet"

#: 品种 -> 映射主题（登记表口径）。
PRODUCT_THEMES: dict[str, list[str]] = {
    "SC": ["mideast_conflict", "oil_price"],
    "AU": ["mideast_conflict", "metal_price", "fed_policy"],
    "AG": ["metal_price", "fed_policy"],
    "CU": ["fed_policy", "us_china_trade"],
    "M": ["us_china_trade"],
}
HORIZONS = (1, 2, 3, 5, 10, 15)
ZWIN, ZMIN = 4800, 960
CLIP = (0.02, 0.98)


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, CLIP[0], CLIP[1])
    return np.log(q / (1.0 - q))


def zscore(s: pd.Series, win: int = ZWIN, minp: int = ZMIN) -> pd.Series:
    m = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - m) / sd.replace(0.0, np.nan)


# ---------------------------------------------------------------- PM 分钟聚合
def pm_minute_theme(con, registry: pd.DataFrame, theme: str, product: str,
                    start_utc: str, end_utc: str) -> pd.DataFrame:
    """主题分钟聚合：``[ts(BJ 分钟末), dl, flow, usdc, n, n_mkts]``。

    - 分钟桶 [T-60s, T)，标签 T（严格早于 T 的成交）；
    - 市场级 vwap -> logit 差分（对市场自身上一活跃分钟），orientation ×
      sqrt(usdc_win) 加权聚合；
    - 时点化准入（admit_ts 前剔除）与结算截断（resolved_at 后剔除）。
    """
    reg = registry[(registry["theme"] == theme)
                   & (registry["product"] == product)]
    if reg.empty:
        return pd.DataFrame(columns=["ts", "dl", "flow", "usdc", "n", "n_mkts"])
    ids = ",".join("'" + c + "'" for c in reg["condition_id"])
    cols = "condition_id, block_timestamp, p_event, D, usdc_amount"
    sql = f"""
        SELECT condition_id,
               (block_timestamp // 60) * 60 + 60          AS m_end,
               sum(p_event * usdc_amount) / sum(usdc_amount) AS vwap,
               sum(D * usdc_amount)                        AS signed,
               sum(usdc_amount)                            AS usdc,
               count(*)                                    AS n
        FROM {tape.union_sql(cols)}
        WHERE condition_id IN ({ids}) AND p_event IS NOT NULL
          AND block_timestamp >= epoch(TIMESTAMP '{start_utc}')
          AND block_timestamp <  epoch(TIMESTAMP '{end_utc}')
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df = con.execute(sql).fetch_df()
    if df.empty:
        return pd.DataFrame(columns=["ts", "dl", "flow", "usdc", "n", "n_mkts"])
    meta = reg.set_index("condition_id")
    df["admit"] = df["condition_id"].map(meta["admit_ts"])
    res = pd.to_datetime(meta["resolved_at"], utc=True)
    res_ep = (res - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(
        seconds=1)
    df["res_ep"] = df["condition_id"].map(res_ep)
    df = df[(df["m_end"] > df["admit"])
            & (df["res_ep"].isna() | (df["m_end"] <= df["res_ep"]))]

    df = df.sort_values(["condition_id", "m_end"])
    df["dl_m"] = logit(df["vwap"].to_numpy()) - logit(
        df.groupby("condition_id")["vwap"].shift(1).to_numpy())
    df["w"] = (df["condition_id"].map(meta["orientation"]).astype(float)
               * np.sqrt(df["condition_id"].map(meta["usdc_win"]).astype(float)))
    df["flow_m"] = (df["condition_id"].map(meta["orientation"]).astype(float)
                    * df["signed"])
    df["wdl"] = df["w"] * df["dl_m"]

    g = df.groupby("m_end")
    out = pd.DataFrame(
        {
            "wdl": g["wdl"].sum(), "aw": g.apply(
                lambda b: float(np.abs(b["w"]).sum()), include_groups=False),
            "flow": g["flow_m"].sum(), "usdc": g["usdc"].sum(),
            "n": g["n"].sum(), "n_mkts": g["condition_id"].nunique(),
        }
    ).reset_index()
    out["dl"] = out["wdl"] / out["aw"]
    out["ts"] = (pd.to_datetime(out["m_end"], unit="s", utc=True)
                 .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))
    return out[["ts", "dl", "flow", "usdc", "n", "n_mkts"]]


# ---------------------------------------------------------------- CN 分钟面板
def cn_panel(product: str) -> pd.DataFrame:
    """期货分钟面板：段内前向收益 + 归一化量价特征。"""
    df = fut_store.read_minute(product)
    df = df[(df["trade_date"] >= "2026-01-05")
            & (df["trade_date"] <= "2026-07-13")].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    # 连续分钟段：相邻 bar 间差 > 1 分钟即断开（午休 / 小节 / 日夜盘边界）。
    df["seg"] = (df["ts"].diff() != pd.Timedelta(minutes=1)).cumsum()
    logc = np.log(df["close"].to_numpy())
    df["r1"] = pd.Series(logc).diff()
    df.loc[df["seg"].diff() != 0, "r1"] = np.nan
    for k in HORIZONS:
        fwd = pd.Series(logc).shift(-k) - logc
        same = df["seg"].shift(-k) == df["seg"]
        df[f"fwd_{k}"] = fwd.where(same)
    grp = df.groupby("seg", sort=False)
    df["mom15"] = grp["r1"].transform(
        lambda s: s.rolling(15, min_periods=5).sum())
    df["rv15"] = grp["r1"].transform(
        lambda s: np.sqrt((s**2).rolling(15, min_periods=5).sum()))
    df["volu15"] = grp["volume"].transform(
        lambda s: s.rolling(15, min_periods=5).sum())
    df["amt15"] = grp["money"].transform(
        lambda s: s.rolling(15, min_periods=5).sum())
    for c in ("mom15", "rv15", "volu15", "amt15"):
        df[f"{c}_z"] = zscore(df[c])
    df["locked"] = (df["high"] == df["low"]).astype(int)
    return df


# ---------------------------------------------------------------- 信号构造
def add_signals(panel: pd.DataFrame, themes: list[str]) -> pd.DataFrame:
    """在面板上添加 N1-N10 / C1-C10 / K1-K10（公式见报告 §3）。"""
    df = panel.copy()
    prim = themes[0]

    def col(theme: str, name: str) -> pd.Series:
        c = f"{name}_{theme}"
        return df[c] if c in df.columns else pd.Series(0.0, index=df.index)

    # ---- 数值信号 N ----
    df["N1"] = col(prim, "dl").fillna(0.0)                       # 1min 主题创新
    df["N2"] = df["N1"].rolling(15, min_periods=1).sum()          # 15min 累积
    flow15 = col(prim, "flow").fillna(0.0).rolling(15, min_periods=1).sum()
    df["N3"] = zscore(flow15).fillna(0.0)                         # 15min 签名流 z
    df["N4"] = zscore(df["N2"]).fillna(0.0)                       # 创新 z
    usdc15 = col(prim, "usdc").fillna(0.0).rolling(15, min_periods=1).sum()
    df["N5"] = (flow15 / usdc15.replace(0, np.nan)).fillna(0.0)   # 流不平衡比
    n15 = col(prim, "n").fillna(0.0).rolling(15, min_periods=1).sum()
    df["N6"] = zscore(np.log1p(n15)).fillna(0.0)                  # 成交强度 z
    # N7 多主题复合创新（各主题 15min 创新的等权和后 z）
    comp = sum(col(t, "dl").fillna(0.0) for t in themes)
    df["N7"] = zscore(pd.Series(comp, index=df.index)
                      .rolling(15, min_periods=1).sum()).fillna(0.0)
    # N8 主题内头部市场集中度变化（herding）：n_mkts 的倒数变化 z
    nm = col(prim, "n_mkts").fillna(0.0)
    df["N8"] = zscore(nm.rolling(15, min_periods=1).mean()).fillna(0.0)
    # N9 信念波动：dl 的 60min 滚动波动 z
    df["N9"] = zscore(df["N1"].rolling(60, min_periods=20).std()).fillna(0.0)
    # N10 信念不确定度创新：|dl| 的 15min 和减去其长期均值（z）
    df["N10"] = zscore(df["N1"].abs().rolling(15, min_periods=1).sum()).fillna(0.0)

    # ---- 量价组合信号 C ----
    mz = df["mom15_z"].fillna(0.0)
    rz = df["rv15_z"].fillna(0.0)
    vz = df["volu15_z"].fillna(0.0)
    az = df["amt15_z"].fillna(0.0)
    df["C1"] = df["N4"] * mz                                       # 确认
    df["C2"] = df["N4"] * (np.abs(mz) < 0.5)                       # 事件动价未动
    df["C3"] = df["N4"] / (1.0 + rz.clip(lower=0.0))               # 波动折减
    df["C4"] = df["N4"] * vz                                       # 放量确认
    df["C5"] = df["N3"] * (np.sign(df["N3"] * mz) > 0)             # 流与价同向
    df["C6"] = df["N4"] * rz                                       # 高波动状态
    # C7 残差信号：N4 对 mom15_z 的滚动回归残差
    cov = (df["N4"] * mz).rolling(ZWIN, min_periods=ZMIN).mean()
    var = (mz**2).rolling(ZWIN, min_periods=ZMIN).mean()
    beta = (cov / var.replace(0, np.nan)).fillna(0.0)
    df["C7"] = df["N4"] - beta * mz
    # C8 未兑现累积：120min PM 创新 z 减 120min 价格动量 z
    dl120 = df["N1"].rolling(120, min_periods=30).sum()
    r120 = df.groupby("seg", sort=False)["r1"].transform(
        lambda s: s.rolling(120, min_periods=30).sum())
    df["C8"] = zscore(dl120).fillna(0.0) - zscore(r120).fillna(0.0)
    df["C9"] = df["N5"] * az                                       # 不平衡×成交额
    df["C10"] = df["N9"] - rz                                      # 信念波动-实现波动差

    # ---- 离散事件与复杂统计指标 K ----
    # 原始事件：主题 1min 创新超过稳健阈值 kappa（30 日滚动 MAD，下限 0.05）。
    dl = df["N1"]
    nz = dl.where(dl != 0.0)
    mad = (nz - nz.rolling(ZWIN, min_periods=ZMIN).median()).abs() \
        .rolling(ZWIN, min_periods=ZMIN).median()
    kappa = (3.0 * 1.4826 * mad).clip(lower=0.05).ffill().fillna(0.05)
    df["kappa"] = kappa
    df["E_up"] = (dl > kappa).astype(int)
    df["E_dn"] = (dl < -kappa).astype(int)
    df["E_any"] = df["E_up"] | df["E_dn"]
    usdc1 = col(prim, "usdc").fillna(0.0)
    q99 = usdc1.where(usdc1 > 0).rolling(ZWIN, min_periods=ZMIN) \
        .quantile(0.99).ffill()
    df["E_big"] = ((usdc1 > q99) & (usdc1 > 0)).astype(int)
    sgn = df["E_up"].astype(int) - df["E_dn"].astype(int)

    df["K1"] = (sgn.rolling(60, min_periods=1).sum())              # 净事件计数
    cnt60 = df["E_any"].rolling(60, min_periods=1).sum()
    hod = df["ts"].dt.hour
    mu = cnt60.groupby(hod).transform(
        lambda s: s.rolling(30 * 6, min_periods=60).mean())
    sd = cnt60.groupby(hod).transform(
        lambda s: s.rolling(30 * 6, min_periods=60).std())
    df["K2"] = ((cnt60 - mu) / sd.replace(0, np.nan)).fillna(0.0)  # 分时频率 z
    e_all = sum(
        (col(t, "dl").fillna(0.0).abs() > kappa).astype(int) for t in themes)
    df["K3"] = pd.Series(e_all, index=df.index).rolling(
        60, min_periods=1).sum()                                   # 跨主题截面计数
    decay = np.zeros(len(df))
    lam = np.exp(-1.0 / 30.0)
    acc = 0.0
    sv = sgn.to_numpy(dtype=float)
    for i in range(len(df)):
        acc = acc * lam + sv[i]
        decay[i] = acc
    df["K4"] = decay                                               # 指数衰减强度
    same_dir = (sgn != 0) & (sgn == sgn.where(sgn != 0).ffill().shift(1))
    run = same_dir.groupby((~same_dir).cumsum()).cumsum()
    df["K5"] = run.rolling(30, min_periods=1).max().fillna(0.0)    # 同向连发
    last_idx = pd.Series(np.where(df["E_any"] == 1, np.arange(len(df)), np.nan),
                         index=df.index).ffill()
    gap_min = (np.arange(len(df)) - last_idx).fillna(480.0).clip(upper=480.0)
    df["K6"] = np.exp(-gap_min / 60.0) * np.sign(
        sgn.where(sgn != 0).ffill().fillna(0.0))                   # 近因得分
    tail = (dl.abs() * (dl.abs() > kappa)).rolling(60, min_periods=1).sum()
    df["K7"] = zscore(tail).fillna(0.0)                            # 尾部幅度和 z
    up120 = df["E_up"].rolling(120, min_periods=1).sum()
    dn120 = df["E_dn"].rolling(120, min_periods=1).sum()
    df["K8"] = ((up120 - dn120)
                / (up120 + dn120).replace(0, np.nan)).fillna(0.0)  # 方向一致率
    ev_pos = pd.Series(np.where(df["E_any"] == 1, np.arange(len(df)), np.nan),
                       index=df.index)
    inter = ev_pos.dropna().diff()
    med_inter = inter.rolling(5, min_periods=2).median()
    heat = (1.0 / (1.0 + med_inter)).reindex(df.index).ffill().fillna(0.0)
    df["K9"] = heat                                                # 到达率热度
    multi = sum(
        ((col(t, "dl").fillna(0.0).abs() > kappa).astype(int)
         .rolling(15, min_periods=1).max()) for t in themes)
    df["K10"] = (pd.Series(multi, index=df.index) >= 2).astype(float) * np.sign(
        df["N2"])                                                  # 跨主题共振
    return df


NUM_SIGNALS = [f"N{i}" for i in range(1, 11)]
COMBO_SIGNALS = [f"C{i}" for i in range(1, 11)]
K_SIGNALS = [f"K{i}" for i in range(1, 11)]
ALL_SIGNALS = NUM_SIGNALS + COMBO_SIGNALS + K_SIGNALS


# ---------------------------------------------------------------- 评估
def ic_table(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """数值信号 IC / RankIC / ICIR（逐日 IC 的 mean/std）。"""
    rows = []
    for sig in ALL_SIGNALS:
        s = df[sig]
        active = s.replace(0.0, np.nan).notna()
        for k in HORIZONS:
            y = df[f"fwd_{k}"]
            ok = active & y.notna()
            n = int(ok.sum())
            if n < 200:
                continue
            sv, yv = s[ok], y[ok]
            pear = float(np.corrcoef(sv, yv)[0, 1])
            rank = float(sv.rank().corr(yv.rank()))
            daily = (
                pd.DataFrame({"d": df.loc[ok, "trade_date"], "s": sv, "y": yv})
                .groupby("d")
                .apply(lambda b: b["s"].corr(b["y"]) if len(b) >= 10 else np.nan,
                       include_groups=False)
                .dropna()
            )
            icir = (float(daily.mean() / daily.std())
                    if len(daily) >= 20 and daily.std() > 0 else np.nan)
            rows.append(
                {"product": product, "signal": sig, "horizon": k, "n": n,
                 "ic": pear, "rank_ic": rank, "ic_daily_mean": float(daily.mean()),
                 "ic_daily_std": float(daily.std()), "icir": icir,
                 "n_days": int(len(daily))}
            )
    return pd.DataFrame(rows)


def signal_stats(df: pd.DataFrame, product: str) -> pd.DataFrame:
    rows = []
    n_days = df["trade_date"].nunique()
    for sig in ALL_SIGNALS:
        s = df[sig].replace([np.inf, -np.inf], np.nan)
        nz = s[(s != 0.0) & s.notna()]
        rows.append(
            {
                "product": product, "signal": sig,
                "n_minutes": int(len(s)), "nonzero": int(len(nz)),
                "nonzero_rate": float(len(nz) / max(len(s), 1)),
                "per_day": float(len(nz) / max(n_days, 1)),
                "mean": float(nz.mean()) if len(nz) else np.nan,
                "std": float(nz.std()) if len(nz) else np.nan,
                "skew": float(nz.skew()) if len(nz) > 10 else np.nan,
                **{f"q{q}": (float(np.percentile(nz, q)) if len(nz) else np.nan)
                   for q in (1, 25, 50, 75, 99)},
            }
        )
    return pd.DataFrame(rows)


def event_study(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """离散事件研究：E_up / E_dn / E_big / K5 连发，1..15min 与直到下一事件。"""
    rows = []
    thresh = 1e-3
    base = {k: float((df[f"fwd_{k}"].abs() > thresh).mean())
            for k in HORIZONS}
    base_absmean = {k: float(df[f"fwd_{k}"].abs().mean() * 1e4)
                    for k in HORIZONS}
    n_months = max(df["trade_date"].nunique() / 21.0, 1e-9)
    specs = {
        "E_up(利多事件)": (df["E_up"] == 1, +1),
        "E_dn(利空事件)": (df["E_dn"] == 1, -1),
        "E_big(美元量爆发)": (df["E_big"] == 1, 0),
        "E_burst(同向连发>=3)": (df["K5"] >= 3, 0),
    }
    ev_idx = np.where(df["E_any"] == 1)[0]
    for name, (mask, direction) in specs.items():
        idx = np.where(mask.to_numpy())[0]
        if len(idx) < 10:
            continue
        row = {"product": product, "event": name, "n": int(len(idx)),
               "per_month": float(len(idx) / n_months)}
        sgn_ev = (np.sign(df["N1"].to_numpy()[idx]) if direction == 0
                  else np.full(len(idx), direction))
        for k in HORIZONS:
            fwd = df[f"fwd_{k}"].to_numpy()[idx]
            ok = np.isfinite(fwd)
            if ok.sum() < 10:
                continue
            row[f"ret_bp_{k}"] = float(np.nanmean(fwd[ok] * sgn_ev[ok]) * 1e4)
            row[f"p_move_{k}"] = float((np.abs(fwd[ok]) > thresh).mean())
            row[f"lift_{k}"] = row[f"p_move_{k}"] / base[k]
        # 直到下一事件（同类不区分方向，用 E_any；上限 240 分钟）
        nxt = np.searchsorted(ev_idx, idx, side="right")
        has = nxt < len(ev_idx)
        tgt = np.where(has, ev_idx[np.minimum(nxt, len(ev_idx) - 1)], -1)
        span = np.where(has, tgt - idx, 240).clip(1, 240)
        logc = np.log(df["close"].to_numpy())
        seg = df["seg"].to_numpy()
        upto = np.full(len(idx), np.nan)
        for j, (i0, sp) in enumerate(zip(idx, span, strict=True)):
            i1 = min(i0 + int(sp), len(df) - 1)
            if seg[i1] == seg[i0]:
                upto[j] = logc[i1] - logc[i0]
        okc = np.isfinite(upto)
        if okc.sum() >= 10:
            row["ret_bp_next"] = float(np.nanmean(upto[okc] * sgn_ev[okc]) * 1e4)
            row["med_span_min"] = float(np.median(span))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["baseline_p"] = base
    out.attrs["baseline_absmean_bp"] = base_absmean
    return out


# ---------------------------------------------------------------- 全历史频率
def history_monthly(con, skip: bool) -> pd.DataFrame:
    """2022-11 至今主题事件的逐月频率（规则匹配全历史市场，vwap 分钟差分）。"""
    if skip:
        return pd.DataFrame()
    cols = "condition_id, market_slug, block_timestamp, p_event, usdc_amount"
    slugs = con.execute(
        f"SELECT condition_id, any_value(market_slug) slug FROM "
        f"{tape.union_sql('condition_id, market_slug')} GROUP BY 1"
    ).fetch_df()
    hits = []
    for rec in slugs.itertuples(index=False):
        m = spm.match_rule(rec.slug or "")
        if m is not None:
            hits.append({"condition_id": rec.condition_id, "theme": m[0],
                         "sigma": m[1]})
    hit = pd.DataFrame(hits)
    print(f"  全历史规则命中市场：{len(hit)}")
    ids = ",".join("'" + c + "'" for c in hit["condition_id"])
    sql = f"""
        SELECT condition_id,
               (block_timestamp // 60) * 60 + 60 AS m_end,
               sum(p_event * usdc_amount) / sum(usdc_amount) AS vwap,
               sum(usdc_amount) AS usdc
        FROM {tape.union_sql(cols)}
        WHERE condition_id IN ({ids}) AND p_event IS NOT NULL
        GROUP BY 1, 2
    """
    mn = con.execute(sql).fetch_df()
    mn = mn.merge(hit, on="condition_id")
    mn = mn.sort_values(["condition_id", "m_end"])
    dl = (logit(mn["vwap"].to_numpy())
          - logit(mn.groupby("condition_id")["vwap"].shift(1).to_numpy()))
    mn["dl"] = mn["sigma"] * dl
    mn["month"] = pd.to_datetime(mn["m_end"], unit="s").dt.strftime("%Y-%m")
    mn["event"] = (np.abs(mn["dl"]) > 0.10).astype(int)  # 全历史统一阈值
    g = mn.groupby(["theme", "month"])
    out = g.agg(
        n_trade_minutes=("dl", "size"),
        n_events=("event", "sum"),
        usdc=("usdc", "sum"),
        n_markets=("condition_id", "nunique"),
    ).reset_index()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-history", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    con = store.connect()
    registry = pd.read_parquet(REGISTRY)

    ics, stats, events, meta = [], [], [], {}
    for product, themes in PRODUCT_THEMES.items():
        print(f"[{product}] 面板 ...")
        panel = cn_panel(product)
        for th in themes:
            pm = pm_minute_theme(
                con, registry, th, product,
                "2026-01-02 00:00:00", "2026-07-13 16:00:00")
            pm = pm.rename(columns={c: f"{c}_{th}" for c in
                                    ("dl", "flow", "usdc", "n", "n_mkts")})
            panel = panel.merge(pm, on="ts", how="left")
        df = add_signals(panel, themes)
        df.to_parquet(OUT / f"panel_{product}.parquet", index=False)
        print(f"[{product}] IC / 统计 / 事件研究 ...")
        ics.append(ic_table(df, product))
        stats.append(signal_stats(df, product))
        ev = event_study(df, product)
        events.append(ev)
        meta[product] = {
            "n_minutes": int(len(df)),
            "n_days": int(df["trade_date"].nunique()),
            "locked_share": float(df["locked"].mean()),
            "pm_active_minute_share": float(
                (df["N1"] != 0).mean()),
            "baseline_p_move": ev.attrs["baseline_p"],
            "baseline_absmean_bp": ev.attrs["baseline_absmean_bp"],
        }

    pd.concat(ics, ignore_index=True).to_parquet(
        OUT / "ic_table.parquet", index=False)
    pd.concat(stats, ignore_index=True).to_parquet(
        OUT / "signal_stats.parquet", index=False)
    pd.concat(events, ignore_index=True).to_parquet(
        OUT / "event_study.parquet", index=False)

    print("全历史逐月频率 ...")
    hist = history_monthly(con, args.skip_history)
    if not hist.empty:
        hist.to_parquet(OUT / "history_monthly.parquet", index=False)

    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"完成，耗时 {time.time() - t0:.0f}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
