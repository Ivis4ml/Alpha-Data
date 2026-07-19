"""J 族：Polymarket 跳变因子的构建与检验。

动机（正文 §4.4 与用户假设）：套利钉住 Yes+No 恒等式后，价格水平的
跳变只剩两种解释——公共信息到达（信念突变）或私有信息入场（知情流）。
本脚本把跳变形式化为因子族并检验其对国内期货分钟收益的影响。

设计：

1. 跳检测沿用第一部分 E1 的注册口径（15 分钟桶，|Δℓ| > 3×1.4826×
   MAD₄₈ 且桶内成交 ≥ $10k），不另设新阈值（避免探测器层面的自由度）。
2. 每个跳记录属性向量：带方向跳幅 J = orientation×Δℓ、前导流比率
   （跳桶开始前 60 分钟的带方向净流 / 总流）、同桶同主题其他市场跳数
   （同步性，实时可知）；持续性（跳后 4 桶漂移）与结算临近度（用实际
   resolved_at 计算）是**事后属性**，只用于分组事件研究，不进实时因子。
3. 实时因子 J1-J7（日历分钟网格上滚动聚合，采样到期货面板分钟，
   滚动 z 归一化，口径与第三部分 N/C 族一致）：
   J1 跳强度（120 分钟跳数）；J2 带方向跳幅和；J3 跳占比
   JV/(JV+BV)（240 分钟，信念变化中"跳"贡献的份额）；J4 前导流加权
   跳幅和（知情流通道）；J5 孤立跳幅和（同桶无同主题共跳）；
   J6 同步跳幅和（同桶有共跳，公共新闻通道）；J7 跳方向偏度
   （240 分钟升级跳与降级跳的不对称）。
4. 检验：五品种面板 IC / RankIC / 逐日 ICIR（第三部分同协议）+ 族内
   BH-FDR（登记入检验总账）+ 冲突期内外分解 + 按事后属性分组的
   期货响应事件研究（双基线）。

产出 ``data/cn_futures/analysis/v3/jump/``：jumps.parquet、
factors_{P}.parquet、ic.parquet、event_groups.parquet、
monthly.parquet、meta.json。

用法::

    .venv/bin/python scripts/v3_jump_factors.py
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

from alpha_data.polymarket import cn_features  # noqa: E402
from alpha_data.polymarket import store as poly_store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

V3_DIR = ROOT / "data" / "cn_futures" / "analysis" / "v3"
DEF_DIR = V3_DIR / "defense"
OUT_DIR = V3_DIR / "jump"
REGISTRY_V3 = poly_store.FEATURES_DIR / "cn_registry_v3.parquet"

PRODUCT_THEMES: dict[str, list[str]] = {
    "SC": ["mideast_conflict", "oil_price"],
    "AU": ["mideast_conflict", "metal_price", "fed_policy"],
    "AG": ["metal_price", "fed_policy"],
    "CU": ["fed_policy", "us_china_trade"],
    "M": ["us_china_trade"],
}
HORIZONS = (1, 2, 3, 5, 10, 15)
ZWIN, ZMIN = 4800, 960
#: E1 注册口径。
JUMP_MAD_K = 3 * 1.4826
JUMP_MIN_USDC = 10_000
#: 前导流窗口（分钟）。
PREFLOW_MIN = 60
#: 冲突期边界（与第三部分一致）。
HOT_START, HOT_END = "2026-02-01", "2026-03-31"


def zscore(s: pd.Series, win: int = ZWIN, minp: int = ZMIN) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / sd.replace(0.0, np.nan)


# ------------------------------------------------------------ 1. 跳检测
def detect_jumps(themes: set[str]) -> pd.DataFrame:
    """全部映射主题市场上的 E1 口径跳与属性向量。"""
    registry = pd.read_parquet(REGISTRY_V3)
    sub = registry[registry["theme"].isin(themes)].drop_duplicates(
        "condition_id")
    con = poly_store.connect()
    rows: list[dict] = []
    t0 = time.time()
    recs = list(sub.itertuples(index=False))
    for i, rec in enumerate(recs, 1):
        tr = tape.fetch_trades_cn(con, rec.condition_id)
        if tr is None or len(tr) < 50:
            continue
        agg = cn_features.aggregate_price(tr, agg_minutes=15)
        if len(agg) < 60:
            continue
        be = pd.to_datetime(agg["bucket_end"])
        lo = pd.Series(cn_features.logit(agg["p_agg"].to_numpy()),
                       index=be)
        consecutive = lo.index.to_series().diff() == pd.Timedelta(minutes=15)
        dlo = lo.diff().where(consecutive)
        usdc = pd.Series(agg["usdc"].to_numpy(), index=be)
        mad = (dlo - dlo.rolling(48, min_periods=24).median()).abs() \
            .rolling(48, min_periods=24).median()
        is_jump = (dlo.abs() > JUMP_MAD_K * mad) & (usdc >= JUMP_MIN_USDC)

        tr = tr.sort_values("cn_ts")
        tr_ts = pd.to_datetime(tr["cn_ts"]).to_numpy()
        flow = (tr["D"].to_numpy(dtype="float64")
                * tr["usdc_amount"].to_numpy(dtype="float64"))
        vol = tr["usdc_amount"].to_numpy(dtype="float64")
        cum_flow = np.concatenate([[0.0], np.cumsum(flow)])
        cum_vol = np.concatenate([[0.0], np.cumsum(vol)])

        res_cn = None
        if pd.notna(getattr(rec, "resolved_at", None)):
            res_cn = (pd.to_datetime(rec.resolved_at, utc=True)
                      .tz_convert("Asia/Shanghai").tz_localize(None))

        dlo_np = dlo.to_numpy()
        for pos in np.flatnonzero(is_jump.fillna(False).to_numpy()):
            ts = be.iloc[pos]
            b_start = ts - pd.Timedelta(minutes=15)
            i0 = np.searchsorted(tr_ts, (b_start - pd.Timedelta(
                minutes=PREFLOW_MIN)).to_datetime64())
            i1 = np.searchsorted(tr_ts, b_start.to_datetime64())
            pf_flow = cum_flow[i1] - cum_flow[i0]
            pf_vol = cum_vol[i1] - cum_vol[i0]
            post4 = np.nansum(dlo_np[pos + 1:pos + 5]) \
                if pos + 1 < len(dlo_np) else np.nan
            rows.append({
                "condition_id": rec.condition_id, "theme": rec.theme,
                "ts": ts, "dlo": float(dlo_np[pos]),
                "J": float(rec.orientation) * float(dlo_np[pos]),
                "usdc": float(usdc.iloc[pos]),
                "preflow_ratio": float(rec.orientation) * float(
                    pf_flow / pf_vol) if pf_vol > 0 else np.nan,
                "preflow_vol": float(pf_vol),
                "post4_drift": float(rec.orientation) * float(post4)
                if np.isfinite(post4) else np.nan,
                "days_to_res": float((res_cn - ts).days)
                if res_cn is not None else np.nan,
            })
        if i % 50 == 0:
            print(f"  跳检测 {i}/{len(recs)}，累计 {len(rows)} 跳，"
                  f"{time.time() - t0:.0f}s")
    jumps = pd.DataFrame(rows)
    # 同步性：同主题同桶的其他市场跳数（bucket_end 相同，实时可知）
    grp = jumps.groupby(["theme", "ts"])["condition_id"].transform("count")
    jumps["n_cojump"] = (grp - 1).astype(int)
    return jumps.sort_values("ts").reset_index(drop=True)


# ------------------------------------------------- 2. 分钟级因子构建
def build_factors(jumps: pd.DataFrame, product: str) -> pd.DataFrame:
    """产品面板分钟网格上的 J1-J7（滚动 z 归一化）。"""
    panel = pd.read_parquet(DEF_DIR / f"panel_{product}.parquet")
    themes = PRODUCT_THEMES[product]
    jp = jumps[jumps["theme"].isin(themes)].copy()
    jp["minute"] = jp["ts"].dt.ceil("min")

    t0 = panel["ts"].min() - pd.Timedelta(days=3)
    t1 = panel["ts"].max()
    grid = pd.date_range(t0, t1, freq="min")

    def impulse(values: pd.Series, idx: pd.Series) -> pd.Series:
        s = pd.Series(values.to_numpy(), index=idx.to_numpy())
        s = s.groupby(level=0).sum()
        return s.reindex(grid).fillna(0.0)

    j_sgn = impulse(jp["J"], jp["minute"])
    j_cnt = impulse(pd.Series(1.0, index=jp.index), jp["minute"])
    pf = jp["preflow_ratio"].fillna(0.0)
    j_pf = impulse(jp["J"] * pf, jp["minute"])
    iso = (jp["n_cojump"] == 0).astype(float)
    j_iso = impulse(jp["J"] * iso, jp["minute"])
    j_syn = impulse(jp["J"] * (1.0 - iso), jp["minute"])
    n_up = impulse((jp["J"] > 0).astype(float), jp["minute"])
    n_dn = impulse((jp["J"] < 0).astype(float), jp["minute"])
    # 跳占比：全部信念变化平方中跳的份额（映射主题的全部桶）
    # 用跳幅平方 vs 主题全部 Δℓ 平方——后者以 15 分钟桶总方差近似：
    # 由跳表不可得非跳桶，改用 |J| 和与跳数的组合近似强度；J3 定义为
    # 240 分钟跳幅平方和的滚动 z（纯跳能量，声明与 BNS 分解的差异）。
    j_sq = impulse(jp["J"] ** 2, jp["minute"])

    roll120 = {"J1": j_cnt, "J2": j_sgn, "J4": j_pf, "J5": j_iso,
               "J6": j_syn}
    out = pd.DataFrame(index=grid)
    for name, s in roll120.items():
        out[name] = s.rolling(120, min_periods=1).sum()
    out["J3"] = j_sq.rolling(240, min_periods=1).sum()
    up240 = n_up.rolling(240, min_periods=1).sum()
    dn240 = n_dn.rolling(240, min_periods=1).sum()
    tot = up240 + dn240
    out["J7"] = ((up240 - dn240) / tot.replace(0.0, np.nan)).fillna(0.0)

    sampled = out.reindex(panel["ts"].to_numpy())
    sampled.index = panel.index
    for c in sampled.columns:
        panel[c] = zscore(sampled[c])
        panel[f"{c}_raw"] = sampled[c]
    return panel


# ------------------------------------------------------------ 3. IC 检验
def ic_table(panel: pd.DataFrame, product: str) -> pd.DataFrame:
    """J 族 × 视界的 IC / RankIC / 逐日 ICIR（第三部分同协议）。"""
    from scipy import stats as sps

    rows: list[dict] = []
    hot = (panel["trade_date"] >= HOT_START) & (panel["trade_date"] <= HOT_END)
    for sig in [f"J{k}" for k in range(1, 8)]:
        if sig not in panel.columns:
            continue
        for h in HORIZONS:
            x = panel[sig]
            y = panel[f"fwd_{h}"]
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n < 1000 or x[ok].std() == 0:
                continue
            pear = sps.pearsonr(x[ok], y[ok])
            spear = sps.spearmanr(x[ok], y[ok])
            daily = []
            for _, g in panel.loc[ok, ["trade_date", sig, f"fwd_{h}"]] \
                    .groupby("trade_date"):
                if len(g) >= 10 and g[sig].std() > 0:
                    daily.append(float(
                        sps.pearsonr(g[sig], g[f"fwd_{h}"]).statistic))
            icir = (float(np.mean(daily) / np.std(daily, ddof=1))
                    if len(daily) >= 20 else np.nan)
            sub = panel.loc[ok & hot], panel.loc[ok & ~hot]
            ic_hot = (float(sps.spearmanr(sub[0][sig],
                                          sub[0][f"fwd_{h}"]).statistic)
                      if len(sub[0]) >= 500 else np.nan)
            ic_cold = (float(sps.spearmanr(sub[1][sig],
                                           sub[1][f"fwd_{h}"]).statistic)
                       if len(sub[1]) >= 500 else np.nan)
            rows.append({
                "product": product, "signal": sig, "h": h, "n": n,
                "ic": float(pear.statistic), "p_ic": float(pear.pvalue),
                "rank_ic": float(spear.statistic),
                "p_rank": float(spear.pvalue),
                "n_days": len(daily),
                "ic_mean": float(np.mean(daily)) if daily else np.nan,
                "icir": icir,
                "rank_ic_hot": ic_hot, "rank_ic_cold": ic_cold,
            })
    return pd.DataFrame(rows)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    q = np.full(m, np.nan)
    cummin = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        cummin = min(cummin, p[i] * m / (k + 1))
        q[i] = cummin
    return q


# ----------------------------------------- 4. 按属性分组的事件研究
def event_groups(jumps: pd.DataFrame) -> pd.DataFrame:
    """SC 上按事后属性分组的跳后期货响应（双基线口径）。"""
    panel = pd.read_parquet(DEF_DIR / "panel_SC.parquet")
    ts_idx = panel["ts"].to_numpy()
    jp = jumps[jumps["theme"].isin(PRODUCT_THEMES["SC"])].copy()
    jp["minute"] = jp["ts"].dt.ceil("min")

    def response(sub: pd.DataFrame, h: int) -> tuple[float, float, int]:
        pos = np.searchsorted(ts_idx, sub["minute"].to_numpy())
        ok = pos < len(ts_idx)
        pos = pos[ok]
        sgn = np.sign(sub["J"].to_numpy()[ok])
        fwd = panel[f"fwd_{h}"].to_numpy()[pos]
        val = np.isfinite(fwd)
        if val.sum() < 10:
            return np.nan, np.nan, int(val.sum())
        signed = fwd[val] * sgn[val]
        return (float(np.nanmean(signed)) * 1e4,
                float((signed > 0).mean()), int(val.sum()))

    base_abs = {h: float(panel[f"fwd_{h}"].dropna().abs().mean()) * 1e4
                for h in (5, 15)}
    groups: list[tuple[str, pd.DataFrame]] = []
    pf = jumps["preflow_ratio"]
    q_lo, q_hi = pf.quantile(0.33), pf.quantile(0.67)
    groups.append(("前导流高（≥q67）", jp[jp["preflow_ratio"] >= q_hi]))
    groups.append(("前导流低（≤q33）", jp[jp["preflow_ratio"] <= q_lo]))
    groups.append(("孤立跳（同桶无共跳）", jp[jp["n_cojump"] == 0]))
    groups.append(("同步跳（同桶 ≥1 共跳）", jp[jp["n_cojump"] >= 1]))
    groups.append(("临近结算（≤7 天）", jp[jp["days_to_res"] <= 7]))
    groups.append(("远离结算（>30 天）", jp[jp["days_to_res"] > 30]))
    groups.append(("持续跳（post4 同号）",
                   jp[jp["post4_drift"] * jp["J"] > 0]))
    groups.append(("回吐跳（post4 反号）",
                   jp[jp["post4_drift"] * jp["J"] < 0]))
    groups.append(("全部跳（无条件）", jp))

    rows = []
    for name, sub in groups:
        for h in (5, 15):
            mean_bp, share, n = response(sub, h)
            rows.append({
                "group": name, "h": h, "n": n,
                "signed_bp": mean_bp, "share_pos": share,
                "base_abs_bp": base_abs[h],
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    themes = {t for ts in PRODUCT_THEMES.values() for t in ts}

    jumps = detect_jumps(themes)
    jumps.to_parquet(OUT_DIR / "jumps.parquet", index=False)
    print(f"[1/4] 跳 {len(jumps)} 个（{jumps['condition_id'].nunique()} "
          f"市场），{time.time() - t0:.0f}s")

    ics = []
    for product in PRODUCT_THEMES:
        panel = build_factors(jumps, product)
        keep = ["ts", "trade_date"] + \
            [c for c in panel.columns if c.startswith("J")]
        panel[keep].to_parquet(OUT_DIR / f"factors_{product}.parquet",
                               index=False)
        ics.append(ic_table(panel, product))
        print(f"[2/4] {product} 因子与 IC 完成，{time.time() - t0:.0f}s")
    ic = pd.concat(ics, ignore_index=True)
    ic["q_bh"] = bh_fdr(ic["p_rank"].to_numpy())
    ic.to_parquet(OUT_DIR / "ic.parquet", index=False)

    ev = event_groups(jumps)
    ev.to_parquet(OUT_DIR / "event_groups.parquet", index=False)
    print(f"[3/4] 分组事件研究 {len(ev)} 行，{time.time() - t0:.0f}s")

    jm = jumps.assign(month=jumps["ts"].dt.strftime("%Y-%m")) \
        .groupby(["month", "theme"]).size().rename("n").reset_index()
    jm.to_parquet(OUT_DIR / "monthly.parquet", index=False)

    meta = {
        "n_jumps": int(len(jumps)),
        "n_markets": int(jumps["condition_id"].nunique()),
        "family_cells": int(len(ic)),
        "detector": "E1 口径：|dlogit|>3*1.4826*MAD48 且桶成交>=1e4",
        "preflow_min": PREFLOW_MIN,
        "realtime_factors": ["J1 跳强度", "J2 带方向跳幅和", "J3 跳能量",
                             "J4 前导流加权", "J5 孤立跳", "J6 同步跳",
                             "J7 方向偏度"],
        "expost_attrs": ["post4_drift", "days_to_res"],
        "attr_stats": {
            "preflow_ratio": jumps["preflow_ratio"].describe()
            .round(3).to_dict(),
            "share_isolated": float((jumps["n_cojump"] == 0).mean()),
        },
    }
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[4/4] 完成，共 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
