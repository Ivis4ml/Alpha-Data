"""P3 第二阶段：聪明钱加权的窗口流信号与登记检验。

第一阶段发现钱包技能真实且集中于活跃尾部（持续性秩相关随活跃度单调
上升：>=100 市场 0.11、>=10 万美元 0.15、>=100 万美元 0.16），且技能
钱包占登记主题市场成交额 2-5%。本阶段把 taker 流按 PIT 技能分层，
检验"聪明钱的方向流"是否比"全量流"或"价格信号"携带更多信息。

技能分层（PIT）：逐月 asof 快照内，活跃门槛 n_mkts >= 10 的钱包按
pnl_per_usd 分位数分层，smart = 前 10%、dumb = 后 10%、mid = 中间；
无快照或低于门槛的为 unscored。交易日 t 的成交用 t 所在月初的快照，
任何时点的分层不含该时点之后的结算信息。

窗口：与窗口级研究同口径的 s_pre 窗（前一交易日 15:00 至当日 09:00
北京时间），信号 = sum(orientation * D * usdc) 分层求和后的展开 z
（min 20 日、只用 t-1 前历史）。

事前登记检验族（不扩展）：
  T1 预测（核心问题：知情流是否先于价格）：r_day ~ SF_z（smart 层）
     与 r_day ~ AF_z（全量层对照），锚配对 8 个 x 2 = 16 格。
  T2 吸收增量：r_open ~ s_pre 价格 z + SF_z，SF 系数是否显著，
     mideast x SC 与 oil x SC，2 格。
  T3 截面：P2 相同宇宙与 E1 载荷，把主题价格 z 换成 SF_z 后的
     D2 预测与 D1 吸收截面 IC，2 格。
共 20 格，全部并入检验总账。

产物：data/cn_futures/analysis/v3/wallet/
  smart_flow.parquet    (theme, trade_date, tier) 窗口流
  tests_t1.parquet / tests_t2.parquet / tests_t3.parquet
  meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_cross_section import (  # noqa: E402
    MIN_UNIVERSE, e1_weights, load_returns, load_theme_z)

PM = ROOT / "data" / "polymarket"
V3 = ROOT / "data" / "cn_futures" / "analysis" / "v3"
OUT = V3 / "wallet"
HAC_LAGS = 5
Z_MIN = 20

#: T1 锚配对（主题, 品种）——与窗口级研究的注册映射一致
ANCHORS = [("mideast_conflict", "SC"), ("mideast_conflict", "AU"),
           ("oil_price", "SC"), ("russia_ukraine", "SC"),
           ("russia_ukraine", "AU"), ("fed_policy", "AU"),
           ("fed_policy", "CU"), ("metal_price", "AG")]


def build_windows() -> pd.DataFrame:
    """s_pre 窗口边界（北京 15:00 前收盘 -> 09:00 开盘 = UTC 07:00->01:00）。"""
    d = pd.read_parquet(ROOT / "data" / "cn_futures" / "daily" / "SC.parquet",
                        columns=["trade_date"])
    dates = d["trade_date"].tolist()
    rows = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        start = pd.Timestamp(f"{prev} 07:00:00", tz="UTC")
        end = pd.Timestamp(f"{cur} 01:00:00", tz="UTC")
        rows.append({"trade_date": cur,
                     "start_epoch": int(start.timestamp()),
                     "end_epoch": int(end.timestamp())})
    return pd.DataFrame(rows)


def build_smart_flow(con: duckdb.DuckDBPyConnection,
                     windows: pd.DataFrame) -> pd.DataFrame:
    reg = pd.read_parquet(PM / "features" / "cn_registry_v3.parquet")
    reg_m = (reg[["condition_id", "theme", "orientation", "exploratory"]]
             .drop_duplicates("condition_id"))
    skill = pd.read_parquet(PM / "features" / "wallet_skill_monthly.parquet")
    sk = skill[skill["n_mkts"] >= 10].copy()
    sk["pct"] = sk.groupby("asof")["pnl_per_usd"].rank(pct=True)
    sk["tier"] = pd.cut(sk["pct"], [0, 0.1, 0.9, 1.0],
                        labels=["dumb", "mid", "smart"])
    con.register("reg", reg_m)
    con.register("sk", sk[["wallet", "asof", "tier"]])
    con.register("win", windows)
    q = """
    SELECT r.theme, w.trade_date,
           coalesce(CAST(s.tier AS VARCHAR), 'unscored') AS tier,
           sum(r.orientation * x.D * x.usdc_amount) AS signed_usdc,
           sum(x.usdc_amount) AS usdc,
           count(*) AS n_trades
    FROM (
      SELECT taker, condition_id, usdc_amount, D, block_timestamp,
             strftime(to_timestamp(block_timestamp), '%Y-%m-01') AS asof
      FROM read_parquet(['data/polymarket/daily_aligned/2026-*.parquet',
                         'data/polymarket/features/extension_tape/*.parquet'])
    ) x
    JOIN reg r ON x.condition_id = r.condition_id
    JOIN win w ON x.block_timestamp >= w.start_epoch
              AND x.block_timestamp < w.end_epoch
    LEFT JOIN sk s ON x.taker = s.wallet AND x.asof = s.asof
    GROUP BY 1, 2, 3
    """
    return con.execute(q).df()


def z_expanding(s: pd.Series) -> pd.Series:
    mu = s.expanding(Z_MIN).mean().shift(1)
    sd = s.expanding(Z_MIN).std().shift(1)
    return ((s - mu) / sd).where(sd > 0)


def tier_z(flow: pd.DataFrame, tier: str | None) -> pd.DataFrame:
    """按层（None = 全量）聚合到 (trade_date x theme) 的展开 z 表。"""
    sub = flow if tier is None else flow[flow["tier"] == tier]
    piv = (sub.groupby(["trade_date", "theme"])["signed_usdc"].sum()
           .unstack("theme"))
    return piv.apply(z_expanding).fillna(0.0)


def hac_reg(y: pd.Series, X: pd.DataFrame) -> dict:
    df = pd.concat([y, X], axis=1).dropna()
    if len(df) < 30:
        return {"n": len(df)}
    res = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1:])).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    out = {"n": int(res.nobs)}
    for c in df.columns[1:]:
        out[f"b_{c}"] = float(res.params[c])
        out[f"t_{c}"] = float(res.tvalues[c])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    windows = build_windows()
    flow = build_smart_flow(con, windows)
    flow.to_parquet(OUT / "smart_flow.parquet", index=False)
    print(f"smart_flow {len(flow):,} 行，主题 {flow['theme'].nunique()}，"
          f"层 {sorted(flow['tier'].unique())}")

    sf = tier_z(flow, "smart")
    af = tier_z(flow, None)
    rets = load_returns()
    ret_p = {p: g.set_index("trade_date") for p, g in rets.groupby("product")}

    # smart 层覆盖度：稀疏回归元防护。中位窗口流低于 $1,000 的主题，
    # 展开 z 由尘埃值主导，HAC t 不可靠，标记 thin 并以非零日秩相关复核。
    raw_sm = (flow[flow["tier"] == "smart"]
              .groupby(["trade_date", "theme"])["signed_usdc"].sum()
              .unstack("theme"))
    cover = {}
    for th in raw_sm.columns:
        s = raw_sm[th].fillna(0.0)
        nz = s[s.abs() > 1]
        cover[th] = {"nz_days": int(len(nz)),
                     "med_abs_usd": float(nz.abs().median()) if len(nz)
                     else 0.0}

    # T1：锚配对预测（smart 层 vs 全量层）
    rows = []
    for theme, prod in ANCHORS:
        if theme not in sf.columns:
            continue
        rp = ret_p[prod]["r_day"]
        for name, zz in (("SF", sf), ("AF", af)):
            r = hac_reg(rp.rename("y"), zz[theme].rename("x").to_frame())
            row = {"theme": theme, "product": prod, "signal": name, **r}
            if name == "SF":
                cv = cover.get(theme, {})
                row["thin"] = cv.get("med_abs_usd", 0.0) < 1000
                nz = raw_sm[theme].fillna(0.0)
                pair = pd.concat([nz[nz.abs() > 1].rename("f"), rp],
                                 axis=1).dropna()
                if len(pair) >= 20:
                    sp = spearmanr(pair["f"], pair["r_day"])
                    row["nzday_rank_r"] = float(sp.statistic)
                    row["nzday_rank_p"] = float(sp.pvalue)
            rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_parquet(OUT / "tests_t1.parquet", index=False)

    # T2：吸收增量（价格 z + smart 流 z）
    pz = load_theme_z()
    rows = []
    for theme, prod in [("mideast_conflict", "SC"), ("oil_price", "SC")]:
        rp = ret_p[prod]["r_open"]
        X = pd.DataFrame({"price_z": pz[theme], "smart_z": sf[theme]})
        rows.append({"theme": theme, "product": prod,
                     **hac_reg(rp.rename("y"), X)})
    t2 = pd.DataFrame(rows)
    t2.to_parquet(OUT / "tests_t2.parquet", index=False)

    # T3：截面（E1 载荷，SF 主题 z）
    products = sorted(rets["product"].unique())
    w1 = e1_weights(products)
    themes = [t for t in w1.columns if t in sf.columns]
    score = sf[themes] @ w1[themes].T
    uni = rets[rets["liquid"] & ~rets["roll"]]
    rows = []
    for target in ("r_day", "r_open"):
        ics = []
        for date, g in uni.groupby("trade_date"):
            if date not in score.index:
                continue
            gg = g.set_index("product").join(
                score.loc[date].rename("sc")).dropna(subset=["sc", target])
            gg = gg[gg["sc"].abs() > 0]
            if len(gg) < MIN_UNIVERSE:
                continue
            ics.append({"trade_date": date,
                        "ic": spearmanr(gg["sc"], gg[target]).statistic})
        s = pd.DataFrame(ics).set_index("trade_date")["ic"]
        res = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        rows.append({"target": target, "mean_ic": float(res.params[0]),
                     "hac_t": float(res.tvalues[0]), "n_days": int(len(s))})
    t3 = pd.DataFrame(rows)
    t3.to_parquet(OUT / "tests_t3.parquet", index=False)

    meta = {"design": "P3 第二阶段：登记 20 格（T1 16 + T2 2 + T3 2）",
            "tiers": "PIT 月度快照，n_mkts>=10，pnl_per_usd 前/后 10%",
            "window": "s_pre 同口径（前收盘 15:00 -> 09:00 北京）",
            "hac_lags": HAC_LAGS,
            "smart_coverage": cover}
    (OUT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))

    print("\n=== T1 锚配对预测（r_day ~ 流 z）===")
    cols = [c for c in ("theme", "product", "signal", "n", "b_x", "t_x",
                        "thin", "nzday_rank_r", "nzday_rank_p")
            if c in t1.columns]
    print(t1[cols].round(4).to_string(index=False))
    print("\n=== T2 吸收增量（r_open ~ 价格 z + smart 流 z）===")
    print(t2.round(4).to_string(index=False))
    print("\n=== T3 截面（SF 信号，E1 载荷）===")
    print(t3.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
