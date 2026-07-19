"""第六轮预注册假设的定义与冻结样本初值（R1 / R2 / R3）。

本脚本同时承担两件事：(a) 以代码形式固化三个预注册假设的精确定义，
(b) 在已冻结的 2026-01-05 至 07-13 样本上计算初值。该样本已被本研究
反复使用，初值一律标注为"数据重用后的探索值"；裁决以 2026-07-18 之后
的未触碰样本为准（判定标准见 meta 与报告 §12.16）。

R1　吸收赢家的次日截面回吐（D3）
    宇宙 = P2 同口径（流动性门槛、非换月、|E1 得分| > 0）。逐日以
    r_open(t)（开盘前累积收益）截面排序，目标 = 品种 t+1..t+3 收对收
    累计收益的截面去均值。预注册方向：截面 RankIC 为负（事件日重定价
    过冲的相对回吐），高信号离散日更负。
R2　SC 相对残差的事件条件回归
    残差 = SC 收对收收益对 [Brent(t), Brent(t-1), USDCNH(t)] 的展开窗
    （min 40 日、严格 < t）回归残差；Brent 为 EIA 日频现货、跨时区
    以双滞后覆盖，登记为代理口径。事件日 = 中东与油价参考 z 绝对值
    之和 > 2。预注册方向：resid(t+1..t+3) 对 resid(t) 的系数在事件日
    显著为负（回归），非事件日不显著。
R3　尾部风险旗标（overlay 的正确损失函数）
    旗标(t) = 八主题 |z_pre| 之和的展开分位 >= 0.9（min 30 日、仅用
    < t 信息）。结果 = SC |r_cc(t)| >= 展开 90 分位（PIT）。预注册
    判定：精确率相对基率提升（lift）>= 2 且 Fisher 精确 p < 0.05；
    另报告"旗标日降杠杆至 0"对持有 SC 组合的回撤影响（描述性）。

产物：data/cn_futures/analysis/v3/prereg/{r1_d3,r2_resid,r3_flag}.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[1]
XSEC = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "xsec"
DAILY = ROOT / "data" / "cn_futures" / "daily"
INTL = ROOT / "data" / "intl"
OUT = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "prereg"
HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 5}}


def hac_mean(s: pd.Series) -> tuple[float, float, int]:
    s = s.dropna()
    if len(s) < 15:
        return float("nan"), float("nan"), len(s)
    r = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(**HAC)
    return float(r.params[0]), float(r.tvalues[0]), len(s)


def load_cc() -> pd.DataFrame:
    rows = []
    for fp in sorted(DAILY.glob("*.parquet")):
        d = pd.read_parquet(fp, columns=["trade_date", "r_cc", "roll"])
        d["product"] = fp.stem
        rows.append(d)
    cc = pd.concat(rows, ignore_index=True)
    cc.loc[cc["roll"].astype(bool), "r_cc"] = np.nan
    return cc[["product", "trade_date", "r_cc"]]


def r1_d3() -> pd.DataFrame:
    panel = pd.read_parquet(XSEC / "panel.parquet")
    cc = load_cc()
    dates = sorted(panel["trade_date"].unique())
    dpos = {d: i for i, d in enumerate(dates)}
    cc_piv = cc.pivot(index="trade_date", columns="product", values="r_cc")
    cc_piv = cc_piv.reindex(dates)
    fwd3 = (cc_piv.shift(-1) + cc_piv.shift(-2) + cc_piv.shift(-3))

    uni = panel[panel["liquid"] & ~panel["roll"]
                & (panel["score_e1"].abs() > 0)].dropna(subset=["r_open"])
    disp = uni.groupby("trade_date")["score_e1"].std()
    hi_cut = disp.quantile(2 / 3)
    rows = []
    for date, g in uni.groupby("trade_date"):
        if dpos[date] >= len(dates) - 3 or len(g) < 30:
            continue
        f = fwd3.loc[date, g["product"]].to_numpy()
        ok = np.isfinite(f)
        if ok.sum() < 30:
            continue
        rel = f[ok] - np.nanmean(f[ok])
        ic = spearmanr(g["r_open"].to_numpy()[ok], rel).statistic
        rows.append({"trade_date": date, "ic": ic, "n_cs": int(ok.sum()),
                     "hi_disp": bool(disp.loc[date] >= hi_cut)})
    d = pd.DataFrame(rows)
    out = []
    for name, sub in (("全部日", d), ("高离散", d[d["hi_disp"]]),
                      ("低离散", d[~d["hi_disp"]])):
        m, t, n = hac_mean(sub.set_index("trade_date")["ic"])
        out.append({"cell": f"R1 {name}", "mean_ic": m, "hac_t": t,
                    "n_days": n})
    return pd.DataFrame(out)


def r2_resid() -> pd.DataFrame:
    sc = pd.read_parquet(DAILY / "SC.parquet",
                         columns=["trade_date", "r_cc", "roll"])
    sc.loc[sc["roll"].astype(bool), "r_cc"] = np.nan
    brent = pd.read_csv(INTL / "brent_daily.csv")
    brent = brent[pd.to_numeric(brent["value"], errors="coerce").notna()]
    brent["value"] = brent["value"].astype(float)
    brent = brent.sort_values("timestamp")
    brent["ret"] = np.log(brent["value"]).diff()
    cnh = pd.read_csv(INTL / "usdcnh_daily.csv").sort_values("timestamp")
    cnh["ret"] = np.log(cnh["close"]).diff()

    df = sc.rename(columns={"trade_date": "d"})[["d", "r_cc"]]
    df = df.merge(brent[["timestamp", "ret"]].rename(
        columns={"timestamp": "d", "ret": "brent"}), on="d", how="left")
    df = df.merge(cnh[["timestamp", "ret"]].rename(
        columns={"timestamp": "d", "ret": "cnh"}), on="d", how="left")
    df["brent"] = df["brent"].ffill(limit=3)
    df["cnh"] = df["cnh"].ffill(limit=3)
    df["brent_l1"] = df["brent"].shift(1)
    df = df.dropna().reset_index(drop=True)

    # 展开窗残差（严格 < t）
    X = np.column_stack([np.ones(len(df)),
                         df[["brent", "brent_l1", "cnh"]].to_numpy()])
    y = df["r_cc"].to_numpy()
    resid = np.full(len(df), np.nan)
    for t in range(40, len(df)):
        beta, *_ = np.linalg.lstsq(X[:t], y[:t], rcond=None)
        resid[t] = y[t] - X[t] @ beta
    df["resid"] = resid

    z = pd.read_parquet(XSEC / "theme_signals.parquet").set_index(
        "trade_date")
    ev = (z["mideast_conflict"].abs() + z["oil_price"].abs()) > 2
    df["event"] = df["d"].map(ev).fillna(False)
    df["resid_fwd3"] = (pd.Series(resid).shift(-1) + pd.Series(resid)
                        .shift(-2) + pd.Series(resid).shift(-3))

    out = []
    for name, sub in (("事件日", df[df["event"]]),
                      ("非事件日", df[~df["event"]])):
        s = sub.dropna(subset=["resid", "resid_fwd3"])
        if len(s) < 10:
            out.append({"cell": f"R2 {name}", "beta": float("nan"),
                        "hac_t": float("nan"), "n": len(s)})
            continue
        r = sm.OLS(s["resid_fwd3"].to_numpy(),
                   sm.add_constant(s["resid"].to_numpy())).fit(**HAC)
        out.append({"cell": f"R2 {name}", "beta": float(r.params[1]),
                    "hac_t": float(r.tvalues[1]), "n": int(r.nobs)})
    return pd.DataFrame(out)


def r3_flag() -> pd.DataFrame:
    z = pd.read_parquet(XSEC / "theme_signals.parquet").set_index(
        "trade_date")
    act = z.abs().sum(axis=1)
    thr = act.expanding(30).quantile(0.9).shift(1)
    flag = act >= thr
    sc = pd.read_parquet(DAILY / "SC.parquet",
                         columns=["trade_date", "r_cc"]).set_index(
        "trade_date")
    a = sc["r_cc"].abs()
    ethr = a.expanding(30).quantile(0.9).shift(1)
    extreme = (a >= ethr)
    df = pd.DataFrame({"flag": flag, "extreme": extreme,
                       "r_cc": sc["r_cc"]}).dropna()
    tab = pd.crosstab(df["flag"], df["extreme"])
    odds, p = fisher_exact(tab)
    base = df["extreme"].mean()
    prec = df.loc[df["flag"], "extreme"].mean() if df["flag"].any() else np.nan
    rec = (df.loc[df["extreme"], "flag"].mean()
           if df["extreme"].any() else np.nan)
    # overlay 描述：持有 SC，旗标日仓位 0
    ret_hold = df["r_cc"].fillna(0.0)
    ret_ovl = ret_hold.where(~df["flag"], 0.0)

    def mdd(r: pd.Series) -> float:
        nav = (1 + r).cumprod()
        return float((nav / nav.cummax() - 1).min())

    return pd.DataFrame([{
        "cell": "R3 旗标", "n_days": int(len(df)),
        "n_flag": int(df["flag"].sum()),
        "base_rate": float(base), "precision": float(prec),
        "recall": float(rec), "lift": float(prec / base) if base else np.nan,
        "fisher_p": float(p),
        "mdd_hold": mdd(ret_hold), "mdd_overlay": mdd(ret_ovl),
        "vol_hold": float(ret_hold.std() * np.sqrt(244)),
        "vol_overlay": float(ret_ovl.std() * np.sqrt(244)),
    }])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = r1_d3(); r1.to_parquet(OUT / "r1_d3.parquet", index=False)
    print(r1.round(4).to_string(index=False))
    r2 = r2_resid(); r2.to_parquet(OUT / "r2_resid.parquet", index=False)
    print(r2.round(4).to_string(index=False))
    r3 = r3_flag(); r3.to_parquet(OUT / "r3_flag.parquet", index=False)
    print(r3.round(4).to_string(index=False))
    meta = {
        "status": "预注册（2026-07-18），冻结样本初值为数据重用后的探索值",
        "verdict_sample": "2026-07-18 之后未触碰样本，一次性裁决",
        "r1": "全部日 IC<0 且 HAC|t|>=2，高离散更负",
        "r2": "事件日 beta<0 且 HAC|t|>=2，非事件日不显著",
        "r3": "lift>=2 且 Fisher p<0.05",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                              indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
