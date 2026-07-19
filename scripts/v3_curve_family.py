"""R4　期限结构族：事件信号对近远月价差的响应（预注册，6 格）。

机制：交割篮子供给风险住在曲线前端。中东 / 油价升级 -> 近月稀缺溢价 ->
近远月价差（近-远）走阔；事后回归。价差的对冲腿是远月合约本身，全球
共同因子被差分掉；价差保证金低，是杠杆的合法对象。

口径（先于计算注册）：
  合约对 = 当日 OI 最高的两个合约，按交割月排序为近 / 远；
  spread(t) = ln(近月结算价) − ln(远月结算价)；
  spread_chg(t) 仅当 t 与 t−1 的合约对相同（避免换对伪影）；
  事件 z：SC 用 z(mideast)+z(oil)，FU 用 z(oil)，AU 用 z(metal)+z(fed)。
6 个注册格：
  同期响应：spread_chg(t) ~ z(t)，3 品种（预注册方向：正）；
  事件回归：spread_chg(t+1..t+3) ~ spread_chg(t)，|z|>1 日，3 品种
  （预注册方向：负）。
产物：analysis/v3/prereg/r4_curve.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
INTL = ROOT / "data" / "intl"
XSEC = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "xsec"
OUT = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "prereg"
HAC = {"cov_type": "HAC", "cov_kwds": {"maxlags": 5}}
THEME_MAP = {"SC": ["mideast_conflict", "oil_price"],
             "FU": ["oil_price"],
             "AU": ["metal_price", "fed_policy"]}


def build_spreads(curve: pd.DataFrame, product: str) -> pd.DataFrame:
    sub = curve[curve["product"] == product].copy()
    sub = sub[pd.to_numeric(sub["settle"], errors="coerce").notna()
              & (pd.to_numeric(sub["oi"], errors="coerce") > 0)]
    rows = []
    for d, g in sub.groupby("trade_date"):
        top = g.nlargest(2, "oi").sort_values("delivery_month")
        if len(top) < 2:
            continue
        near, far = top.iloc[0], top.iloc[1]
        rows.append({"trade_date": d,
                     "pair": f"{near['delivery_month']}-"
                             f"{far['delivery_month']}",
                     "spread": float(np.log(float(near["settle"]))
                                     - np.log(float(far["settle"])))})
    s = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    same = s["pair"].eq(s["pair"].shift(1))
    s["spread_chg"] = (s["spread"] - s["spread"].shift(1)).where(same)
    return s


def main() -> int:
    curve = pd.read_parquet(INTL / "curve_daily.parquet")
    z = pd.read_parquet(XSEC / "theme_signals.parquet").set_index(
        "trade_date")
    out = []
    for prod, themes in THEME_MAP.items():
        s = build_spreads(curve, prod).set_index("trade_date")
        zz = sum(z[t] for t in themes).reindex(s.index).fillna(0.0)
        df = pd.DataFrame({"chg": s["spread_chg"], "z": zz}).dropna()
        if len(df) >= 30:
            r = sm.OLS(df["chg"].to_numpy() * 1e4,
                       sm.add_constant(df["z"].to_numpy())).fit(**HAC)
            out.append({"cell": f"R4 同期 {prod}",
                        "beta_bp": float(r.params[1]),
                        "hac_t": float(r.tvalues[1]), "n": int(r.nobs)})
        # 事件后回归
        df["fwd3"] = (df["chg"].shift(-1) + df["chg"].shift(-2)
                      + df["chg"].shift(-3))
        ev = df[(df["z"].abs() > 1)].dropna(subset=["fwd3"])
        if len(ev) >= 15:
            r = sm.OLS(ev["fwd3"].to_numpy() * 1e4,
                       sm.add_constant(ev["chg"].to_numpy() * 1e4)).fit(
                **HAC)
            out.append({"cell": f"R4 回归 {prod}(|z|>1)",
                        "beta_bp": float(r.params[1]),
                        "hac_t": float(r.tvalues[1]), "n": int(r.nobs)})
        else:
            out.append({"cell": f"R4 回归 {prod}(|z|>1)",
                        "beta_bp": float("nan"), "hac_t": float("nan"),
                        "n": int(len(ev))})
    res = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "r4_curve.parquet", index=False)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
