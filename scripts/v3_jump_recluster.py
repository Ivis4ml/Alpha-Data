"""外部审计复核二：J 族跳后响应的有效样本重推断。

审计指控：(a) IC 表把滚动保留 120/240 分钟的因子当独立分钟计数
（J5 × M 15 分钟格 n=18,163，实际只有约 53 次孤立跳、31 个交易日），
q 值不可作确认证据；(b) 时段剖面的自助 CI 在跳层面重抽、未按交易日
聚类，SC 盘中日盘 +9.75bp 的区间被高估。

复核设计（对 SC 盘中日盘 15 分钟格）：
  1. (theme, ts) 折叠为事件桶（期限结构共跳记一个事件）；
  2. 事件映射到 SC 分钟网格取符号化 fwd_15；
  3. 同一交易日的事件响应求均值，得逐日序列；
  4. 对逐日序列做 HAC(5) 均值检验，并报告映射延迟分布
     （searchsorted 落点与事件分钟的间隔，检验交易日历粗分类的误差）。
同时输出 J5 × M 的有效样本核算（跳数 / 交易日数 vs IC 表 n）。

产物：analysis/v3/jump/recluster.parquet
"""
from __future__ import annotations

from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
SC_THEMES = ["mideast_conflict", "oil_price"]


def classify_session(ts: pd.Timestamp, night_end: dtime) -> str:
    wd, t = ts.weekday(), ts.time()
    if wd >= 5 and not (wd == 5 and t <= night_end):
        return "闭市·周末"
    if dtime(9, 0) <= t <= dtime(15, 0) and wd < 5:
        return "盘中·日盘"
    if t >= dtime(21, 0) or t <= night_end:
        return "盘中·夜盘"
    if dtime(15, 0) < t < dtime(21, 0):
        return "闭市·傍晚"
    return "闭市·凌晨"


def main() -> int:
    jumps = pd.read_parquet(JD / "jumps.parquet")
    panel = pd.read_parquet(DEF / "panel_SC.parquet",
                            columns=["ts", "trade_date", "fwd_15"])
    ts_idx = panel["ts"].to_numpy()

    # 事件桶折叠 + 时段分类（同原实现：SC 夜盘收 02:30）
    jp = jumps[jumps["theme"].isin(SC_THEMES)].copy()
    ev = (jp.groupby(["theme", "ts"])
          .apply(lambda x: float(np.average(
              x["J"], weights=np.sqrt(x["usdc"]))), include_groups=False)
          .rename("J_evt").reset_index())
    ev["sess"] = ev["ts"].apply(
        lambda x: classify_session(x, dtime(2, 30)))
    day = ev[ev["sess"] == "盘中·日盘"].copy()
    day["minute"] = day["ts"].dt.ceil("min")

    pos = np.searchsorted(ts_idx, day["minute"].to_numpy())
    ok = pos < len(ts_idx)
    day = day[ok].assign(pos=pos[ok])
    day["fwd"] = panel["fwd_15"].to_numpy()[day["pos"]]
    day["map_ts"] = panel["ts"].to_numpy()[day["pos"]]
    day["delay_min"] = ((day["map_ts"] - day["minute"])
                        .dt.total_seconds() / 60)
    day["trade_date"] = panel["trade_date"].to_numpy()[day["pos"]]
    day = day.dropna(subset=["fwd"])
    day["signed"] = np.sign(day["J_evt"]) * day["fwd"]

    exact = float((day["delay_min"] <= 1).mean())
    late = float((day["delay_min"] > 60).mean())

    daily = day.groupby("trade_date")["signed"].mean() * 1e4
    res = sm.OLS(daily.to_numpy(), np.ones(len(daily))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})

    ic = pd.read_parquet(JD / "ic.parquet")
    j5 = ic[(ic["product"] == "M") & (ic["signal"] == "J5")
            & (ic["h"] == 15)].iloc[0]
    iso = jumps[(jumps["theme"] == "us_china_trade")
                & (jumps["n_cojump"] == 1)]

    out = pd.DataFrame([{
        "cell": "SC 盘中日盘 15' (事件桶+日聚类)",
        "n_events": int(len(day)), "n_days": int(len(daily)),
        "mean_bp": float(res.params[0]), "hac_t": float(res.tvalues[0]),
        "orig_mean_bp": 9.75, "orig_n": 455,
        "map_exact_share": exact, "map_late60_share": late,
    }, {
        "cell": "J5×M 15' 有效样本核算",
        "n_events": int(len(iso)),
        "n_days": int(iso["ts"].dt.date.nunique()),
        "mean_bp": float("nan"), "hac_t": float("nan"),
        "orig_mean_bp": float(j5["rank_ic"]), "orig_n": int(j5["n"]),
        "map_exact_share": float("nan"), "map_late60_share": float("nan"),
    }])
    out.to_parquet(JD / "recluster.parquet", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
