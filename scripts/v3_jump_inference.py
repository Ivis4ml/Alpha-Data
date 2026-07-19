"""P0-2：J 族跳后响应的正确推断（真实日历 + 双重聚类级自助与置换）。

修复 §12.14 登记的两个推断缺陷：
  1. 时段分类改用<b>真实期货分钟网格</b>：跳映射到品种面板的下一分钟，
     延迟 <= 2 分钟记盘中（时段取该分钟的 session 列），否则为闭市跳；
     节假日自动落入闭市（网格里没有分钟），不再依赖星期与钟点规则。
  2. 推断只认事件与交易日两级聚类：
     a) 日聚类 HAC(5)（基准）；
     b) wild cluster bootstrap（Rademacher 权重按交易日整块翻转日均值
        的去均值分量，999 次，检验 H0: 均值 = 0）；
     c) 日期块置换（把事件所在交易日整块循环移位到其他交易日，999 次，
        保留事件的日内聚集与期货收益的自相关，重算均值的零分布）。

注册格（共 4 格，先于计算登记，不扩展）：
  SC 盘中日盘事件跳 x {5', 15'}；M 孤立事件跳（全时段映射）x {5', 15'}。
判定：5% 水平结论要求 wild p < 0.05 且 置换 p < 0.05；
     仅一者通过或均在 [0.05, 0.10) 记 10% 探索证据。

产物：analysis/v3/jump/inference.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
N_BOOT = 999
RNG = np.random.default_rng(20260718)

CELLS = [
    ("SC", ["mideast_conflict", "oil_price"], "day_insession", "盘中日盘"),
    ("M", ["us_china_trade"], "isolated_all", "孤立跳全时段"),
]


def event_collapse(jumps: pd.DataFrame, themes: list[str],
                   isolated_only: bool) -> pd.DataFrame:
    jp = jumps[jumps["theme"].isin(themes)]
    if isolated_only:
        jp = jp[jp["n_cojump"] == 1]
    ev = (jp.groupby(["theme", "ts"])
          .apply(lambda x: float(np.average(
              x["J"], weights=np.sqrt(x["usdc"]))), include_groups=False)
          .rename("J_evt").reset_index())
    return ev


def map_to_panel(ev: pd.DataFrame, panel: pd.DataFrame,
                 mode: str) -> pd.DataFrame:
    ts_idx = panel["ts"].to_numpy()
    minute = ev["ts"].dt.ceil("min")
    pos = np.searchsorted(ts_idx, minute.to_numpy())
    ok = pos < len(ts_idx)
    ev = ev[ok].assign(pos=pos[ok], minute=minute[ok].to_numpy())
    ev["map_ts"] = panel["ts"].to_numpy()[ev["pos"]]
    ev["delay_min"] = ((ev["map_ts"] - ev["minute"])
                       .dt.total_seconds() / 60)
    ev["trade_date"] = panel["trade_date"].to_numpy()[ev["pos"]]
    ev["session"] = panel["session"].to_numpy()[ev["pos"]]
    if mode == "day_insession":
        # 真实网格判据：映射延迟 <= 2 分钟 = 盘中，且该分钟属日盘
        ev = ev[(ev["delay_min"] <= 2) & (ev["session"] == "day")]
    return ev


def infer(vals: pd.Series, days: pd.Series, signs: np.ndarray,
          minutes: pd.Series, panel: pd.DataFrame, h: int,
          all_days: list[str]) -> dict:
    """日聚类 HAC + wild cluster bootstrap + 日期块置换。

    置换设计：事件保持符号与日内时刻，所在交易日整块循环移位 k 天，
    期货响应改取移位后交易日<b>同一时刻</b>的前向收益。保留事件的
    日内聚集与期货收益自相关，破坏事件与收益日的配对。
    """
    daily = (pd.DataFrame({"d": days.to_numpy(), "v": vals.to_numpy()})
             .groupby("d")["v"].mean())
    n_d = len(daily)
    mean_bp = float(vals.mean()) * 1e4
    if n_d < 10:
        return {"mean_bp": mean_bp, "n_events": len(vals), "n_days": n_d}
    res = sm.OLS(daily.to_numpy(), np.ones(n_d)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    hac_t = float(res.tvalues[0])

    # wild cluster bootstrap（日块 Rademacher，t 统计量自助，H0 强加）
    dev = daily.to_numpy() - daily.mean()
    t_star = []
    for _ in range(N_BOOT):
        w = RNG.choice([-1.0, 1.0], size=n_d)
        boot = dev * w
        se = boot.std(ddof=1) / np.sqrt(n_d)
        if se > 0:
            t_star.append(boot.mean() / se)
    t0 = daily.mean() / (daily.std(ddof=1) / np.sqrt(n_d))
    p_wild = float((np.abs(t_star) >= abs(t0)).mean())

    # 日期块置换：预构建 (交易日 -> 日内时刻数组, 前向收益数组)
    tod = (panel["ts"] - panel["ts"].dt.normalize()
           ).dt.total_seconds().to_numpy()
    fwd_all = panel[f"fwd_{h}"].to_numpy()
    by_day: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    day_arr = panel["trade_date"].to_numpy()
    for d in all_days:
        m = day_arr == d
        by_day[d] = (tod[m], fwd_all[m])
    date_pos = {d: i for i, d in enumerate(all_days)}
    ev_day_idx = days.map(date_pos).to_numpy()
    ev_tod = (minutes - minutes.dt.normalize()).dt.total_seconds() \
        .to_numpy()
    n_cal = len(all_days)
    m_null = []
    for _ in range(N_BOOT):
        k = int(RNG.integers(5, n_cal - 5))
        resp = np.full(len(ev_day_idx), np.nan)
        for i, (di, td) in enumerate(zip(ev_day_idx, ev_tod)):
            tgt_t, tgt_f = by_day[all_days[(di + k) % n_cal]]
            if len(tgt_t) == 0:
                continue
            j = min(np.searchsorted(tgt_t, td), len(tgt_t) - 1)
            resp[i] = tgt_f[j]
        ok = np.isfinite(resp)
        if ok.sum() >= 10:
            m_null.append(float(np.mean(signs[ok] * resp[ok])))
    p_perm = (float((np.abs(m_null) >= abs(vals.mean())).mean())
              if m_null else float("nan"))
    return {"mean_bp": mean_bp,
            "daily_mean_bp": float(daily.mean()) * 1e4,
            "n_events": len(vals), "n_days": n_d, "hac_t": hac_t,
            "p_wild": p_wild, "p_perm": p_perm}


def main() -> int:
    jumps = pd.read_parquet(JD / "jumps.parquet")
    rows = []
    for prod, themes, mode, label in CELLS:
        panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                                columns=["ts", "trade_date", "session",
                                         "fwd_5", "fwd_15"])
        all_days = sorted(panel["trade_date"].unique())
        ev = event_collapse(jumps, themes, mode == "isolated_all")
        ev = map_to_panel(ev, panel, mode)
        for h in (5, 15):
            fwd = panel[f"fwd_{h}"].to_numpy()[ev["pos"]]
            val = np.isfinite(fwd)
            sgn = np.sign(ev["J_evt"].to_numpy()[val])
            signed = pd.Series(sgn * fwd[val])
            res = infer(signed, ev["trade_date"][val], sgn,
                        pd.Series(ev["minute"][val]), panel, h, all_days)
            res.update({"product": prod, "cell": f"{label} {h}'",
                        "h": h})
            rows.append(res)
            print(f"{prod} {label} {h}': mean {res['mean_bp']:+.2f}bp "
                  f"n_ev={res['n_events']} n_days={res['n_days']} "
                  f"HAC t={res.get('hac_t', float('nan')):+.2f} "
                  f"wild p={res.get('p_wild', float('nan')):.3f} "
                  f"perm p={res.get('p_perm', float('nan')):.3f}")
    out = pd.DataFrame(rows)
    out.to_parquet(JD / "inference.parquet", index=False)
    print("written inference.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
