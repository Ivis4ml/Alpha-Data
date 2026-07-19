"""P0-2：J 族跳后响应的正确推断（第二版，按第二轮复审修正）。

第二轮复审修正三处（原版缺陷已在报告 §12.15 勘误登记）：
  1. 孤立跳筛选 n_cojump == 0（原版误用 == 1，选中的是双市场共跳；
     n_cojump 定义为同主题同桶的其他市场跳数，v3_jump_factors.py）。
  2. wild cluster bootstrap 的观察统计量与每个自助样本均用 HAC(5)
     学生化（原版用 iid 标准误，与主推断口径不一致）。
  3. 日期块置换：目标分钟按 (session, 钟点秒) 精确匹配，找不到即缺失
     （不回退到当日末根 bar）；有效循环位移全部枚举（不随机抽样）。

注册格（4 格）：SC 盘中日盘事件跳 x {5', 15'}；M 孤立事件跳
（n_cojump=0，全时段映射）x {5', 15'}。
判定：5% 结论要求 wild(HAC) p < 0.05 且置换 p < 0.05。

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


def filter_isolated(jumps: pd.DataFrame) -> pd.DataFrame:
    """孤立跳 = 同主题同桶无其他市场共跳（n_cojump == 0）。"""
    return jumps[jumps["n_cojump"] == 0]


def event_collapse(jumps: pd.DataFrame, themes: list[str],
                   isolated_only: bool) -> pd.DataFrame:
    jp = jumps[jumps["theme"].isin(themes)]
    if isolated_only:
        jp = filter_isolated(jp)
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
        ev = ev[(ev["delay_min"] <= 2) & (ev["session"] == "day")]
    return ev


def hac_t_of(daily: np.ndarray) -> float:
    """常数回归的 HAC(5) t 值（观察与自助共用同一学生化）。"""
    if len(daily) < 10 or np.std(daily) == 0:
        return float("nan")
    res = sm.OLS(daily, np.ones(len(daily))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(res.tvalues[0])


def wild_cluster_p(daily: np.ndarray,
                   rng: np.random.Generator) -> tuple[float, float]:
    """日块 Rademacher wild bootstrap，观察量与自助量均为 HAC(5) t。"""
    t0 = hac_t_of(daily)
    dev = daily - daily.mean()
    t_star = []
    for _ in range(N_BOOT):
        w = rng.choice([-1.0, 1.0], size=len(daily))
        t_star.append(hac_t_of(dev * w))
    t_star_arr = np.array([t for t in t_star if np.isfinite(t)])
    p = float((np.abs(t_star_arr) >= abs(t0)).mean())
    return t0, p


def perm_lookup(panel: pd.DataFrame, all_days: list[str],
                session: str | None) -> dict[str, dict[int, float]]:
    """(交易日 -> {日内秒: 前向收益}) 的精确查找表；session 可选过滤。"""
    sub = panel if session is None else panel[panel["session"] == session]
    tod = ((sub["ts"] - sub["ts"].dt.normalize())
           .dt.total_seconds().astype(int))
    out: dict[str, dict[int, float]] = {d: {} for d in all_days}
    for d, t, f in zip(sub["trade_date"], tod, sub["fwd"]):
        out[d][int(t)] = float(f)
    return out


def date_block_perm_p(ev_days: pd.Series, ev_tod: np.ndarray,
                      signs: np.ndarray, obs_mean: float,
                      lookup: dict[str, dict[int, float]],
                      all_days: list[str]) -> float:
    """枚举全部循环位移；目标分钟精确匹配，缺失即剔除。"""
    date_pos = {d: i for i, d in enumerate(all_days)}
    idx = ev_days.map(date_pos).to_numpy()
    n_cal = len(all_days)
    m_null = []
    for k in range(1, n_cal):
        vals, sg = [], []
        for j, (i, td) in enumerate(zip(idx, ev_tod)):
            day = all_days[(i + k) % n_cal]
            f = lookup[day].get(int(td))
            if f is not None and np.isfinite(f):
                vals.append(f)
                sg.append(signs[j])
        if len(vals) >= 10:
            m_null.append(float(np.mean(np.array(sg) * np.array(vals))))
    if not m_null:
        return float("nan")
    return float((np.abs(m_null) >= abs(obs_mean)).mean())


def infer_cell(signed: pd.Series, days: pd.Series, signs: np.ndarray,
               tod: np.ndarray, lookup: dict[str, dict[int, float]],
               all_days: list[str]) -> dict[str, float | int]:
    daily = (pd.DataFrame({"d": days.to_numpy(), "v": signed.to_numpy()})
             .groupby("d")["v"].mean())
    res: dict[str, float | int] = {
        "mean_bp": float(signed.mean()) * 1e4,
        "daily_mean_bp": float(daily.mean()) * 1e4,
        "n_events": int(len(signed)), "n_days": int(len(daily))}
    if len(daily) < 10:
        return res
    t0, p_wild = wild_cluster_p(daily.to_numpy(), RNG)
    res["hac_t"] = t0
    res["p_wild"] = p_wild
    res["p_perm"] = date_block_perm_p(
        days, tod, signs, float(signed.mean()), lookup, all_days)
    return res


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
        sess = "day" if mode == "day_insession" else None
        for h in (5, 15):
            fwd = panel[f"fwd_{h}"].to_numpy()[ev["pos"]]
            val = np.isfinite(fwd)
            sgn = np.sign(ev["J_evt"].to_numpy()[val])
            signed = pd.Series(sgn * fwd[val])
            tod = ((pd.Series(ev["minute"][val])
                    - pd.Series(ev["minute"][val]).dt.normalize())
                   .dt.total_seconds().to_numpy().astype(int))
            pl = panel.assign(fwd=panel[f"fwd_{h}"])
            lookup = perm_lookup(pl, all_days, sess)
            res = infer_cell(signed, ev["trade_date"][val], sgn, tod,
                             lookup, all_days)
            res.update({"product": prod, "cell": f"{label} {h}'", "h": h})
            rows.append(res)
            print(f"{prod} {label} {h}': mean {res['mean_bp']:+.2f}bp "
                  f"日均 {res.get('daily_mean_bp', float('nan')):+.2f}bp "
                  f"n_ev={res['n_events']} n_days={res['n_days']} "
                  f"HAC t={res.get('hac_t', float('nan')):+.2f} "
                  f"wild(HAC) p={res.get('p_wild', float('nan')):.3f} "
                  f"perm p={res.get('p_perm', float('nan')):.3f}")
    out = pd.DataFrame(rows)
    out.to_parquet(JD / "inference.parquet", index=False)
    print("written inference.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
