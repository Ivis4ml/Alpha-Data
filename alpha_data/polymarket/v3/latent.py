"""Layer 1：从逐笔成交到潜在信念（logit 状态空间滤波）。

框架 §5 的可得数据适配：无订单簿，观测为 15 分钟桶内**买卖两向中位数的均值**
（两向平均近似去除 bid-ask bounce，v1.1 已注册的聚合口径），观测噪声由桶内
微观结构代理建模：

    R_k = (spread_k / 2)^2 + r0^2 / n_k

- ``spread_k``：桶内买方向与卖方向中位数之差（有效点差代理；单向桶回退为
  该市场点差中位数）。
- ``n_k``：桶内笔数（中位数抽样误差随 1/n 收缩）。
- ``r0``：市场级残余噪声尺度（MLE）。

状态方程为局部水平（随机游走）：

    z_k = z_{k-1} + eta_k,   Var(eta_k) = q * dt_k * m(tau_k)

``q`` 为每小时状态方差（MLE），``m(tau)`` 为剩余期限乘子（框架 §5.3 期限异方差；
由 Polymarket 内部合并估计，见 :func:`term_multiplier_curve`）。机械收敛段不进入
方差乘子外推，靠结算前剔除窗口处理（框架 §5.3 第 3 条）。

数据隔离（框架 §5.4 / §11.2）：本模块所有参数（``q`` / ``r0`` / 期限乘子）由
one-step 预测似然在 Polymarket 数据内部选取，不接触期货收益。

实现说明：滤波在全市场共用的桶网格上向量化（时间步循环 + 市场维 numpy 向量），
``(q, r0)`` 网格对全部市场同时求似然，逐市场取 MLE 后再跑一遍抽取滤波状态与
标准化 innovation。
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from alpha_data.polymarket.v3 import tape

#: 默认 (q, r0) 网格（q：每小时 logit 方差；r0：观测噪声尺度）。
Q_GRID: tuple[float, ...] = tuple(float(x) for x in np.geomspace(3e-5, 3.0, 14))
R_GRID: tuple[float, ...] = tuple(float(x) for x in np.geomspace(0.02, 1.0, 8))

#: 期限乘子的剩余天数分桶（右开；最后一桶为 [0, 2)）。
TAU_BUCKETS: tuple[float, ...] = (np.inf, 30.0, 7.0, 2.0)


def bucket_observations(
    con: duckdb.DuckDBPyConnection,
    condition_ids: list[str],
    *,
    agg_minutes: int = 15,
    clip: tuple[float, float] = (0.02, 0.98),
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pd.DataFrame:
    """统一 tape -> 桶级观测（DuckDB 内聚合，全市场一次取回）。

    桶为左闭右开 ``[end - agg, end)``、标签取右端（防前视口径与 v1.1 一致）。
    桶价：按 taker 方向 ``D`` 分侧取中位数再平均（与 v1.1 的 USDC 加权中位数
    差异在于权重；DuckDB 无加权中位数，两向平均才是去 bounce 的关键，偏差
    以观测噪声 ``R_k`` 建模）。

    Returns:
        ``[condition_id, bucket_end, y, n_trades, usdc, spread, last_ts]``，
        ``bucket_end`` 为 UTC 秒，``y`` 为截断 logit，``spread`` 单向桶为 NaN。
    """
    ids = ",".join("'" + c.replace("'", "''") + "'" for c in condition_ids)
    step = int(agg_minutes) * 60
    cols = "condition_id, block_timestamp, p_event, D, usdc_amount"
    time_filter = ""
    if start_ts is not None:
        time_filter += f" AND block_timestamp >= {int(start_ts)}"
    if end_ts is not None:
        time_filter += f" AND block_timestamp < {int(end_ts)}"
    sql = f"""
        WITH sides AS (
            SELECT condition_id,
                   (block_timestamp // {step}) * {step} + {step} AS bucket_end,
                   CASE WHEN D >= 0 THEN 1 ELSE -1 END           AS side,
                   median(p_event)                               AS p_side,
                   count(*)                                      AS n,
                   sum(usdc_amount)                              AS usdc,
                   max(block_timestamp)                          AS last_ts
            FROM {tape.union_sql(cols)}
            WHERE condition_id IN ({ids}) AND p_event IS NOT NULL{time_filter}
            GROUP BY 1, 2, 3
        )
        SELECT condition_id,
               bucket_end,
               avg(p_side)                                       AS p_agg,
               abs(max(CASE WHEN side = 1 THEN p_side END)
                   - max(CASE WHEN side = -1 THEN p_side END))   AS spread,
               sum(n)                                            AS n_trades,
               sum(usdc)                                         AS usdc,
               max(last_ts)                                      AS last_ts
        FROM sides
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df = con.execute(sql).fetch_df()
    p = np.clip(df["p_agg"].to_numpy(dtype="float64"), clip[0], clip[1])
    df["y"] = np.log(p / (1.0 - p))
    # 点差从概率空间换算到 logit 空间（观测方程在 logit 域）：d logit/dp = 1/(p(1-p))。
    df["spread"] = df["spread"].to_numpy(dtype="float64") / (p * (1.0 - p))
    return df[["condition_id", "bucket_end", "y", "n_trades", "usdc", "spread", "last_ts"]]


@dataclass
class FilterOutput:
    """滤波结果。

    Attributes:
        params: ``[condition_id, q, r0, loglik, n_obs]``。
        states: ``[condition_id, bucket_end, z, P, nu, nu_std, S, last_ts]``，
            每个观测桶一行（``z`` 为更新后状态）。
    """

    params: pd.DataFrame
    states: pd.DataFrame


def _dense_arrays(
    obs: pd.DataFrame, *, agg_minutes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """观测长表 -> 桶网格稠密数组 ``(y, half_spread_sq, inv_n, dt_hours, ids, grid)``。

    网格覆盖 ``[min_bucket, max_bucket]`` 的所有 ``agg_minutes`` 步；缺测为 NaN。
    """
    step = agg_minutes * 60
    ids = sorted(obs["condition_id"].unique())
    id_idx = {c: i for i, c in enumerate(ids)}
    t0, t1 = int(obs["bucket_end"].min()), int(obs["bucket_end"].max())
    grid = np.arange(t0, t1 + step, step, dtype="int64")
    t_idx = ((obs["bucket_end"].to_numpy(dtype="int64") - t0) // step).astype("int64")
    m_idx = obs["condition_id"].map(id_idx).to_numpy(dtype="int64")

    shape = (len(grid), len(ids))
    y = np.full(shape, np.nan, dtype="float64")
    hs2 = np.full(shape, np.nan, dtype="float64")
    inv_n = np.full(shape, np.nan, dtype="float64")

    y[t_idx, m_idx] = obs["y"].to_numpy(dtype="float64")
    spread = obs["spread"].to_numpy(dtype="float64")
    med_spread = obs.assign(spread=spread).groupby("condition_id")["spread"].median()
    fallback = med_spread.reindex(ids).fillna(0.02).to_numpy()
    spread_filled = np.where(np.isnan(spread), fallback[m_idx], spread)
    hs2[t_idx, m_idx] = (spread_filled / 2.0) ** 2
    inv_n[t_idx, m_idx] = 1.0 / np.maximum(obs["n_trades"].to_numpy(dtype="float64"), 1.0)

    dt_hours = np.full(shape, np.nan, dtype="float64")
    return y, hs2, inv_n, dt_hours, ids, grid


def _run_filter(
    y: np.ndarray,
    hs2: np.ndarray,
    inv_n: np.ndarray,
    q: np.ndarray,
    r0: np.ndarray,
    *,
    step_hours: float,
    q_mult: np.ndarray | None = None,
    collect_states: bool = False,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """向量化 Kalman 一遍：返回逐市场对数似然（与可选的逐步状态）。

    Args:
        y: ``(T, N)`` 观测（NaN = 缺测）。
        hs2 / inv_n: 同形观测噪声组件。
        q / r0: ``(N,)`` 参数向量。
        step_hours: 网格步长（小时）。
        q_mult: ``(T, N)`` 状态方差乘子（期限异方差），None 为 1。
        collect_states: 是否收集逐步 ``(t, z, P, nu, S)``（仅最终抽取时开启）。
    """
    T, N = y.shape
    z = np.full(N, np.nan)
    P = np.full(N, np.nan)
    loglik = np.zeros(N)
    elapsed = np.zeros(N)  # 自上次观测累计的小时数
    states: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    log2pi = np.log(2.0 * np.pi)

    nan_vec = np.full(N, np.nan)
    for t in range(T):
        elapsed += step_hours
        obs = ~np.isnan(y[t])
        if not obs.any():
            continue
        mult = q_mult[t] if q_mult is not None else 1.0
        R = hs2[t] + (r0 * r0) * inv_n[t]

        first = obs & np.isnan(z)
        cont = obs & ~np.isnan(z)

        nu_out = nan_vec
        S_out = nan_vec
        if cont.any():
            P_pred = P + q * elapsed * mult
            S = P_pred + R
            nu = y[t] - z
            K = P_pred / S
            ll = -0.5 * (log2pi + np.log(S) + nu * nu / S)
            z = np.where(cont, z + K * nu, z)
            P = np.where(cont, (1.0 - K) * P_pred, P)
            loglik = np.where(cont, loglik + ll, loglik)
            if collect_states:
                nu_out = np.where(cont, nu, np.nan)
                S_out = np.where(cont, S, np.nan)
        if first.any():
            z = np.where(first, y[t], z)
            P = np.where(first, 1.0, P)
        if collect_states:
            states.append((t, z.copy(), P.copy(), nu_out, S_out))
        elapsed = np.where(obs, 0.0, elapsed)
    return loglik, states


def fit_filter(
    obs: pd.DataFrame,
    *,
    agg_minutes: int = 15,
    q_grid: tuple[float, ...] = Q_GRID,
    r_grid: tuple[float, ...] = R_GRID,
    q_mult: pd.DataFrame | None = None,
) -> FilterOutput:
    """对全部市场做 ``(q, r0)`` 网格 MLE，再抽取滤波状态与标准化 innovation。

    Args:
        obs: :func:`bucket_observations` 输出。
        agg_minutes: 桶宽（须与 obs 构造一致）。
        q_grid / r_grid: 参数网格。
        q_mult: 可选 ``[condition_id, bucket_end, mult]``（期限乘子长表）。

    Returns:
        :class:`FilterOutput`。``nu_std = nu / sqrt(S)``（一步预测标准化创新）。
    """
    y, hs2, inv_n, _, ids, grid = _dense_arrays(obs, agg_minutes=agg_minutes)
    T, N = y.shape
    step_hours = agg_minutes / 60.0

    mult_arr: np.ndarray | None = None
    if q_mult is not None and not q_mult.empty:
        mult_arr = np.ones((T, N))
        id_idx = {c: i for i, c in enumerate(ids)}
        step = agg_minutes * 60
        t0 = int(grid[0])
        sub = q_mult[q_mult["condition_id"].isin(id_idx)]
        ti = ((sub["bucket_end"].to_numpy(dtype="int64") - t0) // step).astype("int64")
        mi = sub["condition_id"].map(id_idx).to_numpy(dtype="int64")
        keep = (ti >= 0) & (ti < T)
        mult_arr[ti[keep], mi[keep]] = sub["mult"].to_numpy(dtype="float64")[keep]

    best_ll = np.full(N, -np.inf)
    best_q = np.full(N, np.nan)
    best_r = np.full(N, np.nan)
    for q_val in q_grid:
        for r_val in r_grid:
            qv = np.full(N, q_val)
            rv = np.full(N, r_val)
            ll, _ = _run_filter(
                y, hs2, inv_n, qv, rv, step_hours=step_hours, q_mult=mult_arr
            )
            better = ll > best_ll
            best_ll = np.where(better, ll, best_ll)
            best_q = np.where(better, q_val, best_q)
            best_r = np.where(better, r_val, best_r)

    ll_final, states_raw = _run_filter(
        y, hs2, inv_n, best_q, best_r,
        step_hours=step_hours, q_mult=mult_arr, collect_states=True,
    )

    n_obs = (~np.isnan(y)).sum(axis=0)
    params = pd.DataFrame(
        {"condition_id": ids, "q": best_q, "r0": best_r,
         "loglik": ll_final, "n_obs": n_obs}
    )

    frames: list[pd.DataFrame] = []
    obs_mask = ~np.isnan(y)
    last_ts_map = obs.set_index(["condition_id", "bucket_end"])["last_ts"]
    for t, z_t, p_t, nu_t, s_t in states_raw:
        idx = np.flatnonzero(obs_mask[t])
        if idx.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "condition_id": [ids[i] for i in idx],
                    "bucket_end": int(grid[t]),
                    "z": z_t[idx],
                    "P": p_t[idx],
                    "nu": nu_t[idx],
                    "S": s_t[idx],
                }
            )
        )
    states = pd.concat(frames, ignore_index=True)
    states["nu_std"] = states["nu"] / np.sqrt(states["S"])
    key = pd.MultiIndex.from_frame(states[["condition_id", "bucket_end"]])
    states["last_ts"] = last_ts_map.reindex(key).to_numpy()
    return FilterOutput(params=params, states=states)


def term_multiplier_curve(
    states: pd.DataFrame,
    deadlines: pd.DataFrame,
    *,
    buckets: tuple[float, ...] = TAU_BUCKETS,
) -> pd.DataFrame:
    """合并估计期限乘子曲线 ``m(tau)``（框架 §5.3；Polymarket 内部目标）。

    对标准化创新 ``nu_std`` 按剩余天数分桶求方差；若模型完全正确则各桶方差为 1，
    偏离即期限异方差，作为下一轮 ``q_mult`` 的乘子。

    Args:
        states: :func:`fit_filter` 的 ``states``。
        deadlines: ``[condition_id, deadline_ts]``（UTC 秒；市场结算 / 截止时刻）。

    Returns:
        ``[tau_lo_days, tau_hi_days, var_ratio, n]``（按剩余天数降序）。
    """
    df = states.merge(deadlines, on="condition_id", how="inner")
    tau_days = (df["deadline_ts"] - df["bucket_end"]) / 86400.0
    df = df.loc[(tau_days > 0) & df["nu_std"].notna()].copy()
    df["tau_days"] = tau_days[df.index]

    edges = list(buckets) + [0.0]
    rows: list[dict] = []
    for hi, lo in zip(edges[:-1], edges[1:], strict=True):
        blk = df.loc[(df["tau_days"] < hi) & (df["tau_days"] >= lo), "nu_std"]
        rows.append(
            {
                "tau_lo_days": lo,
                "tau_hi_days": hi,
                "var_ratio": float(np.nanvar(blk)) if len(blk) >= 50 else np.nan,
                "n": int(len(blk)),
            }
        )
    return pd.DataFrame(rows)


def state_at_boundaries(
    states: pd.DataFrame,
    boundaries: pd.DataFrame,
    *,
    p_age_max_minutes: float | None = 120.0,
) -> pd.DataFrame:
    """在任意边界时刻取滤波状态的 LOCF（口径与 v1.1 的 ``_locf_at`` 一致）。

    Args:
        states: ``fit_filter().states``（须含 ``last_ts``）。
        boundaries: ``[condition_id, boundary_ts]``（UTC 秒，可重复）。
        p_age_max_minutes: 边界距最后成交超过该分钟数则置 NaN（不做无限 LOCF）。

    Returns:
        输入行序对应的 ``[condition_id, boundary_ts, z, P, p_age_min]``。
    """
    out = boundaries[["condition_id", "boundary_ts"]].copy()
    out["_order"] = np.arange(len(out))
    st = states.sort_values("bucket_end")
    by_cid = {cid: blk for cid, blk in st.groupby("condition_id", sort=False)}
    empty = st.iloc[0:0]
    merged_parts: list[pd.DataFrame] = []
    for cid, blk in out.groupby("condition_id", sort=False):
        s = by_cid.get(cid, empty)
        blk = blk.sort_values("boundary_ts")
        m = pd.merge_asof(
            blk,
            s[["bucket_end", "z", "P", "last_ts"]],
            left_on="boundary_ts",
            right_on="bucket_end",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_parts.append(m)
    merged = pd.concat(merged_parts, ignore_index=True).sort_values("_order")
    age_min = (merged["boundary_ts"] - merged["last_ts"]) / 60.0
    z = merged["z"].to_numpy(dtype="float64")
    if p_age_max_minutes is not None:
        z = np.where(age_min > p_age_max_minutes, np.nan, z)
    return pd.DataFrame(
        {
            "condition_id": merged["condition_id"].to_numpy(),
            "boundary_ts": merged["boundary_ts"].to_numpy(),
            "z": z,
            "P": merged["P"].to_numpy(dtype="float64"),
            "p_age_min": age_min.to_numpy(dtype="float64"),
        }
    )
