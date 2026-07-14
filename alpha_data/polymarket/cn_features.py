"""Polymarket 信号按国内期货交易时段拆分（北京时间网格，防前视）。

与 ``features.py``（美东 RTH 分钟网格）互补：本模块把单个 Polymarket 市场的逐笔
成交转为**国内期货交易日**上的三分量 logit innovation 信号（研究规范 §3.3）：

- ``s_night``：夜盘窗口（前一交易日 21:00 至品种夜盘收盘）内的 innovation。
- ``s_gap``：闭市窗口（前收盘 15:00 至 21:00，加夜盘收盘至当日 09:00）内的 innovation。
- ``s_day``：日盘窗口（09:00 至 15:00）内的 innovation。

方法要点（对应研究规范 §1 扩展与 hint）：

1. **聚合价抑制 bid-ask bounce**：把逐笔按 ``agg_minutes``（默认 15，可调 5 / 15 / 30）
   分桶，桶内按 taker 方向分别取 USDC 加权中位数，再取两个方向的均值；只有单方向
   成交的桶直接用该方向的加权中位数。买卖方向各自的成交价横跨点差，两方向平均
   近似去除 bounce（``D`` 识别 bounce）。
2. **logit 空间**：``p`` 截断到 ``clip``（默认 [0.02, 0.98]）后取 logit，窗口信号 =
   窗口两端 LOCF logit 之差。三窗口信号可加：无缺失时
   ``s_night + s_gap + s_day = 当日收盘 logit − 前日收盘 logit``。
3. **防前视（严格早于标签）**：分桶按左闭右开 ``[left, right)``、桶标签取右端，
   故标签 ``T`` 的桶只含 ``ts < T`` 的成交；窗口 ``[start, end)`` 的信号只用
   ``ts < end`` 的成交，恰好在 ``end`` 时刻的成交归属下一窗口。
4. **p_age 时效**：窗口端点距最后一笔成交超过 ``p_age_max_minutes``（默认 120，可调；
   ``None`` 关闭）时该端点视为失效，对应窗口信号置 NaN，不做无限 LOCF。
5. **结算截断**：窗口结束时刻不早于 ``resolved_at`` 的窗口信号置 NaN（结算后的
   常数概率不是信号）。

时间口径：Polymarket ``block_timestamp`` 为秒级 UTC，转换用 ``Asia/Shanghai``
（UTC+8，无夏令时）；窗口边界由 ``alpha_data.cn_futures.sessions.signal_windows``
给出（北京时间、tz-naive）。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from alpha_data.polymarket import store

#: 三个信号窗口名（列前缀）。
WINDOWS: tuple[str, ...] = ("night", "gap", "day")


def fetch_trades_cn(con: duckdb.DuckDBPyConnection, condition_id: str) -> pd.DataFrame:
    """拉取某市场全部成交（时间转北京本地，按时间升序）。

    Returns:
        ``[cn_ts, p_event, D, usdc_amount, resolved_cn]``；``cn_ts`` 为 tz-naive
        北京时间 Timestamp。
    """
    glob = store.daily_aligned_glob().replace("'", "''")
    sql = f"""
        SELECT timezone('Asia/Shanghai', to_timestamp(block_timestamp)) AS cn_ts,
               p_event,
               D,
               usdc_amount,
               timezone('Asia/Shanghai', resolved_at) AS resolved_cn
        FROM read_parquet('{glob}')
        WHERE condition_id = ?
        ORDER BY block_timestamp
    """
    df = con.execute(sql, [condition_id]).fetch_df()
    if not df.empty:
        df["cn_ts"] = pd.to_datetime(df["cn_ts"])
    return df


def logit(p: np.ndarray | pd.Series, clip: tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    """截断后的 logit 变换。"""
    q = np.clip(np.asarray(p, dtype="float64"), clip[0], clip[1])
    return np.log(q / (1.0 - q))


def aggregate_price(trades: pd.DataFrame, *, agg_minutes: int = 15) -> pd.DataFrame:
    """逐笔 -> 桶级去 bounce 聚合价。

    分桶为左闭右开 ``[left, right)``，桶标签为右端 ``bucket_end``（标签 ``T`` 只含
    ``ts < T`` 的成交）。桶内价格：按 ``D``（+1 买 / -1 卖）分方向取 USDC 加权中位数，
    两方向都有成交时取均值，否则用现有方向。

    Args:
        trades: ``[cn_ts, p_event, D, usdc_amount]``，按 ``cn_ts`` 升序。
        agg_minutes: 桶宽（分钟）。

    Returns:
        ``[bucket_end, p_agg, last_trade_ts, n_trades, usdc]``，按 ``bucket_end`` 升序。
    """
    if trades.empty:
        return pd.DataFrame(columns=["bucket_end", "p_agg", "last_trade_ts", "n_trades", "usdc"])
    df = trades[["cn_ts", "p_event", "D", "usdc_amount"]].copy()
    freq = f"{int(agg_minutes)}min"
    df["bucket_end"] = df["cn_ts"].dt.floor(freq) + pd.Timedelta(minutes=int(agg_minutes))

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        v, w = values[order], weights[order]
        cw = np.cumsum(w)
        cutoff = 0.5 * cw[-1]
        return float(v[np.searchsorted(cw, cutoff)])

    rows: list[dict] = []
    for bucket_end, block in df.groupby("bucket_end", sort=True):
        sides: list[float] = []
        for _, side_block in block.groupby(np.sign(block["D"].astype("float64"))):
            sides.append(
                weighted_median(
                    side_block["p_event"].to_numpy(dtype="float64"),
                    side_block["usdc_amount"].to_numpy(dtype="float64"),
                )
            )
        rows.append(
            {
                "bucket_end": bucket_end,
                "p_agg": float(np.mean(sides)),
                "last_trade_ts": block["cn_ts"].max(),
                "n_trades": len(block),
                "usdc": float(block["usdc_amount"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _locf_at(
    boundaries: pd.Series,
    buckets: pd.DataFrame,
    *,
    clip: tuple[float, float],
    p_age_max_minutes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """在边界时刻求 LOCF 聚合价的 logit 与 p_age（分钟）。

    桶标签本身只含严格早于标签的成交，因此按 ``bucket_end <= T`` 取最近桶即满足
    "严格早于 ``T``"。返回 ``(logit 值, p_age 分钟)``；LOCF 失效（无先行成交或
    超时效）处为 NaN。
    """
    bd = pd.DataFrame({"t": pd.to_datetime(boundaries).astype("datetime64[ns]")})
    bd["_order"] = np.arange(len(bd))
    valid = bd.dropna(subset=["t"]).sort_values("t")
    buckets = buckets.assign(
        bucket_end=pd.to_datetime(buckets["bucket_end"]).astype("datetime64[ns]")
    )
    merged = pd.merge_asof(
        valid,
        buckets[["bucket_end", "p_agg", "last_trade_ts"]],
        left_on="t",
        right_on="bucket_end",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_order")
    lo = np.full(len(bd), np.nan)
    age = np.full(len(bd), np.nan)
    idx = merged["_order"].to_numpy()
    p = merged["p_agg"].to_numpy(dtype="float64")
    age_min = (
        (merged["t"] - merged["last_trade_ts"]).dt.total_seconds().to_numpy() / 60.0
    )
    ok = ~np.isnan(p)
    if p_age_max_minutes is not None:
        ok &= age_min <= float(p_age_max_minutes)
    lo[idx[ok]] = logit(p[ok], clip)
    age[idx] = age_min
    return lo, age


def window_signals(
    trades: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    agg_minutes: int = 15,
    clip: tuple[float, float] = (0.02, 0.98),
    p_age_max_minutes: int | None = 120,
) -> pd.DataFrame:
    """单市场逐笔 -> 逐交易日三分量信号与活动量。

    Args:
        trades: ``fetch_trades_cn`` 的输出（或含同名列的合成数据）。
        windows: ``alpha_data.cn_futures.sessions.signal_windows`` 的输出。
        agg_minutes: 聚合价桶宽（分钟），窗口边界须为其整数倍对齐（15:00、21:00、
            23:00、01:00、02:30、09:00 对 5 / 15 / 30 分钟桶均对齐）。
        clip: logit 前的概率截断区间。
        p_age_max_minutes: 端点时效上限（分钟），``None`` 关闭。

    Returns:
        逐交易日 DataFrame：``trade_date``、``s_night / s_gap / s_day``（logit
        innovation）、细分闭市两段 ``s_gap_pm``（前收盘 15:00 至夜盘开盘 21:00，
        无夜盘品种为整段闭市）与 ``s_gap_am``（夜盘收盘至当日 09:00，无夜盘品种
        为 NaN）、``usdc_* / flow_* / n_*``（各窗口成交名义额 / 带方向净额 /
        笔数）、``p_day_open``（当日 09:00 端点的 LOCF 概率经 logit 反变换前的
        聚合价，供水平特征）、``p_age_day_open``（分钟）。
    """
    out_cols = (
        ["trade_date"]
        + [f"s_{w}" for w in WINDOWS]
        + ["s_gap_pm", "s_gap_am"]
        + [f"usdc_{w}" for w in WINDOWS]
        + [f"flow_{w}" for w in WINDOWS]
        + [f"n_{w}" for w in WINDOWS]
        + ["p_day_open", "p_age_day_open"]
    )
    if trades.empty or windows.empty:
        return pd.DataFrame(columns=out_cols)

    trades = trades.dropna(subset=["cn_ts"]).sort_values("cn_ts").reset_index(drop=True)
    buckets = aggregate_price(trades, agg_minutes=agg_minutes)

    resolved: pd.Timestamp | None = None
    if "resolved_cn" in trades.columns:
        r = trades["resolved_cn"].dropna()
        if not r.empty:
            resolved = pd.Timestamp(r.iloc[0])

    # 各类边界的 LOCF logit 与 p_age。
    boundary_cols = ["gap1_start", "night_start", "night_end", "day_start", "day_end",
                     "gap1_end", "gap2_start", "gap2_end"]
    lo: dict[str, np.ndarray] = {}
    age: dict[str, np.ndarray] = {}
    for col in boundary_cols:
        lo[col], age[col] = _locf_at(
            windows[col], buckets, clip=clip, p_age_max_minutes=p_age_max_minutes
        )

    def seg(start_col: str, end_col: str) -> np.ndarray:
        return lo[end_col] - lo[start_col]

    s_night = seg("night_start", "night_end")
    # gap 为两段之和；无夜盘品种只有 gap1（gap2 为 NaT -> NaN），此时 gap 即整段。
    gap1 = seg("gap1_start", "gap1_end")
    gap2 = seg("gap2_start", "gap2_end")
    has_night_window = windows["night_start"].notna().to_numpy()
    s_gap = np.where(has_night_window, gap1 + gap2, gap1)
    s_day = seg("day_start", "day_end")

    # 细分闭市两段：gap_pm = 前收盘至夜盘开（无夜盘品种为整段闭市），
    # gap_am = 夜盘收至当日 09:00（无夜盘品种无此段）。
    s_gap_pm = gap1
    s_gap_am = np.where(has_night_window, gap2, np.nan)

    # 结算截断：窗口结束时刻 >= resolved 的窗口置 NaN。
    if resolved is not None:
        resolved64 = np.datetime64(resolved)
        for arr, end_col in ((s_night, "night_end"), (s_day, "day_end"),
                             (s_gap_pm, "gap1_end"), (s_gap_am, "gap2_end")):
            end_ts = pd.to_datetime(windows[end_col]).to_numpy()
            arr[pd.notna(end_ts) & (end_ts >= resolved64)] = np.nan
        gap_end = pd.to_datetime(windows["day_start"]).to_numpy()
        s_gap[gap_end >= resolved64] = np.nan

    # 各窗口活动量：逐笔按 [start, end) 归属。
    ts_np = trades["cn_ts"].to_numpy()
    signed = (trades["D"].astype("float64").to_numpy()
              * trades["usdc_amount"].to_numpy(dtype="float64"))
    usdc = trades["usdc_amount"].to_numpy(dtype="float64")
    cum_signed = np.concatenate([[0.0], np.cumsum(signed)])
    cum_usdc = np.concatenate([[0.0], np.cumsum(usdc)])

    def window_sums(start_col: str, end_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = pd.to_datetime(windows[start_col]).to_numpy()
        ends = pd.to_datetime(windows[end_col]).to_numpy()
        flow = np.full(len(windows), np.nan)
        amt = np.full(len(windows), np.nan)
        cnt = np.full(len(windows), np.nan)
        ok = ~(pd.isna(starts) | pd.isna(ends))
        i0 = np.searchsorted(ts_np, starts[ok], side="left")
        i1 = np.searchsorted(ts_np, ends[ok], side="left")
        flow[ok] = cum_signed[i1] - cum_signed[i0]
        amt[ok] = cum_usdc[i1] - cum_usdc[i0]
        cnt[ok] = (i1 - i0).astype("float64")
        return flow, amt, cnt

    flow_night, usdc_night, n_night = window_sums("night_start", "night_end")
    fg1, ug1, ng1 = window_sums("gap1_start", "gap1_end")
    fg2, ug2, ng2 = window_sums("gap2_start", "gap2_end")
    flow_gap = np.where(has_night_window, fg1 + fg2, fg1)
    usdc_gap = np.where(has_night_window, ug1 + ug2, ug1)
    n_gap = np.where(has_night_window, ng1 + ng2, ng1)
    flow_day, usdc_day, n_day = window_sums("day_start", "day_end")

    inv = 1.0 / (1.0 + np.exp(-lo["day_start"]))
    return pd.DataFrame(
        {
            "trade_date": windows["trade_date"].to_numpy(),
            "s_night": s_night, "s_gap": s_gap, "s_day": s_day,
            "s_gap_pm": s_gap_pm, "s_gap_am": s_gap_am,
            "usdc_night": usdc_night, "usdc_gap": usdc_gap, "usdc_day": usdc_day,
            "flow_night": flow_night, "flow_gap": flow_gap, "flow_day": flow_day,
            "n_night": n_night, "n_gap": n_gap, "n_day": n_day,
            "p_day_open": inv,
            "p_age_day_open": age["day_start"],
        }
    )
