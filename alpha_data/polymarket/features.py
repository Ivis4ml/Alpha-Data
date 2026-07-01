"""Polymarket 市场级分钟特征（美东网格，防前视）。

对单个市场（``condition_id``）在 RTH 分钟网格上产出特征。所有特征在 bar 起始时刻 ``T`` 只
使用严格早于 ``T`` 的成交（防前视）；市场解析（``resolved_at``）之后置空，避免结算后的常数
概率泄漏为信号。

特征列（单市场）：

- ``p``            : 事件概率 ``p_event`` 的 LOCF（截至 ``T`` 前最后一笔成交，严格 ``< T``）。
- ``dp_intraday``  : ``p(T)`` 与当日 09:30 的 ``p`` 之差（日内相对开盘变化）。
- ``dp_overnight`` : 当日开盘 ``p`` 与前一交易日收盘 ``p`` 之差（隔夜变化，按日广播）。
- ``flow_session`` : 自当日开盘至 ``T``（不含 ``T``）累计带方向净名义额 ``Σ D×usdc``。
- ``usdc_session`` : 自当日开盘至 ``T`` 累计成交名义额。
- ``n_session``    : 自当日开盘至 ``T`` 累计成交笔数。

``build_feature_panel`` 把多个市场按 ``key`` 前缀合并为一张宽表，供 AlphaForge 作为市场级
替代数据在 ``(trade_date, minute)`` 上注入 ``build_panel``（对所有标的一致广播）。
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from alpha_data.common import calendar
from alpha_data.polymarket import store

FEATURE_COLS: tuple[str, ...] = (
    "p",
    "dp_intraday",
    "dp_overnight",
    "flow_session",
    "usdc_session",
    "n_session",
)


def _fetch_trades(con: duckdb.DuckDBPyConnection, condition_id: str) -> pd.DataFrame:
    """拉取某市场全部成交（时间转美东本地，按时间升序）。"""
    glob = store.daily_aligned_glob().replace("'", "''")
    sql = f"""
        SELECT timezone('America/New_York', to_timestamp(block_timestamp)) AS et_ts,
               p_event,
               D,
               usdc_amount,
               timezone('America/New_York', resolved_at) AS resolved_et
        FROM read_parquet('{glob}')
        WHERE condition_id = ?
        ORDER BY block_timestamp
    """
    df = con.execute(sql, [condition_id]).fetch_df()
    if not df.empty:
        df["et_ts"] = pd.to_datetime(df["et_ts"])
    return df


def build_market_minute_features(con: duckdb.DuckDBPyConnection, condition_id: str) -> pd.DataFrame:
    """单市场分钟特征 ``[trade_date, minute, *FEATURE_COLS]``（拉取成交后委托纯函数计算）。"""
    return features_from_trades(_fetch_trades(con, condition_id))


def features_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """从单市场成交计算分钟特征（纯函数，无 IO，便于单测）。

    Args:
        trades: 含 ``et_ts``（美东本地 Timestamp）、``p_event``、``D``、``usdc_amount``，
            可选 ``resolved_et``（解析时刻，美东本地）。

    Returns:
        ``[trade_date, minute, *FEATURE_COLS]``。
    """
    empty = pd.DataFrame(columns=["trade_date", "minute", *FEATURE_COLS])
    if trades is None or trades.empty:
        return empty
    trades = trades.dropna(subset=["et_ts"]).sort_values("et_ts").reset_index(drop=True)
    if trades.empty:
        return empty

    resolved_et = None
    if "resolved_et" in trades.columns:
        resolved_series = trades["resolved_et"].dropna()
        if not resolved_series.empty:
            resolved_et = pd.Timestamp(resolved_series.iloc[0])

    d0 = trades["et_ts"].min().strftime("%Y-%m-%d")
    d1 = trades["et_ts"].max().strftime("%Y-%m-%d")
    grid = calendar.rth_minute_grid(d0, d1)
    if grid.empty:
        return empty
    grid = grid.sort_values("ts").reset_index(drop=True)

    # p 的 LOCF（严格 < T，防前视）。
    asof = pd.merge_asof(
        grid[["ts"]],
        trades[["et_ts", "p_event"]],
        left_on="ts",
        right_on="et_ts",
        direction="backward",
        allow_exact_matches=False,
    )
    grid["p"] = asof["p_event"].to_numpy()

    # 每分钟成交聚合（floor 到分钟）。
    tr = trades[["et_ts", "D", "usdc_amount"]].copy()
    tr["minute_floor"] = tr["et_ts"].dt.floor("min")
    tr["signed"] = tr["D"].astype("float64") * tr["usdc_amount"].astype("float64")
    permin = tr.groupby("minute_floor").agg(
        net=("signed", "sum"),
        usdc=("usdc_amount", "sum"),
        n=("usdc_amount", "size"),
    )

    grid["dp_intraday"] = np.nan
    grid["flow_session"] = 0.0
    grid["usdc_session"] = 0.0
    grid["n_session"] = 0.0
    day_open_p: dict[str, float] = {}
    day_close_p: dict[str, float] = {}

    for day, block in grid.groupby("trade_date", sort=True):
        day_index = pd.DatetimeIndex(block["ts"])
        s_net = permin["net"].reindex(day_index).fillna(0.0)
        s_usdc = permin["usdc"].reindex(day_index).fillna(0.0)
        s_n = permin["n"].reindex(day_index).fillna(0.0)
        # cumsum 到 T 后 shift(1)：bar T 只含 [开盘, T) 的活动（排除 T 自身，防前视）。
        grid.loc[block.index, "flow_session"] = s_net.cumsum().shift(1).fillna(0.0).to_numpy()
        grid.loc[block.index, "usdc_session"] = s_usdc.cumsum().shift(1).fillna(0.0).to_numpy()
        grid.loc[block.index, "n_session"] = s_n.cumsum().shift(1).fillna(0.0).to_numpy()
        p_open = block["p"].iloc[0]
        day_open_p[day] = p_open
        day_close_p[day] = block["p"].iloc[-1]
        grid.loc[block.index, "dp_intraday"] = (block["p"] - p_open).to_numpy()

    # 隔夜：当日开盘 p - 前一交易日收盘 p。
    overnight: dict[str, float] = {}
    prev_day: str | None = None
    for day in day_open_p:
        if prev_day is not None:
            open_p = day_open_p[day]
            close_p = day_close_p[prev_day]
            both_known = pd.notna(open_p) and pd.notna(close_p)
            overnight[day] = (open_p - close_p) if both_known else np.nan
        else:
            overnight[day] = np.nan
        prev_day = day
    grid["dp_overnight"] = grid["trade_date"].map(overnight).astype("float64")

    # 结算后置空（避免结算常数概率泄漏为信号）。
    if resolved_et is not None:
        mask = grid["ts"] >= resolved_et
        for col in FEATURE_COLS:
            grid.loc[mask, col] = np.nan

    return grid[["trade_date", "minute", *FEATURE_COLS]].reset_index(drop=True)


def build_feature_panel(con: duckdb.DuckDBPyConnection, markets: dict[str, str]) -> pd.DataFrame:
    """把多个市场合并为宽表。

    Args:
        con: DuckDB 连接。
        markets: ``{key: condition_id}``；每个市场的特征列以 ``key`` 前缀命名。

    Returns:
        ``[trade_date, minute, {key}_{feature}...]``，按时间排序（外连接各市场活跃区间）。
    """
    panel: pd.DataFrame | None = None
    for key, condition_id in markets.items():
        feat = build_market_minute_features(con, condition_id)
        if feat.empty:
            continue
        feat = feat.rename(columns={col: f"{key}_{col}" for col in FEATURE_COLS})
        if panel is None:
            panel = feat
        else:
            panel = panel.merge(feat, on=["trade_date", "minute"], how="outer")
    if panel is None:
        return pd.DataFrame(columns=["trade_date", "minute"])
    return panel.sort_values(["trade_date", "minute"]).reset_index(drop=True)
