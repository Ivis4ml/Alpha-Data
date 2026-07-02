"""Polymarket 市场级分钟特征（美东网格，防前视）。

对单个市场（``condition_id``）在 RTH 分钟网格上产出特征。网格标签 ``T`` 为分钟瞬时
（09:30..收盘，含端点）；所有特征在标签 ``T`` 只使用严格早于 ``T`` 的成交（防前视）；
市场解析（``resolved_at``）之后置空，避免结算后的常数概率泄漏为信号。

标签语义（与 AlphaForge 连接时务必注意）：AlphaForge 分钟面板的 ``minute`` 是 bar 收盘戳
（09:31..16:00）。按 ``(trade_date, minute)`` 等值连接后，特征行 ``T`` 的含义是"截至该 bar
收盘（不含收盘瞬间）已知的信息"，与由该 bar 自身 OHLCV 计算的价格特征处于同一时点口径：
可用于预测 ``T`` 之后的 bar，不可用于解释或"预测"该 bar 自身的收益（那是前视）。

特征列（单市场）：

- ``p``              : 事件概率 ``p_event`` 的 LOCF（截至 ``T`` 前最后一笔成交，严格 ``< T``）。
- ``dp_intraday``    : ``p(T)`` 与当日 09:30 的 ``p`` 之差（日内相对开盘变化）。
- ``dp_overnight``   : 当日开盘 ``p`` 与前一交易日收盘 ``p`` 之差（隔夜变化，按日广播）。
- ``flow_session``   : 自当日开盘至 ``T``（不含 ``T``）累计带方向净名义额 ``Σ D×usdc``。
- ``usdc_session``   : 自当日开盘至 ``T`` 累计成交名义额。
- ``n_session``      : 自当日开盘至 ``T`` 累计成交笔数。
- ``flow_overnight`` : 闭市窗口（前一交易日网格末标签，即 16:00 / 半日 13:00，到当日 09:30，
  含周末与假日）内的带方向净名义额，按日广播；首日无前收盘为 NaN。
- ``usdc_overnight`` : 同窗口累计成交名义额。
- ``n_overnight``    : 同窗口累计成交笔数。

Polymarket 为 7×24 市场，美股闭市时段（往往是事件密集时段，如选举夜）的活动经
``*_overnight`` 聚合到下一交易日，全天广播。

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
    "flow_overnight",
    "usdc_overnight",
    "n_overnight",
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

    # 隔夜活动：闭市窗口 [前一交易日网格末标签, 当日 09:30) 内的成交聚合，按日广播。
    # 窗口起点取前日末标签（16:00 / 半日 13:00，含端点：末标签 bar 只含 < 末标签的成交），
    # 终点 09:30 排除（09:30 起的成交属当日 session，自 09:31 标签起计入 *_session）。
    # 广播到当日全部标签是安全的：窗口内成交均严格早于当日 09:30。
    bounds = grid.groupby("trade_date", sort=True)["ts"].agg(["min", "max"])
    day_list = list(bounds.index)
    ts_np = tr["et_ts"].to_numpy()
    cum_signed = np.concatenate([[0.0], np.cumsum(tr["signed"].to_numpy(dtype="float64"))])
    cum_usdc = np.concatenate([[0.0], np.cumsum(tr["usdc_amount"].to_numpy(dtype="float64"))])
    lo_idx = np.searchsorted(ts_np, bounds["max"].to_numpy()[:-1], side="left")
    hi_idx = np.searchsorted(ts_np, bounds["min"].to_numpy()[1:], side="left")
    on_flow: dict[str, float] = {day_list[0]: np.nan}
    on_usdc: dict[str, float] = {day_list[0]: np.nan}
    on_n: dict[str, float] = {day_list[0]: np.nan}
    for i, day in enumerate(day_list[1:]):
        on_flow[day] = float(cum_signed[hi_idx[i]] - cum_signed[lo_idx[i]])
        on_usdc[day] = float(cum_usdc[hi_idx[i]] - cum_usdc[lo_idx[i]])
        on_n[day] = float(hi_idx[i] - lo_idx[i])
    grid["flow_overnight"] = grid["trade_date"].map(on_flow).astype("float64")
    grid["usdc_overnight"] = grid["trade_date"].map(on_usdc).astype("float64")
    grid["n_overnight"] = grid["trade_date"].map(on_n).astype("float64")

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
