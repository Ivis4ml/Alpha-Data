"""美股 NYSE 交易日历与 RTH 分钟网格（美东时间）。

基于 ``pandas_market_calendars``（XNYS），自动处理节假日与半日提前收盘。供美股转换与
Polymarket 特征归并共用，保证两侧 ``(trade_date, minute)`` 口径一致。

分钟网格对每个交易日产出 09:30..收盘（含端点，即含 16:00，半日含 13:00）的全部分钟标签，
``minute`` 为 ``'HH:MM'`` 字符串。端点齐全的标签集合使得无论 equity 侧以 bar 起始
（09:30..15:59）还是 bar 收盘（09:31..16:00）标注，都能在 ``(trade_date, minute)`` 上 join。
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal

_TZ = "America/New_York"


@lru_cache(maxsize=1)
def _nyse():
    return mcal.get_calendar("XNYS")


def schedule(start: str, end: str) -> pd.DataFrame:
    """返回 ``[trade_date, open_et, close_et]``（美东本地、tz-naive），逐交易日。"""
    sch = _nyse().schedule(start_date=str(start)[:10], end_date=str(end)[:10])
    if sch.empty:
        return pd.DataFrame(columns=["trade_date", "open_et", "close_et"])
    return pd.DataFrame(
        {
            "trade_date": sch.index.strftime("%Y-%m-%d"),
            "open_et": sch["market_open"].dt.tz_convert(_TZ).dt.tz_localize(None).to_numpy(),
            "close_et": sch["market_close"].dt.tz_convert(_TZ).dt.tz_localize(None).to_numpy(),
        }
    ).reset_index(drop=True)


def trading_days(start: str, end: str) -> list[str]:
    """区间内 NYSE 交易日列表（``'YYYY-MM-DD'``）。"""
    return schedule(start, end)["trade_date"].tolist()


def rth_minute_grid(start: str, end: str) -> pd.DataFrame:
    """RTH 分钟网格 ``DataFrame[trade_date(str), minute('HH:MM'), ts(Timestamp, 美东本地)]``。

    每交易日从 09:30 到当日收盘（含端点），逐分钟；半日自动截断到 13:00。
    """
    sch = schedule(start, end)
    frames: list[pd.DataFrame] = []
    for row in sch.itertuples(index=False):
        idx = pd.date_range(pd.Timestamp(row.open_et), pd.Timestamp(row.close_et), freq="1min")
        frames.append(
            pd.DataFrame(
                {"trade_date": row.trade_date, "minute": idx.strftime("%H:%M"), "ts": idx}
            )
        )
    if not frames:
        return pd.DataFrame(columns=["trade_date", "minute", "ts"])
    return pd.concat(frames, ignore_index=True)
