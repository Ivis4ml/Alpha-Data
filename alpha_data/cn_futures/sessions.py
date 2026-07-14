"""国内期货交易时段定义与信号窗口边界（北京时间）。

用途：把 Polymarket 的连续时间信号按国内期货交易节奏拆为三个窗口（研究规范 §3.3）：

- ``night``：交易日 t 的夜盘窗口，自然时间为**前一交易日**的晚间 21:00 起
  （周一交易日的夜盘是上周五晚间），归属交易日 t。
- ``gap``：闭市且无夜盘覆盖的窗口，分两段：前一交易日 15:00 至夜盘开盘 21:00、
  夜盘收盘至当日 09:00。无夜盘品种整段为前一交易日 15:00 至当日 09:00。
- ``day``：当日日盘 09:00..15:00。

夜盘收盘时刻**优先取数据实测值**（``product_specs.parquet`` 的 ``night_end`` 列，
由构建脚本从分钟 bar 统计），``NIGHT_END_FALLBACK`` 仅作为无实测数据时的参考
（依据交易所公开时段表）。所有窗口参数均可由调用方覆盖（窗口可调）。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd

#: 日盘边界（商品期货；中金所日盘 09:30 起，对窗口拆分影响可忽略，统一 09:00）。
DAY_OPEN = time(9, 0)
DAY_CLOSE = time(15, 0)
NIGHT_OPEN = time(21, 0)

#: 品种 -> 夜盘收盘时刻参考表（``None`` 表示无夜盘），依据交易所公开时段表。
#: 仅在品种缺少实测 ``night_end`` 时使用；以数据实测为准。
NIGHT_END_FALLBACK: dict[str, time | None] = {
    "AU": time(2, 30), "AG": time(2, 30), "SC": time(2, 30),
    "CU": time(1, 0), "AL": time(1, 0), "ZN": time(1, 0), "PB": time(1, 0),
    "NI": time(1, 0), "SN": time(1, 0), "SS": time(1, 0), "AO": time(1, 0),
    "AD": time(1, 0), "BC": time(1, 0),
}


def parse_night_end(value: str | None) -> time | None:
    """把 ``product_specs.night_end`` 的 ``'HH:MM'`` 字符串解析为 ``datetime.time``。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    hh, mm = str(value).split(":")
    return time(int(hh), int(mm))


def signal_windows(
    trade_days: list[str],
    night_end: time | None,
) -> pd.DataFrame:
    """给出逐交易日的三窗口边界（北京时间、tz-naive、左闭右开）。

    交易日 t 的窗口（``prev`` 为前一交易日）：

    - ``night``: ``[prev 21:00, night_end)``，跨自然日的收盘时刻（01:00 / 02:30）落在
      ``prev`` 的次一自然日；``night_end`` 为 ``None``（无夜盘）时为 NaT 空窗口。
    - ``gap``:   两段闭市窗口 ``[prev 15:00, prev 21:00)`` 与 ``[night_end, t 09:00)``；
      无夜盘品种合并为一段 ``[prev 15:00, t 09:00)``（``gap2`` 为 NaT）。
    - ``day``:   ``[t 09:00, t 15:00)``。

    首个交易日缺前收盘，night / gap 窗口为 NaT，调用方应跳过。

    Args:
        trade_days: 升序交易日列表（``YYYY-MM-DD``）。
        night_end: 夜盘收盘时刻（``None`` 表示无夜盘）。来源建议为
            ``product_specs.night_end`` 实测值，经 :func:`parse_night_end` 解析。

    Returns:
        ``[trade_date, night_start, night_end, gap1_start, gap1_end, gap2_start,
        gap2_end, day_start, day_end]``，时间戳均为 ``pd.Timestamp``。
    """
    rows: list[dict] = []
    prev: str | None = None
    for day in trade_days:
        d = datetime.strptime(day, "%Y-%m-%d")
        day_start = pd.Timestamp(datetime.combine(d, DAY_OPEN))
        day_end = pd.Timestamp(datetime.combine(d, DAY_CLOSE))
        row = {
            "trade_date": day,
            "night_start": pd.NaT, "night_end": pd.NaT,
            "gap1_start": pd.NaT, "gap1_end": pd.NaT,
            "gap2_start": pd.NaT, "gap2_end": pd.NaT,
            "day_start": day_start, "day_end": day_end,
        }
        if prev is not None:
            p = datetime.strptime(prev, "%Y-%m-%d")
            prev_close = pd.Timestamp(datetime.combine(p, DAY_CLOSE))
            if night_end is None:
                row["gap1_start"], row["gap1_end"] = prev_close, day_start
            else:
                ne_date = p + timedelta(days=1) if night_end < NIGHT_OPEN else p
                night_s = pd.Timestamp(datetime.combine(p, NIGHT_OPEN))
                night_e = pd.Timestamp(datetime.combine(ne_date, night_end))
                row["night_start"], row["night_end"] = night_s, night_e
                row["gap1_start"], row["gap1_end"] = prev_close, night_s
                row["gap2_start"], row["gap2_end"] = night_e, day_start
        rows.append(row)
        prev = day
    return pd.DataFrame(rows)
