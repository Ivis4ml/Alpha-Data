"""``alpha_data.equity.provider`` 的单元测试（假客户端，无需网络）。

覆盖收盘标签换算（含夏令时两侧）、RTH 边界过滤、限速状态映射与内容哈希敏感性。
"""

from __future__ import annotations

import pandas as pd

from alpha_data.equity.massive_client import MassiveError
from alpha_data.equity.provider import MassiveProvider, _close_ts_et


def _ms(et: str) -> int:
    """美东本地时刻 -> 毫秒 epoch（UTC）。"""
    return int(pd.Timestamp(et, tz="America/New_York").value // 1_000_000)


def _bar(et_start: str, close: float = 1.5) -> dict:
    return {"t": _ms(et_start), "o": 1.0, "h": 2.0, "l": 0.5, "c": close, "v": 100.0}


class _FakeClient:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def paginate(self, path: str, params: dict | None = None):
        return iter(self._rows)


class _RateLimitedClient:
    def paginate(self, path: str, params: dict | None = None):
        raise MassiveError("重试 6 次仍失败：限速 429", status_code=429)


def test_close_ts_et_across_dst():
    # EDT（UTC-4）：09:30 起始 -> 09:31 收盘。
    assert _close_ts_et(_ms("2024-06-13 09:30")) == "2024-06-13 09:31:00"
    # EST（UTC-5）：15:59 起始 -> 16:00 收盘。
    assert _close_ts_et(_ms("2024-12-13 15:59")) == "2024-12-13 16:00:00"


def test_fetch_rth_boundary_filter():
    rows = [
        _bar("2024-06-13 09:29"),  # 收盘 09:30，不属 RTH（严格 >09:30）
        _bar("2024-06-13 09:30"),  # 收盘 09:31，RTH 首 bar
        _bar("2024-06-13 15:59"),  # 收盘 16:00，RTH 末 bar
        _bar("2024-06-13 16:00"),  # 收盘 16:01，盘后
    ]
    res = MassiveProvider(client=_FakeClient(rows)).fetch("AAPL", "2024-06-13", "2024-06-13")
    assert res.status == "ok"
    assert [b["ts"] for b in res.bars] == [
        "2024-06-13 09:31:00",
        "2024-06-13 16:00:00",
    ]


def test_fetch_extended_hours_keeps_all():
    rows = [_bar("2024-06-13 09:29"), _bar("2024-06-13 16:00")]
    res = MassiveProvider(client=_FakeClient(rows)).fetch(
        "AAPL", "2024-06-13", "2024-06-13", extended_hours=True
    )
    assert len(res.bars) == 2


def test_fetch_rate_limited_status():
    res = MassiveProvider(client=_RateLimitedClient()).fetch(
        "AAPL", "2024-06-13", "2024-06-13"
    )
    assert res.status == "rate_limited"
    assert res.bars == []


def test_raw_hash_sensitive_to_bar_content():
    rows_a = [_bar("2024-06-13 09:30", close=1.5)]
    rows_b = [_bar("2024-06-13 09:30", close=1.6)]  # 同区间同 bar 数，仅价格不同
    res_a = MassiveProvider(client=_FakeClient(rows_a)).fetch(
        "AAPL", "2024-06-13", "2024-06-13"
    )
    res_b = MassiveProvider(client=_FakeClient(rows_b)).fetch(
        "AAPL", "2024-06-13", "2024-06-13"
    )
    assert res_a.raw_hash != res_b.raw_hash
