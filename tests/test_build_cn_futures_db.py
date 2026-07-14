"""中国期货库构建的针对性测试（研究规范 §2.3 要求的交易日还原 / 换月 / 时段测试）。

前半部分为纯函数单元测试（合成数据）；结尾两个测试依赖已构建的
``data/cn_futures``（缺库时自动跳过），对真实数据断言交易日还原语义。
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_cn_futures_db import assign_trade_date, build_daily  # noqa: E402

from alpha_data.cn_futures import sessions, store  # noqa: E402

TRADING_DAYS = np.array(
    ["2026-01-09", "2026-01-12", "2026-01-13"], dtype="datetime64[D]"
)


def test_assign_trade_date_day_session() -> None:
    ts = pd.Series(pd.to_datetime(["2026-01-09 09:01:00", "2026-01-12 15:00:00"]))
    got = assign_trade_date(ts, TRADING_DAYS)
    assert got.tolist() == ["2026-01-09", "2026-01-12"]


def test_assign_trade_date_friday_night_belongs_to_monday() -> None:
    # 周五晚 21:01 与周六凌晨 02:30 均归属下周一交易日。
    ts = pd.Series(pd.to_datetime(["2026-01-09 21:01:00", "2026-01-10 02:30:00"]))
    got = assign_trade_date(ts, TRADING_DAYS)
    assert got.tolist() == ["2026-01-12", "2026-01-12"]


def test_assign_trade_date_weeknight() -> None:
    # 周一晚夜盘（含跨零点段）归属周二。
    ts = pd.Series(pd.to_datetime(["2026-01-12 21:01:00", "2026-01-13 01:00:00"]))
    got = assign_trade_date(ts, TRADING_DAYS)
    assert got.tolist() == ["2026-01-13", "2026-01-13"]


def test_assign_trade_date_out_of_range_is_none() -> None:
    # 样本末交易日的晚间夜盘属于样本外的下一交易日，应标 None 而非错归。
    ts = pd.Series(pd.to_datetime(["2026-01-13 21:01:00"]))
    got = assign_trade_date(ts, TRADING_DAYS)
    assert got.tolist() == [None]


def _synthetic_minute() -> pd.DataFrame:
    """两个交易日：d1（合约 C1）、d2（换月为 C2，夜盘起切换）。"""
    rows = [
        # d1 日盘（无夜盘数据的首日）
        ("2026-01-09 09:01:00", "2026-01-09", "day", 100.0, "C1"),
        ("2026-01-09 15:00:00", "2026-01-09", "day", 102.0, "C1"),
        # d2 夜盘（周五晚，合约切换为 C2）
        ("2026-01-09 21:01:00", "2026-01-12", "night", 200.0, "C2"),
        ("2026-01-09 23:00:00", "2026-01-12", "night", 202.0, "C2"),
        # d2 日盘
        ("2026-01-12 09:01:00", "2026-01-12", "day", 204.0, "C2"),
        ("2026-01-12 15:00:00", "2026-01-12", "day", 208.0, "C2"),
    ]
    df = pd.DataFrame(rows, columns=["ts", "trade_date", "session", "px", "contract"])
    df["ts"] = pd.to_datetime(df["ts"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df["px"]
    df["volume"] = 10.0
    df["money"] = 1000.0
    df["open_interest"] = 50.0
    return df.drop(columns=["px"])


def test_build_daily_sessions_and_returns() -> None:
    daily = build_daily(_synthetic_minute())
    assert daily["trade_date"].tolist() == ["2026-01-09", "2026-01-12"]
    d2 = daily.iloc[1]
    # 分时段 OHLC
    assert d2["night_open"] == 200.0 and d2["night_close"] == 202.0
    assert d2["day_open"] == 204.0 and d2["day_close"] == 208.0
    # 夜盘内与夜盘收至日盘开的收益不跨合约，正常计算
    assert d2["r_night"] == pytest.approx(np.log(202.0 / 200.0))
    assert d2["r_gap_am"] == pytest.approx(np.log(204.0 / 202.0))
    # 换月日：跨合约收益置 NaN，原始值保留
    assert bool(d2["roll"])
    assert np.isnan(d2["r_gap_pm"]) and np.isnan(d2["r_gap_full"]) and np.isnan(d2["r_cc"])
    assert d2["r_gap_pm_raw"] == pytest.approx(np.log(200.0 / 102.0))
    # 首日无前收盘
    d1 = daily.iloc[0]
    assert np.isnan(d1["r_gap_full_raw"]) and not bool(d1["roll"])


def test_signal_windows_weekend_and_overnight() -> None:
    win = sessions.signal_windows(["2026-01-09", "2026-01-12"], night_end=time(2, 30))
    d2 = win.iloc[1]
    # 周一交易日的夜盘窗口 = 上周五 21:00 .. 周六 02:30
    assert d2["night_start"] == pd.Timestamp("2026-01-09 21:00:00")
    assert d2["night_end"] == pd.Timestamp("2026-01-10 02:30:00")
    assert d2["gap1_start"] == pd.Timestamp("2026-01-09 15:00:00")
    assert d2["gap1_end"] == pd.Timestamp("2026-01-09 21:00:00")
    assert d2["gap2_start"] == pd.Timestamp("2026-01-10 02:30:00")
    assert d2["gap2_end"] == pd.Timestamp("2026-01-12 09:00:00")
    # 首日窗口缺前收盘
    d1 = win.iloc[0]
    assert pd.isna(d1["night_start"]) and pd.isna(d1["gap1_start"])


def test_signal_windows_no_night() -> None:
    win = sessions.signal_windows(["2026-01-09", "2026-01-12"], night_end=None)
    d2 = win.iloc[1]
    assert pd.isna(d2["night_start"])
    assert d2["gap1_start"] == pd.Timestamp("2026-01-09 15:00:00")
    assert d2["gap1_end"] == pd.Timestamp("2026-01-12 09:00:00")
    assert pd.isna(d2["gap2_start"])


@pytest.mark.skipif(not store.DB_DIR.exists(), reason="cn_futures 库未构建")
def test_real_monday_contains_friday_night() -> None:
    """规范 §2.3 第 1 条要求的测试：周一交易日包含上周五晚间的夜盘快照。"""
    au = store.read_minute("AU")
    monday = au[au["trade_date"] == "2026-01-12"]
    night = monday[monday["session"] == "night"]
    assert not night.empty
    assert night["ts"].min() == pd.Timestamp("2026-01-09 21:01:00")
    assert night["ts"].max() == pd.Timestamp("2026-01-10 02:30:00")


@pytest.mark.skipif(not store.DB_DIR.exists(), reason="cn_futures 库未构建")
def test_real_contract_unique_per_trade_day() -> None:
    for product in ["AU", "SC", "CU", "M", "I", "CF"]:
        m = store.read_minute(product)
        nuniq = m.groupby("trade_date")["contract"].nunique()
        assert (nuniq == 1).all(), f"{product} 存在交易日内多合约"
