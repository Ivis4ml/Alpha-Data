"""Polymarket CN 时段特征（cn_features）的防前视与口径测试。"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.cn_futures import sessions  # noqa: E402
from alpha_data.polymarket import cn_features  # noqa: E402

TRADE_DAYS = ["2026-01-09", "2026-01-12", "2026-01-13"]


def make_trades(rows: list[tuple[str, float, int, float]]) -> pd.DataFrame:
    """rows: (cn_ts, p_event, D, usdc_amount)。"""
    df = pd.DataFrame(rows, columns=["cn_ts", "p_event", "D", "usdc_amount"])
    df["cn_ts"] = pd.to_datetime(df["cn_ts"])
    df["resolved_cn"] = pd.NaT
    return df


def windows_au() -> pd.DataFrame:
    return sessions.signal_windows(TRADE_DAYS, night_end=time(2, 30))


def test_bounce_debias_two_sides() -> None:
    trades = make_trades(
        [
            ("2026-01-12 10:00:30", 0.60, 1, 100.0),
            ("2026-01-12 10:05:00", 0.50, -1, 100.0),
        ]
    )
    agg = cn_features.aggregate_price(trades, agg_minutes=15)
    assert len(agg) == 1
    assert agg["p_agg"].iloc[0] == pytest.approx(0.55)
    assert agg["bucket_end"].iloc[0] == pd.Timestamp("2026-01-12 10:15:00")


def test_no_lookahead_trade_at_window_end_excluded() -> None:
    """恰在日盘结束 15:00:00 的成交不得进入当日 s_day，应归属其后窗口。"""
    base = [
        ("2026-01-09 10:00:00", 0.50, 1, 10.0),   # 前日基准
        ("2026-01-12 09:30:00", 0.50, 1, 10.0),   # 周一日盘内
        ("2026-01-12 15:00:00", 0.90, 1, 10.0),   # 恰在窗口右端
    ]
    sig = cn_features.window_signals(
        make_trades(base), windows_au(), agg_minutes=15, p_age_max_minutes=None
    )
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    # 0.50 -> 0.50：日盘内无变化；15:00 的 0.90 不得计入
    assert monday["s_day"] == pytest.approx(0.0)
    assert monday["n_day"] == 1  # 15:00:00 的成交按 [start, end) 不属于日盘
    tuesday = sig[sig["trade_date"] == "2026-01-13"].iloc[0]
    assert tuesday["s_gap"] == pytest.approx(
        cn_features.logit(np.array([0.90]))[0] - cn_features.logit(np.array([0.50]))[0]
    )


def test_friday_night_trade_belongs_to_monday_night_window() -> None:
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),   # 周五日盘（基准）
            ("2026-01-09 22:00:00", 0.60, 1, 20.0),   # 周五晚（周一夜盘窗口）
        ]
    )
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    assert monday["n_night"] == 1
    assert monday["usdc_night"] == pytest.approx(20.0)
    assert monday["s_night"] == pytest.approx(
        cn_features.logit(np.array([0.60]))[0] - cn_features.logit(np.array([0.40]))[0]
    )


def test_three_components_telescope() -> None:
    """无缺失时 s_night + s_gap + s_day = 当日 15:00 前 LOCF logit − 前日 15:00 前 LOCF logit。"""
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),
            ("2026-01-09 16:30:00", 0.45, -1, 10.0),  # gap1（15:00-21:00）
            ("2026-01-09 23:30:00", 0.55, 1, 10.0),   # 夜盘
            ("2026-01-12 08:00:00", 0.60, 1, 10.0),   # gap2（02:30-09:00）
            ("2026-01-12 11:00:00", 0.70, -1, 10.0),  # 日盘
        ]
    )
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    total = monday["s_night"] + monday["s_gap"] + monday["s_day"]
    expect = (
        cn_features.logit(np.array([0.70]))[0] - cn_features.logit(np.array([0.40]))[0]
    )
    assert total == pytest.approx(expect)


def test_p_age_stale_boundary_gives_nan() -> None:
    trades = make_trades(
        [
            ("2026-01-09 10:00:00", 0.40, 1, 10.0),
            ("2026-01-12 11:00:00", 0.70, 1, 10.0),
        ]
    )
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=120)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    # 夜盘与 gap 端点全部超时效
    assert np.isnan(monday["s_night"]) and np.isnan(monday["s_gap"])
    # 日盘结束端点（15:00）距 11:00 成交 240 分钟，同样超时效
    assert np.isnan(monday["s_day"])
    relaxed = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    assert not np.isnan(relaxed[relaxed["trade_date"] == "2026-01-12"]["s_day"].iloc[0])


def test_clip_keeps_logit_finite() -> None:
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.001, 1, 10.0),
            ("2026-01-12 11:00:00", 0.999, 1, 10.0),
        ]
    )
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    vals = sig[[f"s_{w}" for w in cn_features.WINDOWS]].to_numpy(dtype="float64")
    assert np.isfinite(vals[~np.isnan(vals)]).all()
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    expect = cn_features.logit(np.array([0.98]))[0] - cn_features.logit(np.array([0.02]))[0]
    total = monday["s_night"] + monday["s_gap"] + monday["s_day"]
    assert total == pytest.approx(expect)


def test_resolution_cutoff() -> None:
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),
            ("2026-01-12 10:00:00", 0.99, 1, 10.0),
        ]
    )
    trades["resolved_cn"] = pd.Timestamp("2026-01-12 12:00:00")
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    # 日盘窗口结束（15:00）晚于结算时刻 -> NaN；夜盘 / gap 窗口早于结算 -> 保留
    assert np.isnan(monday["s_day"])
    assert not np.isnan(monday["s_night"]) and not np.isnan(monday["s_gap"])


def test_no_night_product_gap_covers_whole_close() -> None:
    win = sessions.signal_windows(TRADE_DAYS, night_end=None)
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),
            ("2026-01-09 22:00:00", 0.60, 1, 20.0),   # 闭市窗口内（无夜盘品种）
        ]
    )
    sig = cn_features.window_signals(trades, win, p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    assert np.isnan(monday["s_night"])
    assert monday["n_gap"] == 1
    assert monday["s_gap"] == pytest.approx(
        cn_features.logit(np.array([0.60]))[0] - cn_features.logit(np.array([0.40]))[0]
    )


def test_gap_split_components_sum_to_gap() -> None:
    """s_gap_pm + s_gap_am 应等于 s_gap（有夜盘品种、无缺失时）。"""
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),
            ("2026-01-09 16:30:00", 0.45, -1, 10.0),  # gap_pm（15:00-21:00）
            ("2026-01-09 23:30:00", 0.55, 1, 10.0),   # 夜盘
            ("2026-01-12 08:00:00", 0.60, 1, 10.0),   # gap_am（02:30-09:00）
        ]
    )
    sig = cn_features.window_signals(trades, windows_au(), p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    assert monday["s_gap_pm"] + monday["s_gap_am"] == pytest.approx(monday["s_gap"])
    assert monday["s_gap_pm"] == pytest.approx(
        cn_features.logit(np.array([0.45]))[0] - cn_features.logit(np.array([0.40]))[0]
    )
    assert monday["s_gap_am"] == pytest.approx(
        cn_features.logit(np.array([0.60]))[0] - cn_features.logit(np.array([0.55]))[0]
    )


def test_gap_split_no_night_product() -> None:
    """无夜盘品种：s_gap_pm 即整段闭市，s_gap_am 为 NaN。"""
    from datetime import time as _time  # noqa: F401

    win = sessions.signal_windows(TRADE_DAYS, night_end=None)
    trades = make_trades(
        [
            ("2026-01-09 14:00:00", 0.40, 1, 10.0),
            ("2026-01-09 22:00:00", 0.60, 1, 20.0),
        ]
    )
    sig = cn_features.window_signals(trades, win, p_age_max_minutes=None)
    monday = sig[sig["trade_date"] == "2026-01-12"].iloc[0]
    assert monday["s_gap_pm"] == pytest.approx(monday["s_gap"])
    assert np.isnan(monday["s_gap_am"])
