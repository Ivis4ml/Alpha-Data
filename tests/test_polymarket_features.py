"""``alpha_data.polymarket.features`` 的单元测试（合成成交，无需下载数据）。

验证防前视口径与累计不变量：bar 起始 T 只计入严格早于 T 的成交；日内累计单调；解析后置空。
2024-06-13（周四）与 2024-06-14（周五）均为 NYSE 交易日（全日，16:00 收盘）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_data.polymarket.features import FEATURE_COLS, features_from_trades


def _trades(resolved_et=None) -> pd.DataFrame:
    rows = [
        # (et_ts, p_event, D, usdc_amount)
        ("2024-06-13 08:00:00", 0.50, 1, 10.0),   # 盘前：确立当日开盘 p
        ("2024-06-13 09:45:30", 0.60, 1, 100.0),  # BUY，落在 09:45 分钟
        ("2024-06-13 10:30:00", 0.55, -1, 50.0),  # SELL
        ("2024-06-14 11:00:00", 0.62, 1, 80.0),   # 次日 BUY
    ]
    df = pd.DataFrame(rows, columns=["et_ts", "p_event", "D", "usdc_amount"])
    df["et_ts"] = pd.to_datetime(df["et_ts"])
    if resolved_et is not None:
        df["resolved_et"] = pd.Timestamp(resolved_et)
    return df


def _bar(df: pd.DataFrame, day: str, minute: str, col: str):
    hit = df[(df["trade_date"] == day) & (df["minute"] == minute)]
    assert not hit.empty, f"缺 bar {day} {minute}"
    return hit[col].iloc[0]


def test_empty_input_returns_empty():
    out = features_from_trades(pd.DataFrame(columns=["et_ts", "p_event", "D", "usdc_amount"]))
    assert list(out.columns) == ["trade_date", "minute", *FEATURE_COLS]
    assert out.empty


def test_open_bar_is_clean_and_lookahead_safe():
    out = features_from_trades(_trades())
    day = "2024-06-13"
    # 开盘 09:30：p 为盘前 LOCF，日内变化/累计全为 0（不泄漏同 bar 及未来）。
    assert _bar(out, day, "09:30", "p") == pytest.approx(0.50)
    assert _bar(out, day, "09:30", "dp_intraday") == pytest.approx(0.0)
    assert _bar(out, day, "09:30", "flow_session") == pytest.approx(0.0)
    assert _bar(out, day, "09:30", "usdc_session") == pytest.approx(0.0)
    assert _bar(out, day, "09:30", "n_session") == pytest.approx(0.0)
    # 09:45 bar：09:45:30 的成交严格晚于 T=09:45，不得计入。
    assert _bar(out, day, "09:45", "p") == pytest.approx(0.50)
    assert _bar(out, day, "09:45", "flow_session") == pytest.approx(0.0)
    assert _bar(out, day, "09:45", "n_session") == pytest.approx(0.0)


def test_cumulative_after_trades():
    out = features_from_trades(_trades())
    day = "2024-06-13"
    # 09:46：09:45:30 的 BUY 已完成（属 [开盘, 09:46)）。
    assert _bar(out, day, "09:46", "p") == pytest.approx(0.60)
    assert _bar(out, day, "09:46", "n_session") == pytest.approx(1.0)
    assert _bar(out, day, "09:46", "usdc_session") == pytest.approx(100.0)
    assert _bar(out, day, "09:46", "flow_session") == pytest.approx(100.0)
    # 10:31：叠加 10:30 的 SELL（D=-1, 50）。
    assert _bar(out, day, "10:31", "n_session") == pytest.approx(2.0)
    assert _bar(out, day, "10:31", "usdc_session") == pytest.approx(150.0)
    assert _bar(out, day, "10:31", "flow_session") == pytest.approx(50.0)  # 100 - 50
    assert _bar(out, day, "10:31", "dp_intraday") == pytest.approx(0.05)   # 0.55 - 0.50


def test_invariants_pvalue_and_monotonic():
    out = features_from_trades(_trades())
    p = out["p"].dropna()
    assert p.between(0.0, 1.0).all()
    for _, block in out.groupby("trade_date"):
        n = block["n_session"].dropna().to_numpy()
        assert (n[1:] >= n[:-1] - 1e-9).all()
        u = block["usdc_session"].dropna().to_numpy()
        assert (u[1:] >= u[:-1] - 1e-6).all()


def test_resolution_masks_features():
    out = features_from_trades(_trades(resolved_et="2024-06-14 12:00:00"))
    # 解析前有值，解析后置空。
    assert _bar(out, "2024-06-14", "11:30", "p") == pytest.approx(0.62)
    assert pd.isna(_bar(out, "2024-06-14", "12:30", "p"))
    assert pd.isna(_bar(out, "2024-06-14", "12:30", "flow_session"))
    assert pd.isna(_bar(out, "2024-06-14", "12:30", "flow_overnight"))


def test_overnight_features():
    rows = [
        # (et_ts, p_event, D, usdc_amount)
        ("2024-06-13 10:00:00", 0.50, 1, 40.0),   # 日内成交，不属隔夜
        ("2024-06-13 16:00:00", 0.52, 1, 30.0),   # 恰在收盘标签：16:00 bar 只含 <16:00，属隔夜
        ("2024-06-13 22:00:00", 0.55, -1, 20.0),  # 盘后
        ("2024-06-14 09:15:00", 0.58, 1, 10.0),   # 次日盘前
        ("2024-06-14 09:30:30", 0.60, 1, 5.0),    # 开盘后：属当日 session，不属隔夜
    ]
    df = pd.DataFrame(rows, columns=["et_ts", "p_event", "D", "usdc_amount"])
    df["et_ts"] = pd.to_datetime(df["et_ts"])
    out = features_from_trades(df)
    # 首日无前收盘，隔夜为 NaN。
    assert pd.isna(_bar(out, "2024-06-13", "10:00", "flow_overnight"))
    # 次日隔夜窗口 [06-13 16:00, 06-14 09:30) 含三笔：+30、-20、+10。
    assert _bar(out, "2024-06-14", "09:30", "n_overnight") == pytest.approx(3.0)
    assert _bar(out, "2024-06-14", "09:30", "usdc_overnight") == pytest.approx(60.0)
    assert _bar(out, "2024-06-14", "09:30", "flow_overnight") == pytest.approx(20.0)
    # 按日广播：全天恒定。
    assert _bar(out, "2024-06-14", "15:59", "n_overnight") == pytest.approx(3.0)
    # 开盘后的成交计入 session（自 09:31 标签起），不重复计入隔夜。
    assert _bar(out, "2024-06-14", "09:31", "n_session") == pytest.approx(1.0)
    assert _bar(out, "2024-06-14", "09:31", "usdc_session") == pytest.approx(5.0)
