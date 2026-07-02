"""``alpha_data.equity.reference`` 的单元测试（纯函数 + 假客户端，无需网络）。

覆盖三个曾出错的口径：符号归一保留类别股 ``.`` 后缀、listing 去重 active 优先、
同日多笔分红求和合并。
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_data.equity import reference


def test_normalize_symbol_strips_exchange_suffixes():
    assert reference.normalize_symbol(" aapl.us ") == "AAPL"
    assert reference.normalize_symbol("AAPL:NASDAQ") == "AAPL"
    assert reference.normalize_symbol("ibm.n") == "IBM"
    assert reference.normalize_symbol("msft.oq") == "MSFT"


def test_normalize_symbol_preserves_native_dot_tickers():
    # 类别股 / SPAC 单位 / 权证 / 优先股的 "." 是标的原生标识，必须保留。
    assert reference.normalize_symbol("brk.a") == "BRK.A"
    assert reference.normalize_symbol("BRK.B") == "BRK.B"
    assert reference.normalize_symbol("AAC.U") == "AAC.U"
    assert reference.normalize_symbol("achr.ws") == "ACHR.WS"
    assert reference.normalize_symbol("BAC.PR.A") == "BAC.PR.A"


class _FakeClient:
    """最小假客户端：按 ``active`` 参数返回预置分页结果。"""

    def __init__(self, active_rows: list[dict], delisted_rows: list[dict]) -> None:
        self._active = active_rows
        self._delisted = delisted_rows

    def paginate(self, path: str, params: dict | None = None):
        rows = self._active if (params or {}).get("active") == "true" else self._delisted
        return iter(rows)


def test_fetch_listing_prefers_active_on_recycled_ticker():
    active = [
        {"ticker": "AAA", "name": "New Co", "primary_exchange": "XNAS", "active": True},
    ]
    delisted = [
        {
            "ticker": "AAA",
            "name": "Old Co",
            "primary_exchange": "XNYS",
            "active": False,
            "delisted_utc": "2021-05-01T00:00:00Z",
        },
        {
            "ticker": "BBB",
            "name": "Common",
            "primary_exchange": "XNYS",
            "active": False,
            "delisted_utc": "2022-01-01T00:00:00Z",
        },
        {
            "ticker": "BBB.WS",
            "name": "Warrant",
            "primary_exchange": "XNYS",
            "active": False,
            "delisted_utc": "2023-01-01T00:00:00Z",
        },
    ]
    listing = reference.fetch_listing(_FakeClient(active, delisted))
    # ticker 回收：AAA 以当前在市主体（active）为准。
    aaa = listing[listing["symbol"] == "AAA"]
    assert len(aaa) == 1
    assert aaa["status"].iloc[0] == "active"
    assert aaa["name"].iloc[0] == "New Co"
    assert aaa["delist_date"].iloc[0] == ""
    # 权证与正股不再被归一并到同一符号。
    assert set(listing["symbol"]) == {"AAA", "BBB", "BBB.WS"}


def test_fetch_listing_keeps_latest_delist_date():
    delisted = [
        {"ticker": "CCC", "name": "First", "active": False,
         "delisted_utc": "2020-01-01T00:00:00Z"},
        {"ticker": "CCC", "name": "Second", "active": False,
         "delisted_utc": "2024-06-01T00:00:00Z"},
    ]
    listing = reference.fetch_listing(_FakeClient([], delisted))
    assert len(listing) == 1
    assert listing["delist_date"].iloc[0] == "2024-06-01"


def test_build_corp_actions_sums_same_day_dividends():
    splits = pd.DataFrame(
        [{"symbol": "AAPL", "ex_date": "2020-08-31", "split_ratio": 4.0}]
    )
    dividends = pd.DataFrame(
        [
            {"symbol": "KO", "ex_date": "2024-03-15", "cash_div": 0.46},
            {"symbol": "KO", "ex_date": "2024-03-15", "cash_div": 1.00},
            {"symbol": "AAPL", "ex_date": "2020-08-31", "cash_div": 0.20},
        ]
    )
    ca = reference.build_corp_actions(splits, dividends)
    ko = ca[ca["symbol"] == "KO"]
    assert len(ko) == 1
    assert ko["cash_div"].iloc[0] == pytest.approx(1.46)
    assert ko["split_ratio"].iloc[0] == pytest.approx(1.0)
    aapl = ca[ca["symbol"] == "AAPL"]
    assert len(aapl) == 1
    assert aapl["split_ratio"].iloc[0] == pytest.approx(4.0)
    assert aapl["cash_div"].iloc[0] == pytest.approx(0.20)


def test_build_corp_actions_compounds_same_day_splits():
    splits = pd.DataFrame(
        [
            {"symbol": "XYZ", "ex_date": "2023-01-10", "split_ratio": 2.0},
            {"symbol": "XYZ", "ex_date": "2023-01-10", "split_ratio": 3.0},
        ]
    )
    dividends = pd.DataFrame(columns=["symbol", "ex_date", "cash_div"])
    ca = reference.build_corp_actions(splits, dividends)
    assert len(ca) == 1
    assert ca["split_ratio"].iloc[0] == pytest.approx(6.0)


def test_remap_to_native_symbols():
    canonical = ["BF.A", "BF.B", "MOG.A", "AAPL", "AB", "A.B"]
    df = pd.DataFrame(
        {
            "symbol": ["BFB", "MOGA", "AAPL", "ZZZ", "AB"],
            "cash_div": [0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    out = reference.remap_to_native_symbols(df, canonical)
    # 无点类别股映射回带点原生形式。
    assert list(out["symbol"]) == ["BF.B", "MOG.A", "AAPL", "ZZZ", "AB"]
    # 其余列不变。
    assert list(out["cash_div"]) == [0.2, 0.3, 0.4, 0.5, 0.6]


def test_remap_to_native_symbols_conservative():
    # 'AB' 同时是在册符号与 'A.B' 的去点形式：在册优先，不映射。
    canonical = ["A.B", "AB", "X.A", "XA.", "X.B"]
    df = pd.DataFrame({"symbol": ["AB", "XA", "XB"]})
    out = reference.remap_to_native_symbols(df, canonical)
    assert list(out["symbol"])[0] == "AB"
    # 'XA' 有两个候选（X.A 与 XA.），歧义则不映射。
    assert list(out["symbol"])[1] == "XA"
    # 'XB' 唯一对应 X.B，映射。
    assert list(out["symbol"])[2] == "X.B"


def test_build_corp_actions_empty_inputs():
    splits = pd.DataFrame(columns=["symbol", "ex_date", "split_ratio"])
    dividends = pd.DataFrame(columns=["symbol", "ex_date", "cash_div"])
    ca = reference.build_corp_actions(splits, dividends)
    assert list(ca.columns) == ["symbol", "ex_date", "split_ratio", "cash_div"]
    assert ca.empty
