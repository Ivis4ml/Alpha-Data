"""``alpha_data.cn_equity.minute_store`` 的构建语义测试（合成中证 CSV，无需源数据）。

覆盖：个股 / 指数符号口径（含前导零）、列序映射（收盘价在最高/最低之前）、真实成交额、
指数无成交量记 0、日线聚合（首/末 bar、max/min、求和）、listing 名称与交易所、空 corp_actions。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_data.cn_equity import minute_store

_STOCK_HEADER = "时间,代码,名称,开盘价,收盘价,最高价,最低价,成交量,成交额,涨幅,振幅\n"
_INDEX_HEADER = "时间,代码,名称,开盘价,收盘价,最高价,最低价,成交额,涨幅,振幅\n"


def _write_stock(path: Path, code: str, name: str, rows: list[tuple]) -> Path:
    """rows: (ts, open, close, high, low, volume, amount)。列序与源一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_STOCK_HEADER]
    for ts, o, c, h, low, v, amt in rows:
        lines.append(f"{ts},{code},{name},{o},{c},{h},{low},{v},{amt},0.0,0.0\n")
    path.write_text("".join(lines), encoding="utf-8-sig")
    return path


def _write_index(path: Path, code: str, name: str, rows: list[tuple]) -> Path:
    """rows: (ts, open, close, high, low, amount)。指数无成交量列。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_INDEX_HEADER]
    for ts, o, c, h, low, amt in rows:
        lines.append(f"{ts},{code},{name},{o},{c},{h},{low},{amt},0.0,0.0\n")
    path.write_text("".join(lines), encoding="utf-8-sig")
    return path


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("PRAGMA preserve_insertion_order=false")
    return c


def test_stock_minute_symbol_and_columns(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    f = _write_stock(
        tmp_path / "sh600000_2024.csv",
        "sh600000",
        "浦发银行",
        [
            ("2024-01-02 09:30:00", 6.63, 6.63, 6.63, 6.63, 1530, 1014390),
            ("2024-01-02 09:31:00", 6.63, 6.60, 6.65, 6.59, 7192, 4768371),
        ],
    )
    minute_store.build_minute(con, [f], tmp_path / "db", kind="stock")
    m = pd.read_parquet(tmp_path / "db" / "minute" / "600000.SH" / "2024.parquet")
    assert list(m["symbol"].unique()) == ["600000.SH"]
    # 列序映射正确：收盘价映射到 close，非 high。
    r = m.iloc[1]
    assert (r.open, r.close, r.high, r.low) == pytest.approx((6.63, 6.60, 6.65, 6.59))
    assert int(r.volume) == 7192 and int(r.trade_count) == 0
    assert r.source == minute_store.SOURCE
    assert str(m["ts"].dtype).startswith("datetime64")
    # 逐分钟成交额忠实保留，vwap=成交额/成交量（真实分钟 VWAP）。
    assert r.amount == pytest.approx(4768371.0)
    assert r.vwap == pytest.approx(4768371.0 / 7192, rel=1e-5)


def test_index_symbol_preserves_leading_zeros_and_zero_volume(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    fsh = _write_index(
        tmp_path / "000300_2024.csv",
        "000300",
        "沪深300",
        [("2024-01-02 09:31:00", 3426.27, 3425.49, 3430.0, 3420.0, 1.26e9)],
    )
    fsz = _write_index(
        tmp_path / "399006_2024.csv",
        "399006",
        "创业板指",
        [("2024-01-02 09:31:00", 1889.02, 1888.0, 1890.0, 1887.0, 1.55e9)],
    )
    fbj = _write_index(
        tmp_path / "899050_2024.csv",
        "899050",
        "北证50",
        [("2024-01-02 09:31:00", 1081.76, 1080.0, 1082.0, 1079.0, 2.3e8)],
    )
    minute_store.build_minute(con, [fsh, fsz, fbj], tmp_path / "db", kind="index")
    dirs = sorted(p.name for p in (tmp_path / "db" / "minute").glob("*") if p.is_dir())
    assert dirs == ["000300.SH", "399006.SZ", "899050.BJ"]
    m = pd.read_parquet(tmp_path / "db" / "minute" / "000300.SH" / "2024.parquet")
    assert int(m.iloc[0].volume) == 0  # 指数无成交量记 0
    assert m.iloc[0].close == pytest.approx(3425.49)
    # 指数成交额仍忠实保留；无成交量故 vwap 记 0。
    assert m.iloc[0].amount == pytest.approx(1.26e9)
    assert m.iloc[0].vwap == pytest.approx(0.0)


def test_daily_aggregation(con: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    rows = [
        ("2024-01-02 09:30:00", 10.0, 10.0, 10.0, 10.0, 100, 1000),
        ("2024-01-02 09:31:00", 10.1, 10.5, 10.8, 10.0, 200, 2100),
        ("2024-01-02 15:00:00", 10.4, 10.3, 10.4, 10.2, 300, 3090),
    ]
    f = _write_stock(tmp_path / "sz000001_2024.csv", "sz000001", "平安银行", rows)
    minute_store.build_daily(con, [f], tmp_path / "db", kind="stock")
    d = pd.read_parquet(tmp_path / "db" / "daily" / "000001.SZ.parquet").iloc[0]
    assert d.open == pytest.approx(10.0)  # 首 bar (09:30)
    assert d.close == pytest.approx(10.3)  # 末 bar (15:00)
    assert d.high == pytest.approx(10.8)
    assert d.low == pytest.approx(10.0)
    assert int(d.volume) == 600  # 100+200+300
    assert d.amount == pytest.approx(6190.0)  # 真实成交额之和，非 close×volume


def test_listing_and_empty_corp_actions(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    f23 = _write_stock(
        tmp_path / "2023" / "sh600000_2023.csv",
        "sh600000",
        "浦发银行",
        [("2023-01-03 15:00:00", 7.0, 7.1, 7.2, 6.9, 100, 700)],
    )
    f24 = _write_stock(
        tmp_path / "2024" / "sh600000_2024.csv",
        "sh600000",
        "浦发银行",
        [("2024-01-02 15:00:00", 6.6, 6.6, 6.7, 6.5, 100, 660)],
    )
    fdead = _write_stock(
        tmp_path / "2023" / "sz000123_2023.csv",
        "sz000123",
        "某退市股",
        [("2023-06-01 15:00:00", 3.0, 3.0, 3.1, 2.9, 50, 150)],
    )
    db = tmp_path / "db"
    minute_store.build_daily(con, [f23, f24, fdead], tmp_path / "db", kind="stock")
    latest = {"600000.SH": f24, "000123.SZ": fdead}
    active = {"600000.SH"}  # 仅 600000 出现在最新一年
    listing = minute_store.build_listing(con, latest, db, active_symbols=active)
    by_sym = listing.set_index("symbol")
    assert by_sym.loc["600000.SH", "status"] == "active"
    assert by_sym.loc["600000.SH", "exchange"] == "XSHG"
    assert by_sym.loc["600000.SH", "name"] == "浦发银行"
    assert by_sym.loc["600000.SH", "delist_date"] == ""
    assert by_sym.loc["000123.SZ", "status"] == "delisted"
    assert by_sym.loc["000123.SZ", "exchange"] == "XSHE"
    assert by_sym.loc["000123.SZ", "delist_date"] == "2023-06-01"

    minute_store.write_empty_corp_actions(db)
    corp = pd.read_parquet(db / "corp_actions.parquet")
    assert list(corp.columns) == ["symbol", "ex_date", "split_ratio", "cash_div"]
    assert corp.empty
