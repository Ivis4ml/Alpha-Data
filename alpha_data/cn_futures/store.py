"""中国期货本地库路径与读取接口。

库布局（``data/cn_futures/``）::

    minute/{PRODUCT}/{YEAR}.parquet   # 主力连续分钟 bar，ts 为北京时间 bar 收盘戳
    daily/{PRODUCT}.parquet           # 分时段日线（night_* / day_*、收益列、换月标记）
    dominant_table.parquet            # [product, contract, active_from, active_to, rule]
    product_specs.parquet             # [product, exchange, night_end(实测), has_night, ...]
    qc_summary.parquet                # 逐品种质量核查

数据来源为聚宽风格主力连续（``XX9999.交易所``）分钟 CSV。**含夜盘**：56 个品种有
夜盘 bar（收盘 23:00 / 01:00 / 02:30 三档，实测于 ``product_specs.night_end``）。
分钟表的 ``trade_date`` 列为交易日归属（夜盘归属下一交易日：周一交易日含上周五
夜盘），``session`` 列区分 ``night`` / ``day``。构建规则详见
``scripts/build_cn_futures_db.py`` 模块文档。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "data" / "cn_futures"
MINUTE_DIR = DB_DIR / "minute"
DAILY_DIR = DB_DIR / "daily"

#: 交易所代码后缀 -> 标准简称。
EXCHANGE_MAP: dict[str, str] = {
    "XSGE": "SHFE",
    "XDCE": "DCE",
    "XZCE": "CZCE",
    "XINE": "INE",
    "GFEX": "GFEX",
    "CCFX": "CFFEX",
}


def list_products() -> list[str]:
    """返回库中已有分钟数据的品种列表（按字母序）。"""
    if not MINUTE_DIR.exists():
        return []
    return sorted(p.name for p in MINUTE_DIR.iterdir() if p.is_dir())


def read_minute(product: str) -> pd.DataFrame:
    """读取单品种全部年份分钟 bar，按 ``ts`` 升序。"""
    files = sorted((MINUTE_DIR / product).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no minute parquet for product {product!r}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("ts").reset_index(drop=True)


def read_daily(product: str) -> pd.DataFrame:
    """读取单品种日线（含 ``r_gap`` / ``r_day`` / ``roll``），按 ``trade_date`` 升序。"""
    path = DAILY_DIR / f"{product}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no daily parquet for product {product!r}")
    return pd.read_parquet(path).sort_values("trade_date").reset_index(drop=True)


def read_product_specs() -> pd.DataFrame:
    """读取品种规格表。"""
    return pd.read_parquet(DB_DIR / "product_specs.parquet")


def read_dominant_table() -> pd.DataFrame:
    """读取主力合约区间表。"""
    return pd.read_parquet(DB_DIR / "dominant_table.parquet")


def trading_days() -> list[str]:
    """返回库覆盖的全部交易日（``YYYY-MM-DD``，升序，各品种并集）。"""
    days: set[str] = set()
    for product in list_products():
        days.update(read_daily(product)["trade_date"].tolist())
    return sorted(days)
