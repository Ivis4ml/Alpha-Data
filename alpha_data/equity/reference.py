"""massive.com 参考数据 → AlphaForge 表。

- 标的全集（含退市，survivorship-free）→ ``listing`` 表 ``[symbol, name, exchange, status,
  ipo_date, delist_date]``。
- 拆股 ``/v3/reference/splits`` + 分红 ``/v3/reference/dividends`` → ``corp_actions`` 表
  ``[symbol, ex_date, split_ratio, cash_div]``（AlphaForge ``MinuteDB.write_corp_actions`` 口径）。

参考端点不受历史时间档位限制，可在升级订阅前先行拉取（注意免费档约 5 次/分限速）。
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from alpha_data.equity.massive_client import MassiveClient


def normalize_symbol(raw: str) -> str:
    """符号归一（镜像 AlphaForge ``default_normalize_symbol``）：去空白、大写、去交易所后缀。"""
    s = str(raw).strip().upper()
    for sep in (":", "."):
        if sep in s:
            head = s.split(sep)[0]
            if head:
                return head
    return s


def iter_tickers(
    client: MassiveClient,
    *,
    market: str = "stocks",
    active: bool | None = None,
    date: str | None = None,
    limit: int = 1000,
) -> Iterator[dict]:
    """迭代 ``/v3/reference/tickers``。``active=None`` 不限；``date`` 取历史时点的成分。"""
    params: dict[str, object] = {"market": market, "limit": limit}
    if active is not None:
        params["active"] = "true" if active else "false"
    if date:
        params["date"] = date
    yield from client.paginate("/v3/reference/tickers", params)


def fetch_listing(
    client: MassiveClient,
    *,
    market: str = "stocks",
    date: str | None = None,
    include_delisted: bool = True,
) -> pd.DataFrame:
    """拉取标的全集（默认含退市）并构造 ``listing`` 表。

    分别以 ``active=true`` 与 ``active=false`` 拉取，合并去重（按 symbol 保留最后一条）。
    ``ipo_date`` 需 ticker-details 端点，此处留空，后续可补。
    """
    rows: list[dict] = []
    actives: tuple[bool, ...] = (True, False) if include_delisted else (True,)
    for act in actives:
        for ticker in iter_tickers(client, market=market, active=act, date=date):
            symbol = normalize_symbol(ticker.get("ticker", ""))
            if not symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": ticker.get("name", "") or "",
                    "exchange": ticker.get("primary_exchange", "") or "",
                    "status": "active" if ticker.get("active") else "delisted",
                    "ipo_date": "",
                    "delist_date": (ticker.get("delisted_utc") or "")[:10],
                }
            )
    cols = ["symbol", "name", "exchange", "status", "ipo_date", "delist_date"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    return df.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def fetch_splits(
    client: MassiveClient,
    ticker: str | None = None,
    *,
    since: str | None = None,
    until: str | None = None,
) -> pd.DataFrame:
    """拉取拆股 → ``[symbol, ex_date, split_ratio]``。``split_ratio = split_to / split_from``。

    ``since`` / ``until``（``'YYYY-MM-DD'``）按 ``execution_date`` 限定区间。
    """
    params: dict[str, object] = {"limit": 1000}
    if ticker:
        params["ticker"] = ticker
    if since:
        params["execution_date.gte"] = since
    if until:
        params["execution_date.lte"] = until
    rows: list[dict] = []
    for record in client.paginate("/v3/reference/splits", params):
        split_from = float(record.get("split_from") or 1.0)
        split_to = float(record.get("split_to") or 1.0)
        ratio = (split_to / split_from) if split_from else 1.0
        rows.append(
            {
                "symbol": normalize_symbol(record.get("ticker", "")),
                "ex_date": str(record.get("execution_date", "")),
                "split_ratio": ratio,
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "split_ratio"])


def fetch_dividends(
    client: MassiveClient,
    ticker: str | None = None,
    *,
    since: str | None = None,
    until: str | None = None,
) -> pd.DataFrame:
    """拉取分红 → ``[symbol, ex_date, cash_div]``（每股现金分红）。

    ``since`` / ``until``（``'YYYY-MM-DD'``）按 ``ex_dividend_date`` 限定区间。
    """
    params: dict[str, object] = {"limit": 1000}
    if ticker:
        params["ticker"] = ticker
    if since:
        params["ex_dividend_date.gte"] = since
    if until:
        params["ex_dividend_date.lte"] = until
    rows: list[dict] = []
    for record in client.paginate("/v3/reference/dividends", params):
        rows.append(
            {
                "symbol": normalize_symbol(record.get("ticker", "")),
                "ex_date": str(record.get("ex_dividend_date", "")),
                "cash_div": float(record.get("cash_amount") or 0.0),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "ex_date", "cash_div"])


def build_corp_actions(splits: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """合并拆股与分红为 AlphaForge ``corp_actions`` 表。

    列：``[symbol, ex_date, split_ratio, cash_div]``。同一 ``(symbol, ex_date)`` 同时存在拆股
    与分红时，分别取拆股比与现金额（取 max 合并：拆股行 cash_div=0、分红行 split_ratio=1.0，
    max 即保留各自非平凡值）。
    """
    cols = ["symbol", "ex_date", "split_ratio", "cash_div"]
    s = splits.copy()
    s["cash_div"] = 0.0
    d = dividends.copy()
    d["split_ratio"] = 1.0
    both = pd.concat([s[cols], d[cols]], ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=cols)
    both = both[both["ex_date"].astype(str).str.len() >= 8]
    agg = both.groupby(["symbol", "ex_date"], as_index=False).agg(
        split_ratio=("split_ratio", "max"),
        cash_div=("cash_div", "max"),
    )
    return agg[cols].sort_values(["symbol", "ex_date"]).reset_index(drop=True)
