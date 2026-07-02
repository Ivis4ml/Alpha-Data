"""massive.com 参考数据 → AlphaForge 表。

- 标的全集（含退市，survivorship-free）→ ``listing`` 表 ``[symbol, name, exchange, status,
  ipo_date, delist_date]``。
- 拆股 ``/v3/reference/splits`` + 分红 ``/v3/reference/dividends`` → ``corp_actions`` 表
  ``[symbol, ex_date, split_ratio, cash_div]``（AlphaForge ``MinuteDB.write_corp_actions`` 口径）。

参考端点不受历史时间档位限制，可在升级订阅前先行拉取（注意免费档约 5 次/分限速）。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

from alpha_data.equity.massive_client import MassiveClient

# "." 后缀仅当命中此白名单才视为交易所后缀剥离；其余 "." 后缀（BRK.A / AAC.U / ACHR.WS 等
# 类别股 / 单位 / 权证）是 Polygon/massive 原生 ticker 的一部分，必须保留，与分钟库的
# upper(ticker) 保持一致，否则 listing / corp_actions 与分钟数据在 symbol 上失配。
_EXCHANGE_DOT_SUFFIXES = frozenset({"US", "O", "N", "OQ"})


def normalize_symbol(raw: str) -> str:
    """符号归一（镜像 AlphaForge ``default_normalize_symbol``）。

    仅剥离 ``:`` 后缀（AAPL:NASDAQ）与白名单内的 ``.`` 后缀（AAPL.US / AAPL.O）；
    类别股 / 单位 / 权证的 ``.`` 后缀保留（BRK.A -> BRK.A）。
    """
    s = str(raw).strip().upper()
    if ":" in s:
        head = s.split(":", 1)[0]
        if head:
            s = head
    if "." in s:
        head, tail = s.rsplit(".", 1)
        if head and tail in _EXCHANGE_DOT_SUFFIXES:
            s = head
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

    分别以 ``active=true`` 与 ``active=false`` 拉取后合并去重。同一 symbol 同时出现
    active 与 delisted 记录时（ticker 回收：旧公司退市后新公司复用该代码）保留 active
    行，即以当前在市主体为准；多条 delisted 记录取 ``delist_date`` 最新的一条。
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
    # 'active' < 'delisted' 字典序，稳定排序后 keep='first' 即 active 优先；
    # 同为 delisted 时 delist_date 降序，保留最近一次退市记录。
    df = df.sort_values(
        ["symbol", "status", "delist_date"],
        ascending=[True, True, False],
        kind="stable",
    )
    return df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


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


def remap_to_native_symbols(df: pd.DataFrame, canonical_symbols: Iterable[str]) -> pd.DataFrame:
    """把 splits / dividends 端点的无点类别股 ticker 映射回带点原生形式。

    massive/Polygon 的 ``/v3/reference/splits`` 与 ``/v3/reference/dividends`` 对类别股返回
    无点 ticker（``BFB`` / ``MOGA`` / ``HEIA``），而 tickers 端点与分钟 Flat Files 用带点
    原生形式（``BF.B`` / ``MOG.A`` / ``HEI.A``）。不做映射时这些标的的公司行动在下游永远
    匹配不上分钟数据（复权因子静默保持 1.0）。

    映射规则（保守，宁可不映射也不错映射）：仅当 ``symbol`` 不在 ``canonical_symbols``
    中，且恰有一个带点标准符号去点后与之相等时，才替换为该带点形式；否则原样保留。

    Args:
        df: 含 ``symbol`` 列的表（splits / dividends）。
        canonical_symbols: 标准符号全集（listing 表的 ``symbol`` 列，来自 tickers 端点）。

    Returns:
        ``symbol`` 列替换后的副本（其余列不变）。
    """
    known = {str(s) for s in canonical_symbols}
    collapse: dict[str, list[str]] = {}
    for s in known:
        if "." in s:
            collapse.setdefault(s.replace(".", ""), []).append(s)

    def _fix(sym: str) -> str:
        if sym in known:
            return sym
        cands = collapse.get(sym)
        if cands is not None and len(cands) == 1:
            return cands[0]
        return sym

    out = df.copy()
    if not out.empty:
        out["symbol"] = out["symbol"].map(_fix)
    return out


def build_corp_actions(splits: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """合并拆股与分红为 AlphaForge ``corp_actions`` 表。

    列：``[symbol, ex_date, split_ratio, cash_div]``。同一 ``(symbol, ex_date)``：

    - 多笔分红（常规 + 特别股息）``cash_div`` 求和；
    - 多次拆股 ``split_ratio`` 取乘积（复合比例）；
    - 拆股与分红同日则并入同一行。
    """
    cols = ["symbol", "ex_date", "split_ratio", "cash_div"]
    key = ["symbol", "ex_date"]

    def _valid(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df[df["ex_date"].astype(str).str.len() >= 8]

    s = _valid(splits)
    d = _valid(dividends)
    s_agg = (
        s.groupby(key, as_index=False)["split_ratio"].prod()
        if not s.empty
        else pd.DataFrame(columns=[*key, "split_ratio"])
    )
    d_agg = (
        d.groupby(key, as_index=False)["cash_div"].sum()
        if not d.empty
        else pd.DataFrame(columns=[*key, "cash_div"])
    )
    merged = s_agg.merge(d_agg, on=key, how="outer")
    if merged.empty:
        return pd.DataFrame(columns=cols)
    merged["split_ratio"] = merged["split_ratio"].astype("float64").fillna(1.0)
    merged["cash_div"] = merged["cash_div"].astype("float64").fillna(0.0)
    return merged[cols].sort_values(key).reset_index(drop=True)
