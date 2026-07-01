"""市场目录：从 ``daily_aligned`` 汇总各市场（condition_id），供按关键词选取宏观市场。

一次性对全量 ``daily_aligned`` 按 ``condition_id`` 聚合（DuckDB 出核哈希聚合），保留成交额
达到阈值的市场（过滤海量微小市场）。``market_slug`` / 类别 / 生命周期字段为市场级常量。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alpha_data.polymarket import store


def build_catalog(con: duckdb.DuckDBPyConnection, *, min_usdc: float = 50_000.0) -> pd.DataFrame:
    """按 ``condition_id`` 汇总市场目录。

    Args:
        con: DuckDB 连接。
        min_usdc: 仅保留 ``total_usdc >= min_usdc`` 的市场。

    Returns:
        目录 DataFrame，按 ``total_usdc`` 降序。
    """
    glob = store.daily_aligned_glob().replace("'", "''")
    sql = f"""
        SELECT condition_id,
               any_value(market_slug)           AS market_slug,
               any_value(category)              AS category,
               any_value(category_refined)      AS category_refined,
               count(*)                         AS n_trades,
               sum(usdc_amount)                 AS total_usdc,
               min(block_timestamp)             AS first_ts,
               max(block_timestamp)             AS last_ts,
               any_value(opens_at)              AS opens_at,
               any_value(close_at)              AS close_at,
               any_value(resolved_at)           AS resolved_at,
               any_value(winning_outcome_label) AS winning_outcome_label
        FROM read_parquet('{glob}')
        GROUP BY condition_id
        HAVING sum(usdc_amount) >= {float(min_usdc)}
        ORDER BY total_usdc DESC
    """
    return con.execute(sql).fetch_df()


def search(
    catalog: pd.DataFrame,
    keywords: str | list[str],
    *,
    top: int | None = None,
) -> pd.DataFrame:
    """在 ``market_slug`` 中按关键词（任一命中，大小写不敏感，字面匹配）筛选，按成交额降序。"""
    kws = [keywords] if isinstance(keywords, str) else list(keywords)
    slug = catalog["market_slug"].fillna("").str.lower()
    mask = pd.Series(False, index=catalog.index)
    for kw in kws:
        mask = mask | slug.str.contains(kw.lower(), regex=False)
    hit = catalog[mask].sort_values("total_usdc", ascending=False)
    return hit.head(top) if top else hit


def save_catalog(catalog: pd.DataFrame, path: Path | None = None) -> Path:
    """把目录写入 ``data/polymarket/features/market_catalog.parquet``。"""
    store.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    target = path or (store.FEATURES_DIR / "market_catalog.parquet")
    catalog.to_parquet(target, index=False)
    return target
