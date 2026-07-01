"""检查已下载的 Polymarket-v1 各层：schema、行数、时间范围、样例行。

用 DuckDB 直接对本地 Parquet 查询（不全量载入内存）。
"""

from __future__ import annotations

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "data" / "polymarket"


def _q(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchdf()


def main() -> int:
    con = duckdb.connect()

    da_glob = str(POLY / "daily_aligned" / "*.parquet")
    print("=" * 70)
    print("daily_aligned —— schema")
    schema = _q(con, f"DESCRIBE SELECT * FROM read_parquet('{da_glob}') LIMIT 1")
    print(schema.to_string(index=False))

    print("\ndaily_aligned —— 行数 / 时间范围")
    stats = _q(
        con,
        f"""
        SELECT count(*) AS rows,
               min(block_timestamp) AS ts_min,
               max(block_timestamp) AS ts_max,
               to_timestamp(min(block_timestamp)) AS dt_min_utc,
               to_timestamp(max(block_timestamp)) AS dt_max_utc
        FROM read_parquet('{da_glob}')
        """,
    )
    print(stats.to_string(index=False))

    print("\ndaily_aligned —— 样例 3 行（部分列）")
    sample = _q(
        con,
        f"""
        SELECT block_timestamp, condition_id, outcome_label, taker_direction,
               price, p_event, D, usdc_amount, category_refined, market_slug
        FROM read_parquet('{da_glob}')
        ORDER BY block_timestamp
        LIMIT 3
        """,
    )
    print(sample.to_string(index=False)[:2000])

    for name in ("preparations", "splits", "merges", "resolutions", "redemptions"):
        path = POLY / "CTF" / f"{name}.parquet"
        if not path.exists():
            continue
        print("\n" + "=" * 70)
        print(f"CTF/{name} —— schema + 行数")
        cols = _q(con, f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 1")
        n = _q(con, f"SELECT count(*) AS rows FROM read_parquet('{path}')")
        print("  列:", list(cols["column_name"]))
        print("  行数:", int(n["rows"].iloc[0]))

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
