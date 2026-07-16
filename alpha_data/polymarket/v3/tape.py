"""统一 tape（Layer 0 的可得近似）：HF ``daily_aligned`` 与自爬链上成交的拼接。

分界与同质性
------------

- HF ``daily_aligned`` 覆盖至旧交易所合约最后一笔成交（区块 86,126,998，
  2026-04-28 11:00:40 UTC）。
- 自爬链上日分区（``data/polymarket_chain/daily/trades``）自区块 86,127,000 起，
  两段按区块号天然不重叠。
- 迁移重叠期（v2 交易所 2026-04-03 上线至 04-28 切换）v2 流量占比实测 < 0.04%
  （抽样六个 3,000 块窗口，v2 6..157 条对 v1 约 42..74 万条），故 HF 段缺失 v2
  成交不构成流量断层；实测记录见 v3 结果报告。

扩展段构造规则与公开数据集 ``daily_aligned`` 逐条对齐（enrich.py 的实证反推）：
剔除中继腿、只保留已核验 Polymarket 场馆、只保留非 negRisk 的二元市场、内连接
市场元数据；``p_event`` / ``D`` 用同一真值表。全部变换在 DuckDB 内完成。

``resolved_at`` 来源：扩展段用自爬 ``ConditionResolution`` 事件（分片无时间戳列，
以成交表的区块-时间对做 ASOF 最近邻回填，Polygon 出块约 2.1s，误差为秒级）。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alpha_data.polymarket import store

ROOT = Path(__file__).resolve().parents[3]
CHAIN_DIR = ROOT / "data" / "polymarket_chain"
EXT_DIR = store.POLY_DIR / "features" / "extension_tape"

#: HF daily_aligned 覆盖的最后一个区块（旧交易所最后一笔成交所在块）。
HF_LAST_BLOCK = 86_126_998


def resolutions_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """自爬 ``ConditionResolution`` 分片 -> ``[condition_id, resolved_at]``（UTC）。

    分片行无时间戳，从 ``id``（``137_{block}_{logIndex}``）取区块号，与成交事件的
    ``(block_number, block_timestamp)`` 对做 ASOF 最近邻（Polygon 出块约 2.1s，
    误差秒级）。区块-时间对来自两处：自爬成交日分区（区块 >= 86,127,000）与
    HF ``OrderFilled`` 原始层（更早区块，从 ``id`` 解析块号）。同一 condition
    多次结算（UMA 争议重报）取首次。
    """
    res_glob = str(CHAIN_DIR / "shards" / "resolutions" / "*.parquet").replace("'", "''")
    trades_glob = str(CHAIN_DIR / "daily" / "trades" / "*.parquet").replace("'", "''")
    hf_of_glob = str(store.POLY_DIR / "OrderFilled" / "2026_*.parquet").replace("'", "''")
    hf_blocks = ""
    if list((store.POLY_DIR / "OrderFilled").glob("2026_*.parquet")):
        hf_blocks = f"""
            UNION ALL
            SELECT CAST(split_part(id, '_', 2) AS BIGINT) AS block_number,
                   block_timestamp
            FROM read_parquet('{hf_of_glob}')
        """
    sql = f"""
        WITH res AS (
            SELECT condition_id,
                   CAST(split_part(id, '_', 2) AS BIGINT) AS block_number
            FROM read_parquet('{res_glob}')
        ),
        pairs AS (
            SELECT block_number, block_timestamp FROM read_parquet('{trades_glob}')
            {hf_blocks}
        ),
        blocks AS (
            SELECT block_number, min(block_timestamp) AS ts
            FROM pairs
            GROUP BY block_number
        )
        SELECT r.condition_id,
               to_timestamp(min(b.ts)) AS resolved_at
        FROM res r
        ASOF JOIN blocks b ON r.block_number >= b.block_number
        GROUP BY r.condition_id
    """
    return con.execute(sql).fetch_df()


def build_extension(
    out_dir: Path = EXT_DIR,
    *,
    days: list[str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> int:
    """把自爬链上日分区转为 ``daily_aligned`` 同构的扩展 tape，返回写出的天数。

    Args:
        out_dir: 输出目录（``{YYYY-MM-DD}.parquet``，与 ``daily_aligned`` 同布局）。
        days: 只处理指定日期（默认全部）。
        con: 复用的 DuckDB 连接（默认新建，会话时区 UTC）。
    """
    con = con or store.connect()
    out_dir.mkdir(parents=True, exist_ok=True)

    res = resolutions_table(con)
    con.register("res_tbl", res)

    asset_map = str(CHAIN_DIR / "asset_map.parquet").replace("'", "''")
    daily_dir = CHAIN_DIR / "daily" / "trades"
    all_days = sorted(p.stem for p in daily_dir.glob("*.parquet"))
    if days is not None:
        all_days = [d for d in all_days if d in set(days)]

    written = 0
    for day in all_days:
        src = str(daily_dir / f"{day}.parquet").replace("'", "''")
        dst = out_dir / f"{day}.parquet"
        sql = f"""
            COPY (
                WITH am AS (
                    SELECT *,
                           max(outcome_seq) OVER (PARTITION BY condition_id) = 2
                               AS is_binary
                    FROM read_parquet('{asset_map}')
                )
                SELECT t.token_asset_id                        AS asset_id,
                       t.block_timestamp,
                       t.price,
                       t.maker,
                       t.taker,
                       t.taker_direction,
                       t.usdc_amount,
                       t.fee_usdc,
                       a.condition_id,
                       a.outcome_seq,
                       a.neg_risk,
                       CAST(NULL AS VARCHAR)                   AS category,
                       CAST(NULL AS VARCHAR)                   AS category_refined,
                       a.outcome_label,
                       a.winning_outcome_label,
                       CASE WHEN r.resolved_at IS NOT NULL
                            THEN 'resolved' END                AS resolution_status,
                       a.taker_base_fee,
                       a.maker_base_fee,
                       CAST(NULL AS TIMESTAMP)                 AS opens_at,
                       CAST(NULL AS TIMESTAMP)                 AS close_at,
                       CAST(r.resolved_at AS TIMESTAMP)        AS resolved_at,
                       a.market_slug,
                       CASE WHEN a.outcome_seq = 1 THEN t.price
                            ELSE 1.0 - t.price END             AS p_event,
                       CAST((CASE WHEN t.taker_direction = 'BUY' THEN 1 ELSE -1 END)
                            * (CASE WHEN a.outcome_seq = 1 THEN 1 ELSE -1 END)
                            AS TINYINT)                        AS D
                FROM read_parquet('{src}') t
                JOIN am a ON t.token_asset_id = a.asset_id
                LEFT JOIN res_tbl r ON a.condition_id = r.condition_id
                WHERE NOT t.is_relay
                  AND t.venue_class = 'polymarket'
                  AND NOT a.neg_risk
                  AND a.is_binary
                  AND t.block_number > {HF_LAST_BLOCK}
                ORDER BY t.block_timestamp, t.block_number, t.log_index
            ) TO '{str(dst).replace("'", "''")}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
        written += 1
    return written


def union_sql(columns: str) -> str:
    """两段 tape 的 UNION ALL 子查询（列名以 ``daily_aligned`` 口径给出）。"""
    hf = store.daily_aligned_glob().replace("'", "''")
    ext = str(EXT_DIR / "*.parquet").replace("'", "''")
    return f"""(
        SELECT {columns} FROM read_parquet('{hf}')
        UNION ALL
        SELECT {columns} FROM read_parquet('{ext}')
    )"""


def fetch_trades_cn(con: duckdb.DuckDBPyConnection, condition_id: str) -> pd.DataFrame:
    """统一 tape 上按市场拉取全部成交（接口与 ``cn_features.fetch_trades_cn`` 一致）。

    ``resolved_at``：HF 段与扩展段可能只有一段有值（例如市场在 04-28 后才结算），
    取该市场任一非空值（首次结算）。
    """
    cols = "condition_id, block_timestamp, p_event, D, usdc_amount, resolved_at"
    sql = f"""
        SELECT timezone('Asia/Shanghai', to_timestamp(block_timestamp)) AS cn_ts,
               p_event,
               D,
               usdc_amount,
               timezone('Asia/Shanghai',
                        min(resolved_at) OVER ()) AS resolved_cn
        FROM {union_sql(cols)}
        WHERE condition_id = ?
        ORDER BY block_timestamp
    """
    df = con.execute(sql, [condition_id]).fetch_df()
    if not df.empty:
        df["cn_ts"] = pd.to_datetime(df["cn_ts"])
        df["resolved_cn"] = pd.to_datetime(df["resolved_cn"])
    return df
