"""构建 v3 统一 tape 扩展段并重跑时点化准入登记表（窗口延至 2026-07-14）。

步骤：

1. ``alpha_data.polymarket.v3.tape.build_extension``：自爬链上日分区 ->
   ``daily_aligned`` 同构的扩展 tape（``data/polymarket/features/extension_tape/``）。
2. 边界一致性检验：扩展段首日（2026-04-28，区块 > 86,126,998）与 HF 段同日
   活动的市场交集、价格衔接。
3. 重跑事前注册映射（规则与 v1.1 完全一致，仅窗口右端延长）：
   ``cn_registry_v3.parquet``。扩展期新市场无 ``category_refined``（作者派生列），
   对 NULL 类别放行，仅靠 slug 模式与排除词（诚实标注）。

用法::

    .venv/bin/python scripts/v3_build_tape.py [--skip-build]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import select_polymarket_markets as spm  # noqa: E402

from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

#: 扩展窗口右端（UTC）：北京 2026-07-14 00:00，期货库最后交易日为 2026-07-13。
WINDOW_END_V3 = "2026-07-13 16:00:00"

REGISTRY_V3 = store.FEATURES_DIR / "cn_registry_v3.parquet"


def window_activity_v3(min_usdc: float) -> pd.DataFrame:
    """统一 tape 上的窗口活动与时点化准入（与 v1.1 同口径，窗口右端延长）。

    HF 段沿用 ``category_refined`` 过滤；扩展段该列为 NULL（作者派生列不可得），
    对 NULL 放行——slug 模式与排除词是实际的选择规则。
    """
    con = store.connect()
    cats = ", ".join(f"'{c}'" for c in spm.CATEGORIES)
    cols = ("condition_id, market_slug, category_refined, resolved_at, "
            "block_timestamp, usdc_amount")
    sql = f"""
        WITH base AS (
            SELECT condition_id, market_slug, category_refined, resolved_at,
                   block_timestamp, usdc_amount,
                   sum(usdc_amount) OVER (
                       PARTITION BY condition_id ORDER BY block_timestamp
                       ROWS UNBOUNDED PRECEDING
                   ) AS cum_usdc
            FROM {tape.union_sql(cols)}
            WHERE block_timestamp >= epoch(TIMESTAMP '{spm.WINDOW_START}')
              AND block_timestamp <  epoch(TIMESTAMP '{WINDOW_END_V3}')
              AND (category_refined IN ({cats}) OR category_refined IS NULL)
        )
        SELECT condition_id,
               any_value(market_slug)      AS slug,
               any_value(category_refined) AS category,
               max(resolved_at)            AS resolved_at,
               sum(usdc_amount)            AS usdc_win,
               count(*)                    AS n_win,
               min(block_timestamp)        AS first_ts,
               max(block_timestamp)        AS last_ts,
               min(block_timestamp) FILTER (WHERE cum_usdc >= {float(min_usdc)})
                                           AS admit_ts
        FROM base
        GROUP BY condition_id
        HAVING sum(usdc_amount) >= {float(min_usdc)}
    """
    act = con.execute(sql).fetch_df()
    # tape 行内的 resolved_at 元数据稀疏（HF 段作者字段缺失多），
    # 以链上 ConditionResolution 事件补齐（事件时刻优先，元数据兜底）。
    res = tape.resolutions_table(con).rename(columns={"resolved_at": "_res_chain"})
    act = act.merge(res, on="condition_id", how="left")
    act["resolved_at"] = pd.to_datetime(act["_res_chain"], utc=True).combine_first(
        pd.to_datetime(act["resolved_at"], utc=True))
    return act.drop(columns=["_res_chain"])


def boundary_check() -> None:
    """扩展段与 HF 段在 2026-04-28 的衔接检验（同市场两侧末端 / 起端价差）。"""
    con = store.connect()
    hf = store.daily_aligned_glob().replace("'", "''")
    ext = str(tape.EXT_DIR / "2026-04-28.parquet").replace("'", "''")
    sql = f"""
        WITH hf_last AS (
            SELECT condition_id,
                   arg_max(p_event, block_timestamp) AS p_hf,
                   max(block_timestamp) AS ts_hf
            FROM read_parquet('{hf}')
            WHERE block_timestamp >= epoch(TIMESTAMP '2026-04-28 08:00:00')
            GROUP BY condition_id
        ),
        ext_first AS (
            SELECT condition_id,
                   arg_min(p_event, block_timestamp) AS p_ext,
                   min(block_timestamp) AS ts_ext,
                   count(*) AS n_ext
            FROM read_parquet('{ext}')
            GROUP BY condition_id
        )
        SELECT count(*) AS n_common,
               median(abs(p_ext - p_hf)) AS med_gap,
               quantile_cont(abs(p_ext - p_hf), 0.9) AS p90_gap,
               median(ts_ext - ts_hf) AS med_dt_sec
        FROM hf_last JOIN ext_first USING (condition_id)
        WHERE n_ext >= 5
    """
    print("边界衔接（04-28 上午 HF 末笔 vs 扩展段首笔，同市场）：")
    print(con.execute(sql).fetch_df().to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 v3 统一 tape 与登记表")
    parser.add_argument("--skip-build", action="store_true", help="只重跑登记表")
    parser.add_argument("--min-usdc", type=float, default=100_000.0)
    args = parser.parse_args()

    if not args.skip_build:
        t0 = time.time()
        n = tape.build_extension()
        print(f"扩展 tape：写出 {n} 个日分区，耗时 {time.time() - t0:.0f}s")
        boundary_check()

    activity = window_activity_v3(args.min_usdc)
    print(f"\n窗口内候选市场（usdc >= {args.min_usdc:,.0f}）：{len(activity)}")
    registry = spm.build_registry(activity)
    n_markets = registry["condition_id"].nunique()
    print(f"命中映射：{n_markets} 个市场，{len(registry)} 个 (市场, 品种) 对")
    print("\n主题 × 品种分布：")
    print(
        registry.pivot_table(
            index="theme", columns="product", values="condition_id",
            aggfunc="nunique", fill_value=0,
        ).to_string()
    )
    REGISTRY_V3.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(REGISTRY_V3, index=False)
    print(f"\n登记表已写入 {REGISTRY_V3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
