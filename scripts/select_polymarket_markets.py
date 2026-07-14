"""按事前注册映射表（docs/mapping_taxonomy.md）选取与国内期货相关的 Polymarket 市场。

从 ``daily_aligned`` 统计重叠窗口内各市场活动，按注册的 slug 模式规则打主题与
方向先验，输出 orientation 登记表：

    data/polymarket/features/cn_registry.parquet
    [condition_id, slug, category, theme, sigma, product, orientation,
     usdc_win, n_win, first_ts, last_ts, admit_ts, resolved_at, exploratory]

``admit_ts`` 为时点化准入时刻（窗口内累计成交额首次达到门槛的秒级 UTC 时间戳），
下游对早于该时刻的信号置 NaN。

一行 = 一个 (市场, 品种) 对；``orientation = sigma * m_product``。

用法::

    .venv/bin/python scripts/select_polymarket_markets.py [--min-usdc 100000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.polymarket import store  # noqa: E402

#: 重叠窗口（北京时间日期界，转 UTC 秒在查询内处理）。
WINDOW_START = "2026-01-04 16:00:00"   # UTC，对应北京 2026-01-05 00:00
WINDOW_END = "2026-04-28 16:00:00"     # UTC，daily_aligned 尾部

CATEGORIES = ("Politics", "Finance", "Other", "Sci-Tech")

#: 排除模式（新奇 / 元市场）。
EXCLUDES = ("-say-", "wear-a-", "-mention", "odds-of-")

#: 规则：(theme, sigma, 模式列表, {product: m}, exploratory, 排除子串)。
#: 模式为子串匹配（小写）；首个命中的规则生效（规则顺序即优先级，例如
#: ``ceasefire-end``（升级）先于 ``us-x-iran-ceasefire``（降级），
#: ``refund-tariffs`` 先于泛化的 ``tariff``）。命中排除子串的市场跳过该条规则。
Rule = tuple[str, int, list[str], dict[str, int], bool, tuple[str, ...]]

RULES: list[Rule] = [
    # 主题一：中东冲突。
    # 停火"结束 / 破裂"是升级事件，必须先于降级规则匹配（v1.1 修正，
    # 原版被 ``us-x-iran-ceasefire`` 子串误伤反号）。
    ("mideast_conflict", +1,
     ["ceasefire-end", "ceasefire-broken", "ceasefire-has-been-broken"],
     {"SC": +1, "AU": +1}, False, ()),
    ("mideast_conflict", +1,
     ["us-strikes-iran", "us-forces-enter-iran", "israel-strikes-iran",
      "iran-strike-israel", "will-the-us-invade-iran", "iranian-regime-fall",
      "khamenei-out", "close-the-strait-of-hormuz", "kharg-island",
      "ground-offensive-in-lebanon", "us-forces-seize-another-oil-tanker"],
     {"SC": +1, "AU": +1}, False, ()),
    ("mideast_conflict", -1,
     ["us-x-iran-ceasefire", "permanent-peace-deal", "us-x-iran-meeting",
      "military-action-against-iran-ends",
      "strait-of-hormuz-traffic-returns-to-normal", "israel-x-hezbollah-ceasefire"],
     {"SC": +1, "AU": +1}, False, ()),
    # 主题二：原油价格直连
    ("oil_price", +1,
     ["crude-oil-cl-hit-high", "wti-crude-oil-wti-hit-high"],
     {"SC": +1}, False, ()),
    ("oil_price", -1,
     ["crude-oil-cl-hit-low", "wti-crude-oil-wti-hit-low"],
     {"SC": +1}, False, ()),
    # 主题三：贵金属价格直连
    ("metal_price", +1, ["gold-gc-hit-high"], {"AU": +1}, False, ()),
    ("metal_price", -1, ["gold-gc-hit-low"], {"AU": +1}, False, ()),
    ("metal_price", +1, ["silver-si-hit-high"], {"AG": +1}, False, ()),
    ("metal_price", -1, ["silver-si-hit-low"], {"AG": +1}, False, ()),
    # 主题四：美联储政策
    ("fed_policy", +1, ["fed-rate-cut"], {"AU": +1, "AG": +1, "CU": +1}, False, ()),
    ("fed_policy", -1, ["fed-rate-hike"], {"AU": +1, "AG": +1, "CU": +1}, False, ()),
    ("fed_policy", +1,
     ["powell-out-as-fed-chair", "try-to-fire-powell", "sue-powell",
      "powell-federally-charged"],
     {"AU": +1, "AG": +1, "CU": +1}, True, ()),
    # 主题五：俄乌冲突
    ("russia_ukraine", +1,
     ["russia-strike-", "russia-capture", "russia-invade-a-nato",
      "ukraine-strikes-another-tanker"],
     {"SC": +1, "AU": +1}, False, ()),
    ("russia_ukraine", -1,
     ["russia-x-ukraine-ceasefire", "zelenskyy-talk-to-putin"],
     {"SC": +1, "AU": +1}, False, ()),
    # 主题六：中美贸易（探索性）。格陵兰关税与中美机制无关，排除（v1.1 修正）。
    ("us_china_trade", -1,
     ["refund-tariffs", "trump-visit-china", "trump-talk-to-xi"],
     {"M": +1, "CF": +1, "CU": -1, "I": -1}, True, ()),
    ("us_china_trade", +1,
     ["tariff"],
     {"M": +1, "CF": +1, "CU": -1, "I": -1}, True, ("greenland",)),
    # 主题七：台海风险（探索性）
    ("taiwan_risk", +1,
     ["china-invade-taiwan", "blockade-taiwan", "china-x-taiwan-military-clash"],
     {"AU": +1, "IF": -1}, True, ()),
    # 主题八：美国政府停摆（探索性）
    ("us_shutdown", +1,
     ["government-shutdown", "dhs-shutdown"],
     {"AU": +1}, True, ()),
]


def window_activity(min_usdc: float) -> pd.DataFrame:
    """统计重叠窗口内各市场活动（DuckDB 流式，不载入逐笔）。

    额外给出**时点化准入时刻** ``admit_ts``：窗口内累计成交额首次达到 ``min_usdc``
    的秒级时间戳。下游把严格早于该时刻的信号置 NaN，使任一时刻的面板构成只依赖
    当时已实现的流动性（避免"全窗口流动性筛选"把样本内信息带入选样）。
    """
    con = store.connect()
    glob = store.daily_aligned_glob().replace("'", "''")
    cats = ", ".join(f"'{c}'" for c in CATEGORIES)
    sql = f"""
        WITH base AS (
            SELECT condition_id, market_slug, category_refined, resolved_at,
                   block_timestamp, usdc_amount,
                   sum(usdc_amount) OVER (
                       PARTITION BY condition_id ORDER BY block_timestamp
                       ROWS UNBOUNDED PRECEDING
                   ) AS cum_usdc
            FROM read_parquet('{glob}')
            WHERE block_timestamp >= epoch(TIMESTAMP '{WINDOW_START}')
              AND block_timestamp <  epoch(TIMESTAMP '{WINDOW_END}')
              AND category_refined IN ({cats})
        )
        SELECT condition_id,
               any_value(market_slug)      AS slug,
               any_value(category_refined) AS category,
               any_value(resolved_at)      AS resolved_at,
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
    return con.execute(sql).fetch_df()


def match_rule(slug: str) -> tuple[str, int, dict[str, int], bool] | None:
    """返回首个命中的规则 (theme, sigma, products, exploratory)；无命中返回 None。"""
    s = slug.lower()
    if any(pat in s for pat in EXCLUDES):
        return None
    for theme, sigma, patterns, products, exploratory, rule_excludes in RULES:
        if any(ex in s for ex in rule_excludes):
            continue
        if any(pat in s for pat in patterns):
            return theme, sigma, products, exploratory
    return None


def build_registry(activity: pd.DataFrame) -> pd.DataFrame:
    """对窗口活动表打主题与方向，展开为 (市场, 品种) 行。"""
    rows: list[dict] = []
    for rec in activity.itertuples(index=False):
        hit = match_rule(rec.slug)
        if hit is None:
            continue
        theme, sigma, products, exploratory = hit
        for product, m in products.items():
            rows.append(
                {
                    "condition_id": rec.condition_id,
                    "slug": rec.slug,
                    "category": rec.category,
                    "theme": theme,
                    "sigma": sigma,
                    "product": product,
                    "orientation": sigma * m,
                    "usdc_win": float(rec.usdc_win),
                    "n_win": int(rec.n_win),
                    "first_ts": int(rec.first_ts),
                    "last_ts": int(rec.last_ts),
                    "admit_ts": int(rec.admit_ts),
                    "resolved_at": rec.resolved_at,
                    "exploratory": exploratory,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["theme", "product", "usdc_win"], ascending=[True, True, False]
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="选取与国内期货相关的 Polymarket 市场")
    parser.add_argument("--min-usdc", type=float, default=100_000.0)
    parser.add_argument(
        "--out", default=str(store.FEATURES_DIR / "cn_registry.parquet")
    )
    args = parser.parse_args()

    activity = window_activity(args.min_usdc)
    print(f"窗口内候选市场（usdc >= {args.min_usdc:,.0f}）：{len(activity)}")
    registry = build_registry(activity)
    n_markets = registry["condition_id"].nunique()
    print(f"命中映射：{n_markets} 个市场，{len(registry)} 个 (市场, 品种) 对")
    print("\n主题 × 品种分布：")
    print(
        registry.pivot_table(
            index="theme", columns="product", values="condition_id",
            aggfunc="nunique", fill_value=0,
        ).to_string()
    )
    print("\n主题流动性（窗口内 usdc 合计，百万美元）：")
    theme_usdc = (
        registry.drop_duplicates("condition_id").groupby("theme")["usdc_win"].sum() / 1e6
    )
    print(theme_usdc.round(1).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(out, index=False)
    print(f"\n登记表已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
