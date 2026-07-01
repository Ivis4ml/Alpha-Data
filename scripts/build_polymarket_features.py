"""构建 Polymarket 市场级分钟特征面板（防前视，美东网格）。

流程：建市场目录 -> 按宏观主题关键词选取市场 -> 逐市场建分钟特征 -> 合并宽表 -> 写 Parquet。

用法::

    python scripts/build_polymarket_features.py                 # 默认每主题取 1 个市场
    python scripts/build_polymarket_features.py --per-theme 2   # 每主题取成交额前 2
    python scripts/build_polymarket_features.py --catalog-only  # 仅建目录，便于核对 slug
"""

from __future__ import annotations

import argparse

from alpha_data.polymarket import catalog, features, store

# 宏观主题 -> slug 关键词（任一命中，字面匹配）。v1 每主题取成交额最高的若干市场作为代表。
# 关键词力求精确以避免子串误匹配（如裸 'fed' 会命中 'federal-election'）。这是一份可由用户
# 按研究需要增删的配置。
MACRO_THEMES: dict[str, list[str]] = {
    "fed_rate": ["fed-rate-cut", "rate-cut-by", "rate-hike", "fomc", "basis-points"],
    "recession": ["us-recession", "recession-in"],
    "inflation": ["inflation-reach", "annual-inflation", "-cpi-"],
    "election_pres": ["united-states-presidential-election", "which-party-will-win-the-2024"],
    "shutdown": ["government-shutdown"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Polymarket 市场级分钟特征")
    parser.add_argument("--min-usdc", type=float, default=50_000.0, help="目录成交额下限")
    parser.add_argument("--per-theme", type=int, default=1, help="每主题取成交额前 N 个市场")
    parser.add_argument("--catalog-only", action="store_true", help="仅建目录并展示各主题命中")
    args = parser.parse_args()

    con = store.connect()
    print("构建市场目录（daily_aligned 全量按 condition_id 聚合）……", flush=True)
    cat = catalog.build_catalog(con, min_usdc=args.min_usdc)
    path = catalog.save_catalog(cat)
    print(f"  目录：{len(cat)} 个市场（total_usdc >= {args.min_usdc:.0f}）-> {path}")

    print("\n各主题命中（按成交额）：")
    markets: dict[str, str] = {}
    for theme, keywords in MACRO_THEMES.items():
        hit = catalog.search(cat, keywords, top=args.per_theme)
        if hit.empty:
            print(f"  {theme:16s} （无命中）")
            continue
        for i, row in enumerate(hit.itertuples(index=False)):
            key = theme if args.per_theme == 1 else f"{theme}{i + 1}"
            markets[key] = row.condition_id
            print(f"  {key:16s} usdc={row.total_usdc / 1e6:8.2f}M  {str(row.market_slug)[:64]}")

    if args.catalog_only:
        con.close()
        return 0
    if not markets:
        print("未匹配到任何市场，退出。")
        con.close()
        return 1

    print(f"\n构建 {len(markets)} 个市场的分钟特征……", flush=True)
    panel = features.build_feature_panel(con, markets)
    store.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = store.FEATURES_DIR / "market_features.parquet"
    panel.to_parquet(out, index=False)
    print(f"  面板：{panel.shape[0]} 行 × {panel.shape[1]} 列 -> {out}")
    print("  列：", list(panel.columns))

    feat_cols = [c for c in panel.columns if c not in ("trade_date", "minute")]
    nonnull = panel.dropna(how="all", subset=feat_cols)
    show = ["trade_date", "minute"] + [c for c in panel.columns if c.endswith("_p")][:4]
    print("\n有值样例（最后 5 行，部分列）：")
    print(nonnull[show].tail(5).to_string(index=False))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
