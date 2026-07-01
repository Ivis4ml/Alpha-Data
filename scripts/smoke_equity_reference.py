"""冒烟测试：验证 massive.com 参考数据管线（标的 / 拆股 / 分红）在当前订阅下可用。

仅做少量调用以尊重免费档约 5 次/分限速。
"""

from __future__ import annotations

import itertools

import pandas as pd

from alpha_data.common.env import load_env
from alpha_data.equity import reference
from alpha_data.equity.massive_client import MassiveClient, MassiveError


def main() -> int:
    load_env()
    client = MassiveClient()

    print("== 标的样例（前 3 条，不翻页）==")
    sample = list(itertools.islice(reference.iter_tickers(client, active=True, limit=100), 3))
    for t in sample[:3]:
        print(f"  {t.get('ticker'):8s} {t.get('primary_exchange',''):8s} {t.get('name','')[:40]}")

    print("\n== AAPL 拆股 ==")
    try:
        splits = reference.fetch_splits(client, ticker="AAPL")
        print(splits.to_string(index=False) if not splits.empty else "  （无）")
    except MassiveError as exc:
        print(f"  拆股端点不可用：{exc}")
        splits = pd.DataFrame(columns=["symbol", "ex_date", "split_ratio"])

    print("\n== AAPL 分红（最近若干）==")
    try:
        dividends = reference.fetch_dividends(client, ticker="AAPL")
        if dividends.empty:
            print("  （无）")
        else:
            print(dividends.sort_values("ex_date").tail(4).to_string(index=False))
    except MassiveError as exc:
        print(f"  分红端点不可用：{exc}")
        dividends = pd.DataFrame(columns=["symbol", "ex_date", "cash_div"])

    print("\n== corp_actions 合并（AlphaForge 口径）==")
    ca = reference.build_corp_actions(splits, dividends)
    print(f"  行数={len(ca)}；列={list(ca.columns)}")
    print(ca.tail(5).to_string(index=False) if not ca.empty else "  （空）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
