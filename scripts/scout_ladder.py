"""任务 2：阶梯隐含中位数因子的可行性侦察（零检验成本）。

只回答"能不能造"，不回答"预测不预测"：本脚本不接触任何期货收益、
不计算任何 IC，不进任何检验族。

侦察内容：
1. slug 解析率：``will-{asset}-hit-{high|low}-{strike}-by-{expiry}[-后缀]``；
2. 阶梯结构：每 (标的, 方向, 到期) 有多少档行权价（>=3 档才能插值中位数）；
3. 活跃度：各阶梯的成交额与笔数（registry 的 usdc_win / n_win）；
4. 快照单调性：任选一个交易日收盘快照，P(触及 >= K) 应随 K 递减，
   报告违反率与"隐含中位数可定义"（概率穿越 0.5）的阶梯占比。

产物：data/cn_futures/analysis/v4/scout_ladder.json + 控制台报告

用法::

    .venv/bin/python scripts/scout_ladder.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
#: 快照时刻（UTC）：样本中段任选、事前指定，仅用于结构检查。
SNAPSHOT_UTC = "2026-04-15 12:00:00"

PAT = re.compile(
    r"^will-(?P<asset>[a-z\-]+?)-hit-(?P<side>high|low)-"
    r"(?P<strike>\d+(?:\.\d+)?)-by-(?P<expiry>[a-z0-9\-]+?)"
    r"(?:-\d{3}(?:-\d{3})*)?$")


def parse_slug(slug: str) -> dict | None:
    m = PAT.match(slug)
    if not m:
        return None
    d = m.groupdict()
    d["strike"] = float(d["strike"])
    return d


def main() -> int:
    reg = pd.read_parquet(
        ROOT / "data" / "polymarket" / "features" / "cn_registry_v3.parquet")
    lad = reg[reg["theme"].isin(["oil_price", "metal_price"])] \
        .drop_duplicates("condition_id").copy()

    parsed, failed = [], []
    for _, r in lad.iterrows():
        d = parse_slug(r["slug"])
        if d is None:
            failed.append(r["slug"])
            continue
        parsed.append({**d, "condition_id": r["condition_id"],
                       "theme": r["theme"], "usdc_win": r["usdc_win"],
                       "n_win": r["n_win"]})
    pf = pd.DataFrame(parsed)
    report: dict[str, object] = {
        "snapshot_utc": SNAPSHOT_UTC,
        "n_markets": int(len(lad)),
        "n_parsed": int(len(pf)),
        "parse_rate": round(len(pf) / len(lad), 4),
        "failed_slugs": failed[:10],
    }

    # 阶梯结构：每 (asset, side, expiry) 的档数
    grp = pf.groupby(["asset", "side", "expiry"]).agg(
        n_strikes=("strike", "nunique"), usdc=("usdc_win", "sum"),
        n_trades=("n_win", "sum")).reset_index()
    ladders3 = grp[grp["n_strikes"] >= 3]
    report["n_ladders_total"] = int(len(grp))
    report["n_ladders_ge3"] = int(len(ladders3))
    report["ladder_usdc_ge3"] = float(ladders3["usdc"].sum())

    # 快照单调性：各已解析市场在快照时刻的 LOCF p_event
    con = store.connect()
    ids = ",".join("'" + c + "'" for c in pf["condition_id"])
    sql = f"""
        SELECT condition_id,
               arg_max(p_event, block_timestamp) AS p_last
        FROM {tape.union_sql('condition_id, block_timestamp, p_event')}
        WHERE condition_id IN ({ids}) AND p_event IS NOT NULL
          AND block_timestamp <= epoch(TIMESTAMP '{SNAPSHOT_UTC}')
          AND block_timestamp >= epoch(TIMESTAMP '{SNAPSHOT_UTC}') - 86400*30
        GROUP BY 1
    """
    snap = con.execute(sql).fetch_df()
    pf = pf.merge(snap, on="condition_id", how="left")

    viol, defined, checked = 0, 0, 0
    pairs_checked = 0
    for (_a, side, _e), b in pf.dropna(subset=["p_last"]) \
            .groupby(["asset", "side", "expiry"]):
        if len(b) < 3:
            continue
        b = b.sort_values("strike")
        p = b["p_last"].to_numpy(dtype=float)
        # hit-high：P(触及>=K) 随 K 递减；hit-low：随 K 递增
        diffs = np.diff(p) if side == "low" else -np.diff(p)
        pairs_checked += len(diffs)
        viol += int((diffs < -0.02).sum())      # 2pp 容差外的违反
        checked += 1
        crossed = (p.min() <= 0.5 <= p.max())
        defined += int(crossed)
    report["snapshot_ladders_checked"] = checked
    report["adjacent_pairs_checked"] = pairs_checked
    report["monotonicity_violation_rate"] = (
        round(viol / pairs_checked, 4) if pairs_checked else None)
    report["median_defined_share"] = (
        round(defined / checked, 4) if checked else None)

    V4.mkdir(parents=True, exist_ok=True)
    (V4 / "scout_ladder.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n阶梯明细（>=3 档）：")
    print(ladders3.sort_values("usdc", ascending=False)
          .head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
