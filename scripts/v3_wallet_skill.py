"""P3 第一阶段：钱包级技能表（知情流识别的数据基础）。

动机：J 族已发现孤立跳（私有信息候选）在豆粕 M 上有正残余，但"私有
信息"目前只由跳变形态间接推断。链上逐笔带 taker 地址与市场结算结果，
可以直接核算每个钱包的历史判断记录，把"聪明钱"从全量成交流中分离。

第一阶段只建数据资产，不做市场检验：

1. wallet_market.parquet：每个 (taker 钱包, 市场) 的持仓方向与按结算
   结果的核算。只用已结算 (resolved) 且有获胜结果标签的市场；剔除
   结算时刻之后的残余成交。核算在代币空间进行：
       pnl_row = sign(BUY=+1/SELL=-1) * (usdc/price) * (win_token - price)
   即按"持有到结算"的名义盈亏（份额 = usdc/price，price 下限 0.02 防
   长尾赔率爆炸）。事件空间净方向 net_evt_usdc = sum(D * usdc) 用于
   命中判定（win_evt 由 outcome_seq=1 侧的获胜与否给出）。
2. wallet_skill_monthly.parquet：逐月时点 (asof) 快照，每个钱包只用
   resolved_at < asof 的市场核算——供第二阶段做 PIT 的聪明钱加权，
   任何时点的技能评分不含未来结算信息。
3. 持续性诊断：技能是否为真实属性（而非运气）的必要条件是跨期持续。
   对活跃钱包按市场结算时间对半分组，前后半段 pnl_per_usd 的秩相关
   为正且显著，才有资格进入第二阶段。

数据：HF daily_aligned（至 2026-04-28，区块 <= 86,126,998）+ 自建
extension_tape（区块 >= 86,127,000 起，同构 schema），两段按爬取起点
天然不重叠。仅取 taker 侧（主动方）；maker 多为做市程序，其方向是
被动成交的镜像，不计入判断记录。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / "data" / "polymarket"
OUT = PM / "features"
SCRATCH = Path("/private/tmp/claude-501/duckdb_spill")

HF_GLOB = str(PM / "daily_aligned" / "*.parquet")
EXT_GLOB = str(PM / "features" / "extension_tape" / "*.parquet")
WM_PATH = OUT / "wallet_market.parquet"
SKILL_PATH = OUT / "wallet_skill_monthly.parquet"

#: 月度时点快照（覆盖期货重叠样本 2026-01 至 07 及此前一年热身）
ASOF_DATES = [f"{y}-{m:02d}-01" for y, ms in
              [(2025, range(1, 13)), (2026, range(1, 8))] for m in ms]
MIN_MKTS_PERSIST = 10   # 持续性诊断的最低已结算市场数


def build_wallet_market(con: duckdb.DuckDBPyConnection) -> None:
    t0 = time.time()
    seg_sql = []
    for glob in (HF_GLOB, EXT_GLOB):
        seg_sql.append(f"""
        SELECT taker, condition_id, block_timestamp, usdc_amount, price,
               taker_direction, D, outcome_seq,
               (outcome_label = winning_outcome_label) AS win_token,
               epoch(resolved_at) AS resolved_epoch
        FROM read_parquet('{glob}')
        WHERE resolution_status = 'resolved'
          AND winning_outcome_label IS NOT NULL
          AND usdc_amount > 0 AND price > 0 AND price < 1
        """)
    union = " UNION ALL ".join(seg_sql)
    con.execute(f"""
    COPY (
      SELECT taker AS wallet, condition_id,
             count(*) AS n_trades,
             sum(usdc_amount) AS usdc_gross,
             sum(D * usdc_amount) AS net_evt_usdc,
             sum((CASE WHEN taker_direction = 'BUY' THEN 1 ELSE -1 END)
                 * (usdc_amount / greatest(price, 0.02))
                 * ((CASE WHEN win_token THEN 1 ELSE 0 END) - price))
               AS pnl,
             max(CASE WHEN outcome_seq = 1
                      THEN (CASE WHEN win_token THEN 1 ELSE 0 END)
                      ELSE (CASE WHEN win_token THEN 0 ELSE 1 END) END)
               AS win_evt,
             min(block_timestamp) AS first_ts,
             max(resolved_epoch) AS resolved_epoch
      FROM ({union})
      WHERE resolved_epoch IS NULL OR block_timestamp < resolved_epoch
      GROUP BY 1, 2
      HAVING sum(usdc_amount) >= 10
    ) TO '{WM_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    n = con.execute(
        f"SELECT count(*) FROM '{WM_PATH}'").fetchone()[0]
    print(f"[1/3] wallet_market {n:,} 行，{time.time() - t0:.0f}s")


def build_skill_snapshots(con: duckdb.DuckDBPyConnection) -> None:
    t0 = time.time()
    frames = []
    for asof in ASOF_DATES:
        df = con.execute(f"""
        SELECT wallet,
               count(*) AS n_mkts,
               sum(usdc_gross) AS usdc,
               sum(CASE WHEN (net_evt_usdc > 0) = (win_evt = 1)
                        THEN usdc_gross ELSE 0 END)
                 / sum(usdc_gross) AS hit_w,
               sum(pnl) AS pnl,
               sum(pnl) / sum(usdc_gross) AS pnl_per_usd
        FROM '{WM_PATH}'
        WHERE resolved_epoch < epoch(TIMESTAMP '{asof} 00:00:00')
          AND net_evt_usdc != 0
        GROUP BY 1
        HAVING count(*) >= 3
        """).df()
        df["asof"] = asof
        frames.append(df)
        print(f"    asof {asof}: {len(df):,} 钱包")
    skill = pd.concat(frames, ignore_index=True)
    skill.to_parquet(SKILL_PATH, index=False)
    print(f"[2/3] skill 快照 {len(skill):,} 行，{time.time() - t0:.0f}s")


def persistence_check(con: duckdb.DuckDBPyConnection) -> dict:
    """技能持续性：前后半段（按结算时间）pnl_per_usd 的秩相关。"""
    t0 = time.time()
    wm = con.execute(f"""
    SELECT wallet, condition_id, usdc_gross, pnl, resolved_epoch
    FROM '{WM_PATH}'
    WHERE wallet IN (
        SELECT wallet FROM '{WM_PATH}'
        GROUP BY wallet HAVING count(*) >= {MIN_MKTS_PERSIST})
    """).df()
    med = wm.groupby("wallet")["resolved_epoch"].transform("median")
    wm["half"] = (wm["resolved_epoch"] > med).map({False: "h1", True: "h2"})
    agg = (wm.groupby(["wallet", "half"])
           .agg(pnl=("pnl", "sum"), usdc=("usdc_gross", "sum"))
           .assign(ppu=lambda d: d["pnl"] / d["usdc"])
           .reset_index()
           .pivot(index="wallet", columns="half", values="ppu")
           .dropna())
    rho, p = spearmanr(agg["h1"], agg["h2"])
    res = {"n_wallets": int(len(agg)), "spearman": float(rho),
           "p_value": float(p)}
    print(f"[3/3] 持续性：n={res['n_wallets']:,} 钱包，"
          f"rho={rho:+.4f}（p={p:.2e}），{time.time() - t0:.0f}s")

    # 按活跃度分层的持续性（报告 §12.13 引用，构建时读 parquet）
    stats = wm.groupby("wallet").agg(n_mkts=("pnl", "size"),
                                     usdc=("usdc_gross", "sum"))
    piv = agg.join(stats)
    tiers = []
    for kind, lo, hi, lab in [
            ("n_mkts", 10, 30, "10-29 市场"),
            ("n_mkts", 30, 100, "30-99 市场"),
            ("n_mkts", 100, 10**12, ">=100 市场"),
            ("usdc", 1e4, 10**15, ">=1 万美元"),
            ("usdc", 1e5, 10**15, ">=10 万美元"),
            ("usdc", 1e6, 10**15, ">=100 万美元")]:
        sub = piv[(piv[kind] >= lo) & (piv[kind] < hi)]
        r, pv = spearmanr(sub["h1"], sub["h2"])
        tiers.append({"tier": lab, "n_wallets": int(len(sub)),
                      "spearman": float(r), "p_value": float(pv)})
    pd.DataFrame(tiers).to_parquet(OUT / "persistence_tiers.parquet",
                                   index=False)
    return res


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{SCRATCH}'")
    con.execute("PRAGMA memory_limit='24GB'")
    build_wallet_market(con)
    build_skill_snapshots(con)
    res = persistence_check(con)
    meta = {
        "built_from": ["daily_aligned/*.parquet", "extension_tape/*.parquet"],
        "taker_only": True,
        "min_usdc_per_market": 10,
        "price_floor_for_shares": 0.02,
        "asof_dates": ASOF_DATES,
        "persistence": res,
    }
    (OUT / "wallet_skill_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
