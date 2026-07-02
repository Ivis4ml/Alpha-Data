"""在 Alpha-Data 的 MinuteDB 上驱动 AlphaForge 计算因子与日内 IC（美股 / 中证通用）。

本脚本在 Alpha-Data 仓库内演示如何消费已产出的分钟库，不修改 AlphaForge 源码：
接线两处运行时适配（均为 AlphaForge 的既有扩展点），其余全部走平台原生流程。

- 交易日枚举限定到基准标的（:class:`BenchmarkScopedSource`），避免 ``DbMinuteSource``
  默认扫描全库标的导致的极慢枚举。
- 中证符号带交易所后缀（``600519.SH``，用于区分指数 ``000001.SH`` 与个股 ``000001.SZ``），
  而 ``cn_ashare`` 适配器的 ``normalize_symbol`` 会剥成 6 位裸码，故对 CN 覆盖归一以保留后缀；
  美股原生符号无需覆盖。

前置：需能 import ``alphaforge``。设 ``ALPHAFORGE_ROOT`` 指向其仓库根，或自行加入 PYTHONPATH。

用法::

    ALPHAFORGE_ROOT=/path/to/AlphaForge \\
    .venv/bin/python examples/alphaforge/factor_demo.py --market cn
    .venv/bin/python examples/alphaforge/factor_demo.py --market us \\
        --start 2024-01-02 --end 2024-03-29
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_AF_ROOT = os.environ.get("ALPHAFORGE_ROOT")
if _AF_ROOT and _AF_ROOT not in sys.path:
    sys.path.insert(0, _AF_ROOT)

from alphaforge.config import load_config  # noqa: E402
from alphaforge.core.factors.library import build_default_registry  # noqa: E402
from alphaforge.kernel.runtime.combine import combine_equal_z, multi_horizon_ic  # noqa: E402
from alphaforge.market import get_market  # noqa: E402
from alphaforge.providers.data.panel import build_panel  # noqa: E402
from alphaforge.providers.db.minute_db import MinuteDB  # noqa: E402
from alphaforge.providers.db.source import DbMinuteSource  # noqa: E402

HERE = Path(__file__).resolve().parent

FACTOR_IDS = ["amount_surge_20b", "vwap_close_dev_10b", "rev_20b", "volatility_20b", "mom_20b"]
DIRECTIONS = {
    "amount_surge_20b": 1,
    "vwap_close_dev_10b": 1,
    "rev_20b": -1,
    "volatility_20b": -1,
    "mom_20b": 1,
}

# 各市场默认参数（配置文件、基准、演示标的）。
MARKETS = {
    "cn": {
        "config": HERE / "cn.toml",
        "benchmark": "000300.SH",
        "keep_suffix": True,
        "symbols": [
            "600519.SH", "601318.SH", "600036.SH", "600030.SH", "601166.SH", "600276.SH",
            "600887.SH", "601888.SH", "601012.SH", "600900.SH", "000001.SZ", "000858.SZ",
            "000333.SZ", "002594.SZ", "300750.SZ", "000651.SZ", "002415.SZ", "300059.SZ",
            "002475.SZ", "000725.SZ",
        ],
    },
    "us": {
        "config": HERE / "us.toml",
        "benchmark": "SPY",
        "keep_suffix": False,
        "symbols": [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH",
            "XOM", "JNJ", "WMT", "MA", "HD", "PG", "AVGO", "COST", "KO", "PEP",
        ],
    },
}


class BenchmarkScopedSource(DbMinuteSource):
    """交易日枚举限定到基准标的，避免全库扫描（生产推荐口径）。"""

    def __init__(self, db: MinuteDB, benchmark: str, **kw: object) -> None:
        super().__init__(db, **kw)
        self.benchmark = benchmark

    def trading_days(self, start: str, end: str) -> list[str]:
        return self.db.trading_days(start, end, symbols=[self.benchmark])


def run(market: str, start: str, end: str, symbols: list[str] | None) -> pd.DataFrame:
    """构建面板、计算因子、评估日内多 horizon RankIC，返回 IC 汇总表。"""
    m = MARKETS[market]
    cfg = load_config(str(m["config"]))
    syms = symbols or list(m["symbols"])

    if m["keep_suffix"]:
        # 覆盖 cn_ashare 归一以保留交易所后缀，与库内符号一致。
        spec_cls = type(get_market(cfg.market.market_id))
        spec_cls.normalize_symbol = lambda self, raw: str(raw).strip().upper()

    print(
        f"市场={cfg.market.market_id} 基准={cfg.market.benchmark} 库={cfg.data.db_dir}",
        flush=True,
    )
    src = BenchmarkScopedSource(MinuteDB(cfg.data.db_dir), str(m["benchmark"]), rth_only=True)
    panel, meta = build_panel(cfg, start, end, symbols=syms, use_cache=False, source=src)
    print(f"面板：{len(panel):,} 行 × {panel.shape[1]} 列，"
          f"标的 {meta.get('n_symbols')}，交易日 {meta.get('minute_days_available')}", flush=True)
    if panel.empty:
        raise SystemExit("面板为空：检查符号口径与库路径。")

    reg = build_default_registry()
    for fid in FACTOR_IDS:
        _, func = reg.get(fid)
        panel[fid] = pd.to_numeric(func(panel), errors="coerce")

    def _row(name: str, h: dict) -> dict[str, object]:
        return {
            "factor": name,
            "+1bar": h.get("+1bar"),
            "+4bar": h.get("+4bar"),
            "EOD": h.get("EOD"),
        }

    rows: list[dict[str, object]] = []
    for fid in FACTOR_IDS:
        rows.append(_row(fid, multi_horizon_ic(panel, fid, "10:00")))
    panel["combo"] = combine_equal_z(panel, FACTOR_IDS, directions=DIRECTIONS)
    rows.append(_row("combo(dir-norm)", multi_horizon_ic(panel, "combo", "10:00")))

    out = pd.DataFrame(rows)
    print("\n日内多 horizon RankIC（买点 10:00）：")
    print(out.to_string(index=False))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="AlphaForge 因子 IC 演示（读 Alpha-Data MinuteDB）")
    p.add_argument("--market", choices=["cn", "us"], required=True)
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--end", default="2024-03-29")
    p.add_argument("--symbols", default=None, help="逗号分隔；缺省用内置演示标的")
    args = p.parse_args()
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    run(args.market, args.start, args.end, syms)
    print("\nOK：端到端跑通。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
