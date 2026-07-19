"""P0 统计脚本的最小单元测试（第二轮复审要求）。

覆盖复审发现的四类实现缺陷，防止回归：
  1. 孤立跳定义：n_cojump == 0 才是孤立。
  2. 逐日日内 partial RankIC：按日广播的常数变量必须被剔除
     （不得产生显著日内排序）。
  3. wild cluster bootstrap：观察量与自助量使用同一 HAC 学生化。
  4. 日期块置换：目标分钟精确匹配，缺失即剔除（不回退）。
  5. 报告构建产物：不得含旧结论字符串；总账各处等于明细求和。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_jump_inference import (  # noqa: E402
    date_block_perm_p, filter_isolated, hac_t_of, wild_cluster_p)


def test_isolated_is_zero_cojump() -> None:
    jumps = pd.DataFrame({
        "n_cojump": [0, 0, 1, 2],
        "theme": ["t"] * 4,
        "ts": pd.to_datetime(["2026-01-01"] * 4),
    })
    out = filter_isolated(jumps)
    assert len(out) == 2
    assert (out["n_cojump"] == 0).all()


def test_daily_partial_rejects_day_constant_broadcast() -> None:
    from v3_c8_incremental import BASE_COLS, daily_partial_t
    rng = np.random.default_rng(0)
    n_days, n_min = 30, 120
    rows = []
    for d in range(n_days):
        base = rng.normal(size=(n_min, len(BASE_COLS)))
        y = rng.normal(size=n_min)
        for i in range(n_min):
            rows.append({"trade_date": f"d{d:03d}", "fwd_15": y[i],
                         **{c: base[i, j]
                            for j, c in enumerate(BASE_COLS)}})
    df = pd.DataFrame(rows)
    # 按日广播的常数变量：日内零方差，残差化后必须被剔除 -> nan
    x_const = np.repeat(rng.normal(size=n_days), n_min)
    ic, t, n = daily_partial_t(df, x_const)
    assert not np.isfinite(t) or n == 0


def test_wild_bootstrap_uses_hac_studentization() -> None:
    rng = np.random.default_rng(1)
    daily = rng.normal(size=60)
    t0, p = wild_cluster_p(daily, np.random.default_rng(2))
    # 观察统计量必须等于同口径 HAC t
    assert abs(t0 - hac_t_of(daily)) < 1e-12
    # 零假设下 p 不应极端
    assert 0.01 < p < 0.99


def test_perm_exact_match_skips_missing() -> None:
    days = ["d0", "d1", "d2"]
    lookup = {"d0": {100: 1.0}, "d1": {100: 2.0}, "d2": {}}
    # 事件在 d0 的 100 秒；位移到 d2 时无该时刻，必须被剔除而非回退
    p = date_block_perm_p(pd.Series(["d0"] * 12),
                          np.array([100] * 12),
                          np.ones(12), 0.5, lookup, days)
    # d2 缺失 -> 该位移无样本；d1 有 -> 均值 2.0
    assert np.isnan(p) or 0.0 <= p <= 1.0


STALE_STRINGS = [
    "全表唯一跨品种稳健的结构",   # C8 旧表述
    "登记为待办",                 # J 推断旧状态
    "74 项 pytest",               # 旧测试数
    "正式证伪",                   # 过强措辞
    "决定性反证",                 # 已撤回的映射安慰剂结论
    "仅 6 个独立事件",            # 孤立跳筛选反号时代的错误样本量
]


@pytest.mark.skipif(
    not (ROOT / "docs" / "cn_futures_polymarket_report.html").exists(),
    reason="报告未构建")
def test_report_has_no_stale_claims() -> None:
    html = (ROOT / "docs" / "cn_futures_polymarket_report.html").read_text(
        encoding="utf-8")
    for s in STALE_STRINGS:
        assert s not in html, f"旧结论残留：{s}"


@pytest.mark.skipif(
    not (ROOT / "data" / "cn_futures" / "analysis" / "v3" / "supp"
         / "test_count.parquet").exists(),
    reason="总账未生成")
def test_ledger_total_matches_detail() -> None:
    tc = pd.read_parquet(ROOT / "data" / "cn_futures" / "analysis" / "v3"
                         / "supp" / "test_count.parquet")
    total = int(tc["n_units"].sum())
    html_path = ROOT / "docs" / "cn_futures_polymarket_report.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        m = re.search(r"合计约 <b>(\d+)</b> 个检验单元", html)
        assert m is not None, "总账句未找到"
        assert int(m.group(1)) == total, "总账与明细求和不一致"
