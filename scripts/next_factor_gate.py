"""P6 因子质量七关门控：下一阶段因子的硬性准入检查表。

回应评审意见 6（"现有 40 个因子对质量没有要求，后面会有要求"）。
任一因子进入确认层之前必须逐关通过；未过者只能留探索层。门控本身
不产生对外检验、不进任何族；其作用是缩小进入确认层的假设数，而确认
必须在未触碰样本上进行。

七关（全部机械可查）：

  1 prereg     事前经济假说与预期符号已登记（freeze/prereg_signs.json）
  2 coverage   频率覆盖可行（非零分钟占比 >= 1% 且日均有效观测 >= 30）
  3 orthogonal 对期货量价四因子（mom15_z/rv15_z/volu15_z/amt15_z）逐日
               残差化后，残差 RankIC 的日间 |t| 是否仍达 2（描述性门，
               非族级判定）
  4 monotone   P3 分位单调（k=15，任一品种通过即记通过）
  5 stability  跨品种 IC(k=15) 符号一致率 >= 60%
  6 cost       P3 策略净收益（成本后）在任一品种为正
  7 no_lookahead 前视自检：截去尾部 5 个交易日重算，公共区间取值必须
               逐值不变（构造只依赖 <= t 的信息）

产物：data/cn_futures/analysis/next/quality_scorecard.parquet

用法::

    .venv/bin/python scripts/next_factor_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v4_market_dispersion import DCOLS, PRODUCT_THEMES_V4, aggregate, market_minutes  # noqa: E402
from v4_signal_grid import classify_locked, clean_labels, find_panel  # noqa: E402

from alpha_data.polymarket import store  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
NEXT = ROOT / "data" / "cn_futures" / "analysis" / "next"
PREREG = ROOT / "freeze" / "prereg_signs.json"
QP = V4 / "quantile_grid.parquet"
DG = V4 / "dispersion_grid.parquet"
MKT_COLS = ("mom15_z", "rv15_z", "volu15_z", "amt15_z")


def daily_resid_ic_t(df: pd.DataFrame, col: str, h: int = 15) -> float:
    """逐日：因子对期货量价四因子日内残差化后与 fwd 的 Spearman；日间 t。"""
    ics = []
    for _, b in df.groupby("trade_date", sort=False):
        x = b[col].to_numpy(dtype=float)
        y = b[f"fwd_{h}"].to_numpy(dtype=float)
        z = b[list(MKT_COLS)].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
        if ok.sum() < 40:
            continue
        x, y, z = x[ok], y[ok], z[ok]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        zz = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(zz, x, rcond=None)
        r = x - zz @ beta
        if np.std(r) == 0:
            continue
        c = stats.spearmanr(r, y).statistic
        if np.isfinite(c):
            ics.append(float(c))
    if len(ics) < 20:
        return float("nan")
    a = np.asarray(ics)
    return float(a.mean() / a.std(ddof=1) * np.sqrt(len(a)))


def lookahead_check(product: str, factor_theme: str) -> bool:
    """截尾重算：丢掉最后 5 个交易日后，公共分钟的取值必须逐值不变。"""
    con = store.connect()
    registry = pd.read_parquet(
        ROOT / "data" / "polymarket" / "features" / "cn_registry_v3.parquet")
    full = market_minutes(con, registry, factor_theme, product)
    if full.empty:
        return False
    agg_full = aggregate(full)
    cut_ts = np.sort(full["m_end"].unique())
    if len(cut_ts) < 1000:
        return False
    cutoff = float(cut_ts[int(len(cut_ts) * 0.9)])
    agg_cut = aggregate(full[full["m_end"] <= cutoff])
    a = agg_full[agg_full["m_end"] <= cutoff].reset_index(drop=True)
    b = agg_cut.reset_index(drop=True)
    if len(a) != len(b):
        return False
    for c in ("disp", "cancel", "hhi", "sign_ratio", "head_share", "skew"):
        if not np.allclose(a[c].to_numpy(dtype=float),
                           b[c].to_numpy(dtype=float),
                           rtol=1e-9, atol=1e-12, equal_nan=True):
            return False
    return True


def main() -> int:
    NEXT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(PREREG.read_text())["signs"]
    qgrid = pd.read_parquet(QP) if QP.exists() else pd.DataFrame()
    dgrid = pd.read_parquet(DG) if DG.exists() else pd.DataFrame()

    # 候选 = 预注册的分散度矩族（因子名 D_<col>；逐主题实例视为同一因子）
    rows = []
    la_cache: dict[str, bool] = {}
    for fname in prereg:
        col = fname[2:]                      # D_disp -> disp
        if col not in DCOLS:
            continue
        inst = dgrid[dgrid["signal"].str.startswith(f"D_{col}_")] \
            if len(dgrid) else pd.DataFrame()
        rec: dict[str, object] = {"factor": fname}
        # 1 prereg
        rec["g1_prereg"] = True              # 迭代自 prereg 本身
        # 2 coverage
        cov_ok = bool(len(inst)
                      and (inst["nonzero_share"] >= 0.01).any()
                      and (inst["n_valid"] / inst["days"] >= 30).any())
        rec["g2_coverage"] = cov_ok
        # 3 orthogonal（对每个实例算，任一 |t|>=2 记过；描述性门）
        best_t = float("nan")
        for _, r in (inst.iterrows() if len(inst) else []):
            panel = pd.read_parquet(find_panel(r["product"]))
            disp = pd.read_parquet(V4 / f"dispersion_{r['product']}.parquet")
            panel["ts"] = pd.to_datetime(panel["ts"])
            disp["ts"] = pd.to_datetime(disp["ts"])
            df = clean_labels(classify_locked(
                panel.merge(disp, on="ts", how="left")))
            t = daily_resid_ic_t(df, r["signal"])
            if np.isfinite(t) and (not np.isfinite(best_t)
                                   or abs(t) > abs(best_t)):
                best_t = t
        rec["g3_orth_t"] = round(best_t, 3) if np.isfinite(best_t) else None
        rec["g3_orthogonal"] = bool(np.isfinite(best_t) and abs(best_t) >= 2)
        # 4 monotone（P3 输出中同名分位行；分散度未入分位则记 False）
        if len(qgrid):
            qs = qgrid[qgrid["signal"].str.startswith(f"D_{col}_")] \
                if qgrid["signal"].str.startswith("D_").any() else \
                pd.DataFrame()
            rec["g4_monotone"] = bool(len(qs) and qs.get(
                "monotone_15", pd.Series(dtype=bool)).any())
        else:
            rec["g4_monotone"] = False
        # 5 stability：跨品种 ic_s_15 符号一致率
        if len(inst) >= 3:
            s = np.sign(inst["ic_s_15"].to_numpy(dtype=float))
            s = s[s != 0]
            rec["g5_stability"] = bool(
                len(s) and max((s > 0).sum(), (s < 0).sum()) / len(s) >= 0.6)
        else:
            rec["g5_stability"] = False
        # 6 cost（P3 策略净收益任一品种为正；分散度未入分位则 False）
        rec["g6_cost"] = bool(len(qgrid)
                              and qgrid["signal"].str.startswith("D_").any()
                              and (qgrid.loc[
                                  qgrid["signal"].str.startswith(f"D_{col}_"),
                                  "strat_net_bp_day"] > 0).any())
        # 7 no_lookahead（每主题算一次，缓存）
        theme = None
        if len(inst):
            first_sig = inst["signal"].iloc[0]
            theme = first_sig[len(f"D_{col}_"):]
        if theme is None:
            rec["g7_no_lookahead"] = False
        else:
            key = theme
            if key not in la_cache:
                prod = next(p for p, ths in PRODUCT_THEMES_V4.items()
                            if theme in ths)
                la_cache[key] = lookahead_check(prod, theme)
            rec["g7_no_lookahead"] = la_cache[key]
        gates = [rec[k] for k in rec if k.startswith("g") and
                 isinstance(rec[k], bool)]
        rec["passed"] = int(sum(gates))
        rec["admitted"] = bool(all(gates))
        rows.append(rec)
        print(f"{fname}: {rec['passed']}/7 关"
              f"{'，准入' if rec['admitted'] else '，留探索层'}")

    out = pd.DataFrame(rows)
    out.to_parquet(NEXT / "quality_scorecard.parquet", index=False)
    print(f"written {NEXT}/quality_scorecard.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
