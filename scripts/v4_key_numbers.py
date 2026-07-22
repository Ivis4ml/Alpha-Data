"""扩展轮关键数字汇总：全部从 v4 产物计算，写 key_numbers_v2.json。

报告（docs/concise/main.tex 扩展节）只准引用本文件输出的数字，不手抄。
扩展主族 = v3 主族 1,476 个检验 + 扩展品种(CF/I/IF) + 分散度矩族 +
分位注册终点(k=15)，门槛按实际总数重算（只升不降）。

用法::

    .venv/bin/python scripts/v4_key_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
NEXT = ROOT / "data" / "cn_futures" / "analysis" / "next"
OUT = ROOT / "docs" / "concise" / "key_numbers_v2.json"

HORIZONS = [1, 2, 3, 5, 10, 15]
IC_T = [f"ic_t_{h}" for h in HORIZONS]
EV_T = [f"{d}_t_{h}" for d in ("up", "dn") for h in HORIZONS]
PURE_PM = ([f"N{i}" for i in range(1, 11)] + [f"K{i}" for i in range(1, 11)]
           + ["X5", "X7", "X10"])
NEW_PRODUCTS = ("CF", "I", "IF")


def proper_t(g: pd.DataFrame) -> np.ndarray:
    """适当口径 |t| 全集；事件列按实际存在者取（非负信号无下行触发，
    dn_t_* 列可整组缺席）。"""
    parts = []
    cont = g[g["kind"] != "离散"]
    disc = g[g["kind"] == "离散"]
    if len(cont):
        parts.append(np.abs(cont[IC_T].to_numpy(dtype=float)).ravel())
    if len(disc):
        cols = [c for c in EV_T if c in disc.columns]
        if cols:
            parts.append(np.abs(disc[cols].to_numpy(dtype=float)).ravel())
    a = np.concatenate(parts) if parts else np.empty(0)
    return a[np.isfinite(a)]


def bonf(n: int) -> float:
    return float(stats.norm.ppf(1.0 - 0.05 / 2.0 / max(n, 1)))


def rnd(x: float, nd: int = 3) -> float | None:
    return None if not np.isfinite(x) else round(float(x), nd)


def main() -> int:
    key: dict[str, object] = {}

    # ---- P2 涨跌停 ----
    imp = pd.read_parquet(V4 / "locked_impact.parquet")
    key["locked"] = {
        r["product"]: {
            "true_lock_share": rnd(r["true_lock_share"], 5),
            "pseudo_lock_share": rnd(r["pseudo_lock_share"], 5),
            "dropped_share_15": rnd(r["dropped_share_15"], 5),
        } for _, r in imp.iterrows()}

    cln = pd.read_parquet(V4 / "signal_grid_clean.parquet")
    raw = pd.read_parquet(DEF / "signal_grid.parquet")
    cln5 = cln[cln["product"].isin(raw["product"].unique())]
    key["clean_vs_raw"] = {
        "pure_pm_max_t_raw": rnd(proper_t(
            raw[raw["signal"].isin(PURE_PM)]).max()),
        "pure_pm_max_t_clean": rnd(proper_t(
            cln5[cln5["signal"].isin(PURE_PM)]).max()),
    }

    # ---- P5 扩展品种 ----
    ext = cln[cln["product"].isin(NEW_PRODUCTS)]
    key["ext_products"] = {}
    for p in NEW_PRODUCTS:
        a = proper_t(ext[ext["product"] == p])
        pp = proper_t(ext[(ext["product"] == p)
                          & (ext["signal"].isin(PURE_PM))])
        key["ext_products"][p] = {
            "tests": int(len(a)), "max_t": rnd(a.max()),
            "pure_pm_tests": int(len(pp)),
            "pure_pm_max_t": rnd(pp.max()),
        }

    # ---- P4 分散度 ----
    dg = pd.read_parquet(V4 / "dispersion_grid.parquet")
    a = proper_t(dg)
    key["dispersion"] = {
        "cells": int(len(dg)), "tests": int(len(a)),
        "max_t": rnd(a.max()),
        "max_cell": None,
    }
    per = dg.assign(m=dg[IC_T].abs().max(axis=1))
    top = per.loc[per["m"].idxmax()]
    key["dispersion"]["max_cell"] = f"{top['signal']}·{top['product']}"

    # ---- P3 分位 ----
    q = pd.read_parquet(V4 / "quantile_grid.parquet")
    el = q[q["eligible"].astype(bool)]
    t15 = el["ls_t_15"].to_numpy(dtype=float)
    t15 = t15[np.isfinite(t15)]
    key["quantile"] = {
        "pairs_total": int(len(q)),
        "pairs_eligible": int(len(el)),
        "registered_tests_k15": int(len(t15)),
        "max_ls_t_15": rnd(np.abs(t15).max()),
        "monotone_15_n": int(el["monotone_15"].sum()),
        "strat_net_positive_n": int((el["strat_net_bp_day"] > 0).sum()),
        "strat_net_bp_day_median": rnd(el["strat_net_bp_day"].median(), 2),
        "strat_gross_bp_day_median": rnd(
            el["strat_gross_bp_day"].median(), 2),
        "breakeven_ratio_median": rnd(el["breakeven_ratio"].median(), 2),
        "pure_pm_max_ls_t_15": rnd(np.abs(
            el.loc[el["signal"].isin(PURE_PM), "ls_t_15"]
            .to_numpy(dtype=float)).max()),
    }

    # ---- 扩展主族总账（A'）----
    n_v3 = len(proper_t(raw))                       # 1476（冻结轮，不变）
    n_ext = sum(v["tests"] for v in key["ext_products"].values())
    n_disp = key["dispersion"]["tests"]
    n_q = key["quantile"]["registered_tests_k15"]
    n_total = n_v3 + n_ext + n_disp + n_q
    thr = bonf(n_total)
    all_t = np.concatenate([
        proper_t(cln),                              # 8 品种 clean 网格
        proper_t(dg),
        np.abs(t15),
    ])
    pure_parts = np.concatenate([
        proper_t(cln[cln["signal"].isin(PURE_PM)]),
        proper_t(dg),                               # 分散度全为纯 PM 构造
        np.abs(el.loc[el["signal"].isin(PURE_PM), "ls_t_15"]
               .to_numpy(dtype=float)),
    ])
    pure_parts = pure_parts[np.isfinite(pure_parts)]
    key["family_ext"] = {
        "n_v3": int(n_v3), "n_ext_products": int(n_ext),
        "n_dispersion": int(n_disp), "n_quantile_k15": int(n_q),
        "n_total": int(n_total),
        "threshold": rnd(thr),
        "threshold_v3": rnd(bonf(n_v3)),
        "max_t_all": rnd(all_t.max()),
        "n_pass": int((all_t > thr).sum()),
        "pure_pm_n": int(len(pure_parts)),
        "pure_pm_max_t": rnd(pure_parts.max()),
        "pure_pm_pass": int((pure_parts > thr).sum()),
    }
    # 通过门槛的格归属（按信号族）
    pass_fams: set[str] = set()
    for g_ in (cln, dg):
        avail_ev = [c for c in EV_T if c in g_.columns]
        for _, r in g_.iterrows():
            cols = avail_ev if r["kind"] == "离散" else IC_T
            v = np.abs(np.asarray([r[c] for c in cols], dtype=float))
            v = v[np.isfinite(v)]
            if len(v) and v.max() > thr:
                pass_fams.add(str(r["signal"])[0])
    key["family_ext"]["pass_families"] = sorted(pass_fams)

    # ---- P6 门控 ----
    sc = pd.read_parquet(NEXT / "quality_scorecard.parquet")
    key["gate"] = {
        "candidates": int(len(sc)),
        "admitted": int(sc["admitted"].sum()),
        "median_passed": int(sc["passed"].median()),
        "detail": {r["factor"]: int(r["passed"]) for _, r in sc.iterrows()},
    }

    # ---- P1 映射 ----
    tier = pd.read_parquet(
        ROOT / "data" / "polymarket" / "features" / "tier_registry.parquet")
    key["mapping"] = {
        "pairs": int(len(tier)),
        "products_mapped": int(tier["product"].nunique()),
        "primary": int((tier["tier"] == "primary").sum()),
        "secondary": int((tier["tier"] == "secondary").sum()),
        "exploratory": int((tier["tier"] == "exploratory").sum()),
    }

    OUT.write_text(json.dumps(key, ensure_ascii=False, indent=2))
    print(json.dumps(key["family_ext"], ensure_ascii=False, indent=2))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
