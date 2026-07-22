"""全信号网格：40 信号 x 5 品种 x 6 视界的频率、取值统计、IC / ICIR 与事件研究。

口径（答辩文档 §指标 IC 计算）：

- 视界 h in {1, 2, 3, 5, 10, 15} 分钟，前向对数收益 fwd_h（段内，跨休市置空）。
- 连续 / 计数信号：与 fwd_h 的 Spearman（RankIC）与 Pearson IC；逐日 IC 的
  均值 / 标准差给出 ICIR = mean(IC_d) / std(IC_d)，t = ICIR * sqrt(D)。
- 离散签名信号：取 +1（与取 -1）后各视界的平均收益、|收益| 超过 10bp 的比例，
  与同品种全样本无条件基准比较，差异 t 统计量按交易日聚类。
- 终止条件视界：从触发分钟到下一次触发前一分钟（同段内，上限 60 分钟）。

产物：data/cn_futures/analysis/v3/defense/signal_grid.parquet
      data/cn_futures/analysis/v3/defense/signal_grid_meta.json
命令：.venv/bin/python scripts/v3_signal_grid.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"

PRODUCTS = ["SC", "AU", "AG", "CU", "M"]
HORIZONS = [1, 2, 3, 5, 10, 15]
FAMILIES = {"N": "数值信号", "C": "量价组合信号", "K": "离散事件统计指标",
            "X": "条件 / 共振信号"}
SIGNALS = [f"{f}{i}" for f in "NCKX" for i in range(1, 11)]

MOVE_BP = 10.0          # 千分之一 = 10bp 的价格变动门槛
MAX_TERM = 60           # 终止条件视界上限（分钟）
TRADING_DAYS_PER_MONTH = 21.0


def classify(s: pd.Series) -> str:
    """信号取值类型：离散签名 / 计数 / 连续。"""
    v = s.dropna()
    if v.empty:
        return "连续"
    u = np.unique(v.to_numpy())
    if len(u) <= 7:
        return "离散"
    if np.allclose(v.to_numpy(), np.round(v.to_numpy())) and len(u) <= 200:
        return "计数"
    return "连续"


def daily_ic(df: pd.DataFrame, sig: str, h: int,
             method: str = "spearman") -> tuple[float, float, int]:
    """逐日 IC 的均值 / 标准差 / 有效日数（日内取值恒定的日被剔除）。"""
    ics: list[float] = []
    y = f"fwd_{h}"
    for _, b in df.groupby("trade_date", sort=False):
        x = b[sig].to_numpy(dtype=float)
        r = b[y].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(r)
        if ok.sum() < 30:
            continue
        x, r = x[ok], r[ok]
        if np.std(x) == 0 or np.std(r) == 0:
            continue
        c = (stats.spearmanr(x, r).statistic if method == "spearman"
             else stats.pearsonr(x, r).statistic)
        if np.isfinite(c):
            ics.append(float(c))
    if len(ics) < 5:
        return float("nan"), float("nan"), len(ics)
    a = np.asarray(ics)
    return float(a.mean()), float(a.std(ddof=1)), len(a)


def pooled_ic(df: pd.DataFrame, sig: str, h: int, method: str) -> float:
    x = df[sig].to_numpy(dtype=float)
    r = df[f"fwd_{h}"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(r)
    if ok.sum() < 100 or np.std(x[ok]) == 0:
        return float("nan")
    c = (stats.spearmanr(x[ok], r[ok]).statistic if method == "spearman"
         else stats.pearsonr(x[ok], r[ok]).statistic)
    return float(c) if np.isfinite(c) else float("nan")


def cluster_t(vals: np.ndarray, days: np.ndarray, mu0: float) -> float:
    """H0: E[vals] = mu0，按交易日聚类的 t 统计量。"""
    ok = np.isfinite(vals)
    vals, days = vals[ok], days[ok]
    if len(vals) < 10:
        return float("nan")
    d = vals - mu0
    ser = pd.Series(d).groupby(pd.Series(days)).agg(["sum", "count"])
    n = float(ser["count"].sum())
    if n == 0:
        return float("nan")
    mean = float(ser["sum"].sum() / n)
    # 聚类稳健方差：sum_g (sum_g(d) - n_g*mean)^2 / n^2
    resid = ser["sum"].to_numpy() - ser["count"].to_numpy() * mean
    var = float((resid ** 2).sum()) / (n ** 2)
    if var <= 0:
        return float("nan")
    return mean / np.sqrt(var)


def term_return(df: pd.DataFrame, idx: np.ndarray,
                trig: np.ndarray) -> tuple[float, float]:
    """终止条件视界：触发点到下一触发前一分钟的收益均值与平均持有分钟数。"""
    logc = np.log(df["close"].to_numpy(dtype=float))
    seg = df["seg"].to_numpy()
    trig_pos = np.flatnonzero(trig)
    if len(trig_pos) == 0:
        return float("nan"), float("nan")
    nxt = np.searchsorted(trig_pos, idx, side="right")
    rets, holds = [], []
    for i, p in enumerate(idx):
        j = nxt[i]
        end = trig_pos[j] - 1 if j < len(trig_pos) else p + MAX_TERM
        end = min(end, p + MAX_TERM, len(df) - 1)
        if end <= p or seg[end] != seg[p]:
            # 跨段则截到本段末尾
            same = np.flatnonzero(seg[p:end + 1] != seg[p])
            if len(same) == 0:
                continue
            end = p + same[0] - 1
            if end <= p:
                continue
        rets.append(logc[end] - logc[p])
        holds.append(end - p)
    if not rets:
        return float("nan"), float("nan")
    return float(np.mean(rets) * 1e4), float(np.mean(holds))


def build(product: str) -> list[dict]:
    df = pd.read_parquet(DEF / f"panel_{product}.parquet")
    return build_from_frame(df, product)


def build_from_frame(df: pd.DataFrame, product: str,
                     signals: list[str] | None = None) -> list[dict]:
    """在给定面板上计算全信号网格（v4 复用入口：标签列可为 clean 口径）。

    ``signals`` 为 None 时用默认 40 信号；否则评价给定列（如 v4 的
    分散度矩族）。标签一律读 ``fwd_{h}`` 列，调用方可在传入前把 clean
    口径换入同名列。
    """
    days = df["trade_date"].nunique()
    months = days / TRADING_DAYS_PER_MONTH
    dayarr = df["trade_date"].to_numpy()

    # 品种无条件基准（全样本，同一有效样本上计算）
    base: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        r = df[f"fwd_{h}"].to_numpy(dtype=float)
        ok = np.isfinite(r)
        base[h] = {
            "mean_bp": float(np.mean(r[ok]) * 1e4),
            "absmean_bp": float(np.mean(np.abs(r[ok])) * 1e4),
            "p_move": float(np.mean(np.abs(r[ok]) * 1e4 > MOVE_BP)),
            "n": int(ok.sum()),
        }

    rows: list[dict] = []
    for sig in (SIGNALS if signals is None else signals):
        if sig not in df.columns:
            continue
        s = df[sig].astype(float)
        kind = classify(s)
        rec: dict[str, object] = {
            "product": product, "signal": sig, "family": sig[0],
            "kind": kind, "days": days,
            "n_rows": int(len(df)),
            "n_valid": int(s.notna().sum()),
        }
        for h in HORIZONS:
            rec[f"base_mean_bp_{h}"] = base[h]["mean_bp"]
            rec[f"base_p_move_{h}"] = base[h]["p_move"]

        # ---- 取值统计量 ----
        v = s.dropna().to_numpy()
        nz = v[v != 0]
        rec["nonzero_share"] = float(len(nz) / max(len(v), 1))
        if kind == "离散":
            vc = pd.Series(v).value_counts().sort_index()
            rec["valuecount"] = json.dumps(
                {str(int(k)) if float(k).is_integer() else f"{k:.4g}": int(n)
                 for k, n in vc.items()})
            rec["mean"] = rec["median"] = rec["std"] = None
            rec["skew"] = rec["kurt"] = None
        else:
            rec["valuecount"] = None
            rec["mean"] = float(np.mean(v))
            rec["median"] = float(np.median(v))
            rec["std"] = float(np.std(v, ddof=1))
            src = nz if len(nz) > 30 else v
            rec["skew"] = float(stats.skew(src))
            rec["kurt"] = float(stats.kurtosis(src, fisher=True))
            rec["q05"] = float(np.quantile(v, 0.05))
            rec["q95"] = float(np.quantile(v, 0.95))

        # ---- 频率 ----
        if kind == "离散":
            n_trig = int((v != 0).sum())
            rec["n_trigger"] = n_trig
            rec["per_month"] = float(n_trig / months)
            rec["freq_desc"] = "事件型触发"
        else:
            n_eff = int((v != 0).sum())
            rec["n_trigger"] = n_eff
            rec["per_month"] = float(n_eff / months)
            rec["freq_desc"] = "逐分钟"

        # ---- IC / ICIR（全部信号类型都算，离散信号的 IC 即符号 IC）----
        for h in HORIZONS:
            rec[f"ic_s_{h}"] = pooled_ic(df, sig, h, "spearman")
            rec[f"ic_p_{h}"] = pooled_ic(df, sig, h, "pearson")
            m, sd, nd = daily_ic(df, sig, h, "spearman")
            rec[f"icd_mean_{h}"] = m
            rec[f"icd_std_{h}"] = sd
            rec[f"icir_{h}"] = (m / sd) if (sd and np.isfinite(sd)
                                            and sd > 0) else float("nan")
            rec[f"ic_t_{h}"] = ((m / sd) * np.sqrt(nd)
                                if (sd and np.isfinite(sd) and sd > 0)
                                else float("nan"))
            rec[f"ic_days_{h}"] = nd

        # ---- 事件研究（离散信号）----
        if kind == "离散":
            arr = s.to_numpy(dtype=float)
            for tag, mask in (("up", arr > 0), ("dn", arr < 0)):
                idx = np.flatnonzero(mask)
                rec[f"n_{tag}"] = int(len(idx))
                rec[f"{tag}_per_month"] = float(len(idx) / months)
                if len(idx) < 10:
                    continue
                for h in HORIZONS:
                    r = df[f"fwd_{h}"].to_numpy(dtype=float)[idx]
                    ok = np.isfinite(r)
                    if ok.sum() < 10:
                        continue
                    rec[f"{tag}_ret_{h}"] = float(np.mean(r[ok]) * 1e4)
                    rec[f"{tag}_pmove_{h}"] = float(
                        np.mean(np.abs(r[ok]) * 1e4 > MOVE_BP))
                    rec[f"{tag}_t_{h}"] = cluster_t(
                        r * 1e4, dayarr[idx], base[h]["mean_bp"])
                    rec[f"{tag}_excess_{h}"] = (rec[f"{tag}_ret_{h}"]
                                                - base[h]["mean_bp"])
                tr, hold = term_return(df, idx, mask)
                rec[f"{tag}_ret_term"] = tr
                rec[f"{tag}_hold_term"] = hold
        rows.append(rec)
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for p in PRODUCTS:
        r = build(p)
        all_rows.extend(r)
        print(f"{p}: {len(r)} signals")
    out = pd.DataFrame(all_rows)
    DEF.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DEF / "signal_grid.parquet", index=False)

    meta = {
        "horizons": HORIZONS,
        "move_threshold_bp": MOVE_BP,
        "term_cap_min": MAX_TERM,
        "trading_days_per_month": TRADING_DAYS_PER_MONTH,
        "products": PRODUCTS,
        "n_signals": len(SIGNALS),
        "families": FAMILIES,
        "panel_start": "2026-01-05",
        "panel_end": "2026-07-13",
        "note": ("频率按 125 交易日 / 21 折月；期货分钟数据仅覆盖 2026-01-05 "
                 "至 2026-07-13，无法给出 2022 年以来的联合面板频率，"
                 "Polymarket 侧的全历史到达率见 signal_arrival_history.parquet。"),
    }
    (DEF / "signal_grid_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"written {DEF}/signal_grid.parquet  rows={len(out)} "
          f"cols={len(out.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
