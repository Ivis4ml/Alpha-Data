"""M1 机制因子样板：分歧度预测波动（按冻结规格执行，一次计算一次报告）。

规格：freeze/prereg_endpoints_m1.json（先于本脚本任何运行提交冻结）。
- 主终点：逐日日内 partial Spearman（D_disp 对 [1, rv15_z] 残差化后
  与 fwd_rv15 的秩相关），日序列 t；仅 15 分钟一个视界。
- fwd_rv15 = sqrt(sum_{u=t+1..t+15} r1_u^2)，段内，窗口含真封板置缺失。
- M 族 13 个 (品种, 主题) 对，Bonferroni(13) 现算；退化对不出分母。
- 条件证伪：事件近旁（过去 60 分钟有 |N1|>kappa 事件）对平静分钟的
  partial IC 差，跨对中位数应 > 0（机理判别，不进族计数）。

产物：data/cn_futures/analysis/v4/m1_disp_vol.parquet 与打印报告。

用法::

    .venv/bin/python scripts/m1_disp_vol.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v4_market_dispersion import PRODUCT_THEMES_V4  # noqa: E402
from v4_signal_grid import classify_locked, find_panel  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
N_PAIRS_REGISTERED = 13          # 冻结规格：分母固定，不随结果变


def fwd_rv15(df: pd.DataFrame) -> np.ndarray:
    """前向 15 分钟已实现波动（段内；窗口含真封板置缺失）。"""
    r1 = df["r1"].to_numpy(dtype=float)
    seg = df["seg"].to_numpy()
    tl = df["true_lock"].to_numpy(dtype=float)
    n = len(df)
    r2 = np.where(np.isfinite(r1), r1 ** 2, np.nan)
    csum = np.concatenate([[0.0], np.nancumsum(r2)])
    cnt = np.concatenate([[0], np.cumsum(np.isfinite(r2).astype(int))])
    ctl = np.concatenate([[0.0], np.cumsum(tl)])
    out = np.full(n, np.nan)
    idx = np.arange(n)
    end = idx + 15
    ok = end < n
    ok &= np.where(ok, seg[np.minimum(end, n - 1)] == seg, False)
    # 窗口 (t, t+15] 的观测数须满 15、无真封板
    n_obs = cnt[np.minimum(end, n - 1) + 1] - cnt[idx + 1]
    n_tl = ctl[np.minimum(end, n - 1) + 1] - ctl[idx + 1]
    ok &= (n_obs == 15) & (n_tl == 0)
    s = csum[np.minimum(end, n - 1) + 1] - csum[idx + 1]
    out[ok] = np.sqrt(s[ok])
    return out


def daily_partial_series(df: pd.DataFrame, xcol: str,
                         mask: np.ndarray | None = None) -> list[float]:
    """逐日日内 partial Spearman(resid(x | rv15_z), fwd_rv)。"""
    ics = []
    m_all = np.ones(len(df), dtype=bool) if mask is None else mask
    for _, b in df.groupby("trade_date", sort=False):
        sel = m_all[b.index.to_numpy()]
        x = b[xcol].to_numpy(dtype=float)[sel]
        y = b["_fwd_rv"].to_numpy(dtype=float)[sel]
        z = b["rv15_z"].to_numpy(dtype=float)[sel]
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if ok.sum() < 30:
            continue
        x, y, z = x[ok], y[ok], z[ok]
        zz = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(zz, x, rcond=None)
        r = x - zz @ beta
        if np.std(r) == 0 or np.std(y) == 0:
            continue
        c = stats.spearmanr(r, y).statistic
        if np.isfinite(c):
            ics.append(float(c))
    return ics


def t_of(ics: list[float]) -> tuple[float, float, int]:
    if len(ics) < 20:
        return float("nan"), float("nan"), len(ics)
    a = np.asarray(ics)
    return (float(a.mean()),
            float(a.mean() / a.std(ddof=1) * np.sqrt(len(a))), len(a))


def main() -> int:
    thr = float(stats.norm.ppf(1 - 0.05 / 2 / N_PAIRS_REGISTERED))
    rows = []
    for product, themes in PRODUCT_THEMES_V4.items():
        panel = pd.read_parquet(find_panel(product))
        disp = pd.read_parquet(V4 / f"dispersion_{product}.parquet")
        panel["ts"] = pd.to_datetime(panel["ts"])
        disp["ts"] = pd.to_datetime(disp["ts"])
        df = classify_locked(panel.merge(disp, on="ts", how="left"))
        df["_fwd_rv"] = fwd_rv15(df)
        df = df.reset_index(drop=True)
        ev_recent = (df["E_any"].rolling(60, min_periods=1).sum() > 0) \
            .to_numpy()
        for th in themes:
            col = f"D_disp_{th}"
            rec: dict[str, object] = {"product": product, "theme": th}
            if col not in df.columns:
                rec.update(degenerate=True)
                rows.append(rec)
                continue
            mu, t, nd = t_of(daily_partial_series(df, col))
            mu_raw = float("nan")
            # 次要描述：raw RankIC（pooled）
            x = df[col].to_numpy(dtype=float)
            y = df["_fwd_rv"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 500:
                mu_raw = float(stats.spearmanr(x[ok], y[ok]).statistic)
            # 条件证伪：事件近旁 vs 平静
            ic_e, t_e, nd_e = t_of(daily_partial_series(df, col, ev_recent))
            ic_q, t_q, nd_q = t_of(daily_partial_series(df, col, ~ev_recent))
            # 十分位单调（D_disp 档均值 fwd_rv）
            v = x.copy()
            okv = np.isfinite(v) & np.isfinite(y)
            mono_rho = float("nan")
            if okv.sum() > 2000:
                qq = pd.qcut(pd.Series(v[okv]), 10, labels=False,
                             duplicates="drop")
                dm = pd.Series(y[okv]).groupby(qq).mean()
                if len(dm) >= 5:
                    mono_rho = float(stats.spearmanr(
                        dm.index.to_numpy(), dm.to_numpy()).statistic)
            rec.update(degenerate=bool(not np.isfinite(t)),
                       partial_ic_mean=round(mu, 5) if np.isfinite(mu)
                       else None,
                       partial_t=round(t, 3) if np.isfinite(t) else None,
                       n_days=nd, raw_pooled_ic=round(mu_raw, 5),
                       ic_event=round(ic_e, 5) if np.isfinite(ic_e)
                       else None,
                       ic_quiet=round(ic_q, 5) if np.isfinite(ic_q)
                       else None,
                       cond_diff=(round(ic_e - ic_q, 5)
                                  if np.isfinite(ic_e) and np.isfinite(ic_q)
                                  else None),
                       mono_rho=round(mono_rho, 3) if np.isfinite(mono_rho)
                       else None)
            rows.append(rec)
    out = pd.DataFrame(rows)
    assert len(out) == N_PAIRS_REGISTERED, \
        f"对数 {len(out)} != 注册分母 {N_PAIRS_REGISTERED}"
    out.to_parquet(V4 / "m1_disp_vol.parquet", index=False)

    val = out[~out["degenerate"].astype(bool)]
    ts = val["partial_t"].astype(float)
    print(f"M 族：注册 {N_PAIRS_REGISTERED} 对，有效 {len(val)}，"
          f"门槛 |t| > {thr:.3f}")
    print(val[["product", "theme", "partial_ic_mean", "partial_t",
               "n_days", "cond_diff", "mono_rho"]].to_string(index=False))
    n_pass = int((ts.abs() > thr).sum())
    n_pos = int((ts > 0).sum())
    cd = val["cond_diff"].astype(float).dropna()
    print(f"\n通过门槛：{n_pass}/{N_PAIRS_REGISTERED}；符号为正："
          f"{n_pos}/{len(val)}（注册先验 +1）")
    print(f"条件证伪（事件近旁-平静，中位数应>0）："
          f"中位 {cd.median():+.5f}，为正的对 {int((cd > 0).sum())}/{len(cd)}")
    print(f"单调 rho 中位：{val['mono_rho'].astype(float).median():+.3f}")
    print(f"written {V4}/m1_disp_vol.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
