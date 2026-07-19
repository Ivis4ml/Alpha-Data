"""信号库统计量补全：峰度、离散 valuecount 与 J 族分布。

补齐附录三 §4.1-4.2 缺失的三块（按导师"每个信号定义/公式/取值统计量
都弄清楚"的要求）：
  1. 峰度：40 信号 x 5 品种 + J1-J7 x 5 品种，非零值口径（与既有偏度
     一致），写入 signal_stats_ext.parquet；
  2. 离散信号 valuecount：取值种数 <= 9 的信号（X 族签名型等）逐值
     计数与占比；
  3. J 族（滚动 z 后的 J1-J7）均值/中位数/标准差/偏度/峰度/非零频率，
     并出 SC 的 J 族直方图（f_sigdist_J.png）。

产物：analysis/v3/defense/signal_stats_ext.parquet、
      docs/figures/v3/f_sigdist_J.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "defense"
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
PRODUCTS = ["SC", "AU", "AG", "CU", "M"]
PANEL_SIGNALS = ([f"N{i}" for i in range(1, 11)]
                 + [f"C{i}" for i in range(1, 11)]
                 + [f"K{i}" for i in range(1, 11)]
                 + [f"X{i}" for i in range(1, 11)])
J_SIGNALS = [f"J{i}" for i in range(1, 8)]
DISCRETE_MAX_UNIQUE = 9


def stat_row(prod: str, sig: str, v: pd.Series) -> dict:
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    nz = v[v != 0]
    uniq = int(v.nunique())
    row: dict = {
        "product": prod, "signal": sig, "n": int(len(v)),
        "n_unique": uniq,
        "is_discrete": uniq <= DISCRETE_MAX_UNIQUE,
        "mean": float(v.mean()), "median": float(v.median()),
        "std": float(v.std()),
        "skew": float(nz.skew()) if len(nz) > 2 else np.nan,
        "kurt": float(nz.kurt()) if len(nz) > 3 else np.nan,
        "nonzero_rate": float((v != 0).mean()),
        "valuecount": "",
    }
    if row["is_discrete"]:
        vc = v.value_counts().sort_index()
        row["valuecount"] = "；".join(
            f"{k:+g}: {int(c):,}（{c / len(v):.1%}）" for k, c in vc.items())
    return row


def main() -> int:
    rows = []
    for prod in PRODUCTS:
        panel = pd.read_parquet(DEF / f"panel_{prod}.parquet",
                                columns=PANEL_SIGNALS)
        for sig in PANEL_SIGNALS:
            rows.append(stat_row(prod, sig, panel[sig]))
        jf = pd.read_parquet(JD / f"factors_{prod}.parquet",
                             columns=J_SIGNALS)
        for sig in J_SIGNALS:
            rows.append(stat_row(prod, sig, jf[sig]))
    out = pd.DataFrame(rows)
    out.to_parquet(DEF / "signal_stats_ext.parquet", index=False)
    n_disc = out[out["is_discrete"]]["signal"].nunique()
    print(f"signal_stats_ext: {len(out)} 行（47 信号 x 5 品种），"
          f"离散信号 {n_disc} 个")

    # J 族直方图（SC，滚动 z 后、非零值、截尾展示）
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import pub_style
    pub_style.setup(cn_font=True)
    jf = pd.read_parquet(JD / "factors_SC.parquet", columns=J_SIGNALS)
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.2))
    for ax, sig in zip(axes.ravel(), J_SIGNALS):
        v = jf[sig].replace([np.inf, -np.inf], np.nan).dropna()
        v = v[v != 0]
        if len(v) > 50:
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v.clip(lo, hi), bins=40, color="#3b6db3")
        ax.set_yscale("log")
        ax.set_title(sig, fontsize=8)
        ax.tick_params(labelsize=6)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fp = ROOT / "docs" / "figures" / "v3" / "f_sigdist_J.png"
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f"figure -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
