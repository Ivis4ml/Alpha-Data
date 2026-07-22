"""P4 主题内分散度矩族：把 N1 合并丢掉的横截面结构还原为固定 6 列。

回应评审意见 1 / 2（按主题合并丢信息、N1 定义丢信息）。每 (主题, 品种)
每分钟输出固定 6 个标量（维数不随市场数增长，不放大检验）：

  D_disp        成交额权重下定向创新的加权标准差（主题内分歧度）
  D_cancel      |Σ w·Δℓ| / Σ|w·Δℓ|（合并的抵消存活率，1 = 全同向）
  D_hhi         当分钟市场 usdc 份额的 Herfindahl 集中度
  D_sign_ratio  定向创新多数符号占比
  D_head_share  最大单市场 |w·Δℓ| 贡献占比
  D_skew        定向创新的加权偏度

口径与冻结面板逐项一致（市场级 SQL、时点化准入、结算截断、logit、
时点化累计 usdc 权重均复刻 ``v3_defense_build.pm_minute_theme``，该冻结
脚本只读 import 常量、不改动）；当分钟活跃市场数 K < 2 时全部置缺失
（诚实缩减检验，不虚计）。方向先验已预注册为 0（freeze/prereg_signs.json，
双侧检验）。

产物：data/cn_futures/analysis/v4/dispersion_{PRODUCT}.parquet
（列：ts + 6 列 x 主题；ts 与冻结面板同为北京时 naive）

用法::

    .venv/bin/python scripts/v4_market_dispersion.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_defense_build import (  # noqa: E402  只读 import，不改冻结脚本
    CLIP,
    PRODUCT_THEMES,
    REGISTRY,
    logit,
)

from alpha_data.polymarket import store  # noqa: E402
from alpha_data.polymarket.v3 import tape  # noqa: E402

V4 = ROOT / "data" / "cn_futures" / "analysis" / "v4"
START_UTC = "2026-01-02 00:00:00"
END_UTC = "2026-07-13 16:00:00"
#: 扩展品种的主题映射（与 cn_registry_v3 一致；v4_panel_build 同源使用）。
PRODUCT_THEMES_V4: dict[str, list[str]] = {
    **PRODUCT_THEMES,
    "CF": ["us_china_trade"],
    "I": ["us_china_trade"],
    "IF": ["taiwan_risk"],
}
DCOLS = ("disp", "cancel", "hhi", "sign_ratio", "head_share", "skew")


def market_minutes(con, registry: pd.DataFrame, theme: str,
                   product: str) -> pd.DataFrame:
    """市场分钟中间量（复刻冻结实现的市场级层，聚合前）。"""
    reg = registry[(registry["theme"] == theme)
                   & (registry["product"] == product)]
    if reg.empty:
        return pd.DataFrame()
    ids = ",".join("'" + c + "'" for c in reg["condition_id"])
    cols = "condition_id, block_timestamp, p_event, D, usdc_amount"
    sql = f"""
        SELECT condition_id,
               (block_timestamp // 60) * 60 + 60          AS m_end,
               sum(p_event * usdc_amount) / sum(usdc_amount) AS vwap,
               sum(usdc_amount)                            AS usdc
        FROM {tape.union_sql(cols)}
        WHERE condition_id IN ({ids}) AND p_event IS NOT NULL
          AND block_timestamp >= epoch(TIMESTAMP '{START_UTC}')
          AND block_timestamp <  epoch(TIMESTAMP '{END_UTC}')
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    df = con.execute(sql).fetch_df()
    if df.empty:
        return df
    meta = reg.set_index("condition_id")
    df["admit"] = df["condition_id"].map(meta["admit_ts"])
    res = pd.to_datetime(meta["resolved_at"], utc=True)
    res_ep = (res - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(
        seconds=1)
    df["res_ep"] = df["condition_id"].map(res_ep)
    df = df[(df["m_end"] > df["admit"])
            & (df["res_ep"].isna() | (df["m_end"] <= df["res_ep"]))]
    df = df.sort_values(["condition_id", "m_end"])
    df["dl_m"] = logit(df["vwap"].to_numpy()) - logit(
        df.groupby("condition_id")["vwap"].shift(1).to_numpy())
    cum = df.groupby("condition_id")["usdc"].cumsum()
    df["w"] = (df["condition_id"].map(meta["orientation"]).astype(float)
               * np.sqrt(cum.clip(lower=1.0)))
    # 定向创新（主题轴上）与其权重贡献
    df["x"] = np.sign(df["w"]) * df["dl_m"]        # orientation 定向
    df["aw"] = np.abs(df["w"])
    df["contrib"] = df["w"] * df["dl_m"]
    return df.dropna(subset=["dl_m"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """按分钟聚合为 6 个分散度标量（K < 2 的分钟置缺失）。"""
    rows = []
    for m_end, b in df.groupby("m_end", sort=True):
        k = len(b)
        if k < 2:
            continue
        x = b["x"].to_numpy(dtype=float)
        aw = b["aw"].to_numpy(dtype=float)
        usdc = b["usdc"].to_numpy(dtype=float)
        contrib = b["contrib"].to_numpy(dtype=float)
        wsum = aw.sum()
        mu = float((aw * x).sum() / wsum)
        var = float((aw * (x - mu) ** 2).sum() / wsum)
        disp = float(np.sqrt(max(var, 0.0)))
        denom = float(np.abs(contrib).sum())
        cancel = float(abs(contrib.sum()) / denom) if denom > 0 else np.nan
        share = usdc / usdc.sum()
        hhi = float((share ** 2).sum())
        pos = int((x > 0).sum())
        neg = int((x < 0).sum())
        nz = pos + neg
        sign_ratio = float(max(pos, neg) / nz) if nz else np.nan
        head_share = float(np.abs(contrib).max() / denom) if denom > 0 \
            else np.nan
        sd = float(np.sqrt(var))
        skew = (float((aw * ((x - mu) / sd) ** 3).sum() / wsum)
                if sd > 0 else 0.0)
        rows.append({"m_end": m_end, "disp": disp, "cancel": cancel,
                     "hhi": hhi, "sign_ratio": sign_ratio,
                     "head_share": head_share, "skew": skew, "k": k})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ts"] = (pd.to_datetime(out["m_end"], unit="s", utc=True)
                 .dt.tz_convert("Asia/Shanghai").dt.tz_localize(None))
    return out


def main() -> int:
    t0 = time.time()
    V4.mkdir(parents=True, exist_ok=True)
    con = store.connect()
    registry = pd.read_parquet(REGISTRY)
    clip_note = f"logit 截断 {CLIP}"
    for product, themes in PRODUCT_THEMES_V4.items():
        frames = []
        for th in themes:
            mm = market_minutes(con, registry, th, product)
            if mm.empty:
                continue
            agg = aggregate(mm)
            if agg.empty:
                continue
            agg = agg.rename(columns={c: f"D_{c}_{th}" for c in
                                      (*DCOLS, "k")})
            frames.append(agg.drop(columns=["m_end"]))
        if not frames:
            print(f"[{product}] 无分散度数据")
            continue
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on="ts", how="outer")
        out = out.sort_values("ts").reset_index(drop=True)
        out.to_parquet(V4 / f"dispersion_{product}.parquet", index=False)
        n_active = {th: int(out[f"D_disp_{th}"].notna().sum())
                    for th in themes if f"D_disp_{th}" in out.columns}
        print(f"[{product}] {len(out)} 分钟；活跃分钟/主题 {n_active}")
    print(f"完成（{clip_note}），耗时 {time.time() - t0:.0f}s -> {V4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
