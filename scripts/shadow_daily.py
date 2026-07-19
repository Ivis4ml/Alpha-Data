"""影子记录框架：按冻结口径逐日产出信号档案（追加写入，不回改）。

§12.16 影子验证协议的可运行实现。每个交易日一条 JSON 记录，包含
开盘前已知的全部冻结信号状态与收盘后的当日结果，追加写入
``shadow/records.jsonl``（受 Git 跟踪）；已存在的日期直接跳过，
协议上禁止回改历史行。

流水线（数据可得性逐步检查，缺哪步提示哪步）：
  1. [可选 --crawl]      增量爬链 + tape 重建（调既有脚本，需 RPC 配置）
  2. [可选 --fetch-curve] 交易所逐合约日结算补抓（fetch_cn_curve.py）
  3. 信号计算（全部冻结口径，只用 <= 当日开盘的信息）：
     八主题 z_pre、活跃度与 R3 尾部旗标状态、E1 截面得分头尾品种、
     R2 SC 相对残差、R4 近远月价差变化、当日 SC 跳事件数
  4. 当日结果（收盘后可得）：SC 开盘前收益 / 收对收、当日截面吸收 IC
  5. 追加写入 shadow/records.jsonl

用法：
  .venv/bin/python scripts/shadow_daily.py --date 2026-07-13
  .venv/bin/python scripts/shadow_daily.py            # 最新可得交易日
冻结样本内的日期自动标注 sample="frozen_demo"（示范用），
2026-07-18 之后的日期标注 sample="untouched"（正式影子记录）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
XSEC = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "xsec"
DAILY = ROOT / "data" / "cn_futures" / "daily"
JD = ROOT / "data" / "cn_futures" / "analysis" / "v3" / "jump"
INTL = ROOT / "data" / "intl"
SHADOW = ROOT / "shadow" / "records.jsonl"
FREEZE_DATE = "2026-07-18"


def theme_block(date: str) -> tuple[dict, dict]:
    """八主题 z_pre 与 R3 旗标状态（展开分位阈值只用 < date 历史）。"""
    z = pd.read_parquet(XSEC / "theme_signals.parquet").set_index(
        "trade_date").sort_index()
    if date not in z.index:
        raise SystemExit(f"theme_signals 无 {date}（先重建信号层）")
    act = z.abs().sum(axis=1)
    thr = act.expanding(30).quantile(0.9).shift(1)
    row = z.loc[date]
    return (
        {k: round(float(v), 4) for k, v in row.items()},
        {"activity": round(float(act.loc[date]), 4),
         "threshold_q90": (round(float(thr.loc[date]), 4)
                           if pd.notna(thr.loc[date]) else None),
         "flag_on": bool(pd.notna(thr.loc[date])
                         and act.loc[date] >= thr.loc[date])},
    )


def xsec_block(date: str) -> dict:
    panel = pd.read_parquet(XSEC / "panel.parquet")
    g = panel[(panel["trade_date"] == date) & panel["liquid"]
              & ~panel["roll"] & (panel["score_e1"].abs() > 0)]
    out: dict = {"n_universe": int(len(g))}
    if len(g) >= 10:
        top = g.nlargest(5, "score_e1")["product"].tolist()
        bot = g.nsmallest(5, "score_e1")["product"].tolist()
        out["e1_top5"] = top
        out["e1_bottom5"] = bot
        gg = g.dropna(subset=["r_open"])
        if len(gg) >= 30:
            out["absorption_cs_ic_today"] = round(float(
                spearmanr(gg["score_e1"], gg["r_open"]).statistic), 4)
    return out


def r2_block(date: str) -> dict:
    sc = pd.read_parquet(DAILY / "SC.parquet",
                         columns=["trade_date", "r_cc", "roll"])
    sc.loc[sc["roll"].astype(bool), "r_cc"] = np.nan
    try:
        brent = pd.read_csv(INTL / "brent_daily.csv")
        cnh = pd.read_csv(INTL / "usdcnh_daily.csv")
    except FileNotFoundError:
        return {"note": "data/intl 缺 Brent/USDCNH，跳过（需先更新数据腿）"}
    brent = brent[pd.to_numeric(brent["value"], errors="coerce").notna()]
    brent["value"] = brent["value"].astype(float)
    brent = brent.sort_values("timestamp")
    brent["ret"] = np.log(brent["value"]).diff()
    cnh = cnh.sort_values("timestamp")
    cnh["ret"] = np.log(cnh["close"]).diff()
    df = sc.rename(columns={"trade_date": "d"})[["d", "r_cc"]]
    df = df.merge(brent[["timestamp", "ret"]].rename(
        columns={"timestamp": "d", "ret": "brent"}), on="d", how="left")
    df = df.merge(cnh[["timestamp", "ret"]].rename(
        columns={"timestamp": "d", "ret": "cnh"}), on="d", how="left")
    df["brent"] = df["brent"].ffill(limit=3)
    df["cnh"] = df["cnh"].ffill(limit=3)
    df["brent_l1"] = df["brent"].shift(1)
    df = df.dropna().reset_index(drop=True)
    if date not in set(df["d"]):
        return {"note": f"{date} 无对齐的国际数据"}
    i = df.index[df["d"] == date][0]
    if i < 40:
        return {"note": "展开窗热身期不足 40 日"}
    X = np.column_stack([np.ones(len(df)),
                         df[["brent", "brent_l1", "cnh"]].to_numpy()])
    y = df["r_cc"].to_numpy()
    beta, *_ = np.linalg.lstsq(X[:i], y[:i], rcond=None)
    return {"sc_resid_bp": round(float(y[i] - X[i] @ beta) * 1e4, 2)}


def r4_block(date: str) -> dict:
    p = INTL / "curve_daily.parquet"
    if not p.exists():
        return {"note": "curve_daily 缺失（fetch_cn_curve.py 补抓）"}
    curve = pd.read_parquet(p)
    out = {}
    for prod in ("SC", "AU", "FU"):
        sub = curve[(curve["product"] == prod)
                    & pd.to_numeric(curve["settle"], errors="coerce")
                    .notna() & (pd.to_numeric(curve["oi"],
                                              errors="coerce") > 0)]
        days = sorted(sub["trade_date"].unique())
        if date not in days or days.index(date) == 0:
            continue
        prev = days[days.index(date) - 1]
        def pair(d: str) -> tuple[str, float] | None:
            g = sub[sub["trade_date"] == d].nlargest(2, "oi") \
                .sort_values("delivery_month")
            if len(g) < 2:
                return None
            return ("-".join(g["delivery_month"]),
                    float(np.log(float(g.iloc[0]["settle"]))
                          - np.log(float(g.iloc[1]["settle"]))))
        a, b = pair(prev), pair(date)
        if a and b and a[0] == b[0]:
            out[f"{prod.lower()}_spread_chg_bp"] = round(
                (b[1] - a[1]) * 1e4, 2)
    return out or {"note": f"{date} 无同合约对价差"}


def jump_block(date: str) -> dict:
    p = JD / "jumps.parquet"
    if not p.exists():
        return {"note": "jumps.parquet 缺失"}
    j = pd.read_parquet(p)
    d = j[j["ts"].dt.strftime("%Y-%m-%d") == date]
    sc = d[d["theme"].isin(["mideast_conflict", "oil_price"])]
    m_iso = d[(d["theme"] == "us_china_trade") & (d["n_cojump"] == 0)]
    return {"sc_theme_jumps": int(sc.groupby(["theme", "ts"]).ngroups),
            "m_isolated_jumps": int(m_iso.groupby(["theme", "ts"]).ngroups)}


def outcome_block(date: str) -> dict:
    out = {}
    for prod in ("SC", "M"):
        d = pd.read_parquet(DAILY / f"{prod}.parquet",
                            columns=["trade_date", "r_cc", "roll",
                                     "day_open", "day_close"])
        row = d[d["trade_date"] == date]
        if len(row) and not bool(row.iloc[0]["roll"]):
            r = row.iloc[0]
            out[f"{prod.lower()}_r_cc_bp"] = (
                round(float(r["r_cc"]) * 1e4, 1)
                if pd.notna(r["r_cc"]) else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="影子记录：冻结口径逐日信号档案")
    ap.add_argument("--date", default=None, help="交易日 YYYY-MM-DD")
    ap.add_argument("--crawl", action="store_true",
                    help="先增量爬链 + 重建 tape（需 RPC 配置）")
    ap.add_argument("--fetch-curve", action="store_true",
                    help="先补抓交易所逐合约日结算")
    args = ap.parse_args()

    if args.crawl:
        print("[1] 增量爬链：请运行 crawl_polymarket_chain.py 与 "
              "v3_build_tape.py（需 RPC 配置，此处不自动执行）")
    if args.fetch_curve:
        import subprocess
        subprocess.run([str(ROOT / ".venv" / "bin" / "python"),
                        str(ROOT / "scripts" / "fetch_cn_curve.py")],
                       check=False)

    z = pd.read_parquet(XSEC / "theme_signals.parquet")
    date = args.date or str(z["trade_date"].max())
    sample = "untouched" if date >= FREEZE_DATE else "frozen_demo"

    SHADOW.parent.mkdir(parents=True, exist_ok=True)
    if SHADOW.exists():
        seen = {json.loads(line)["date"]
                for line in SHADOW.read_text().splitlines() if line.strip()}
        if date in seen:
            print(f"{date} 已在档案中，按协议不回改，跳过")
            return 0

    themes, r3 = theme_block(date)
    rec = {
        "date": date,
        "sample": sample,
        "caliber": "frozen per freeze/manifest.json",
        "theme_z_pre": themes,
        "r3_tail_flag": r3,
        "cross_section": xsec_block(date),
        "r2_sc_residual": r2_block(date),
        "r4_curve": r4_block(date),
        "jumps_today": jump_block(date),
        "outcomes": outcome_block(date),
    }
    with open(SHADOW, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"\nappended -> {SHADOW}（sample={sample}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
