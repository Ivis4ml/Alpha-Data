"""下载交易所逐合约日行情（免费公开端点），构建期限结构数据。

来源（已核实可用）：
  INE：https://www.ine.cn/data/tradedata/future/dailydata/kxYYYYMMDD.dat
  SHFE：https://www.shfe.com.cn/data/tradedata/future/dailydata/kxYYYYMMDD.dat
字段：DELIVERYMONTH、SETTLEMENTPRICE、CLOSEPRICE、VOLUME、OPENINTEREST。

只取研究品种：SC（INE）、AU / FU / CU（SHFE）。交易日取自本地期货库。
产物：data/intl/curve_daily.parquet
（product, trade_date, delivery_month, settle, close, volume, oi）
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "intl" / "curve_daily.parquet"
PRODUCTS = {"ine": ["sc"], "shfe": ["au", "fu", "cu"]}
URL = {"ine": "https://www.ine.cn/data/tradedata/future/dailydata/kx{d}.dat",
       "shfe": ("https://www.shfe.com.cn/data/tradedata/future/dailydata/"
                "kx{d}.dat")}
HDR = {"User-Agent": "Mozilla/5.0"}


def fetch_day(exch: str, ymd: str) -> list[dict]:
    req = urllib.request.Request(URL[exch].format(d=ymd), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - 网络容错，缺日记录后跳过
        print(f"  {exch} {ymd}: {e}")
        return []
    rows = []
    for r in data.get("o_curinstrument", []):
        pid = str(r.get("PRODUCTID", "")).strip().lower()
        grp = str(r.get("PRODUCTGROUPID", "")).strip().lower()
        prod = next((p for p in PRODUCTS[exch]
                     if pid.startswith(p) or grp == p), None)
        dm = str(r.get("DELIVERYMONTH", "")).strip()
        if prod is None or not dm.isdigit():
            continue
        rows.append({
            "product": prod.upper(), "trade_date":
            f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", "delivery_month": dm,
            "settle": r.get("SETTLEMENTPRICE"),
            "close": r.get("CLOSEPRICE"),
            "volume": r.get("VOLUME"), "oi": r.get("OPENINTEREST")})
    return rows


def main() -> int:
    days = pd.read_parquet(ROOT / "data" / "cn_futures" / "daily" /
                           "SC.parquet", columns=["trade_date"])
    dates = [d.replace("-", "") for d in days["trade_date"]]
    all_rows: list[dict] = []
    for i, ymd in enumerate(dates, 1):
        for exch in ("ine", "shfe"):
            all_rows.extend(fetch_day(exch, ymd))
        if i % 20 == 0:
            print(f"{i}/{len(dates)} 天，累计 {len(all_rows)} 行")
        time.sleep(0.4)
    df = pd.DataFrame(all_rows)
    for c in ("settle", "close", "volume", "oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(OUT, index=False)
    print(f"完成：{len(df)} 行 -> {OUT}")
    if len(df):
        print(df.groupby("product")["trade_date"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
