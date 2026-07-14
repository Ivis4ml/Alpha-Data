"""把 ``data/future_shares`` 聚宽风格主力连续分钟 CSV 转换为 ``data/cn_futures`` parquet 库。

源数据事实（已逐项核实，勿按惯例假设）：

- 文件按自然日存放（``2026/2026-MM/2026-MM-DD.csv``），GBK 编码，首列表头为中文
  "证券代码"。**夜盘 bar 存放在其开始时刻的次一自然日文件**：周二文件含周一晚
  21:00 起的夜盘，周六文件仅含周五晚的夜盘（归属下周一交易日），周一文件因此
  从不含夜盘。跨文件无重复行。
- ``code`` 形如 ``AU9999.XSGE``：``9999`` 为主力连续，后缀为交易所；``symbol`` 列给出
  实际主力合约。主力切换发生在夜盘开盘（21:01 bar），即交易日边界，故同一交易日内
  合约唯一。
- ``volume`` / ``money`` 为**逐分钟增量**（跨夜盘 / 日盘边界亦然），``open_interest``
  为分钟末水平值；``date`` 为北京时间 bar 收盘戳（21:01 表示 21:00..21:01）。
- 56 个品种有夜盘（收盘 23:00 / 01:00 / 02:30 三档），中金所、广期所等无夜盘。
  夜盘收盘时刻从数据实测（研究规范 §2.3 第 2 条），不硬编码。

交易日归属规则（从 ``ts`` 推导，与文件布局无关）：

- 09:00..15:15 的 bar：交易日 = 自然日；
- 21:00 之后的 bar：交易日 = 严格大于该自然日的下一交易日；
- 03:00 之前的 bar：交易日 = 不小于该自然日的下一交易日（周六凌晨 -> 下周一）。

交易日全集 = 出现日盘 bar 的自然日集合。

产出：

- ``minute/{PRODUCT}/{YEAR}.parquet``：ts、trade_date、session(night/day)、OHLC、
  volume、money、open_interest、contract。
- ``daily/{PRODUCT}.parquet``：分时段日线（night_* 与 day_* 两组 OHLCV）+ 收益列
  （``r_night`` 夜盘内、``r_gap_pm`` 前收盘至夜盘开、``r_gap_am`` 夜盘收至日盘开、
  ``r_gap_full`` 前收盘至日盘开、``r_day`` 日盘内、``r_cc`` 前日收至当日收）+
  ``roll`` 换月标记。跨合约的收益（``r_gap_pm`` / ``r_gap_full`` / ``r_cc``）在换月日
  置 NaN，原始值保留在 ``*_raw`` 列。
- ``dominant_table.parquet``：主力合约起止区间（rule = ``provider_9999_continuous``）。
- ``product_specs.parquet``：品种、交易所、实测夜盘收盘 ``night_end``、覆盖区间。
- ``qc_summary.parquet``：逐品种质量核查。

用法::

    .venv/bin/python scripts/build_cn_futures_db.py \
        [--src data/future_shares] [--out data/cn_futures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_data.cn_futures import store  # noqa: E402

CSV_COLUMNS = ["code", "date", "open", "high", "low", "close",
               "volume", "money", "open_interest", "symbol"]


def read_day_csv(path: Path) -> pd.DataFrame:
    """读取单个自然日 CSV（GBK），规范列名与类型。"""
    df = pd.read_csv(path, encoding="gbk")
    if len(df.columns) != len(CSV_COLUMNS):
        raise ValueError(f"{path}: unexpected column count {len(df.columns)}")
    df.columns = CSV_COLUMNS
    df["ts"] = pd.to_datetime(df["date"])
    parts = df["code"].str.extract(r"^([A-Z]+)9999\.([A-Z]+)$")
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "code"].unique()[:5]
        raise ValueError(f"{path}: unrecognized codes {bad}")
    df["product"] = parts[0]
    df["exchange"] = parts[1].map(store.EXCHANGE_MAP)
    if df["exchange"].isna().any():
        bad = parts[1][df["exchange"].isna()].unique()
        raise ValueError(f"{path}: unknown exchange suffix {bad}")
    return df[["product", "exchange", "ts", "open", "high", "low", "close",
               "volume", "money", "open_interest", "symbol"]]


def assign_trade_date(ts: pd.Series, trading_days: np.ndarray) -> pd.Series:
    """按交易日归属规则把自然时间戳映射为交易日（``YYYY-MM-DD``）。

    Args:
        ts: 北京时间 bar 收盘戳。
        trading_days: 升序交易日数组（``datetime64[D]``）。

    Returns:
        交易日字符串 Series；无法归属（超出交易日列表）的为 ``None``。
    """
    nat = ts.dt.normalize().to_numpy().astype("datetime64[D]")
    hour = ts.dt.hour.to_numpy()
    out = np.full(len(ts), None, dtype=object)

    day_mask = (hour >= 8) & (hour <= 16)
    # 日盘：自然日必须本身是交易日。
    idx = np.searchsorted(trading_days, nat[day_mask], side="left")
    safe = np.minimum(idx, len(trading_days) - 1)
    ok = (idx < len(trading_days)) & (trading_days[safe] == nat[day_mask])
    vals = np.full(day_mask.sum(), None, dtype=object)
    vals[ok] = np.datetime_as_string(nat[day_mask][ok], unit="D")
    out[day_mask] = vals

    evening_mask = hour >= 20
    idx = np.searchsorted(trading_days, nat[evening_mask], side="right")
    ok = idx < len(trading_days)
    vals = np.full(evening_mask.sum(), None, dtype=object)
    safe_idx = np.minimum(idx, len(trading_days) - 1)
    labels = np.datetime_as_string(trading_days[safe_idx], unit="D")
    vals[ok] = labels[ok]
    out[evening_mask] = vals

    early_mask = hour <= 3
    idx = np.searchsorted(trading_days, nat[early_mask], side="left")
    ok = idx < len(trading_days)
    vals = np.full(early_mask.sum(), None, dtype=object)
    safe_idx = np.minimum(idx, len(trading_days) - 1)
    labels = np.datetime_as_string(trading_days[safe_idx], unit="D")
    vals[ok] = labels[ok]
    out[early_mask] = vals

    unhandled = ~(day_mask | evening_mask | early_mask)
    if unhandled.any():
        raise ValueError(f"{unhandled.sum()} bars in unexpected hours: "
                         f"{sorted(set(hour[unhandled]))}")
    return pd.Series(out, index=ts.index, dtype=object)


def build_daily(minute: pd.DataFrame) -> pd.DataFrame:
    """由单品种分钟 bar（含 trade_date / session 列）聚合分时段日线并计算收益。"""
    def agg_session(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        g = df.groupby("trade_date")
        res = pd.DataFrame(
            {
                f"{prefix}_open": g["open"].first(),
                f"{prefix}_high": g["high"].max(),
                f"{prefix}_low": g["low"].min(),
                f"{prefix}_close": g["close"].last(),
                f"{prefix}_volume": g["volume"].sum(),
                f"{prefix}_money": g["money"].sum(),
                f"{prefix}_oi": g["open_interest"].last(),
                f"{prefix}_n_bars": g["open"].size(),
            }
        )
        return res

    day = agg_session(minute[minute["session"] == "day"], "day")
    night = agg_session(minute[minute["session"] == "night"], "night")
    contract = minute.groupby("trade_date").agg(
        contract=("contract", "first"),
        n_contracts=("contract", "nunique"),
    )
    daily = contract.join(day, how="left").join(night, how="left")
    daily.index.name = "trade_date"
    daily = daily.sort_index().reset_index()

    prev_day_close = daily["day_close"].shift(1)
    daily["roll"] = daily["contract"].ne(daily["contract"].shift(1))
    daily.loc[daily.index[0], "roll"] = False

    daily["r_night"] = np.log(daily["night_close"] / daily["night_open"])
    daily["r_gap_pm_raw"] = np.log(daily["night_open"] / prev_day_close)
    daily["r_gap_pm"] = daily["r_gap_pm_raw"].where(~daily["roll"])
    daily["r_gap_am"] = np.log(daily["day_open"] / daily["night_close"])
    daily["r_gap_full_raw"] = np.log(daily["day_open"] / prev_day_close)
    daily["r_gap_full"] = daily["r_gap_full_raw"].where(~daily["roll"])
    daily["r_day"] = np.log(daily["day_close"] / daily["day_open"])
    daily["r_cc_raw"] = np.log(daily["day_close"] / prev_day_close)
    daily["r_cc"] = daily["r_cc_raw"].where(~daily["roll"])
    return daily


def empirical_night_end(minute: pd.DataFrame) -> str | None:
    """实测夜盘收盘时刻：夜盘日内最晚 bar 时刻的众数（无夜盘返回 ``None``）。"""
    night = minute[minute["session"] == "night"]
    if night.empty:
        return None
    last_per_day = night.groupby("trade_date")["ts"].max()
    if len(last_per_day) < 3:
        return None
    return last_per_day.dt.strftime("%H:%M").mode().iloc[0]


def build_dominant_table(daily_by_product: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """由逐日主力合约推导起止区间表。"""
    rows: list[dict] = []
    for product, daily in daily_by_product.items():
        for _, block in daily.groupby(
            (daily["contract"] != daily["contract"].shift(1)).cumsum()
        ):
            rows.append(
                {
                    "product": product,
                    "contract": block["contract"].iloc[0],
                    "active_from": block["trade_date"].iloc[0],
                    "active_to": block["trade_date"].iloc[-1],
                    "rule": "provider_9999_continuous",
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["product", "active_from"])
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="构建中国期货主力连续分钟库")
    parser.add_argument("--src", default=str(ROOT / "data" / "future_shares"))
    parser.add_argument("--out", default=str(store.DB_DIR))
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    files = sorted(src.glob("*/*/*.csv"))
    if not files:
        print(f"未找到 CSV：{src}", file=sys.stderr)
        return 1
    print(f"共 {len(files)} 个自然日文件：{files[0].stem} .. {files[-1].stem}")

    frames: list[pd.DataFrame] = []
    for i, path in enumerate(files, 1):
        frames.append(read_day_csv(path))
        if i % 30 == 0 or i == len(files):
            print(f"  已读 {i}/{len(files)}")
    allbars = pd.concat(frames, ignore_index=True)
    del frames

    hour = allbars["ts"].dt.hour
    allbars["session"] = np.where((hour >= 8) & (hour <= 16), "day", "night")
    day_dates = np.sort(
        allbars.loc[allbars["session"] == "day", "ts"].dt.normalize().unique()
    ).astype("datetime64[D]")
    print(f"分钟 bar {len(allbars):,} 行（夜盘 {(allbars['session'] == 'night').sum():,}），"
          f"品种 {allbars['product'].nunique()}，交易日 {len(day_dates)} 个")

    allbars["trade_date"] = assign_trade_date(allbars["ts"], day_dates)
    dropped = allbars["trade_date"].isna().sum()
    if dropped:
        print(f"警告：{dropped} 行无法归属交易日（样本边界外），丢弃")
        allbars = allbars.dropna(subset=["trade_date"])

    (out / "minute").mkdir(parents=True, exist_ok=True)
    (out / "daily").mkdir(parents=True, exist_ok=True)

    daily_by_product: dict[str, pd.DataFrame] = {}
    qc_rows: list[dict] = []
    specs_rows: list[dict] = []

    for product, block in allbars.groupby("product", sort=True):
        block = block.sort_values("ts").reset_index(drop=True)
        exchange = block["exchange"].iloc[0]

        mdir = out / "minute" / product
        mdir.mkdir(parents=True, exist_ok=True)
        minute_cols = block[["ts", "trade_date", "session", "open", "high", "low",
                             "close", "volume", "money", "open_interest", "symbol"]].rename(
            columns={"symbol": "contract"}
        )
        for year, yblock in minute_cols.groupby(minute_cols["ts"].dt.year):
            yblock.reset_index(drop=True).to_parquet(
                mdir / f"{int(year)}.parquet", index=False, compression="zstd"
            )

        daily = build_daily(minute_cols)
        daily.to_parquet(out / "daily" / f"{product}.parquet", index=False, compression="zstd")
        daily_by_product[product] = daily

        dup_ts = int(block["ts"].duplicated().sum())
        ohlc_bad = int(
            (
                (block["high"] < block["low"])
                | (block["high"] < block[["open", "close"]].max(axis=1))
                | (block["low"] > block[["open", "close"]].min(axis=1))
            ).sum()
        )
        day_mode = int(daily["day_n_bars"].mode().iloc[0])
        night_days = int(daily["night_n_bars"].notna().sum())
        qc_rows.append(
            {
                "product": product,
                "n_minute": len(block),
                "n_days": len(daily),
                "missing_days": int(len(day_dates) - len(daily)),
                "day_bar_mode": day_mode,
                "days_day_bar_deviate": int((daily["day_n_bars"] != day_mode).sum()),
                "night_days": night_days,
                "dup_ts": dup_ts,
                "ohlc_violations": ohlc_bad,
                "zero_volume_bars": int((block["volume"] <= 0).sum()),
                "n_rolls": int(daily["roll"].sum()),
                "multi_contract_days": int(daily["n_contracts"].gt(1).sum()),
            }
        )

        ne = empirical_night_end(minute_cols)
        specs_rows.append(
            {
                "product": product,
                "exchange": exchange,
                "night_end": ne,
                "has_night": ne is not None,
                "night_day_ratio": round(night_days / max(len(daily), 1), 3),
                "n_days": len(daily),
                "first_date": daily["trade_date"].iloc[0],
                "last_date": daily["trade_date"].iloc[-1],
                "day_bar_mode": day_mode,
            }
        )

    dominant = build_dominant_table(daily_by_product)
    dominant.to_parquet(out / "dominant_table.parquet", index=False)
    specs = pd.DataFrame(specs_rows).sort_values("product").reset_index(drop=True)
    specs.to_parquet(out / "product_specs.parquet", index=False)
    qc = pd.DataFrame(qc_rows).sort_values("product").reset_index(drop=True)
    qc.to_parquet(out / "qc_summary.parquet", index=False)

    print(f"\n品种 {len(specs)} 个，主力区间 {len(dominant)} 段；"
          f"夜盘品种 {int(specs['has_night'].sum())} 个：")
    print(specs[specs["has_night"]].groupby("night_end")["product"].agg(list).to_string())
    bad = qc[(qc["ohlc_violations"] > 0) | (qc["dup_ts"] > 0) | (qc["multi_contract_days"] > 0)]
    if not bad.empty:
        print("\n质量异常品种：")
        cols = ["product", "dup_ts", "ohlc_violations", "multi_contract_days"]
        print(bad[cols].to_string(index=False))
    else:
        print("QC：无重复时间戳、无 OHLC 违例、交易日内合约唯一")
    print(f"库已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
