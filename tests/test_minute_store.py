"""``alpha_data.equity.minute_store`` 的构建 / 重建 / 续传语义测试（合成 CSV，无需下载）。

重点覆盖曾出错的重建路径：强制重建必须替换旧分区（而非静默保留过期数据）。
"""

from __future__ import annotations

import gzip
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from alpha_data.equity import minute_store

_HEADER = "ticker,volume,open,close,high,low,window_start,transactions\n"


def _ns(et: str) -> int:
    """美东本地时刻 -> 纳秒 epoch（UTC）。"""
    return int(pd.Timestamp(et, tz="America/New_York").value)


def _write_flat(path: Path, rows: list[tuple]) -> Path:
    """写一个最小 Flat Files 风格的分钟 CSV.gz。rows: (ticker, o, h, l, c, v, et_start)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER]
    for ticker, o, h, low, c, v, et_start in rows:
        lines.append(f"{ticker},{v},{o},{c},{h},{low},{_ns(et_start)},1\n")
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)
    return path


def _read_parquet(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}')").fetch_df()


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def test_build_minute_close_label_conversion(tmp_path: Path, con) -> None:
    csv1 = _write_flat(
        tmp_path / "2024-06-13.csv.gz",
        [("TEST", 10.0, 10.5, 9.5, 10.2, 100, "2024-06-13 09:30")],
    )
    db = tmp_path / "db"
    n = minute_store.build_minute(con, [csv1], db)
    assert n == 1
    out = _read_parquet(con, db / "minute" / "TEST" / "2024.parquet")
    assert len(out) == 1
    # window_start 为 bar 起始，+60s 收盘标签：09:30 起始 -> ts=09:31。
    assert str(out["ts"].iloc[0]) == "2024-06-13 09:31:00"
    assert out["symbol"].iloc[0] == "TEST"


def test_build_minute_skip_done_then_forced_rebuild_replaces(tmp_path: Path, con) -> None:
    csv1 = _write_flat(
        tmp_path / "2024-06-13.csv.gz",
        [("TEST", 10.0, 10.5, 9.5, 10.2, 100, "2024-06-13 09:30")],
    )
    db = tmp_path / "db"
    minute_store.build_minute(con, [csv1], db)
    # 同年补了新一天的数据。
    csv2 = _write_flat(
        tmp_path / "2024-06-14.csv.gz",
        [("TEST", 11.0, 11.5, 10.5, 11.2, 200, "2024-06-14 09:30")],
    )
    # skip_done=True：年份已标记完成，跳过（不含新数据）。
    n = minute_store.build_minute(con, [csv1, csv2], db, skip_done=True)
    assert n == 0
    assert len(_read_parquet(con, db / "minute" / "TEST" / "2024.parquet")) == 1
    # 强制重建：必须替换旧分区并包含两天（修复前旧分区被静默保留）。
    n = minute_store.build_minute(con, [csv1, csv2], db, skip_done=False)
    assert n == 1
    out = _read_parquet(con, db / "minute" / "TEST" / "2024.parquet")
    assert len(out) == 2


def test_build_minute_rebuild_removes_stale_symbol_partition(tmp_path: Path, con) -> None:
    csv_old = _write_flat(
        tmp_path / "old" / "2024-06-13.csv.gz",
        [("GONE", 1.0, 1.0, 1.0, 1.0, 10, "2024-06-13 09:30")],
    )
    db = tmp_path / "db"
    minute_store.build_minute(con, [csv_old], db)
    assert (db / "minute" / "GONE" / "2024.parquet").exists()
    # 重建时输入不再含 GONE：该年旧分区应被清除，不残留过期数据。
    csv_new = _write_flat(
        tmp_path / "new" / "2024-06-13.csv.gz",
        [("TEST", 10.0, 10.5, 9.5, 10.2, 100, "2024-06-13 09:30")],
    )
    minute_store.build_minute(con, [csv_new], db, skip_done=False)
    assert not (db / "minute" / "GONE" / "2024.parquet").exists()
    assert (db / "minute" / "TEST" / "2024.parquet").exists()


def test_build_minute_resume_skips_copy(tmp_path: Path, con) -> None:
    db = tmp_path / "db"
    minute_dir = db / "minute"
    minute_dir.mkdir(parents=True)
    # 手工构造"COPY 已完成、归并未开始"的状态：暂存分区 + copydone 标记。
    tmp_dir = db / ".tmp_minute_2024"
    part = tmp_dir / "_safe=TEST"
    part.mkdir(parents=True)
    con.execute(
        "COPY (SELECT 'TEST' AS symbol, TIMESTAMP '2024-06-13 09:31:00' AS ts, "
        "CAST(1 AS FLOAT) AS open, CAST(1 AS FLOAT) AS high, CAST(1 AS FLOAT) AS low, "
        "CAST(1 AS FLOAT) AS close, CAST(10 AS BIGINT) AS volume, "
        "CAST(1 AS BIGINT) AS trade_count, 'massive' AS source, '' AS ingested_utc, "
        f"'' AS raw_hash) TO '{(part / 'data.parquet').as_posix()}' (FORMAT parquet)"
    )
    # 续传要求 copydone 内的输入指纹与本次一致。
    ghost = tmp_path / "2024-01-02.csv.gz"
    (minute_dir / ".copydone_2024").write_text(minute_store._input_fingerprint([ghost]))
    # csv 路径指向不存在的文件：若续传误走 COPY 会立即报错。
    n = minute_store.build_minute(con, [ghost], db, skip_done=False)
    assert n == 1
    assert (minute_dir / "TEST" / "2024.parquet").exists()
    assert (minute_dir / ".done_2024").exists()
    assert not (minute_dir / ".copydone_2024").exists()


def test_build_minute_stale_copydone_with_new_inputs_rebuilds(tmp_path: Path, con) -> None:
    """COPY 后中断、随后补充新文件再重跑：指纹不符须走全新构建，不得丢新数据。"""
    csv1 = _write_flat(
        tmp_path / "2024-06-13.csv.gz",
        [("TEST", 10.0, 10.5, 9.5, 10.2, 100, "2024-06-13 09:30")],
    )
    db = tmp_path / "db"
    minute_store.build_minute(con, [csv1], db)
    # 模拟"上次仅含 csv1 的 COPY 完成后中断"残留：copydone（旧指纹）+ 暂存目录。
    (db / ".tmp_minute_2024" / "_safe=TEST").mkdir(parents=True)
    (db / "minute" / ".copydone_2024").write_text(minute_store._input_fingerprint([csv1]))
    csv2 = _write_flat(
        tmp_path / "2024-06-14.csv.gz",
        [("TEST", 11.0, 11.5, 10.5, 11.2, 200, "2024-06-14 09:30")],
    )
    minute_store.build_minute(con, [csv1, csv2], db, skip_done=False)
    out = _read_parquet(con, db / "minute" / "TEST" / "2024.parquet")
    assert len(out) == 2


def test_reorg_merges_multifile_partition_atomically(tmp_path: Path, con) -> None:
    """同一分区多个 parquet 走合并分支：经临时文件原子替换，残留 .merge 不影响重跑。"""
    tmp_dir = tmp_path / "tmp"
    part = tmp_dir / "_safe=TEST"
    part.mkdir(parents=True)
    for i, name in enumerate(["a.parquet", "b.parquet"]):
        con.execute(
            f"COPY (SELECT {i} AS v) TO '{(part / name).as_posix()}' (FORMAT parquet)"
        )
    dst_dir = tmp_path / "out"
    dst_dir.mkdir()
    dst = dst_dir / "TEST.parquet"
    # 模拟上次合并中断留下的截断 .merge 残留：应被覆盖，不影响结果。
    (dst_dir / "TEST.parquet.merge").write_bytes(b"garbage")
    n = minute_store._reorg_partitions(con, tmp_dir, lambda safe: dst_dir / f"{safe}.parquet")
    assert n == 1
    out = _read_parquet(con, dst)
    assert sorted(out["v"]) == [0, 1]
    assert not (dst_dir / "TEST.parquet.merge").exists()


def test_build_daily_rebuild_replaces(tmp_path: Path, con) -> None:
    csv1 = _write_flat(
        tmp_path / "2024-06-13.csv.gz",
        [("TEST", 10.0, 10.5, 9.5, 10.2, 100, "2024-06-13 09:30")],
    )
    db = tmp_path / "db"
    n = minute_store.build_daily(con, [csv1], db)
    assert n == 1
    assert len(_read_parquet(con, db / "daily" / "TEST.parquet")) == 1
    # 扩充区间后重跑：日线必须覆盖为两天（修复前永远停留在首次构建范围）。
    csv2 = _write_flat(
        tmp_path / "2024-06-14.csv.gz",
        [("TEST", 11.0, 11.5, 10.5, 11.2, 200, "2024-06-14 09:30")],
    )
    n = minute_store.build_daily(con, [csv1, csv2], db)
    assert n == 1
    out = _read_parquet(con, db / "daily" / "TEST.parquet")
    assert sorted(out["trade_date"]) == ["2024-06-13", "2024-06-14"]
