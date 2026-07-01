"""massive.com Flat Files（S3）批量下载：全市场分钟聚合 CSV.gz。

端点 ``files.massive.com``，桶 ``flatfiles``，分钟前缀
``us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz``（每交易日一个 gzip 文件，含当日全部
ticker，含退市 / OTC，天然无幸存者偏差）。凭证取自环境变量
``MASSIVE_S3_ACCESS_KEY_ID`` / ``MASSIVE_S3_SECRET_ACCESS_KEY``（与 REST key 不同）。

原始 CSV 列：``ticker, volume, open, close, high, low, window_start, transactions``，其中
``window_start`` 为纳秒 epoch（UTC）。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
DAY_PREFIX = "us_stocks_sip/day_aggs_v1"


def s3_client(max_pool: int = 32):
    """按环境变量构造 S3 客户端（s3v4 签名，自定义 endpoint）。"""
    endpoint = os.environ.get("MASSIVE_S3_ENDPOINT", "https://files.massive.com")
    access = os.environ.get("MASSIVE_S3_ACCESS_KEY_ID")
    secret = os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise RuntimeError(
            "缺少 S3 凭证：设置 MASSIVE_S3_ACCESS_KEY_ID / MASSIVE_S3_SECRET_ACCESS_KEY"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(
            signature_version="s3v4",
            max_pool_connections=max_pool,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def minute_key(date_str: str) -> str:
    """某交易日的分钟聚合对象键（``date_str`` 为 ``'YYYY-MM-DD'``）。"""
    return f"{MINUTE_PREFIX}/{date_str[:4]}/{date_str[5:7]}/{date_str}.csv.gz"


def local_path(root: Path, date_str: str) -> Path:
    """本地镜像路径（与桶内前缀结构一致）。"""
    return root / MINUTE_PREFIX / date_str[:4] / date_str[5:7] / f"{date_str}.csv.gz"


def download_one(
    date_str: str,
    root: Path,
    *,
    bucket: str = "flatfiles",
    client=None,
    overwrite: bool = False,
) -> tuple[str, str, int]:
    """下载单个交易日文件。返回 ``(date, status, size)``；status ∈ {ok, skip, missing, error}。"""
    client = client or s3_client()
    dst = local_path(root, date_str)
    if dst.exists() and not overwrite and dst.stat().st_size > 0:
        return (date_str, "skip", dst.stat().st_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    try:
        client.download_file(bucket, minute_key(date_str), str(tmp))
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        tmp.unlink(missing_ok=True)
        if code in ("404", "NoSuchKey"):
            return (date_str, "missing", 0)
        return (date_str, "error", 0)
    tmp.rename(dst)
    return (date_str, "ok", dst.stat().st_size)


def download_range(
    dates: Iterable[str],
    root: Path,
    *,
    bucket: str = "flatfiles",
    workers: int = 16,
    overwrite: bool = False,
    progress: callable | None = None,
) -> list[tuple[str, str, int]]:
    """并行下载多个交易日文件。返回每个日期的 ``(date, status, size)``。"""
    date_list = list(dates)
    client = s3_client(max_pool=max(8, workers * 2))
    results: list[tuple[str, str, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, d, root, bucket=bucket, client=client, overwrite=overwrite): d
            for d in date_list
        }
        for done in as_completed(futures):
            res = done.result()
            results.append(res)
            if progress:
                progress(res, len(results), len(date_list))
    return sorted(results)
