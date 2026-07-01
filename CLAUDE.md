# CLAUDE.md

本文件指导 Claude Code（claude.ai/code）在 **Alpha-Data** 仓库内工作。

## 项目目标

Alpha-Data 为消费方项目 **Alpha-Forge**（本机路径 `/Users/xinyu/Code/ncvx-project/分钟线策略任务-B/AlphaForge`）准备真实数据，替换其当前的合成源。产出必须严格满足 Alpha-Forge 的消费契约（见第 3 节）。本仓库为独立 Git 仓库（remote: `Ivis4ml/Alpha-Data`），与父目录 `ncvx-project` 的工程相互独立。

暂时准备两个相互独立的数据层：

1. **美股分钟线**：来自 massive.com，2020-01-01 至 2026-06-30，全市场 1min OHLCV + 公司行动 + 标的池。
2. **Polymarket 预测市场**：来自 HuggingFace `TimeSeventeen/Polymarket-v1`，2022-11-21 至 2026-04-28，作为宏观 / 事件类替代数据。

## 1. 事实来源（优先依据，不要臆造系统逻辑）

- **massive.com = Polygon.io 更名**（2025-10-30，"Polygon.io is Now Massive"）。既有 Polygon API key、SDK、`api.polygon.io` 域名在迁移期内并行可用；REST 端点与 Flat Files 布局与 Polygon 一致。区别于 `joinmassive.com`（无关的代理公司）。接入细节以 massive.com 官方文档为准；官网部分页面由 JS 渲染，精确字符串（Flat Files 前缀、价格档位）须在 Dashboard 或以 `aws s3 ls` 核实，未核实者标注为待确认。
- **Polymarket** 以 HF 数据集卡片与 `arXiv:2606.04217` 为准。`block_timestamp` 为秒级 Unix 时间（UTC）；无 tx hash，事件键为 `chainId_blockNumber_logIndex`。
- **Alpha-Forge 契约**以其源码为准（已核对 `core/schema.py`、`providers/db/source.py`、`providers/synthetic.py`、`providers/ingest/provider.py`）。

## 2. 约定

- **文档与交付物使用中文撰写。**
- **日期口径以 Alpha-Forge 为准**：`trade_date` 为 `YYYY-MM-DD` 字符串，`minute` 为 `HH:MM` 字符串（bar 起始时刻，美东 / `America/New_York` 本地时间）。注意：这与父项目 A 股流水线的整数 `YYYYMMDD` 口径不同，本项目一律用 Alpha-Forge 的字符串口径。
- **价格存原始未复权**，复权因子在本地由拆股 / 分红计算（与 Alpha-Forge 同口径）。
- **绝不伪造数据**：缺口以状态标注（`empty` / `data_gap`），不得合成填充。
- **不提交大数据与密钥**：`data/`、`.env`、`*.parquet`、`*.csv.gz` 已在 `.gitignore`。
- **Git**：在特性分支工作并向 `main` 发起 PR，不直接提交 `main`。
- **交付物内的示例 / 工具代码**遵循 PEP 8、完整类型注解、异常处理、可运行（与 Alpha-Forge 同标准）；代码中不使用表情符号、不使用 em dash。

## 3. Alpha-Forge 消费契约（Alpha-Data 必须产出）

### 3.1 读取侧 `DataSource`（`build_panel` 依赖）

| 方法 | 返回 |
|---|---|
| `read_minute(dates, symbols)` | `DataFrame[trade_date, minute, symbol, open, high, low, close, volume, amount]` |
| `read_daily(symbols, start, end)` | `DataFrame[symbol, trade_date, open, high, low, close, volume, amount]` |
| `corp_actions(symbols)` | `DataFrame[symbol, ex_date, split_ratio, cash_div]` 或 `None` |
| `universe_candidates(dates)` | `list[str]` |
| `symbol_meta()` | `DataFrame[symbol, exchange, name]` |
| `trading_days(start, end)` | `list[str]`（`YYYY-MM-DD`） |

字段口径：

- `trade_date`：`YYYY-MM-DD` 字符串。
- `minute`：`HH:MM` 字符串，**bar 收盘时刻（区间结束）**，美东本地。RTH 为 `09:31`..`16:00`（390 个；`core/schema.py` 与 `market/us_equity/calendar.is_rth` 定义 RTH 为 `09:30 < 收盘戳 <= 16:00`）；含盘前盘后为 `04:01`..`20:00`。massive 的 `window_start` 是 bar 起始，需 +60s 得收盘戳。
- `symbol`：归一化大写 ticker（去 `.US` / `.O` / `:NYSE` 等交易所后缀，见 `core/schema.py` 的 `default_normalize_symbol`），结果为 1–5 字符大写。
- `open/high/low/close`：原始未复权（`float`）。
- `volume`：成交量；`amount`：成交额 = `vwap×volume`，缺 vwap 时退化为 `close×volume`。
- 日线由分钟聚合：`open`=首 bar，`high`=max，`low`=min，`close`=末 bar，`volume`/`amount`=求和。
- `corp_actions`：`split_ratio`（如 2:1 拆股记 `2.0`，无拆股记 `1.0`）；`cash_div`（每股现金分红，无则 `0.0`）；`ex_date` 为 `YYYY-MM-DD`。

### 3.2 写入侧 `MinuteProvider`（`Ingester` 依赖，用于增量 / 补缺）

```python
name: str            # 供应商标识，写入 ingest_log 与 minute 表的 source 列，本项目用 "massive"
available() -> bool  # 密钥 / 配置是否就绪
fetch(symbol, start, end, *, interval="1min", extended_hours=False) -> FetchResult
```

`FetchResult`：

- `bars: list[dict]`，每个 bar（**原始未复权**）：
  `{"ts": "YYYY-MM-DD HH:MM:SS", "open": float, "high": float, "low": float, "close": float, "volume": float, "vwap": float(可选), "trade_count": int(可选)}`。`ts` 为 **bar 收盘时刻**、美东本地、分钟对齐（本项目口径；虽 `providers/ingest/provider.py` 文档串写"起始"，但面板契约 `schema.py` 与 `is_rth` 以收盘戳为准，故一律存收盘时刻，与批量库口径一致）。
- `status ∈ {ok, empty, data_gap, error, rate_limited}`。
- `note: str`、`raw_hash: str`（内容寻址哈希，供口径变更检测）。

### 3.3 MinuteDB 库布局（直接产出历史时建议产物）

`minute/{symbol}/{year}.parquet`（float32 OHLCV/vwap、int64 volume/trade_count，zstd）、`daily/{symbol}.parquet`、`corp_actions.parquet`、`listing.parquet`、`ingest_log.parquet`。`DbMinuteSource` 包装该库为 `DataSource`。

## 4. massive.com 接入要点

- **批量历史走 Flat Files（S3）**：端点 `https://files.massive.com`，桶 `flatfiles`，签名 s3v4；分钟线前缀（待核实）`us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`，每文件含当日全部 ticker。客户端用 `aws s3` / `boto3`（自定义 `endpoint_url`）/ `rclone` 均可。并行 8–32。
- **原始 CSV 列**：`ticker, volume, open, close, high, low, window_start, transactions`，其中 `window_start` 为纳秒 epoch（UTC）。
- **口径转换**：`window_start` 纳秒 UTC 为 bar 起始，**+60s 得 bar 收盘时刻** → `America/New_York` → 拆出 `trade_date`（`YYYY-MM-DD`）与 `minute`（`HH:MM`，bar 收盘戳；对齐 AlphaForge `is_rth`）。夏令时由时区转换自动处理。已核对：flat file 09:30 起始 bar 转为 `minute='09:31'`，OHLCV 与 REST `adjusted=false` 逐字段一致。
- **参考数据走 REST**：标的全集 `/v3/reference/tickers`（`active=true` 与 `active=false` 各取一次以覆盖退市；`date=` 取历史时点）；拆股 `/v3/reference/splits`；分红 `/v3/reference/dividends`。Auth：`Authorization: Bearer <key>` 或 `?apiKey=<key>`。
- **无幸存者偏差标的池**：对全部每日 Flat 文件的 `ticker` 列取并集即得全集，无需额外 REST。
- **补缺走 REST 聚合**：`/v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}`，`adjusted=false` 取未复权，`limit` 最大 50000，按 `next_url` 翻页。

## 5. Polymarket 接入要点

- **下载**：`huggingface_hub.snapshot_download(repo_id="TimeSeventeen/Polymarket-v1", repo_type="dataset", allow_patterns="daily_aligned/*")`；按层下载避免一次拉满 49 GB。本地用 DuckDB / Polars / PyArrow 查询 Parquet。
- **层选择**：`daily_aligned`（已去中继腿、补元数据，推荐起点）+ `ctf`（生命周期 / 初级市场 / 洗量识别）；`orderfilled`（原始 tape，含中继腿，需自行过滤地址）仅在需要最原始数据时取。
- **关键字段**（`daily_aligned`）：`block_timestamp`（秒级 UTC）、`price`（[0,1] 概率）、`taker_direction`（BUY/SELL，真值方向）、`usdc_amount`、`condition_id`、`p_event`（归一事件概率）、`D`（归一方向 ±1）、`category` / `category_refined`、`opens_at` / `close_at` / `resolved_at`（UTC 时间戳）、`winning_outcome_label`、`market_slug`。
- **时间归并**：所有时间为 UTC 且 7×24 连续；`block_timestamp` 秒 → 美东，按分钟分桶并归并到美股分钟网格；美股闭市时段的活动向下一开盘聚合。注意夏令时使 UTC 到美东偏移在 2022–2026 间变化（4h / 5h），链上区块时间非均匀，须重采样而非假设每分钟一行。
- **防前视**：构造特征时只使用严格早于目标 bar 的 `p_event`；剔除中继腿与铸造 / 销毁洗量后再视成交流为有效信号。
- **作为替代数据**：Polymarket 是宏观 / 事件概率，非逐 symbol 行情。默认作为市场级特征广播（事件概率水平、滚动窗口的概率变化、带方向成交流、解析事件标记），逐 symbol / 板块映射为后续扩展。Alpha-Forge 的情绪模块期望 `{symbol, date, source, headline, url, sentiment_score}`，Polymarket 特征需经适配器接入 `build_panel` 的情绪列。

## 6. 凭证

环境变量（实际 `.env` 不入库，模板见 `.env.example`）：`MASSIVE_API_KEY`（兼容 `POLYGON_API_KEY`）、`MASSIVE_S3_ACCESS_KEY_ID`、`MASSIVE_S3_SECRET_ACCESS_KEY`、`HF_TOKEN`（只读）。

## 7. 不要做的事

- 不在公开仓库内再分发 massive.com 原始数据（受订阅条款约束）；Polymarket 为 CC-BY-4.0，再分发须署名。
- 不提交 `data/`、`.env`、原始下载文件。
- 不在缺数据时伪造或外推填充。
- 不直接提交 `main`。
