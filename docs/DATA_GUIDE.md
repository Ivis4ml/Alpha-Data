# Alpha-Data 数据指南

本文说明 Alpha-Data 已产出的数据、目录结构、口径，以及如何用 API 访问与使用。所有数据位于
`data/`（不入库，见 `.gitignore`）。代码示例默认在项目 venv 下运行：`.venv/bin/python`。

## 1. 概览与统计

| 数据层 | 内容 | 规模 | 时间范围 |
|---|---|---|---|
| 美股分钟线 | 全市场 1min OHLCV（原始未复权） | **25.9 亿** bar，21,855 标的，36 GB | 2020-01-02 .. 2026-06-29 |
| 美股日线 | 由 RTH 分钟聚合 | 1671 万行，21,854 标的，466 MB | 同上 |
| 公司行动 | 拆股 + 分红（同日分红求和、类别股符号已回映射） | 143.3 万行，69,422 标的 | 2019-01-02 起 |
| 上市表 | 标的元数据（含退市，无幸存者偏差；带点标的独立成行） | 35,895 标的（active 12,900） | — |
| Polymarket `daily_aligned` | 清洗后逐笔（防前视分析层） | 6.02 亿行，13.2 GB | 2022-11-21 .. 2026-04-28 |
| Polymarket `ctf` / `orderfilled` | 生命周期 / 原始 tape | 8.39 亿 / 12.0 亿行 | 同上 |
| Polymarket 市场目录 | 成交额≥$10万的市场 | 29,634 个 | — |
| Polymarket 特征面板 | 市场级、防前视、美东分钟网格（含隔夜列） | 284,741 行 × 47 列（示例，5 个宏观市场 × 9 特征） | — |
| 中证 A 股分钟线 | 全市场个股 + 指数 1min OHLCV（原始未复权，北京时间） | **17.9 亿** bar，5,710 标的，约 17 GB | 2020-01-02 .. 2025-12-31 |
| 中证 A 股日线 | 由分钟聚合（真实成交额） | 742.3 万行，5,710 标的 | 同上 |
| 中证上市表 | 个股 + 指数元数据（含退市，无幸存者偏差） | 5,710 标的（active 5,708；深 3,220 / 沪 2,488 / 北 2） | — |

美股分钟 bar 总数含盘前盘后（RTH 仅约六成）。中证 A 股无复权数据（`corp_actions` 为空，见 §6.4）。

## 2. 目录结构

```text
data/
  equity/
    flatfiles/us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz   # 原始下载（构建后可删）
    minute_db/                        # AlphaForge MinuteDB 布局（消费入口）
      minute/{SYMBOL}/{YEAR}.parquet  # 分钟 bar，按 symbol/year 分区
      daily/{SYMBOL}.parquet          # 日线（一 symbol 一文件，含全部交易日）
      corp_actions.parquet            # 公司行动
      listing.parquet                 # 上市表
    reference/                        # 参考数据独立副本（listing/splits/dividends/corp_actions）
  cn_equity/
    minute_db/                        # 中证 A 股 MinuteDB（与美股库同构，见 §6）
      minute/{SYMBOL}/{YEAR}.parquet  # 分钟 bar，SYMBOL 如 600000.SH / 000300.SH
      daily/{SYMBOL}.parquet          # 日线
      listing.parquet                 # 上市表
      corp_actions.parquet            # 空表（本源无拆股 / 分红）
  polymarket/
    daily_aligned/*.parquet           # 1248 个按日分区文件
    CTF/{preparations,splits,merges,resolutions,redemptions}.parquet
    OrderFilled/*.parquet
    features/market_catalog.parquet   # 市场目录
    features/market_features.parquet  # 特征宽表
```

`{SYMBOL}` 为文件系统安全化的大写 ticker（非法字符替 `_`，与 AlphaForge `MinuteDB._safe_symbol` 一致）。

## 3. 美股分钟线

### 3.1 口径（重要）

- **原始未复权**价。复权因子在本地由拆股 / 分红计算（见 §4）。
- `minute` / `ts` 为 **bar 收盘时刻**（区间结束），美东本地时间。RTH 为 `09:31`..`16:00`（390 bar/日），
  含盘前盘后为 `04:01`..`20:00`。这与 AlphaForge `core/schema.py` 及 `is_rth`（`09:30 < 收盘戳 <= 16:00`）一致。
- **符号为 Polygon/massive 原生大写 ticker，`.` 后缀保留**：类别股（`BRK.A`/`BRK.B`）、SPAC 单位
  （`AAC.U`）、权证（`ACHR.WS`）等约 1,000 个带点标的与其正股是不同证券，分钟库、`listing`、
  `corp_actions` 三处一致存原生符号。归一（`normalize_symbol`）只剥交易所后缀
  （`AAPL:NASDAQ`、`AAPL.US`/`.O`/`.N`/`.OQ`），不再剥类别股后缀（AlphaForge 侧
  `default_normalize_symbol` 已同步修正，否则 `BRK.A` 与 `BRK.B` 会被并成同一标的）。
- Flat Files 分钟聚合**无 vwap**；`trade_count` 取原始 `transactions`。下游 `amount` = `close×volume`。
- 半日（提前 13:00 收盘）自动处理。缺数据不填充。

### 3.2 访问方式 A —— AlphaForge `MinuteDB`（推荐，契约口径）

需要装 AlphaForge（`pip install -e <AlphaForge 路径>`）。`read_minute` 返回 AlphaForge 契约列。

```python
from alphaforge.providers.db.minute_db import MinuteDB

db = MinuteDB("data/equity/minute_db")

# 分钟（RTH），返回 [trade_date, minute, symbol, open, high, low, close, volume, trade_count]
m = db.read_minute(["AAPL", "MSFT"], "2024-03-01", "2024-03-05", rth_only=True)
print(m.head())

# 日线，返回 [symbol, trade_date, open, high, low, close, volume, amount, source]
d = db.read_daily(["AAPL"], "2024-01-01", "2024-12-31")

# 公司行动 [symbol, ex_date, split_ratio, cash_div]（无则 None）
ca = db.read_corp_actions(["AAPL"])

syms = db.symbols()          # 约 2.2 万标的，~0.4s
# 交易日：务必传基准标的（如 SPY）。不传 symbols 会扫描全部 21k+ 标的的分钟数据，极慢！
tdays = db.trading_days("2024-03-01", "2024-03-31", symbols=["SPY"])
```

> 大规模注意：本库约 2.2 万标的。凡是**读取全部标的分钟数据**的调用在本规模下很慢（分钟级）：
> `db.trading_days(start, end)`（不传 `symbols`）、`DbMinuteSource.trading_days()`、
> `DbMinuteSource.universe_candidates()`。`read_minute` / `read_daily` / `read_corp_actions`（传定标的）
> 与 `db.symbols()` / `symbol_meta()` 都很快。交易日与全市场选池请传基准标的或改用 DuckDB（见 §3.4）。

取**交易日历**最省事（NYSE 日历，不读数据、瞬时，含半日 / 节假日处理）：

```python
from alpha_data.common import calendar
tdays = calendar.trading_days("2024-03-01", "2024-03-31")   # ['2024-03-01', '2024-03-04', ...]
```

这与「库中实际有数据的交易日」一般一致；唯一差异是尚未下载的日子（如今天的文件未定版），
日历会含它、库不含。需要「库中实际存在」的口径时用 `db.trading_days(..., symbols=["SPY"])` 或 DuckDB（§3.4）。

### 3.3 访问方式 B —— `DbMinuteSource`（DataSource 协议，喂 build_panel）

```python
from alphaforge.providers.db.minute_db import MinuteDB
from alphaforge.providers.db.source import DbMinuteSource

src = DbMinuteSource(MinuteDB("data/equity/minute_db"), rth_only=True)
panel = src.read_minute(["2024-03-01", "2024-03-04"], ["AAPL"])   # 定标的定日期，快；含 amount 列
meta = src.symbol_meta()                          # [symbol, exchange, name]，取自 listing
# 注意：src.trading_days() 与 src.universe_candidates() 会读取全部标的分钟数据，本规模下很慢。
# 交易日 / 全市场选池请用 DuckDB（见 §3.4）。
```

### 3.4 访问方式 C —— DuckDB 直查（快速探索，无需 AlphaForge）

分钟 parquet 内部列：`symbol, ts, open, high, low, close, volume, trade_count, source, ingested_utc, raw_hash`。
`trade_date` / `minute` 由 `ts` 派生。

```python
import duckdb
con = duckdb.connect()

# 单标的一年
con.sql("""
  SELECT strftime(ts,'%Y-%m-%d') AS trade_date, strftime(ts,'%H:%M') AS minute,
         open, high, low, close, volume, trade_count
  FROM read_parquet('data/equity/minute_db/minute/AAPL/2024.parquet')
  WHERE strftime(ts,'%H:%M') > '09:30' AND strftime(ts,'%H:%M') <= '16:00'   -- 仅 RTH
  ORDER BY ts LIMIT 5
""").show()

# 跨标的、按日聚合成交额（读日线更快）
con.sql("""
  SELECT trade_date, sum(amount) AS mkt_amount
  FROM read_parquet('data/equity/minute_db/daily/*.parquet')
  WHERE trade_date BETWEEN '2024-03-01' AND '2024-03-31'
  GROUP BY trade_date ORDER BY trade_date
""").show()

# 交易日列表（从 SPY 日线，快）
con.sql("""SELECT DISTINCT trade_date FROM read_parquet('data/equity/minute_db/daily/SPY.parquet')
           WHERE trade_date BETWEEN '2024-03-01' AND '2024-03-31' ORDER BY 1""").show()

# 某日全市场有成交的选池（读全部日线文件，约数秒；比读全部分钟数据快得多）
con.sql("""SELECT symbol FROM read_parquet('data/equity/minute_db/daily/*.parquet')
           WHERE trade_date='2024-03-01' AND volume>0 ORDER BY symbol""").show()
```

注意：`minute/*/*.parquet` 通配约 8 万个文件，全量扫描较慢；按 `SYMBOL` / `YEAR` 精确定位最快。

### 3.5 增量 / 补缺 —— `MassiveProvider`（REST）

批量历史已入库；单标的、指定区间的按需拉取用此（REST `/v2/aggs`，与 Flat Files 同口径）。

```python
from alpha_data.common.env import load_env
from alpha_data.equity.provider import MassiveProvider

load_env()                                        # 读取 .env 的 MASSIVE_API_KEY
p = MassiveProvider()
res = p.fetch("AAPL", "2026-06-25", "2026-06-29") # FetchResult
print(res.status, res.n_bars, res.bars[0])        # bars[i].ts 为收盘时刻、美东
```

### 3.6 字段

分钟（AlphaForge `read_minute` 口径）：`trade_date`(str YYYY-MM-DD)、`minute`(str HH:MM 收盘戳)、
`symbol`(str)、`open/high/low/close`(float32 原始未复权)、`volume`(int64)、`trade_count`(int64)、
（经 `DbMinuteSource` 时）`amount`=close×volume。

日线：`symbol`、`trade_date`、`open/high/low/close`(float32)、`volume`(int64)、`amount`(float64)、`source`。

## 4. 参考数据与复权

- `listing.parquet`：`symbol, name, exchange, status, ipo_date, delist_date`。`exchange` 为 MIC 码
  （`XNAS`=Nasdaq、`XNYS`=NYSE、`ARCX`=NYSE Arca 等）。`ipo_date` 暂空。
  同一 symbol 兼有 active 与 delisted 记录时（ticker 回收），以当前在市主体（active）为准；
  一表一行的结构无法同时表达历史占用者，需要历史归属时以退市日期回查原始拉取。
- `corp_actions.parquet`：`symbol, ex_date, split_ratio, cash_div`。拆股 2:1 记 `split_ratio=2.0`，
  无拆股记 `1.0`；`cash_div` 为每股现金分红。同一 `(symbol, ex_date)` 的多笔分红（常规 + 特别
  股息）已**求和**，多次拆股取**乘积**。
- **类别股符号回映射**：splits / dividends 端点对类别股返回无点 ticker（`BFB`/`MOGA`），与
  分钟库的带点原生形式（`BF.B`/`MOG.A`）不一致。构建时已按 listing 全集做保守回映射（仅当
  无点符号不在册、且唯一对应一个带点在册符号时才映射），保证公司行动能与分钟数据在
  symbol 上连接。
- `reference/{splits,dividends}.parquet`：拆股 / 分红副本（逐笔，未合并，已回映射）。

复权因子（后复权比值，从最新往回累乘拆股）示例：

```python
import duckdb, pandas as pd
con = duckdb.connect()
daily = con.sql("SELECT trade_date, close FROM read_parquet('data/equity/minute_db/daily/AAPL.parquet') ORDER BY trade_date").df()
ca = con.sql("SELECT ex_date, split_ratio FROM read_parquet('data/equity/minute_db/corp_actions.parquet') WHERE symbol='AAPL' AND split_ratio<>1").df()

# 除权日当日及以后价格 ÷ split_ratio 以拼接连续（仅拆股口径；分红总收益另计 cash_div）
daily = daily.sort_values("trade_date").reset_index(drop=True)
daily["adj_factor"] = 1.0
for _, r in ca.iterrows():
    daily.loc[daily["trade_date"] < r["ex_date"], "adj_factor"] /= r["split_ratio"]
daily["close_adj"] = daily["close"] * daily["adj_factor"]
```

## 5. Polymarket

### 5.1 层与用途

- `daily_aligned`（**推荐分析层**）：已去中继腿、补元数据、归一。逐笔成交 + 真值方向。
- `ctf`：Conditional Tokens 生命周期（铸造 / 销毁 / 解析 / 赎回），做初级市场 / 洗量识别。
- `orderfilled`：原始链上 tape，含中继腿，需自行过滤。

### 5.2 `daily_aligned` 关键字段

`block_timestamp`(int64 秒级 UTC)、`price`([0,1] 概率)、`p_event`(归一事件概率)、`D`(归一方向 ±1)、
`taker_direction`(BUY/SELL 真值)、`usdc_amount`(名义额)、`condition_id`(市场键)、`market_slug`、
`category`/`category_refined`、`outcome_label`/`winning_outcome_label`、`opens_at`/`close_at`/`resolved_at`(tz UTC)。

### 5.3 DuckDB 直查

```python
import duckdb
con = duckdb.connect()
con.execute("SET TimeZone='UTC'")

# 某市场的概率轨迹（转美东时间）
con.sql("""
  SELECT timezone('America/New_York', to_timestamp(block_timestamp)) AS et_ts,
         price, p_event, D, usdc_amount
  FROM read_parquet('data/polymarket/daily_aligned/*.parquet')
  WHERE market_slug = 'fed-rate-cut-by-september-18'
  ORDER BY block_timestamp LIMIT 10
""").show()
```

### 5.4 市场目录与特征

```python
from alpha_data.polymarket import catalog, features, store

con = store.connect()
cat = catalog.build_catalog(con, min_usdc=100_000)            # 约 4s
hits = catalog.search(cat, ["fed-rate-cut", "recession"], top=5)   # 按 slug 关键词

# 单市场分钟特征（防前视）
cid = hits.iloc[0]["condition_id"]
feat = features.build_market_minute_features(con, cid)
```

或一次性构建宏观主题特征宽表：`.venv/bin/python scripts/build_polymarket_features.py`
（产物 `data/polymarket/features/market_features.parquet`）。

### 5.5 特征面板字段

`trade_date`、`minute`（美东，与美股同网格），以及每个市场 `{key}_` 前缀的：

| 列 | 含义 |
|---|---|
| `p` | `p_event` 的 LOCF（截至标签前最后一笔成交） |
| `dp_intraday` | `p` 相对当日开盘的变化 |
| `dp_overnight` | 当日开盘 `p` 相对前一交易日收盘的变化（按日广播） |
| `flow_session` / `usdc_session` / `n_session` | 自当日开盘累计的带方向净额 / 名义额 / 笔数 |
| `flow_overnight` / `usdc_overnight` / `n_overnight` | 闭市窗口（前收盘至当日 09:30，含夜间与周末）的带方向净额 / 名义额 / 笔数，按日广播；首日 NaN |

Polymarket 是 7×24 市场，闭市时段（往往是事件密集时段，如选举夜）的活动经 `*_overnight`
列聚合到下一交易日。市场解析（`resolved_at`）后全部特征置空。

**标签语义（防前视，务必理解后再用）**：特征行标签 `minute=M` 只含**严格早于时刻 M** 的成交。
AlphaForge 面板的 `minute` 是 bar 收盘戳，等值连接后特征行 M 的含义是"截至该 bar 收盘
（不含收盘瞬间）已知的信息"，与由该 bar 自身 OHLCV 计算的价格特征同一时点口径：
**可用于预测 M 之后的 bar，不可用于解释或"预测"该 bar 自身的收益**（那是前视）。
作为宏观 / 事件替代数据在 `(trade_date, minute)` 上广播到全体标的。

## 6. 中证 A 股分钟线（CN）

与美股库**同构**的第三个数据层：源自中证 A 股 1min CSV
（本机 `remote_db/a_stock_data/extracted`），2020-2025 全市场个股 + 主要指数，产出
AlphaForge `MinuteDB` 布局，可由同一套 `DbMinuteSource` / DuckDB 读取。

规模：17.9 亿分钟 bar、742.3 万日线行、5,710 标的（5,180 个股 + 530 指数；含退市，
无幸存者偏差），2020-01-02 .. 2025-12-31 共 1,455 个交易日，约 17 GB。

### 6.1 目录与符号口径

```text
data/cn_equity/minute_db/
  minute/{SYMBOL}/{YEAR}.parquet   # 分钟 bar，列与美股库逐列一致
  daily/{SYMBOL}.parquet           # 日线（一 symbol 一文件）
  listing.parquet                  # 上市表 [symbol, name, exchange, status, ipo_date, delist_date]
  corp_actions.parquet             # 空表（本源无拆股 / 分红，见 6.4）
```

符号为 industry-standard 带交易所后缀（保留前导零）：

- 个股：`sh600000` → `600000.SH`，`sz000001` → `000001.SZ`。
- 指数：`000300` → `000300.SH`（上交所），`399006` → `399006.SZ`（深交所），
  `899050` → `899050.BJ`（北交所）。规则：代码前缀 `39` 记 `.SZ`，`899` 记 `.BJ`，其余 `.SH`。

`.SH`/`.SZ`/`.BJ` 不在 AlphaForge 交易所后缀白名单内，归一时保留，`symbol` 与分钟库目录名一致。

### 6.2 访问方式（与美股相同）

```python
from alphaforge.providers.db.minute_db import MinuteDB

db = MinuteDB("data/cn_equity/minute_db")
# 分钟：默认 rth_only=True 会保留 09:31..15:00 全部连续 bar，仅丢弃 09:30 集合竞价那一根；
# 如需含 09:30 开盘集合竞价 bar，传 rth_only=False。
m = db.read_minute(["600000.SH", "000300.SH"], "2024-01-02", "2024-01-31", rth_only=False)
d = db.read_daily(["600000.SH"], "2024-01-01", "2024-12-31")
```

DuckDB 直查与美股库写法相同（把路径换成 `data/cn_equity/minute_db/...`）。

### 6.3 与美股库的口径差异（重要）

- **`ts` 为北京本地墙钟时间**（源时间戳原样保留，非 UTC、非美东）。下游若按美股日历 / RTH
  解释并不适用：A 股交易时段为 09:30 集合竞价、09:31–11:30 与 13:01–15:00 连续、
  15:00 收盘集合竞价，全部计入日线，构建时**不做 RTH 过滤**。
- **`amount` 为真实成交额**（源 `成交额`），非美股库的 `close×volume` 退化估算。
- **指数无成交量**，`volume` 记 0；`amount` 仍为真实成交额。`trade_count` 源无此列，记 0。

### 6.4 复权缺失（务必知悉）

本源不含拆股 / 分红，故 `corp_actions.parquet` 为空表，**无复权因子**。价格为原始价，
跨除权 / 除息日的收益不可直接使用（例如贵州茅台 2024-06-19 除息，前后价格有跳空）。
如需复权，须从免费源（如 akshare、交易所公告）补拆股 / 分红后重建本表。

### 6.5 重建

```bash
# 全量（2020-2025，个股 + 指数，分钟 + 日线 + listing）
PYTHONPATH=. .venv/bin/python scripts/build_cn_minute_db.py \
    --start-year 2020 --end-year 2025 --memory 24GB
# 只补某几年个股分钟（可断点续传，跳过已完成年份）
PYTHONPATH=. .venv/bin/python scripts/build_cn_minute_db.py --years 2024 2025 --kind stock --no-daily
```

续传 / 重建语义与美股库一致（每年 `.done_{kind}_{year}` 标记、输入指纹校验、归并原子替换）。

## 7. 环境与凭证

- venv：`.venv`（`pandas_market_calendars`、`duckdb`、`boto3`、`huggingface_hub` 等）。
- 凭证：`.env`（不入库，模板 `.env.example`）：`MASSIVE_API_KEY`、`MASSIVE_S3_*`、`HF_TOKEN`。
- 重建：`scripts/download_equity_flatfiles.py` → `scripts/build_minute_db.py` →
  `scripts/build_reference.py`；Polymarket `scripts/download_polymarket.py` → `build_polymarket_features.py`。

分钟 / 日线构建的续传与重建语义（`build_minute_db.py`）：

- 分钟按年构建，成功后写 `minute/.done_{YEAR}` 标记；默认跳过已标记年份（可反复重跑）。
- 单年内两阶段可续传：COPY 完成后把**输入指纹**（文件名 + 大小）写入 `.copydone_{YEAR}`，
  中断后重跑只补未归并的分区、不重读 CSV；若中断后补充下载了新文件，指纹不符会自动改走
  全新构建，不会把旧暂存误当续传而丢新数据。
- **扩充某年数据**（例如把 2026 年从上半年补到全年）：先下载新文件，然后
  `--no-skip-done` 强制重建。全新构建会先清除该年旧分区再整年替换，不会静默保留过期数据。
- 日线一 symbol 一文件、覆盖**全部已下载交易日**，每次重跑整体替换。脚本的日线阶段固定使用
  全部已下载文件（与 `--start/--end` 无关），扩充区间后直接 `--daily-only` 重跑即可。

## 8. 注意事项

- 价格原始未复权；复权自算（§4）。
- 分钟 / 日线时间为 **bar 收盘戳**、美东本地；RTH `09:31`..`16:00`。
- `exchange` 为 MIC 码（可后续映射为 NASDAQ/NYSE 友好名）。
- Polymarket 时间为 UTC、7×24 连续；归并到美股分钟网格须重采样并**防前视**（只用早于目标 bar 的 `p_event`）。
- Polymarket 是宏观 / 事件概率，非逐 symbol 行情；默认市场级广播。
- 缺数据不填充；`corp_actions` 含大量非普通股证券（ETF / 基金 / ADR），按 symbol 过滤即可。
- **规模注意**：MinuteDB 读取全部标的分钟数据的调用（`trading_days()` 不传 symbols、
  `DbMinuteSource.trading_days()` / `universe_candidates()`）在 2.2 万标的下很慢；交易日 / 选池请用
  DuckDB 日线（§3.4）或对 `trading_days` 传基准标的（如 `SPY`）。定标的的 `read_minute` 等都很快。
