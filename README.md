# Alpha-Data

为 **Alpha-Forge**（美股分钟线因子研究系统，仓库内目录 `分钟线策略任务-B/AlphaForge`）准备真实数据的数据工程项目。Alpha-Forge 目前运行在合成数据上（`configs/default.toml: synthetic = true`），本项目负责产出与其消费契约一致的真实数据，替换合成源。

## 1. 数据范围

本项目暂时准备两个相互独立的数据层。

| 数据层 | 来源 | 时间范围 | 体量（压缩 Parquet） | 用途 |
|---|---|---|---|---|
| 美股分钟线 | [massive.com](https://massive.com) | 2020-01-01 至 2026-06-30 | 约 100–150 GB（估计） | Alpha-Forge 的核心价格面板（1min OHLCV + 公司行动 + 标的池） |
| Polymarket 预测市场 | HuggingFace [`TimeSeventeen/Polymarket-v1`](https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1) | 2022-11-21 至 2026-04-28 | 约 49 GB（全量）/ 约 22 GB（推荐子集） | 宏观与事件类替代数据（事件概率、带方向的成交流） |

两个时间范围不同是预期内的。两层各自独立，仅当构造"美股价格 × 预测市场"的联合信号时才需要重叠区间，重叠区间从 2022-11 起。

## 2. 关键事实

### 2.1 massive.com 是 Polygon.io 的更名

massive.com 是 Polygon.io 于 2025-10-30 完成的品牌更名（"Polygon.io is Now Massive"），并非收购，也不是 `joinmassive.com`（一家无关的代理 / 带宽公司）。对本项目的直接影响：

- 既有 Polygon 的 API key、账户、官方 SDK（如 `polygon-api-client`）继续可用；
- 旧域名 `api.polygon.io` 与新域名 `api.massive.com` 在迁移期内并行；
- REST 端点、Flat Files（S3）目录布局、字段命名与 Polygon 完全一致，成熟工具可直接复用。

数据接入采用以下方式：

- **批量历史下载走 Flat Files（S3）**，而非逐条 REST。S3 端点 `https://files.massive.com`，桶 `flatfiles`，分钟线前缀（待在 Dashboard 核实精确字符串）形如 `us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`，每个交易日一个 gzip 文件，包含当日全部 ticker（含退市与 OTC），因此天然无幸存者偏差。2020-01-01 至 2026-06-30 约 1630 个交易日，即约 1630 个文件。
- **REST 用于参考数据与补缺**：标的全集 `/v3/reference/tickers`（`active=false` 取退市标的）、拆股 `/v3/reference/splits`、分红 `/v3/reference/dividends`，以及对个别缺口的分钟线回补 `/v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}`。
- **价格取原始未复权**，拆股 / 分红在本地计算复权因子。Flat Files 的分钟与日线聚合本身为未复权；REST 聚合默认仅按拆股复权（`adjusted=true`），不含分红复权。这与 Alpha-Forge "存原始价、本地复权"的口径一致。

### 2.2 Polymarket-v1 数据集

链上事件日志归档，覆盖 Polymarket v1（运行于 Polygon 网络的预测市场）的完整合约生命周期，约 26.4 亿行、约 49 GB，分三层（config）：

| 层（config） | 行数 | 体量 | 角色 |
|---|---|---|---|
| `orderfilled` | 12.0 亿 | 27.4 GB | 原始链上成交 tape，含中继 / 路由腿，需自行过滤 |
| `daily_aligned` | 6.02 亿 | 13.2 GB | 清洗、去重、补充元数据、归一后的分析层（推荐起点，按日分区） |
| `ctf` | 8.39 亿 | 8.5 GB | Conditional Tokens Framework 生命周期事件（铸造 / 销毁 / 解析 / 赎回） |

许可为 CC-BY-4.0，公开可下载（匿名亦可，提供只读 token 仅为提升吞吐与稳定性）。引用 arXiv:2606.04217。

## 3. 输出契约（Alpha-Forge 消费口径）

Alpha-Data 的产物必须满足 Alpha-Forge 的两个接口。详细字段见 `CLAUDE.md`。

- **读取侧 `DataSource`**（`build_panel` 依赖）：`read_minute` 返回
  `[trade_date, minute, symbol, open, high, low, close, volume, amount]`；价格原始未复权；
  `trade_date` 为 `YYYY-MM-DD` 字符串，`minute` 为 `HH:MM` 字符串（bar 收盘时刻，美东本地时间；RTH `09:31`..`16:00`）。
  另有 `read_daily`、`corp_actions`、`universe_candidates`、`symbol_meta`、`trading_days`。
- **写入侧 `MinuteProvider`**（`Ingester` 依赖，用于增量与补缺）：实现
  `fetch(symbol, start, end, *, interval="1min", extended_hours=False) -> FetchResult`。

落地形态二选一（建议两者都做）：

1. 直接产出 Alpha-Forge 的 `MinuteDB` Parquet 库（`minute/{symbol}/{year}.parquet`、`daily/{symbol}.parquet`、`corp_actions.parquet`、`listing.parquet`），由 `DbMinuteSource` 包装喂给 `build_panel`。批量历史走此路径最快。
2. 实现 `MassiveProvider`（满足 `MinuteProvider` 协议），交给 Alpha-Forge 的 `Ingester` 写库，用于增量与缺口回补。

## 4. 建议的项目结构

```
Alpha-Data/
  README.md
  CLAUDE.md
  .env.example              # 凭证模板（实际 .env 不入库）
  .gitignore
  pyproject.toml            # 依赖与工具配置（后续创建）
  config/
    default.toml            # 时间范围、路径、标的过滤、订阅档位
  alpha_data/
    equity/
      flatfiles.py          # S3 批量下载 minute_aggs_v1
      reference.py          # tickers（含退市）、splits、dividends（REST）
      transform.py          # 原始 CSV.gz -> MinuteDB Parquet（口径转换）
      provider.py           # MassiveProvider（实现 MinuteProvider 协议，REST 补缺）
    polymarket/
      download.py           # huggingface_hub snapshot_download（按层 allow_patterns）
      features.py           # 事件概率 / 带方向成交流特征，归并到美东分钟网格
    common/
      calendar.py           # NYSE 交易日 / 半日 / 夏令时
      io.py                 # Parquet 读写辅助
  scripts/
    download_equity.py
    download_polymarket.py
  data/                     # 输出（不入库）：equity/、polymarket/
  tests/
```

## 5. 环境变量与凭证

复制 `.env.example` 为 `.env` 并填入（实际 `.env` 不入库）：

| 变量 | 用途 | 备注 |
|---|---|---|
| `MASSIVE_API_KEY` | REST：参考数据、拆股 / 分红、分钟线补缺 | 兼容旧名 `POLYGON_API_KEY` |
| `MASSIVE_S3_ACCESS_KEY_ID` | Flat Files（S3）批量下载 | 与 REST key 不同，需在 Dashboard 单独生成 |
| `MASSIVE_S3_SECRET_ACCESS_KEY` | 同上 | |
| `HF_TOKEN` | Polymarket 下载 | 只读 scope；公开数据集非强制，建议提供以提升吞吐 |

## 6. 使用流程（规划）

1. 下载美股 Flat Files：枚举 2020-01-01 至 2026-06-30 交易日，并行下载每日 `minute_aggs` 文件。
2. 拉取参考数据：全量 tickers（含退市）、splits、dividends。
3. 转换口径：纳秒 UTC 时间戳归并到美东分钟网格，拆出 `trade_date` / `minute`，写入 `MinuteDB`。
4. 下载 Polymarket：按层 `snapshot_download` 至本地，用 DuckDB / Polars 查询。
5. 构造 Polymarket 特征：事件概率、带方向成交流，归并到美东分钟网格，防前视。
6. 对接 Alpha-Forge：将 `MinuteDB` 指向 Alpha-Forge 的 `data` 根，关闭其 `synthetic`。

## 7. 体量与存储估算

- 美股分钟线 Flat Files：压缩约 50–100 GB，转为 Parquet 约 100–150 GB。仅 1min 聚合；若另取逐笔 trades / quotes，体量将达 TB 级（暂不取）。
- Polymarket：全量约 49 GB；推荐子集 `daily_aligned` + `ctf` 约 22 GB。
- 建议预留约 250–300 GB 可用磁盘。下载首月数据后据实测校准。

## 8. 待确认决策

下列默认值已写入本文档与 `config`，请确认或调整（详见与本次交付一并给出的"需要你提供 / 确认"清单）：

1. massive.com 订阅档位：覆盖 2020 需 Developer（10 年）起，建议 Advanced / Business（全历史 + Flat Files + 实时）。具体价格须在官网核实。
2. 美股范围：默认全市场（无幸存者偏差，Flat Files 免费包含）。
3. 数据产品：默认仅 1min 聚合，不含逐笔 trades / quotes。
4. 复权：默认存原始价，本地按拆股 / 分红计算复权因子。
5. 盘前盘后：默认入库但读取时默认仅 RTH。
6. Polymarket 层：默认 `daily_aligned` + `ctf`，暂不取 `orderfilled`。
7. Polymarket 到美股的映射：作为市场级宏观 / 事件特征广播，非逐 symbol 行情。
8. 输出落地：默认直接产出 `MinuteDB` 库，并附 `MassiveProvider` 供增量与补缺。

## 9. 已实现进度

- 环境：`.venv` + 依赖（含 duckdb / polars / pandas_market_calendars / boto3 / huggingface_hub）。
- Polymarket 原始数据：全量三层已下载并校验（`daily_aligned` + `ctf` + `orderfilled`，约 46 GB，26.4 亿行）。
- 美股参考数据（`alpha_data/equity/reference.py`）：标的全集（含退市）、拆股、分红 → AlphaForge `listing` / `corp_actions` 表，已入库。
- 美股 1min Flat Files 管线：`flatfiles.py`（S3 下载）+ `transform.py`（口径转换）+ `minute_store.py`（构建 `MinuteDB` 分区库）。全量 2020-01 至 2026-06 已构建完成，并经 AlphaForge 自身 `MinuteDB` 读回验证（单日 390 个 RTH bar、OHLCV 与 REST `adjusted=false` 逐字段一致）。
- `MassiveProvider`（REST 增量 / 补缺，实现 `MinuteProvider` 协议）：已实现并与 Flat Files 构建交叉验证。
- **Polymarket 特征管线（`alpha_data/polymarket/`）**：见下。
- 2026-07-01 修订：符号约定改为保留类别股 / 单位 / 权证的 `.` 后缀（与 AlphaForge
  `default_normalize_symbol` 协同修正），并对 splits / dividends 端点的无点类别股 ticker
  做回映射（`BFB` → `BF.B`），`listing` / `corp_actions` 已按新口径重拉；
  `corp_actions` 同日多笔分红改为求和；分钟 / 日线构建的强制重建路径修正为整年替换 +
  输入指纹校验（此前会静默保留旧分区）；Polymarket 特征新增 `*_overnight` 隔夜列。
- 待办：切换 AlphaForge 至真实源并端到端跑 `build_panel`。

## 10. Polymarket 特征管线

把 `daily_aligned` 转为市场级、防前视、归并到美东分钟网格的替代数据，供 AlphaForge 广播注入。

- `common/calendar.py`：NYSE 交易日 + RTH 分钟网格（`pandas_market_calendars`，自动处理节假日 / 半日）。
- `polymarket/catalog.py`：按 `condition_id` 汇总市场目录（DuckDB，约 4s 完成 6 亿行聚合），按 slug 关键词检索。
- `polymarket/features.py`：单市场分钟特征 `p`（`p_event` LOCF）、`dp_intraday`、`dp_overnight`、`flow_session`、`usdc_session`、`n_session`，以及闭市窗口（夜间 / 周末 / 假日）聚合的 `flow_overnight`、`usdc_overnight`、`n_overnight`（按日广播）。防前视：标签 T 只用严格早于时刻 T 的成交（对 AlphaForge 收盘标签即"截至该 bar 收盘已知"，只能预测 T 之后的 bar）；解析后置空。
- `scripts/build_polymarket_features.py`：建目录 → 按宏观主题（fed_rate / recession / inflation / election / shutdown）选市场 → 建特征宽表 → 写 `data/polymarket/features/`。
- 测试：`tests/test_polymarket_features.py`（合成数据，验证防前视与累计不变量）。

运行：`python scripts/build_polymarket_features.py`（产物 `market_catalog.parquet` + `market_features.parquet`）。

v1 范围与后续：当前每个宏观主题取成交额最高的 1 个代表市场；后续可扩展为同主题多市场链接（跨相继市场连续覆盖）、逐板块 / 逐标的映射、滚动窗口与解析事件标记，以及接入 AlphaForge `build_panel` 的注入适配器。

## 11. 引用与许可

- Polymarket-v1：CC-BY-4.0。引用 `arXiv:2606.04217`（Qin & Yang, 2026, *Polymarket-v1 Database*）及 HF 数据集 `TimeSeventeen/Polymarket-v1`。
- massive.com 数据受其订阅条款约束，不在公开仓库内再分发原始数据。

## 12. 来源

- massive.com 文档与公告：<https://massive.com/docs>、<https://massive.com/blog/polygon-is-now-massive>、<https://github.com/massive-com/mcp_massive>
- Polymarket 数据集与论文：<https://huggingface.co/datasets/TimeSeventeen/Polymarket-v1>、<https://arxiv.org/abs/2606.04217>
