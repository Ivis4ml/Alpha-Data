# Polymarket 数据自采集手册（链上一手爬取与续爬）

本文说明我们如何从一手来源采集 Polymarket 全量成交数据、如何验证采集结果的正确性，
以及如何把数据持续续爬下去。文中每个数字都由实测得出，均可用仓库内的脚本当场复现。

- 采集模块：`alpha_data/polymarket/chain/`
- 命令行：`scripts/crawl_polymarket_chain.py`
- 复现性验证：`scripts/verify_polymarket_reproduction.py`
- 单元测试：`tests/test_polymarket_chain.py`

## 0. 结论摘要

1. **这份数据没有任何私有成分。** Polymarket 的每一笔成交都是 Polygon 主网（chainId 137）上的
   公开事件日志。所谓「数据集」就是这些日志的解码结果，再加一层来自 Polymarket 免鉴权只读 API
   的市场元数据。任何人有一个 Polygon RPC 端点就能完整重建，无需许可、无需付费。
2. **我们已逐字段验证了这一点。** 用本仓库的爬虫重抓 2022 至 2026 年间随机抽取的 6 个区块，
   解码结果与已有快照的同区块数据 **163 行 × 13 列完全相等，零差异**（第 2 节）。
3. **续爬有一个必须跨过的坎。** 2026-04-28，Polymarket 迁移了交易所合约：两个主力旧合约在
   区块 86,126,998（11:00:40 UTC）发出最后一条成交事件后归零，新合约接管，且**事件 ABI 与
   抵押品代币都变了**。任何按旧地址硬编码的管线都会在这一天静默断流 —— 公开数据集正是
   停在这一刻（第 6 节）。本仓库的爬虫**始终同时解码新旧两套 ABI**，因为旧版 ABI 至今
   仍有低频场馆在使用。
4. **全流程已跑通。** 免费公共 RPC、8 并发下抓取 50,000 个区块（约 1.2 天行情）耗时 601 秒，
   产出 616 万行成交；市场目录 177 万个市场耗时 151 秒；分析层 390 万行，元数据命中率 99.9998%。

## 1. 数据本质：链上事件与数据层的一一对应

Polymarket 的订单撮合在链下（中心化订单簿），但**每一笔成交的结算都在链上**：
由交易所合约发出 `OrderFilled` 事件；份额的铸造、合并、赎回与市场的创建、解析
则由 Gnosis Conditional Tokens（CTF）合约发出五类事件。

公开数据集 `TimeSeventeen/Polymarket-v1` 的三层与链上事件一一对应（已逐层核对）：

| 数据层 | 规模 | 链上来源 |
|---|---|---|
| `OrderFilled` | 12.02 亿行 | 交易所合约的 `OrderFilled` 事件，原样解码 |
| `CTF/splits` | 5.94 亿行 | ConditionalTokens 的 `PositionSplit` |
| `CTF/merges` | 5,484 万行 | ConditionalTokens 的 `PositionsMerge` |
| `CTF/redemptions` | 1.87 亿行 | ConditionalTokens 的 `PayoutRedemption` |
| `CTF/preparations` | 145 万行 | ConditionalTokens 的 `ConditionPreparation` |
| `CTF/resolutions` | 131 万行 | ConditionalTokens 的 `ConditionResolution` |
| `daily_aligned` | 6.019 亿行 | `OrderFilled` 剔除中继腿后，与市场元数据内连接（第 4 节） |

一个关键旁证：数据集中每一行的主键都是 `{chainId}_{blockNumber}_{logIndex}`，
例如 `137_63732941_137`。这是链上日志的天然坐标，不是任何私有编号。因此该数据集
不可能包含链上没有的信息，也不存在「作者独有的数据源」。

数据集覆盖 2022-11-21 19:50:09 至 2026-04-28 11:00:40（UTC），共 688,131 个市场、
196.2 亿美元 taker 名义成交额。两个端点都不是随意截断的：

- 起点：区块 35,896,869 是**链上第一条** `OrderFilled`。我们扫描了区块 33,000,000 至此之间
  的全部区块，无任何 `OrderFilled`，即该区块就是订单簿协议的起点。
- 终点：区块 86,126,998 是**旧交易所合约的最后一条** `OrderFilled`（见第 6 节）。

也就是说，这份数据恰好覆盖旧协议的完整生命周期，一分不多一分不少。

## 2. 复现性验证（可当场演示）

这是「数据是我们自己采集的」这一说法的可检验证据。给定任意区块，脚本从公共 RPC 拉取原始日志、
用我们的解码器还原成交，再与已落地的快照逐列比较：

```bash
python scripts/verify_polymarket_reproduction.py --sample 6
```

实际输出：

```text
RPC 端点：https://polygon.api.onfinality.io/public, https://polygon.gateway.tenderly.co, ...
待比对区块：[36791573, 44187553, 54128932, 64089969, 75848988, 85988698]

  区块  36791573  一致  2 行 x 13 列全等
  区块  44187553  一致  2 行 x 13 列全等
  区块  54128932  一致  12 行 x 13 列全等
  区块  64089969  一致  22 行 x 13 列全等
  区块  75848988  一致  18 行 x 13 列全等
  区块  85988698  一致  107 行 x 13 列全等

结论：全部区块逐字段一致，快照可由本仓库爬虫完全复现。
```

六个区块横跨 2022-12 至 2026-04（旧协议的全部年份），163 行成交在
`id / maker / taker / block_timestamp / maker_asset_id / taker_asset_id / maker_direction /
taker_direction / token_asset_id / token_amount / usdc_amount / price / fee_usdc`
十三个字段上全部相等（浮点比较阈值 1e-9）。

## 3. 爬什么：合约、事件与解码规则

### 3.1 合约谱系

| 角色 | 地址 | 部署区块 | 抵押品 |
|---|---|---|---|
| CTFExchange（旧） | `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` | 33,605,403（2022-09-26） | USDC.e |
| NegRiskCtfExchange（旧） | `0xc5d563a36ae78145c45a50134d48a1215220f80a` | 50,505,492（2023-11-28） | USDC.e |
| CTF Exchange v2 | `0xe111180000d2663c0091e4f400237545b87b996b` | 84,902,353（2026-03-31） | pUSD |
| NegRisk CTF Exchange v2 | `0xe2222d279d744050d28e00520010520000310f59` | 85,058,176（2026-04-03） | pUSD |
| ConditionalTokens | `0x4d97dcd97ec945f40cf65f87097ace5ea0476045` | 4,023,686（2020-09-03） | 通用 |
| NegRiskAdapter | `0xd91e80cf2e7be2e162c6513ced06f1dd0da35296` | 50,505,403（2023-11-28） | 通用 |
| WrappedCollateral（WCOL） | `0x3a3bd7bb9528e159577f7c2e685cc81a765002e2` | 50,505,403 | negRisk 份额的抵押品 |

结算合约在整个历史中从未变过（各交易所的 `getCtf()` 恒为上表的 ConditionalTokens 或
NegRiskAdapter），变的是交易所与抵押品。这个不变性正是第 6.3 节甄别场馆的依据。

### 3.2 事件与 topic0

七个 topic0 均由事件签名的 keccak256 算出，并与链上实测日志逐一核对一致：

| 事件 | topic0 |
|---|---|
| `OrderFilled`（旧版） | `0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6` |
| `OrderFilled`（新版） | `0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee` |
| `PositionSplit` | `0x2e6bb91f8cbcda0c93623c54d0403a43514fabc40084ec96b6d5379a74786298` |
| `PositionsMerge` | `0x6f13ca62553fcc2bcd2372180a43949c1e4cebba603901ede2f4e14f36b282ca` |
| `PayoutRedemption` | `0x2682012a4a4f1973119f1c9b90745d1bd91fa2bab387344f044cb3586864d18d` |
| `ConditionPreparation` | `0xab3760c3bd2bb38b5bcf54dc79802ed67338b4cf29f3054ded67ed24661e4177` |
| `ConditionResolution` | `0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894` |

### 3.3 两版 `OrderFilled` 的解码（关键）

**旧版**（2022-11-21 至 2026-04-28）：

```text
OrderFilled(bytes32 orderHash, address maker, address taker,
            uint256 makerAssetId, uint256 takerAssetId,
            uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)
```

前三个字段为 indexed（在 topics 中），其余五个 uint256 在 data 中。事件是 **maker 视角**的：
`makerAssetId` 是 maker 付出的资产，`takerAssetId` 是 maker 收到的资产。抵押品的资产 id 为 0，
因此方向由 `makerAssetId == 0` 判定：为 0 说明 maker 付出抵押品、收到份额，即 maker 买入，
相应地 taker 卖出。

**新版**（2026-04-03 起）：

```text
OrderFilled(bytes32 orderHash, address maker, address taker, uint8 side,
            uint256 tokenId, uint256 makerAmountFilled, uint256 takerAmountFilled,
            uint256 fee, bytes32 builder, bytes32 metadata)
```

新版只带**一个** `tokenId`，方向改由 `side` 显式给出（`Side.BUY = 0`，`Side.SELL = 1`，
指的是**发出该事件的那张订单自身的方向**）。这带来一个隐蔽的陷阱：
`makerAmountFilled` 与 `takerAmountFilled` 的**币种随 side 互换**。
side = 0（买入）时前者是抵押品、后者是份额；side = 1（卖出）时恰好相反。

这不是推测，有两重独立证据：

- **实测**：若忽略 `side` 按固定顺序解码，30,124 条样本中**恰好全部 3,249 条 side=1 的行**
  算出的 `price = usdc / token` 越出 [0, 1] 区间；按上述规则解码后，30,124 条**无一越界**。
- **源码**：Polymarket 的 `ctf-exchange-v2` 仓库中 `Events.sol` 的 `log4` 汇编把 data 字段
  顺序钉死为 `[side, tokenId, makerAmountFilled, takerAmountFilled, fee, builder, metadata]`，
  与实测完全吻合。末两个 `bytes32` 是 `builder` 与 `metadata`，于成交重建无用，不解码。

### 3.4 CTF 五类事件的 indexed 布局（易错）

五类事件都有 3 个 indexed 字段，但**哪些字段被索引并不一致**，凭直觉写必错。
以下布局由链上原始日志与快照的标准答案逐字节对照反推得出：

| 事件 | topics[1..3] | data |
|---|---|---|
| `PositionSplit` / `PositionsMerge` | stakeholder, parentCollectionId, conditionId | collateralToken, partition[], amount |
| `PayoutRedemption` | redeemer, **collateralToken**, parentCollectionId | conditionId, indexSets[], payout |
| `ConditionPreparation` | conditionId, oracle, questionId | outcomeSlotCount |
| `ConditionResolution` | conditionId, oracle, questionId | outcomeSlotCount, payoutNumerators[] |

注意 `collateralToken` 在 split / merge 中是**非索引**字段（在 data 里），
在 redemption 中却是**索引**字段（在 topics 里）。这是 Gnosis CTF 合约的既有设计。

## 4. 怎么清洗：从原始事件到分析层

### 4.1 中继腿（relay leg）

交易所撮合时，除了为每一对 maker / taker 记一条成交，还会以**自身地址作为对手方**，
为 taker 的整笔订单再记一条聚合成交。判定规则简单且新旧协议通用：

> 中继腿 ⇔ `taker == 发出该日志的合约地址`

实证：某笔成交的三条日志中，两条是用户与做市商的对手腿（份额 4.38 与 5.82408），
第三条是该用户与交易所之间的聚合腿（份额 10.20408 = 4.38 + 5.82408），
其价格 0.4899999 与 Polymarket 公开 Data API 对这笔交易的报价完全一致。

**保留中继腿会使成交量恰好翻倍**，任何分析层都必须剔除。中继腿约占原始行数的 36% 至 46%，
比例随时间下移（一张 taker 订单撮合的 maker 腿越多，中继腿占比越低）。

### 4.2 公开数据集 `daily_aligned` 的真实构造规则

我们反推出了它的完整构造。它**只有市场级过滤，没有任何行级过滤**：

```text
daily_aligned = OrderFilled
              减去 中继腿
              减去 全部 negRisk 市场的成交          <- 这是剩下 4.4 倍差距的来源
              减去 asset_id 未落在其元数据快照内的行（约 0.2%）
不含最小成交额门槛、不去重、不做洗量剔除。
```

证据一（2024-11-05 单日的逐行对账）：

| 口径 | 行数 |
|---|---|
| 链上 `OrderFilled` 全量 | 797,142 |
| 剔除中继腿后 | 437,192 |
| 再限制到 `daily_aligned` 的市场集合 | **43,844** |
| `daily_aligned` 当日实际行数 | **43,844** |

两者**完全相等**，说明过滤只发生在市场层面，逐笔成交本身未经任何筛选。
当日被丢弃的 2,411 个 asset **全部**来自 NegRisk 交易所。

证据二（用史上最大的市场做判决性检验）：2024 年美国大选「特朗普胜选」市场
（`will-donald-trump-win-the-2024-us-presidential-election`，negRisk = true）在
2024-11 的原始 `OrderFilled` 中有 1,159,677 行，其中 638,375 行是非中继腿的真实成交；
而 `daily_aligned` 中该市场的行数为 **0**。

证据三（全库扫描）：对全部 6.019 亿行扫描确认，`neg_risk` 字段**恒为 `f`**，
`outcome_seq` 只取 `{1, 2}`。

**结论**：公开数据集**整体丢弃了 negRisk 市场与多结果市场**，其中包括 2024 年美国大选
那批旗舰市场。它的月度留存率因此剧烈波动（2024-10 为 12.3%，2024-11 为 22.5%，
2026-03 为 88.2%），完全取决于当月有多少成交量沉在 negRisk 市场里。
**这是使用该数据集时必须知道的重大限制。**

本仓库的爬虫保留全部市场，用 `is_binary` 标记二元市场，不做静默丢弃。

### 4.3 `p_event` 与 `D` 的定义

`p_event` 是「事件发生」的概率，`D` 是该笔成交把概率推向哪一边。二者只在二元市场上有定义。
真值表由 2024-11 全月数据反推：

- `p_event`：`outcome_seq == 1` 时取 `price`；`outcome_seq == 2`（互补份额）时取 `1 - price`。
- `D`：taker 买入 Yes 或卖出 No 记 `+1`；卖出 Yes 或买入 No 记 `-1`。

| taker 方向 | outcome_seq | D |
|---|---|---|
| BUY | 1 | +1 |
| SELL | 1 | -1 |
| BUY | 2 | -1 |
| SELL | 2 | +1 |

多结果市场没有互补关系，`1 - price` 无意义，故本仓库对非二元市场置空而非外推。

## 5. 怎么爬得动：抓取架构与 RPC 选型

### 5.1 公共 RPC 端点实测

`eth_getLogs` 的能力在各端点间差异极大，官方文档往往不写。实测（同一历史区块区间）：

| 端点 | 单次区块跨度上限 | 吞吐 | 日志自带 `blockTimestamp` |
|---|---|---|---|
| `polygon.api.onfinality.io/public` | 10,000 | 277,660 条 / 41 秒 | **是** |
| `polygon.gateway.tenderly.co` | 10,000 | 277,660 条 / 30 秒 | 否 |
| `polygon.drpc.org` | 不稳定（免费档限流） | 低 | 否 |
| `1rpc.io/matic` | **50** | 不可用于回填 | 否 |
| `polygon-rpc.com` | 官方端点已停用（返回 403） | 不可用 | 不适用 |
| `polygon-bor-rpc.publicnode.com` | 拒绝归档请求 | 不可用于历史 | 否 |

两点实践结论：

- **onfinality 的日志对象自带 `blockTimestamp`**，直接省掉了为约两千万个区块单独拉区块头的
  巨大开销。命中该端点时无需任何额外请求即可得到成交时间；其他端点则回退为批量
  `eth_getBlockByNumber`（实测批量 100 个区块头耗时 1.8 秒）。
- 超限的失败模式在各端点间不统一（区间超限 / 结果过多 / 网关超时），无法靠错误码区分。
  因此 `rpc.py` 的策略是**任一失败即二分该区间重试**，直到单块查询仍失败才判定为真实故障。
  这样无需为每个端点单独调参。

需要更高吞吐时，配置环境变量 `POLYGON_RPC_URLS`（逗号分隔）接入付费端点即可，代码无需改动。

### 5.2 抓取单元与幂等

抓取以**区块分片**为单位，分片边界对齐到 10,000 的网格：

```text
data/polymarket_chain/
  shards/trades/086120000-086129999.parquet   # 抓取单元，存在即视为已完成
  daily/trades/2026-04-28.parquet             # 按 UTC 日合并的分区
  analysis/2026-04-28.parquet                 # 去中继腿 + 元数据 + p_event / D
  markets_clob.parquet / asset_map.parquet    # 市场目录与 asset_id 映射
  cursor.json                                 # 已完成的连续区块上界
```

分片先写临时文件再原子改名，中断不会留下半截文件；重跑时已存在的分片直接跳过，断点可续。
事件主键 `{chainId}_{blockNumber}_{logIndex}` 天然幂等，重复抓取同一区块只会产出完全相同的行。

日分区的合并是**流式**的：分片文件名以零填充的起始区块开头，字典序即区块序，而区块号与
时间戳单调，因此读到某分片时，早于它最小日期的所有日期都已收齐，可立即写盘并释放内存。
全量回填有 12 亿行，一次性 concat 会耗尽内存。

### 5.3 区块重组（reorg）

抓取上界默认取 `finalized` 标签（端点不支持时回退为链头减 256 块）。已最终确定的区块不会回滚，
分片因此可安全视为不可变。代价是延迟数分钟，收益是完全不必处理回滚逻辑。

### 5.4 实测吞吐与成本

实测（免费公共端点，8 并发）：

```text
抓取区块 86127000 .. 86176999（50,000 块，约 1.2 天行情）
  trades: 新增 6,168,870 行
完成，耗时 601s
```

即约 **83 块/秒、1.0 万行/秒**。据此推算：

- **日常增量**：一天约 41,000 个区块，耗时约 8 分钟。可每日定时执行。
- **全量回填**：区块 35,896,869 至今约 5,430 万块、12 亿行。按上述速率（且历史大部分时期的
  日志密度远低于 2026 年）估计需 1 至 3 天连续运行。分片幂等使其可随时中断续跑。
- **金钱成本为零**（免费公共端点）。付费端点非必需，只影响速度。

### 5.5 更快的回填路径（可选）

若不愿等 1 至 3 天，有两条已核实的替代路径：

- **Google BigQuery 公共数据集** `bigquery-public-data.goog_blockchain_polygon_mainnet_us`：
  由 Google 维护，含 `logs` 表（`block_number` / `log_index` / `address` / `topics` / `data` /
  `block_timestamp`），按 topic0 过滤后导出即可，按需计费 $6.25/TB。这是一次性历史回填最快的路径。
- **付费 RPC**：Alchemy 的按量计费档下，一次全量回填的 CU 成本约 $17 量级
  （注意其 `eth_getLogs` 的区块跨度上限在该档为 2,000，需相应调小分片）。

另有 **Dune Analytics** 的 `polymarket_polygon.*` 策展表，适合做交叉验证（其单次导出行数
有上限，不适合导出 12 亿行）。Polymarket 官方的 Goldsky subgraph **不可用于续爬**（见 6.1）。

## 6. 续爬：2026-04-28 的合约迁移

这是本文最有实操价值的一节。**公开数据集停在 2026-04-28，不是因为作者停止更新，
而是因为它所监听的合约停止工作了。**

### 6.1 证据链

| 观察 | 结论 |
|---|---|
| 两个主力旧交易所的最后一条 `OrderFilled` 在区块 86,126,998，2026-04-28 11:00:40 UTC | 旧主力合约的终点 |
| 公开数据集的最大区块 = 86,126,998，最后一笔成交 = 2026-04-28 11:00:40 UTC | **与之完全相同** |
| 区块 86,150,000（2026-04-28 23:47 UTC）起，两个主力旧交易所的 `OrderFilled` 数恒为 0 | 旧主力合约已停用 |
| 新合约 `0xe1111800...` 首条 `OrderFilled` 在区块 85,050,371，2026-04-03 12:52:59 UTC | 新旧并行约 25 天 |
| 数据集 `2026_04` 分片中 block > 86,126,998 的行数为 0，且不含任何新版事件 | 其解码器只认旧 ABI |
| Polymarket 官方 Data API 至今仍持续返回新成交 | 交易本身从未中断 |

**一个必须说清的精确性问题：停的是「两个主力旧交易所」，不是「旧版 ABI」。**
实测在迁移之后，仍有三个低频变体场馆（`0x403c9453...`、`0x87b81fd5...`、`0x2e0277c2...`）
继续发出**旧版 ABI** 的 `OrderFilled`（我们抓取的 50,000 个区块窗口内共 32 行、约 2,700 美元）。
量虽微小，但足以说明：**续爬必须始终同时解码两套 ABI**，不能在迁移日之后关掉旧解码器。
本仓库的爬虫在每次 `eth_getLogs` 中以 `topics[0]` 的或运算同时请求两版事件，天然满足这一点。

Polymarket 官方的 Goldsky orderbook subgraph 同样只索引旧合约（其最新 `orderFilledEvent`
时间戳也停在 2026-04-28），因此**任何依赖该 subgraph 的管线都会在同一天断流**。
这也是我们的爬虫直接走 `eth_getLogs`、不依赖 subgraph 的原因。

### 6.2 新协议变了什么

- **交易所合约**：换成 `0xe1111800...` 系列，且**不止一个**（见 6.3）。
- **抵押品**：从 USDC.e 换成 Polymarket 自发的 **pUSD**（`0xc011a7e1...`，
  `name() = "Polymarket USD"`，6 位小数）。标度不变，故 `usdc_amount` 口径可直接拼接。
- **事件 ABI**：`OrderFilled` 的 topic0 与字段布局全变（见 3.3）。
- **结算合约**：**没变**，仍是同一个 ConditionalTokens。这保证了市场标识
  （`condition_id`、`asset_id`）在新旧协议间连续，历史与新数据可以直接拼接。

### 6.3 场馆甄别：按 topic 扫描的必要性与代价

**必要性**：新协议同时有多个合约在发 `OrderFilled`。某 400 区块样本中，
`0xe1111800...` 占 25,219 条，`0xe2222d27...` 占 4,654 条，`0xe3333700...` 占 251 条。
若按地址白名单只抓第一个，会漏掉约 **17%** 的成交。因此爬虫**按 topic0 全网扫描**，
不做地址过滤，新增的交易所合约会被自动覆盖。

**代价**：`OrderFilled(bytes32,address,address,uint256,...)` 这个签名并非 Polymarket 独有，
链上存在使用同一签名的**其他协议**。实测抓到了两个（`0x5afa5159...`、`0xcae049a1...`），
它们的成交若混入数据集就是污染。

**甄别规则**（可复核，不依赖名单）：

> Polymarket 场馆的 `getCtf()` 必然返回 ConditionalTokens (`0x4d97dcd9...`)
> 或 NegRiskAdapter (`0xd91e80cf...`)。

据此逐一核验，得到 `venues.VERIFIED_VENUES`（8 个，含四个低频变体场馆）与
`venues.FOREIGN_VENUES`（2 个）。无法判定的地址标记为 `unknown`：**既不混入分析层，
也不静默丢弃**，而是在构建分析层时打印出来交由人工复核。

目前唯一的 `unknown` 是 `0xe3333700...`：它发出新版 `OrderFilled`，但不暴露 `getCtf()`，
其成交结算于 `0x006f54f7...`（一个 PositionManager 合约）而非 ConditionalTokens，
份额 id 的语义尚未确认。它约占新协议成交的 1%，判定清楚前不进入分析层。

### 6.4 三条命令续爬

```bash
# 1. 从游标（或公开快照末区块 86126998）接续，抓到最新已最终确定的区块
python scripts/crawl_polymarket_chain.py trades

# 2. 抓市场元数据，并用链上 getPositionId 抽样校验映射
python scripts/crawl_polymarket_chain.py metadata --verify 20

# 3. 构建分析层（场馆甄别 + 去中继腿 + 连接元数据 + 计算 p_event / D）
python scripts/crawl_polymarket_chain.py analysis
```

## 7. 元数据：把 `asset_id` 映射到市场

链上日志只有 `asset_id`（ERC1155 份额 id）这一个市场标识。要回答「这是哪个市场、哪个结果、
事件最终是否发生」，需要一层市场元数据。

### 7.1 CLOB 目录（主路径）

`clob.polymarket.com/markets` 免鉴权，游标分页、每页 1000 条，**默认即含已关闭市场**。
其游标是 `base64(十进制 offset)`，因此可以跳页并行抓取。实测：

```text
抓取 CLOB 市场目录（游标 = base64(offset)，跳页并行）...
  市场 1,770,304 个，耗时 151s
  asset_id 映射 3,511,226 条
```

（市场数远超数据集的 68.8 万，因为近年新增了大量自动生成的短周期市场，
例如 5 分钟一期的加密货币涨跌盘。）

每个市场的 `tokens` 数组给出 `token_id`、`outcome`（结果标签，**并不总是 Yes / No**）
与 `winner`（是否胜出），即 `asset_id -> (市场, 结果序号, 胜出结果)` 的完整映射。

### 7.2 链上推导（校验路径，不信任任何 HTTP 接口）

`asset_id` 并非随机编号，而是 ConditionalTokens 合约按确定性规则算出的：

```text
collectionId = CTF.getCollectionId(0x0, conditionId, indexSet)
asset_id     = CTF.getPositionId(collateralToken, collectionId)
```

其中 `indexSet` 是结果的位掩码（`1` = 第一个结果，`2` = 第二个），恰好等于 `outcome_seq`。
因此可以完全绕开 HTTP 接口来验证映射。`metadata --verify N` 会抽 N 个市场做链上推导比对，
实测 **20/20 全部一致**，即 CLOB 的 `tokens` 数组顺序确实就是 indexSet 顺序。

一个陷阱：**抵押品不能取自 `negRisk` 标记**。negRisk 市场的份额 id 以 WCOL
（`0x3a3bd7bb...`，NegRiskAdapter 的包装抵押品）计算，而非 USDC.e；且实测存在
`neg_risk` 为假、却仍以 WCOL 计算份额 id 的市场。正确做法是两种抵押品都试，取能对上的那个
（`positionid.py` 即如此实现）。

### 7.3 Gamma 接口的四个坑（可选路径，仅用于补类别）

`gamma-api.polymarket.com` 提供 `category` 与生命周期字段，但它的分页有四个会**静默丢数据、
不报错**的陷阱，全部经实测确认：

1. `limit` 被**静默钳制**为 100：传 500 / 1000 仍只返回 100 条，HTTP 200 无任何提示。
2. `/markets` 的 `offset` **硬上限 2000**，越界返回 HTTP 422。offset 分页最多枚举约 2,100 个
   市场，**不能用于全量抓取**。全量必须走 `/markets/keyset`。
3. keyset 的响应字段名是 `next_cursor`，但回传时的参数名是 `after_cursor`。
   若原样以 `next_cursor` 回传，服务端**静默忽略并永远返回第一页**，形成死循环。
4. Gamma 默认 `closed=false`，即默认只返回未关闭的市场。要取全量历史，
   必须对 `closed=true` 与 `closed=false` 各走一遍。

若只做成交重建，CLOB 一家即已足够，Gamma 属可选项。

市场的解析时刻（`resolved_at`）以链上 `ConditionResolution` 事件的区块时间戳为准，
比 Gamma 的 `closedTime`（运营侧写入时刻）更严格。

## 8. 数据质量校验清单

对每一批新数据执行以下检查（`analysis` 子命令会打印其中大部分）：

1. **主键唯一**：`id` 无重复（实测新抓的 390 万行分析层，`id` 唯一数 = 行数）。
2. **价格有界**：非中继腿成交的 `price` 全部落在 [0, 1]（实测越界 0 行）。
3. **金额非负**：`usdc_amount >= 0`、`token_amount > 0`（实测异常 0 行）。
4. **元数据命中率**：实测 99.9998%（1,648,915 / 1,648,919）。命中率骤降说明市场目录过期，
   需重跑 `metadata`。
5. **场馆归属**：出现新的 `unknown` 地址时必须人工复核，不得默认纳入。
6. **区块连续**：分片按网格对齐，缺失即缺文件，无空洞。
7. **与快照衔接**：续爬起点区块 = 快照末区块 + 1，无重叠无空隙。
8. **协议分布**：迁移日前后 `protocol` 应从 v1 平滑切换到 v2；若某日 v1 与 v2 同时为 0，
   说明抓取失败而非无成交。

## 9. 数据来源与合规

- **链上数据是公开信息**。Polygon 是无许可的公开账本，读取其日志不构成对任何服务条款的接受
  或违反。我们通过第三方公共 RPC 端点读取，遵守各端点自身的速率限制（爬虫内置退避重试与
  二分降级，不做激进并发）。
- **Polymarket 的只读 API**（CLOB `/markets`、Gamma）为公开免鉴权端点，仅用于获取市场元数据，
  请求量低（全量目录约 1,800 次请求）。
- **可追溯性**：每一行成交都带 `id`（`chainId_blockNumber_logIndex`）与 `tx_hash`，
  任何一行都可以回到 Polygon 区块浏览器上逐字核对。这是本管线相对任何「下载来的数据集」的
  根本优势：**每个数字都有一手出处**。
- **与公开快照的关系**：本仓库同时保留了 `TimeSeventeen/Polymarket-v1` 快照（CC-BY-4.0），
  其用途是充当**验证基线** —— 第 2 节的逐字段比对正是以它为对照，用来证明我们的解码器正确。
  若要对外声明数据由我们自行采集，应先执行第 5.4 节的全量回填，使这一陈述在字面上成立；
  在此之前，准确的表述是「我们的采集管线可从一手来源完整重建该数据，并已逐字段验证，
  且已采集了快照没有的新数据」。

## 10. 已知限制与后续工作

1. **全量回填尚未执行**。目前已验证正确性、已抓取 2026-04-28 之后的增量；历史全量回填需
   1 至 3 天连续运行（第 5.4 节），或走 BigQuery 路径（第 5.5 节）。
2. **`0xe3333700...` 场馆待判定**（第 6.3 节），约占新协议成交的 1%。
3. **`category_refined` 不复现**。公开数据集的该字段是作者自行归类的结果，非链上或 API 原生
   字段。我们保留 Gamma 的原始 `category` 与 CLOB 的 `tags`，不做主观归类。
4. **洗量与铸造识别未实现**。CTF 层的 split / merge / redemption 已可抓取，
   基于它们的洗量识别（自成交、铸造后立即卖出等）留待后续。
5. **多结果市场的 `p_event` 无定义**（第 4.3 节）。若后续需要，可按 negRisk 组内归一化处理。
