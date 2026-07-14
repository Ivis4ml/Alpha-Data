# Polymarket 市场 -> 国内期货品种映射表（事前注册，v1.1）

本表在运行任何回归之前注册（研究规范 §3.2）。市场选取规则、方向先验与聚合方式
一经注册不再回溯修改；后续修订须另立版本并注明原因。

**v1.1 修订（2026-07-13，多智能体对抗审查确认的缺陷修正）**：

1. 主题一新增升级规则 `ceasefire-end` / `ceasefire-broken` /
   `ceasefire-has-been-broken`（先于降级规则匹配）：停火"结束 / 破裂"是升级事件，
   v1.0 被 `us-x-iran-ceasefire` 子串误伤反号（8 个市场，窗口成交约 $4.1M）。
2. 主题六泛化模式 `tariff` 增加排除子串 `greenland`：格陵兰关税与中美贸易机制
   无关（2 个市场）。
3. **"事前注册"的适用范围澄清**：事前注册的是规则（主题、方向、门槛数值）；
   流动性准入改为**时点化**（`admit_ts` = 窗口内累计成交额首次达到门槛的时刻，
   之前的信号置 NaN），v1.0 的全窗口累计筛选含样本内信息。
4. 预测性检验的 BH-FDR 改用 HAC p 值（v1.0 误用 iid Pearson p 值，对重叠前向
   收益反保守）；`n < 40` 的检验加小样本标记。

- 数据窗口：期货 2026-01-05 起，Polymarket `daily_aligned` 至 2026-04-28 止，
  有效重叠约 75 个交易日。样本内 Polymarket 活动高度集中于美伊冲突主题。
- 选取范围：`category_refined` 属于 Politics / Finance / Other / Sci-Tech；
  Sports / Crypto / Culture / Price Action（加密价格）与商品无干净对应，整类排除。
- 流动性门槛：窗口内成交名义额 `usdc_win >= $100,000`（可调，稳健性用 $500,000）。
- 排除规则（新奇 / 元市场，无经济内容）：slug 含 `-say-`、`wear-a-`、`-mention`、
  `odds-of-` 的市场。
- 符号约定：市场方向先验 = σ(市场语义) × m(品种)。σ = +1 表示"事件概率上升 =
  主题风险 / 价格上行"；m 为主题对品种的作用方向。

## 主题一：中东冲突（mideast_conflict）

样本窗口内流动性最高的主题（美伊军事冲突episode）。机制：供给风险溢价（原油）+
避险（黄金）。品种：SC(+1)、AU(+1)。

| σ | slug 模式 |
|---|---|
| +1（升级） | `us-strikes-iran`、`us-forces-enter-iran`、`israel-strikes-iran`、`iran-strike-israel`、`will-the-us-invade-iran`、`iranian-regime-fall`、`khamenei-out`、`close-the-strait-of-hormuz`、`kharg-island`、`ground-offensive-in-lebanon`、`us-forces-seize-another-oil-tanker` |
| −1（降级） | `us-x-iran-ceasefire`、`permanent-peace-deal`、`us-x-iran-meeting`、`military-action-against-iran-ends`、`strait-of-hormuz-traffic-returns-to-normal`、`israel-x-hezbollah-ceasefire` |

注：`iranian-regime-fall` / `khamenei-out` 的油价方向存在两可（政权更迭可能带来
制裁解除），按样本期语境（军事冲突中的政权崩溃风险）注册为 +1，列入稳健性剔除项。
`netanyahu-out` 方向不可判定，排除。

## 主题二：原油价格直连（oil_price）

Polymarket 上直接押注 CL/WTI 期货价位的市场（`will-crude-oil-cl-hit-high-X` 等）。
机制上这是国际油价的镜像，**最接近国际基准的代理**：该主题对 SC 的"预测力"主要
检验国际传导速度，解释时不得声称独立于国际市场的信息含量（研究规范 §0 约束 1）。
品种：SC(+1)。

| σ | slug 模式 |
|---|---|
| +1 | `crude-oil-cl-hit-high`、`wti-crude-oil-wti-hit-high` |
| −1 | `crude-oil-cl-hit-low`、`wti-crude-oil-wti-hit-low` |

## 主题三：贵金属价格直连（metal_price）

同上，GC/SI 价位市场。品种：`gold-gc` -> AU(+1)；`silver-si` -> AG(+1)；
`hit-high` σ=+1，`hit-low` σ=−1。国际基准代理性质同主题二。

## 主题四：美联储政策（fed_policy）

机制：实际利率与美元通道。品种：AU(+1)、AG(+1)、CU(+1)（降息利多）。

| σ | slug 模式 |
|---|---|
| +1（宽松） | `fed-rate-cut` |
| −1（紧缩） | `fed-rate-hike` |
| +1（制度风险，探索性） | `powell-out-as-fed-chair`、`try-to-fire-powell`、`sue-powell`、`powell-federally-charged` |

注：Powell 去职类市场同时含"宽松预期"与"央行独立性风险"两种利多黄金的机制，
无法区分，标注探索性。Fed 主席提名类市场（`warsh-confirmed`、`judy-shelton`）
候选人立场与市场解读方向不可事前判定，排除。本主题对应 H1 偏零假设：控制国际
金价后预期无增量预测力。

## 主题五：俄乌冲突（russia_ukraine）

机制：制裁 / 解除制裁经由原油供给，避险经由黄金。品种：SC(+1)、AU(+1)。

| σ | slug 模式 |
|---|---|
| +1（升级） | `russia-strike-`、`russia-capture`、`russia-invade-a-nato`、`ukraine-strikes-another-tanker` |
| −1（降级） | `russia-x-ukraine-ceasefire`、`zelenskyy-talk-to-putin` |

## 主题六：中美贸易（us_china_trade，探索性 / H4）

机制：关税升级 -> 中国进口美豆减少 -> 国内蛋白粕供给收紧 -> M 涨；棉花同向；
增长与工业金属受损 -> CU、I 跌。降级（互访、和谈、协议）方向相反。
品种：M(+1)、CF(+1)、CU(−1)、I(−1)（乘 σ）。

| σ | slug 模式 |
|---|---|
| +1（升级） | `tariff`（含 `rule-in-favor-of-trumps-tariffs`；排除 `refund-tariffs`） |
| −1（降级） | `trump-visit-china`、`trump-talk-to-xi`、`court-force-trump-to-refund-tariffs` |

注：样本窗口内纯关税市场流动性单薄（6 个，最大 $1.7M），主题以中美互动市场为主，
异质性强，整体标注探索性（FDR 控制）。

## 主题七：台海风险（taiwan_risk，探索性）

机制：地缘风险避险（AU 涨）+ 中国资产风险溢价（股指期货跌）。
品种：AU(+1)、IF(−1)。

| σ | slug 模式 |
|---|---|
| +1 | `china-invade-taiwan`、`blockade-taiwan`、`china-x-taiwan-military-clash` |

## 主题八：美国政府停摆（us_shutdown，探索性）

机制：财政不确定性避险。品种：AU(+1)。

| σ | slug 模式 |
|---|---|
| +1 | `government-shutdown`、`dhs-shutdown` |

## 聚合与信号定义

- 市场级信号：`alpha_data.polymarket.cn_features.window_signals`，15 分钟量加权
  中位数聚合价（买卖两向均值去 bounce），logit（截断 [0.02, 0.98]）innovation 按
  品种交易时段拆为 `s_night / s_gap / s_day`；`p_age > 120` 分钟置 NaN；结算后置 NaN。
- 主题级聚合：对主题内市场按 `sqrt(usdc_win)` 加权求 σ 加权平均（等权作为稳健性）。
  权重使用全窗口流动性为静态量，含轻微的样本内信息，作为局限记录。
- 近结算市场：临近结算的概率变化有同义反复风险（研究规范 v2），结算前 3 个
  交易日的观测在稳健性检验中剔除。
