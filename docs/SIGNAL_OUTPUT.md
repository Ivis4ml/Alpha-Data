# 信号结构化输出说明（规定格式的生成、位置与命令）

本文说明"已发现信号"的结构化输出如何生成、存放在哪里、如何复算，
以及逐日影子记录框架的用法。全部数字由脚本从产物 parquet 计算，
不存在手抄；正文表格在报告构建时读取同一 JSON 渲染，两处必然一致。

## 一、规定格式的输出（JSON）

**产物位置**：`docs/signal_summary.json`（受 Git 跟踪，8 条记录）。

**生成命令**：

```bash
.venv/bin/python scripts/export_signal_summary.py    # 计算并写 JSON
.venv/bin/python scripts/build_thesis_html.py        # 渲染进报告正文表 2/表 3
```

**每条记录的字段与来源**（规定的 10 个字段 + 补充字段）：

| 字段 | 来源与口径 |
|---|---|
| 交易品种 | SC / AU / AG / CU / M |
| 信号定义 | 文字定义，内嵌审计限定（复合性质、门槛结论），防断章引用 |
| 公式 | 显式公式（C8 滚动 z、E1 稳健阈值、事件跳检出与折叠） |
| 信号频率 | 连续 = 逐分钟；离散 = 事件型触发 |
| 信号月次数 | 连续 = 有效信号分钟数/月；离散 = 触发次数/月（125 交易日按 21 折月） |
| 连续信号 IC（1/3/10 分钟） | `defense/ic_table.parquet` 的 pooled RankIC（描述性；推断以报告 §12.14-12.15 为准） |
| 离散信号（取 1）后 1/3/10 分钟平均收益 | 面板上取 1（上行触发）后前向收益均值，bp 毛值 |
| 取值均值/中位数/标准差/偏度/峰度 | 连续信号取值统计（面板全样本；偏度峰度按非零值口径） |
| 取值 valuecount / 频率 | 离散信号逐值计数与占比、触发频率 |
| 直方图 | `docs/figures/f_signal_summary_hist.png`（面板号在字段内） |
| 品种全区间平均收益 | 样本期收对收累计（剔换月日）+ 无条件 1/3/10 分钟均值，作条件均值的对照基准 |

**输入数据**（脚本读取，均为既有产物）：
`data/cn_futures/analysis/v3/defense/panel_{品种}.parquet`（分钟面板与
前向收益）、同目录 `ic_table.parquet`、`analysis/v3/jump/jumps.parquet`
（事件跳）、`data/cn_futures/daily/{品种}.parquet`（基准收益）。

**全信号库档案**（8 条头部记录之外的完整登记）：
公式与机制 = 报告附录三 §3（40 信号）与 §5b（J 族）；统计量 =
`defense/signal_stats.parquet`（频率/分位数）+
`defense/signal_stats_ext.parquet`（峰度/valuecount，47 信号 × 5 品种）；
直方图 = 报告附录三 §4.2（N/C/K/X 四族 + J 族，共 47 张）。

## 二、逐日影子记录框架（冻结口径，追加不回改）

**脚本**：`scripts/shadow_daily.py`；**档案**：`shadow/records.jsonl`
（每交易日一行 JSON，追加写入；已存在的日期拒绝重写）。

```bash
# 指定交易日（先决条件：该日的 tape 与信号层已重建）
.venv/bin/python scripts/shadow_daily.py --date 2026-07-21
# 缺省取信号层最新可得交易日；可选步骤：
.venv/bin/python scripts/shadow_daily.py --fetch-curve   # 先补抓交易所曲线
```

每行记录的内容（全部只用 ≤ 当日开盘的信息，口径由
`freeze/manifest.json` 哈希锁定）：

- `theme_z_pre`：八主题开盘前信念创新 z；
- `r3_tail_flag`：活跃度、展开 90 分位阈值（PIT）与旗标状态；
- `cross_section`：E1 暴露度得分头尾五品种、当日截面吸收 IC；
- `r2_sc_residual`：SC 对 Brent/USDCNH 展开窗残差（bp）；
- `r4_curve`：SC/AU/FU 近远月价差变化（bp，同合约对）；
- `jumps_today`：SC 主题跳与 M 孤立跳事件数；
- `outcomes`：当日收盘后结果（SC/M 收对收，bp）；
- `sample`：2026-07-18 前为 `frozen_demo`（示范），之后为
  `untouched`（正式影子记录，参与一次性裁决）。

**运行前置**（新交易日）：增量爬链与信号层重建
`crawl_polymarket_chain.py → v3_build_tape.py → v3_cross_section.py`
（Polymarket 侧，链上数据即时可得）；期货日线更新依赖行情源交付
（当前唯一未自动化的外部环节）。

## 三、裁决与投产（不因实施而改变）

影子档案只做记录，不做交易。冻结候选的裁决时点与门槛见
`freeze/manifest.json` 与报告 §12.15-12.16：SC 跳格 2026-12-31 或新增
95 个交易日先到者，R1-R4 与旗标 2027-01-31，各候选仅一次预约定检验。
