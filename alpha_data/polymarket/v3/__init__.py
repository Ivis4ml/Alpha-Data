"""v3 研究框架实现（research/Polymarket_中国期货预测_识别与统计功效_v3.pdf）。

模块对应框架层级：

- :mod:`tape`：Layer 0 的可得近似——HF ``daily_aligned`` 与自爬链上成交的统一 tape。
- :mod:`latent`：Layer 1——logit 状态空间滤波，微观结构噪声建模与标准化 innovation。
- :mod:`families`：市场族识别（日期阶梯 / 价格阈值）与事件类型标注（门控输入）。
- :mod:`coherence`：Layer 2——族内单调软投影与 narrative tension。
- :mod:`geometry`：Layer 3——日期族 hazard 恢复与价格阈值族分布特征。
- :mod:`story`：Layer 3——story 因子聚合、平台公共因子与正交化。
- :mod:`gates`：Event Eligibility Gate 与 Power Gate（surprise / MDE / 降级决策）。
- :mod:`impact`：Layer 4——门控后的 category-level 影响系数与 sign-only 回退。

数据现实与框架的偏离（诚实记录，详见 docs 报告）：

1. 无历史订单簿快照，Layer 0/1 的 effective book / microprice 不可得；以逐笔成交的
   买卖两向聚合价差作有效点差代理，观测噪声由桶内成交结构（点差代理、成交额、笔数、
   时效）建模。
2. 市场条款无编辑历史（CLOB 元数据为当前快照），point-in-time 性只对成交与准入时刻
   （``admit_ts``）成立。
3. 数据隔离协议（框架 §11.2）保持：Layer 1-3 超参数只用 Polymarket 内部目标选择，
   期货数据仅在 Layer 5 出现。
"""
