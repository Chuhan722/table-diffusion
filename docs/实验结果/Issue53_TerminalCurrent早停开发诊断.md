# Issue #53：terminal-current 早停后继续收益开发诊断

> 分类：`development_known_trajectories_not_validation`。
>
> 本诊断继续使用已经看过的六条人工轨迹和固定六 tick 基线：主体检查“B 停止后相对增加不同自然
> 工作量，terminal current 质量怎样变化”，并在状态机接线后追加在线/离线实现一致性审计。它没有
> 增加 seed、比较 patience、选择 C 或形成正式通过/失败结论，不能验证或调定生产参数。

## 1. 为什么改成相对继续收益曲线

旧版把 B 的 terminal current 与完整已知轨迹的最后 current 状态比较。该终点会随完整轨迹长度变化，
而无门控随机过程的 current loss 本身会上下波动，因此它不能作为 ground truth，也不能用来选择 C。

本版固定：

```text
work_tick = floor(cumulative applied participating rows / N)
P = 6 work ticks（仅 development 基线）

一个 tick 内出现严格更低历史 best loss → progress
连续 P 个 tick 没有 progress              → B / early_stopped

B 正式输出 = tau 时的 terminal current table/current loss
```

一旦 B 在自然工作量 `tau` 触发，正式输出身份立即冻结。只有离线影子副本继续运行，并观察：

```text
tau + P
tau + 2P
tau + 4P
```

每个检查点取第一个实际达到或越过目标工作量的 post-round current 状态，不插值、不伪造中间表。
已知轨迹不够长时记为 `right_censored_by_known_trace_horizon`，不得拿轨迹末尾状态补齐。原 40/160 轮
只是历史 development 轨迹可观察范围，不是 C，也不是质量终点。

normalized L1 只在 B 输出和各影子 current 表身份固定后离线计算，绝不反馈给 B。历史 best 只作
progress/诊断，任何检查点都不返回 best 表。

## 2. 实现与审计

入口仍为：

```text
scripts/analyze_issue53_terminal_early_stop_development.py
```

新增纯函数只读取 applied participating-row counts，定位 `tau+P/2P/4P` 的第一个真实状态。对每个
实际可观察检查点，脚本使用相同 seed、固定参数和 `horizon_invariant=True` 重放对应前缀，并核对：

- current-state metrics 与 transition clocks 等于完整轨迹同一前缀；
- terminal table SHA-256 等于该 state 的 current-table SHA-256；
- 从 terminal 表独立重算的 squared loss 与 normalized L1 等于该 state 的权威 current 指标；
- 额外工作量、raw rounds 与 candidate evaluations 均从真实状态身份计算；
- 右删失点的 state/table/loss/L1 保持为空，不替换为最后可用状态。

所有断言通过。旧轨迹没有逐状态累计墙钟，独立前缀重放的耗时又不能相减作为同一运行的续跑成本，
因此本诊断明确报告 `wall_clock_delta_available=false`，没有伪造墙钟差值。

## 3. 逐条 terminal-current 结果

下表中的每格为 `current loss / 离线 normalized L1`；`—` 表示右删失，不是使用轨迹末尾值。

| seed | rho | B stop work | B 输出 | `+6 work` | `+12 work` | `+24 work` |
|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 1.0 | 12.0000 | 12.0 / 0.104167 | 19.5 / 0.156250 | 20.0 / 0.125000 | 4.0 / 0.062500 |
| 20260817 | 1.0 | 20.0000 | 7.0 / 0.083333 | 4.0 / 0.062500 | 8.5 / 0.093750 | — |
| 20260818 | 1.0 | 8.0000 | 7.5 / 0.072917 | 13.5 / 0.093750 | 12.5 / 0.093750 | 5.0 / 0.062500 |
| 20260816 | 0.25 | 7.0625 | 4.0 / 0.062500 | 8.0 / 0.083333 | 4.0 / 0.062500 | 2.5 / 0.052083 |
| 20260817 | 0.25 | 16.0000 | 1.0 / 0.020833 | 2.5 / 0.031250 | 5.5 / 0.072917 | — |

第六条 `seed=20260818/rho=0.25` 在 work=2.8125 先由 A 精确命中，因此不是 B 后继续收益的对象。

## 4. 聚合曲线

差值统一定义为：

```text
影子续跑检查点 current 指标 - B 正式输出 current 指标
```

所以负数表示继续运行后的该张 terminal current 更好，正数表示更差。

| 相对检查点 | 可观察 / 右删失 | loss：继续更好/相同/更差 | loss 平均/中位差 | L1：继续更好/相同/更差 | L1 平均/中位差 |
|---:|---:|---:|---:|---:|---:|
| `+6 work` | 5 / 0 | 1 / 0 / 4 | +3.2 / +4.0 | 1 / 0 / 4 | +0.016667 / +0.020833 |
| `+12 work` | 5 / 0 | 0 / 1 / 4 | +3.8 / +4.5 | 0 / 1 / 4 | +0.020833 / +0.020833 |
| `+24 work` | 3 / 2 | 3 / 0 / 0 | -4.0 / -2.5 | 3 / 0 / 0 | -0.020833 / -0.010417 |

实际平均额外 normalized work 分别为 6.0、12.0625、24.0；第二档略高于 12，是因为只选择真实
post-round 状态。对应的平均额外 raw rounds/candidate evaluations 分别为 13.4、28、49。五条 B
轨迹在各自可观察检查点前均未达到零残差 A。

## 5. 可以得出的结论

1. 新诊断协议可复现：B 的正式 terminal 输出先固定，影子续跑只用于离线评价，不改变输出身份。
2. 继续运行的 terminal current 质量不是单调变化。`+6` 和 `+12` 时大多数轨迹反而更差；更远的
   `+24` 在三个可观察样本上都更好。
3. `+24` 的 3/3 不能解释为“多跑 24 一定更好”：另外两条较晚触发 B 的轨迹恰好被旧 horizon
   右删失，样本既小又不完整，存在明显的可观察性偏差。
4. 这组结果直接支持取消单一 C 终点比较。若只挑 `+6`、`+12` 或 `+24` 中任何一个终点，都可能
   得到不同甚至相反的判断。
5. 本结果既不支持把 P=6 定为生产早停，也不自动否决它。它只说明正式验收必须用结果前冻结的
   多检查点质量—计算口径和未见 seed，不能看完曲线后挑对结论有利的检查点。

固定六 tick 触发时 5/5 current loss 高于当时历史 best、平均差 3.8 的旧 terminal-output 风险仍然
成立，但解决办法不是返回 best、回滚或等待有利 current 状态，而是如实评价 terminal-output 的
计算—质量取舍。

## 6. 在线接线一致性验证

状态机接入 `run_evolution` 后，使用完全相同的六条已知人工轨迹做了一次仅限接线的在线核对。每条
先按旧方式生成完整轨迹并离线回放 A/B，再在相同 seed、参数和旧 horizon 下实际启用
`inner_early_stopping_patience_ticks=6`。逐条要求在线结果严格等于离线决定，不使用数值容差替代表/
状态身份。

| seed | rho | 离线原因 | 在线原因 | stop state | stop work | terminal loss |
|---:|---:|---|---|---:|---:|---:|
| 20260816 | 1.0 | B | B | 12 | 12.0000 | 12.0 |
| 20260817 | 1.0 | B | B | 20 | 20.0000 | 7.0 |
| 20260818 | 1.0 | B | B | 8 | 8.0000 | 7.5 |
| 20260816 | 0.25 | B | B | 27 | 7.0625 | 4.0 |
| 20260817 | 0.25 | B | B | 64 | 16.0000 | 1.0 |
| 20260818 | 0.25 | A | A | 11 | 2.8125 | 0.0 |

六条全部通过以下严格核对：

- 在线 termination reason、stop state、normalized work 与旧离线回放相同；
- 主返回表、`final_table` 与完整轨迹对应 stop state 的 current-table SHA-256 相同；
- current metrics、transition clocks、accept history、candidate evaluations 均等于完整轨迹前缀；
- 停止后的主 RNG SHA-256 等于完整轨迹同一状态，说明观察器没有改变随机前缀；
- 5 条由 B 返回、1 条由 A 返回、0 条被 C 抢先截断；5 条 B 仍全部返回高于历史 best loss 的
  terminal current，确认接线没有偷偷回到 best 表。

该结果只证明“在线实现复现既有离线定义”，不增加关于 P=6 是否合适、C 应设多大、输出质量是否
通过或算法是否收敛的证据。分类固定为
`development_known_trajectories_wiring_consistency_only`。

## 7. 当前边界与下一步

当前不能根据这五条已知 B 轨迹选择 P、质量容限或 C。在线接线已经完成，下一步应先讨论并在读取
新结果前冻结：

```text
允许的 terminal current loss / 离线 true-L1 退化容限
至少要求的 normalized work / candidates / 墙钟收益
正式相对检查点与右删失处理
一个全局 patience 候选
全新未见 seed/family 协议
```

这些内容确认前，不读取新验证数据、不运行真实数据，也不接外层 DP。当前在线接线通过不等于正式
质量验证通过。

复现纯逻辑与六条在线接线契约（15 项）：

```text
PYTHONPATH=src /home/chuhan/projects/.issue49-tools/venv/bin/python \
  -m pytest -q tests/test_issue53_terminal_early_stop_development.py
```

复现 development 报告：

```text
PYTHONPATH=src /home/chuhan/projects/.issue49-tools/venv/bin/python \
  scripts/analyze_issue53_terminal_early_stop_development.py
```
