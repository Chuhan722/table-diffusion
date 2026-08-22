# Issue #53：P=6 两数据归档 smoke 结果

> 结论：`real_dataset_smoke_completed_not_formal_validation`。
>
> `test_300x10` 与 `nltcs` 均由 B/`early_stopped` 在 C=6000 前正常结束，说明当前 P=6 自然工作量
> 早停已经接通真实生成链路；但单 seed smoke 既不是收敛证明，也不是生产质量或计算优势验收。

## 1. 冻结身份与边界

```text
Git commit
  d220ba4d04606c4ed99c89d98da314a31f1d0d71

protocol SHA-256
  3b593ce71c8b4bd147b836dd03986d4e64d27bb782a57d0a9ac5759baf805c17

历史描述性基线 SHA-256
  51aff5414eb15c9cfdda496dc1549c6fba7216043159bd377be429fb11443f64

正式输出
  outputs/issue53_p6_dataset_smoke_seed200/

report SHA-256
  cd1e10f9034f63ec4a4caed39370e1b7bb802720e41c9ed5dae8716667ee90fa
```

runner 固定为 `test_300x10 -> nltcs` 串行、seed 200、`rho=0.01`、`P=6 natural-work ticks`、
`n_rounds=candidate_budget=6000`、relative residual geometry floor 8、scale-invariant fixed alpha 16、
Gibbs sweeps 0、无 gate、无 retry、输出 terminal current。命令行没有科学参数覆盖，运行中没有调参或
重跑。

自然工作量定义为：

```text
cumulative applied participating rows / N
```

这里 C=6000 是 raw rounds/candidates 的绝对安全上限，期望最多约 60 natural-work ticks。P=6 只表示
连续六个 natural-work ticks 没有出现严格更低的历史 best loss 时早停。在当前 `rho=0.01` 下，一个
tick 大约需要 100 raw rounds，所以一次未被刷新耐心约为 600 轮；一旦出现新最好 loss，六 tick 耐心
从该进展点重新累计。因此 600 从来不是总轮数上限。

本次只运行内层无门控生成过程，不做外层选择/测量、不加噪，不使用在线 L1，不访问原始 reference
table，也不消耗隐私预算。历史同 seed、固定 2000 轮结果只作描述性参照，不是验收阈值。

## 2. 正式结果

| 数据 | 规模 | 设备 | 停止原因 | rounds / cap | work | terminal loss | terminal normalized L1 | 墙钟 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `test_300x10` | 300 行、50 queries | NumPy/CPU | `early_stopped` | 2128 / 6000 | 21.0000 | 49.5 | 0.0036666667 | 111.8 s |
| `nltcs` | 16181 行、1001 queries | RTX A6000 | `early_stopped` | 2500 / 6000 | 25.0062 | 17649.5 | 0.0002645522 | 1113.2 s |

两个结果的 `inner_complete=true`、`output_table_identity=terminal_current`，没有资源上限停止。历史 best
loss 仅作停止观察器诊断，分别为 19.0 和 12742.5；正式返回没有回滚到 best table，所以它们不能替代
上表 terminal loss。

## 3. 与同 seed、2000 轮历史结果的描述性对照

| 数据 | 历史 terminal loss | 新旧 loss 差 | 历史 normalized L1 | 新旧 L1 差 | raw rounds 变化 |
|---|---:|---:|---:|---:|---:|
| `test_300x10` | 48.5 | +1.0 | 0.0036666667 | 0 | +128（+6.4%） |
| `nltcs` | 17467.0 | +182.5 | 0.0002652313 | -0.0000006791 | +500（+25.0%） |

这组单 seed 结果是混合而非单向：外部 normalized L1 在 test 上完全相同、在 nltcs 上极小改善；内层
terminal squared loss 在两者上都略高。同时 P=6 比旧的任意 2000 轮终点多运行，而不是节省 raw
rounds。loss 与 L1 是不同指标，无门控 terminal current 还会随机波动，因此不能从其中任一列单独
推出“更好”或“收敛”。

## 4. 可以与不可以得出的结论

可以得出：

1. 当前 A/B/C 状态机和 P=6 natural-work 早停可在两个目标数据链路上完整运行。
2. 两条轨迹都由 B 在 C 前停止，实际只使用上限的 35.47% 和 41.67% raw rounds。
3. 正式输出保持无门控 terminal current 身份，早停观察器没有把返回表换成历史最好表。
4. 在本次有限 smoke 中，提前停止没有造成 normalized L1 的明显恶化。

不可以得出：

1. 算法、loss 或 L1 已经收敛；
2. P=6 对其他 seed、rho、数据或未来带噪过程同样合适；
3. P=6 相对固定 2000 轮节省计算——本次两条恰好都运行得更久；
4. nltcs 的极小 L1 改善具有统计意义，或应据此调整参数；
5. 本结果完成了外层 DP、隐私会计或带噪停止设计。

本 smoke 达到了“归档 PR 前在 test 与 nltcs 各跑一次、确认不会只能依赖 C 才停止”的目的。下一步
可归档当前 PR；不应因这两个已见结果自动调 P/rho、重跑 seed 或扩大结论。

## 5. 产物哈希与环境

```text
nltcs/result.json
  89be9353f615c3c7617e4391cda2ea4a138f27aeddedd0a084950afcd54f7537
nltcs/terminal_current.csv
  efbd686231e770d5e54a1caff6b00314a43dff58100845195a047775958da212
report.json
  cd1e10f9034f63ec4a4caed39370e1b7bb802720e41c9ed5dae8716667ee90fa
test_300x10/result.json
  48163b2e3805fc33c06985be0f6530023ea2dbff44ccd15b82a4a810295c3839
test_300x10/terminal_current.csv
  137091a7246205de5fb9780779d945f0be1204993c1fb66b383552c806d9247b
```

远端环境为 Python 3.11.15、NumPy 2.4.6、Pandas 3.0.3、PyTorch 2.13.0+cu130、CUDA runtime
13.0、RTX A6000。运行前冻结提交的工作树包含 untracked 在内均干净；运行后代码树仍干净，GPU 0
已释放。下载到本地的五个文件与远端逐文件 SHA-256 完全一致。
