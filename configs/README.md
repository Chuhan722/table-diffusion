# configs/ 冻结考卷与工作负载文件说明

本目录下的大型 JSON 考卷文件全部由**确定性生成器**产出并被协议 SHA-256
锚定：审查者无需信任文本本身，用下述命令即可从公开数据逐字节重建并对账。

## all-2way 全家族考卷（`<dataset>/all2way_issue53_v1.json`）

- 生成器：`scripts/gen_all2way_queries.py`（零随机、纯纸面枚举：按列序全部
  C(M,2) 属性对 × 完整 2×2 列联表 4 cell；目标值为数据集精确计数）。
- 逐字节重建验证（不写盘，与在库文件逐字比较，任何差异即报错）：

  ```bash
  python scripts/gen_all2way_queries.py --verify-existing
  ```

  预期输出（2026-09-06 冻结身份）：

  ```text
  verified configs/nltcs/all2way_issue53_v1.json: sha256 = 5821fa4e8dc11e499c468ef618843656f51052ae91a3a2e6e6b338cde9c0552d
  旧池 double 对账通过: 479/479
  verified configs/plants/all2way_issue53_v1.json: sha256 = 9dc37994e912ce52c5859a122150144cad487c06d71de6bb0f8bd8a4a584409a
  旧池 double 对账通过: 460/460
  ```

- 生成时强制自检：每对 4 cell 和 == N；查询指纹全体唯一；与
  `measured_1000query.json` 旧池 double 逐指纹对账（目标值必须逐位一致）。

## 其他冻结工作负载（同机制）

- `<dataset>/heldout_issue53_v1.json`：`scripts/build_issue53_heldout_workloads.py
  --verify-existing`（哈希排序确定性选择，与 measured 无交集）。
- `<dataset>/halfspace_issue53_v1.json`：`scripts/build_issue53_halfspace_workload.py
  --verify-existing`（冻结种子确定性构造）。

重建输入（`data/<dataset>/<dataset>.csv`）已入库；plants 数据来源与授权
说明见 `data/plants/README.md`（nltcs 来源同一基准系列，见该 README 引用）。
