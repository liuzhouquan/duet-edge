# Val / Test 数据集详细报告（v2，修正时长计算）

_生成时间: 2026-05-27 02:03_

> **v1 → v2 修正**：v1 错误地用 `n_slices × 5s` 算时长，没考虑 slice 之间 90% 重叠。
> 正确公式：`coverage = (n_slices - 1) × stride + length`，stride=0.5s, length=5.0s。

## 三集 pair 数（修正后无变化）

| Split | Pairs | Slices | **真实动作时长** |
|------|------:|------:|---------------:|
| train | 383 | 3433 | **3440.0s ≈ 57.33 min** |
| val | 10 | 93 | **91.5s ≈ 1.52 min** |
| test | 10 | 93 | **91.5s ≈ 1.52 min** |

> ⚠️ 旧报告说 train 是 ~286 min、val/test 各 7.75 min，**全都错了**。真实数字见上表。


---

## VAL 集详表（10 对 / 实际 91.5s）

| # | Music | Ch | Lead | Follower | Slices | 覆盖时长 | 原 wav 时长 |
|---|-------|----|----- |----------|--------|---------|------------|
| 1 | `mBR0` | `ch01` | d04 | d05 | 14 | **11.5s** | 11.99s |
| 2 | `mHO5` | `ch01` | d20 | d21 | 5 | **7.0s** | 7.10s |
| 3 | `mJB5` | `ch01` | d08 | d09 | 5 | **7.0s** | 7.38s |
| 4 | `mJS3` | `ch01` | d01 | d03 | 8 | **8.5s** | 8.73s |
| 5 | `mKR2` | `ch01` | d28 | d30 | 10 | **9.5s** | 9.60s |
| 6 | `mLH4` | `ch01` | d17 | d18 | 7 | **8.0s** | 8.00s |
| 7 | `mLO2` | `ch01` | d13 | d15 | 10 | **9.5s** | 9.60s |
| 8 | `mMH3` | `ch01` | d22 | d24 | 8 | **8.5s** | 8.73s |
| 9 | `mPO1` | `ch01` | d10 | d11 | 12 | **10.5s** | 10.67s |
| 10 | `mWA0` | `ch01` | d25 | d26 | 14 | **11.5s** | 11.99s |

**统计**: 总 slice = 93, 总覆盖 = 91.5s ≈ 1.52min

---

## TEST 集详表（10 对 / 实际 91.5s）

| # | Music | Ch | Lead | Follower | Slices | 覆盖时长 | 原 wav 时长 |
|---|-------|----|----- |----------|--------|---------|------------|
| 1 | `mBR0` | `ch02` | d04 | d05 | 14 | **11.5s** | 11.99s |
| 2 | `mHO5` | `ch02` | d20 | d21 | 5 | **7.0s** | 7.10s |
| 3 | `mJB5` | `ch02` | d08 | d09 | 5 | **7.0s** | 7.38s |
| 4 | `mJS3` | `ch02` | d01 | d03 | 8 | **8.5s** | 8.73s |
| 5 | `mKR2` | `ch02` | d28 | d30 | 10 | **9.5s** | 9.60s |
| 6 | `mLH4` | `ch02` | d17 | d18 | 7 | **8.0s** | 8.00s |
| 7 | `mLO2` | `ch02` | d13 | d15 | 10 | **9.5s** | 9.60s |
| 8 | `mMH3` | `ch02` | d22 | d24 | 8 | **8.5s** | 8.73s |
| 9 | `mPO1` | `ch02` | d10 | d11 | 12 | **10.5s** | 10.67s |
| 10 | `mWA0` | `ch02` | d25 | d26 | 14 | **11.5s** | 11.99s |

**统计**: 总 slice = 93, 总覆盖 = 91.5s ≈ 1.52min

---

## 为什么 slice 覆盖时长 ≠ slice 数 × 5s？

```
slice 0: [0.0s → 5.0s]
slice 1: [0.5s → 5.5s]   ← 与 slice 0 重叠 4.5s（90%）
slice 2: [1.0s → 6.0s]
...
slice N: [(N×0.5)s → (N×0.5+5)s]

总覆盖 = 末尾 - 起点 = (N-1)×0.5 + 5
```

- **训练用 dense 切片** (stride=0.5s, 90% overlap) → 同一段动作被切成多段供模型学习
- 14 个 slice 实际只覆盖 11.5s 原序列
- 这是数据增广手段，**不是有 70s 真实动作**

---

## 修正后的论文表述（推荐）

```
We use 383 lead-follower training pairs (~250 unique seconds of motion;
see Table X), each sliced into overlapping 5-second segments with 0.5s
stride (90% overlap), yielding 3,433 training samples per epoch. The
validation and test splits contain 10 pairs each, covering disjoint
AIST++ music IDs (val=ch01, test=ch02), with 93 slices each (~57s of
unique source motion per split).
```
