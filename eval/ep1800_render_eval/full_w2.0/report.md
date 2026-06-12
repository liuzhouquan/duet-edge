# Inference config: `full_w2.0`

- Mode: **full**  (full / lead / music)
- CFG weights: w_music = **2.0**,  w_lead = **2.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.9149 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9917 ± 0.0061 |
| LMA Body component (mean per-metric r) | 0.2094 |
| LMA Effort component (mean per-metric r) | 0.1435 |
| LMA Space component (mean per-metric r) | 0.1295 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.209 ± 0.245 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.235 ± 0.239 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.143 ± 0.316 |
| 4 | `f4_hands_to_head` *(Body)* | 0.243 ± 0.313 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.322 ± 0.353 |
| 6 | `f6_pelvis_height` *(Body)* | 0.208 ± 0.346 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.131 ± 0.385 |
| 8 | `f8_gait_distance` *(Body)* | 0.183 ± 0.381 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.112 ± 0.152 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.294 ± 0.189 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.216 ± 0.176 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.079 ± 0.035 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.141 ± 0.065 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.095 ± 0.060 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.068 ± 0.028 |
| 16 | `f30_distance_covered` *(Space)* | 0.102 ± 0.146 |
| 17 | `f31_area_covered` *(Space)* | 0.157 ± 0.138 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9952 | 0.363 | 0.131 | 0.181 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9935 | 0.009 | 0.048 | 0.195 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9974 | 0.426 | 0.206 | 0.078 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9838 | 0.090 | 0.102 | 0.004 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9788 | 0.107 | 0.222 | 0.267 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9909 | 0.588 | 0.217 | 0.145 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9960 | 0.096 | 0.271 | 0.336 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9945 | 0.182 | 0.099 | 0.065 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9988 | 0.086 | 0.059 | 0.050 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9877 | 0.149 | 0.081 | -0.026 |
