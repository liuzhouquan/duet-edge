# Inference config: `lead_w2.0`

- Mode: **lead**  (full / lead / music)
- CFG weights: w_music = **0.0**,  w_lead = **2.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.3607 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9913 ± 0.0072 |
| LMA Body component (mean per-metric r) | 0.2334 |
| LMA Effort component (mean per-metric r) | 0.1592 |
| LMA Space component (mean per-metric r) | 0.1991 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.289 ± 0.231 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.263 ± 0.253 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.105 ± 0.339 |
| 4 | `f4_hands_to_head` *(Body)* | 0.304 ± 0.350 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.297 ± 0.349 |
| 6 | `f6_pelvis_height` *(Body)* | 0.291 ± 0.248 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.165 ± 0.334 |
| 8 | `f8_gait_distance` *(Body)* | 0.154 ± 0.383 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.155 ± 0.146 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.306 ± 0.225 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.245 ± 0.227 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.073 ± 0.059 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.165 ± 0.106 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.111 ± 0.094 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.061 ± 0.049 |
| 16 | `f30_distance_covered` *(Space)* | 0.161 ± 0.126 |
| 17 | `f31_area_covered` *(Space)* | 0.237 ± 0.143 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9941 | 0.388 | 0.156 | 0.203 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9935 | -0.005 | 0.043 | 0.318 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9961 | 0.524 | 0.292 | 0.100 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9764 | 0.266 | 0.075 | 0.255 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9785 | 0.073 | 0.293 | 0.341 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9947 | 0.632 | 0.250 | 0.244 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9960 | 0.090 | 0.296 | 0.359 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9951 | 0.149 | 0.091 | 0.077 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9984 | 0.001 | 0.043 | 0.103 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9899 | 0.217 | 0.052 | -0.009 |
