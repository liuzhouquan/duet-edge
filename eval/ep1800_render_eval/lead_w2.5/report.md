# Inference config: `lead_w2.5`

- Mode: **lead**  (full / lead / music)
- CFG weights: w_music = **0.0**,  w_lead = **2.5**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.9047 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9919 ± 0.0065 |
| LMA Body component (mean per-metric r) | 0.2345 |
| LMA Effort component (mean per-metric r) | 0.1502 |
| LMA Space component (mean per-metric r) | 0.2096 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.297 ± 0.223 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.254 ± 0.262 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.114 ± 0.333 |
| 4 | `f4_hands_to_head` *(Body)* | 0.306 ± 0.342 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.307 ± 0.348 |
| 6 | `f6_pelvis_height` *(Body)* | 0.281 ± 0.252 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.157 ± 0.338 |
| 8 | `f8_gait_distance` *(Body)* | 0.158 ± 0.380 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.150 ± 0.139 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.298 ± 0.236 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.231 ± 0.217 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.070 ± 0.054 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.146 ± 0.096 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.096 ± 0.079 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.061 ± 0.048 |
| 16 | `f30_distance_covered` *(Space)* | 0.156 ± 0.120 |
| 17 | `f31_area_covered` *(Space)* | 0.263 ± 0.112 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9949 | 0.382 | 0.145 | 0.218 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9946 | 0.009 | 0.039 | 0.294 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9957 | 0.535 | 0.271 | 0.158 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9806 | 0.262 | 0.079 | 0.219 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9787 | 0.073 | 0.273 | 0.338 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9946 | 0.628 | 0.227 | 0.268 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9966 | 0.096 | 0.289 | 0.356 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9949 | 0.141 | 0.098 | 0.125 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9984 | 0.001 | 0.031 | 0.078 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9902 | 0.217 | 0.051 | 0.043 |
