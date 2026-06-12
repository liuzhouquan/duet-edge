# Inference config: `full_w1.0`

- Mode: **full**  (full / lead / music)
- CFG weights: w_music = **1.0**,  w_lead = **1.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 1.7963 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9893 ± 0.0087 |
| LMA Body component (mean per-metric r) | 0.1955 |
| LMA Effort component (mean per-metric r) | 0.1540 |
| LMA Space component (mean per-metric r) | 0.1653 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.219 ± 0.232 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.271 ± 0.190 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.089 ± 0.314 |
| 4 | `f4_hands_to_head` *(Body)* | 0.220 ± 0.282 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.294 ± 0.344 |
| 6 | `f6_pelvis_height` *(Body)* | 0.198 ± 0.255 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.112 ± 0.329 |
| 8 | `f8_gait_distance` *(Body)* | 0.161 ± 0.370 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.118 ± 0.134 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.324 ± 0.187 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.208 ± 0.165 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.074 ± 0.050 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.187 ± 0.097 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.097 ± 0.065 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.071 ± 0.038 |
| 16 | `f30_distance_covered` *(Space)* | 0.112 ± 0.116 |
| 17 | `f31_area_covered` *(Space)* | 0.219 ± 0.097 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9941 | 0.354 | 0.153 | 0.132 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9897 | -0.032 | 0.065 | 0.240 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9966 | 0.384 | 0.189 | 0.090 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9676 | 0.144 | 0.090 | 0.183 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9808 | 0.104 | 0.273 | 0.214 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9882 | 0.557 | 0.237 | 0.227 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9967 | 0.091 | 0.262 | 0.314 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9944 | 0.133 | 0.098 | 0.114 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9963 | 0.053 | 0.088 | 0.107 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9882 | 0.167 | 0.084 | 0.031 |
