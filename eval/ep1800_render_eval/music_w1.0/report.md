# Inference config: `music_w1.0`

- Mode: **music**  (full / lead / music)
- CFG weights: w_music = **1.0**,  w_lead = **0.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.2252 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9775 ± 0.0124 |
| LMA Body component (mean per-metric r) | -0.0009 |
| LMA Effort component (mean per-metric r) | 0.0673 |
| LMA Space component (mean per-metric r) | 0.0767 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.018 ± 0.089 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.020 ± 0.092 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.037 ± 0.103 |
| 4 | `f4_hands_to_head` *(Body)* | -0.037 ± 0.089 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.104 ± 0.098 |
| 6 | `f6_pelvis_height` *(Body)* | -0.072 ± 0.122 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | -0.029 ± 0.089 |
| 8 | `f8_gait_distance` *(Body)* | -0.047 ± 0.072 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.044 ± 0.063 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.061 ± 0.086 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.073 ± 0.099 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.062 ± 0.041 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.083 ± 0.086 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.076 ± 0.051 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.072 ± 0.033 |
| 16 | `f30_distance_covered` *(Space)* | 0.029 ± 0.044 |
| 17 | `f31_area_covered` *(Space)* | 0.124 ± 0.087 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9823 | -0.039 | 0.070 | 0.066 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9924 | -0.033 | 0.076 | 0.086 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9734 | 0.031 | 0.109 | 0.121 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9452 | -0.027 | 0.022 | 0.151 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9856 | 0.008 | -0.037 | -0.024 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9782 | -0.035 | 0.151 | 0.063 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9790 | 0.043 | 0.090 | 0.167 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9847 | 0.024 | 0.004 | 0.028 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9850 | -0.001 | 0.086 | 0.041 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9695 | 0.019 | 0.101 | 0.067 |
