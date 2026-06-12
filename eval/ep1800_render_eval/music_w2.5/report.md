# Inference config: `music_w2.5`

- Mode: **music**  (full / lead / music)
- CFG weights: w_music = **2.5**,  w_lead = **0.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.2994 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9804 ± 0.0117 |
| LMA Body component (mean per-metric r) | 0.0073 |
| LMA Effort component (mean per-metric r) | 0.0703 |
| LMA Space component (mean per-metric r) | 0.0861 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.042 ± 0.106 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.029 ± 0.093 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.010 ± 0.114 |
| 4 | `f4_hands_to_head` *(Body)* | 0.001 ± 0.123 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.073 ± 0.113 |
| 6 | `f6_pelvis_height` *(Body)* | -0.071 ± 0.229 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | -0.020 ± 0.148 |
| 8 | `f8_gait_distance` *(Body)* | -0.006 ± 0.172 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.007 ± 0.053 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.093 ± 0.070 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.122 ± 0.103 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.038 ± 0.040 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.092 ± 0.072 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.093 ± 0.068 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.047 ± 0.036 |
| 16 | `f30_distance_covered` *(Space)* | 0.015 ± 0.050 |
| 17 | `f31_area_covered` *(Space)* | 0.158 ± 0.096 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9902 | -0.108 | 0.083 | 0.007 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9907 | 0.103 | 0.066 | 0.086 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9833 | 0.018 | 0.117 | 0.152 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9486 | 0.012 | 0.048 | 0.083 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9836 | -0.001 | -0.040 | 0.022 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9769 | -0.044 | 0.126 | 0.134 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9803 | 0.059 | 0.110 | 0.170 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9874 | 0.034 | 0.017 | 0.053 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9877 | -0.019 | 0.076 | 0.130 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9756 | 0.019 | 0.099 | 0.024 |
