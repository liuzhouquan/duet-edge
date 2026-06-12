# Inference config: `music_w1.5`

- Mode: **music**  (full / lead / music)
- CFG weights: w_music = **1.5**,  w_lead = **0.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.0318 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9790 ± 0.0118 |
| LMA Body component (mean per-metric r) | -0.0025 |
| LMA Effort component (mean per-metric r) | 0.0632 |
| LMA Space component (mean per-metric r) | 0.0802 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.045 ± 0.097 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.024 ± 0.082 |
| 3 | `f3_lhand_to_rhand` *(Body)* | -0.006 ± 0.091 |
| 4 | `f4_hands_to_head` *(Body)* | -0.021 ± 0.071 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.077 ± 0.108 |
| 6 | `f6_pelvis_height` *(Body)* | -0.064 ± 0.230 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | -0.036 ± 0.135 |
| 8 | `f8_gait_distance` *(Body)* | -0.040 ± 0.163 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.024 ± 0.051 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.068 ± 0.095 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.095 ± 0.069 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.041 ± 0.036 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.084 ± 0.090 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.080 ± 0.059 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.050 ± 0.036 |
| 16 | `f30_distance_covered` *(Space)* | 0.024 ± 0.043 |
| 17 | `f31_area_covered` *(Space)* | 0.137 ± 0.076 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9833 | -0.103 | 0.086 | 0.071 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9913 | 0.054 | 0.077 | 0.191 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9833 | -0.020 | 0.109 | 0.047 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9476 | 0.012 | 0.042 | 0.029 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9852 | 0.001 | -0.058 | 0.016 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9760 | -0.080 | 0.077 | 0.036 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9796 | 0.016 | 0.107 | 0.161 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9875 | 0.036 | 0.044 | 0.121 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9849 | 0.007 | 0.068 | 0.060 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9715 | 0.051 | 0.079 | 0.072 |
