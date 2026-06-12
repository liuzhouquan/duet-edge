# Inference config: `music_w2.0`

- Mode: **music**  (full / lead / music)
- CFG weights: w_music = **2.0**,  w_lead = **0.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.1566 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9798 ± 0.0121 |
| LMA Body component (mean per-metric r) | 0.0073 |
| LMA Effort component (mean per-metric r) | 0.0702 |
| LMA Space component (mean per-metric r) | 0.0823 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.042 ± 0.074 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.042 ± 0.080 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.010 ± 0.086 |
| 4 | `f4_hands_to_head` *(Body)* | 0.017 ± 0.103 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.084 ± 0.124 |
| 6 | `f6_pelvis_height` *(Body)* | -0.077 ± 0.233 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | -0.042 ± 0.141 |
| 8 | `f8_gait_distance` *(Body)* | -0.019 ± 0.149 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.016 ± 0.053 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.083 ± 0.090 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.111 ± 0.098 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.046 ± 0.031 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.089 ± 0.083 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.085 ± 0.063 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.062 ± 0.030 |
| 16 | `f30_distance_covered` *(Space)* | 0.016 ± 0.048 |
| 17 | `f31_area_covered` *(Space)* | 0.148 ± 0.089 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9874 | -0.093 | 0.088 | -0.007 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9912 | 0.074 | 0.053 | 0.112 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9827 | 0.008 | 0.116 | 0.154 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9474 | 0.024 | 0.066 | 0.072 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9830 | 0.007 | -0.039 | -0.017 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9770 | -0.067 | 0.128 | 0.130 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9795 | 0.055 | 0.111 | 0.172 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9879 | 0.037 | 0.028 | 0.060 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9892 | -0.020 | 0.070 | 0.100 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9726 | 0.047 | 0.081 | 0.046 |
