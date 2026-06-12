# Inference config: `music_w3.0`

- Mode: **music**  (full / lead / music)
- CFG weights: w_music = **3.0**,  w_lead = **0.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.5909 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9805 ± 0.0110 |
| LMA Body component (mean per-metric r) | 0.0132 |
| LMA Effort component (mean per-metric r) | 0.0738 |
| LMA Space component (mean per-metric r) | 0.0824 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.040 ± 0.115 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.037 ± 0.108 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.020 ± 0.111 |
| 4 | `f4_hands_to_head` *(Body)* | 0.018 ± 0.154 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.063 ± 0.102 |
| 6 | `f6_pelvis_height` *(Body)* | -0.058 ± 0.228 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | -0.007 ± 0.146 |
| 8 | `f8_gait_distance` *(Body)* | -0.008 ± 0.204 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.012 ± 0.059 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.095 ± 0.078 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.131 ± 0.102 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.043 ± 0.044 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.085 ± 0.070 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.100 ± 0.054 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.051 ± 0.036 |
| 16 | `f30_distance_covered` *(Space)* | 0.022 ± 0.058 |
| 17 | `f31_area_covered` *(Space)* | 0.143 ± 0.100 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9901 | -0.107 | 0.079 | 0.045 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9898 | 0.100 | 0.077 | 0.071 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9837 | 0.022 | 0.107 | 0.154 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9512 | 0.037 | 0.097 | 0.077 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9828 | -0.002 | -0.039 | 0.004 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9730 | -0.029 | 0.119 | 0.111 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9816 | 0.097 | 0.121 | 0.199 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9873 | 0.030 | 0.015 | 0.073 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9878 | -0.024 | 0.071 | 0.096 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9777 | 0.007 | 0.091 | -0.005 |
