# Inference config: `full_w2.5`

- Mode: **full**  (full / lead / music)
- CFG weights: w_music = **2.5**,  w_lead = **2.5**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 3.8414 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9916 ± 0.0060 |
| LMA Body component (mean per-metric r) | 0.2138 |
| LMA Effort component (mean per-metric r) | 0.1319 |
| LMA Space component (mean per-metric r) | 0.1055 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.204 ± 0.237 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.219 ± 0.252 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.144 ± 0.315 |
| 4 | `f4_hands_to_head` *(Body)* | 0.225 ± 0.322 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.319 ± 0.346 |
| 6 | `f6_pelvis_height` *(Body)* | 0.235 ± 0.301 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.179 ± 0.324 |
| 8 | `f8_gait_distance` *(Body)* | 0.187 ± 0.379 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.106 ± 0.142 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.276 ± 0.186 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.204 ± 0.169 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.071 ± 0.028 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.124 ± 0.064 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.081 ± 0.047 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.061 ± 0.028 |
| 16 | `f30_distance_covered` *(Space)* | 0.094 ± 0.143 |
| 17 | `f31_area_covered` *(Space)* | 0.117 ± 0.117 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9952 | 0.349 | 0.117 | 0.168 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9950 | 0.040 | 0.039 | 0.069 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9968 | 0.413 | 0.211 | 0.078 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9826 | 0.151 | 0.117 | 0.050 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9796 | 0.091 | 0.191 | 0.259 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9911 | 0.572 | 0.189 | 0.079 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9958 | 0.107 | 0.264 | 0.277 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9941 | 0.168 | 0.073 | 0.033 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9985 | 0.098 | 0.045 | 0.053 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9875 | 0.148 | 0.072 | -0.010 |
