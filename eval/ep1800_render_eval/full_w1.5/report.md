# Inference config: `full_w1.5`

- Mode: **full**  (full / lead / music)
- CFG weights: w_music = **1.5**,  w_lead = **1.5**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 2.1803 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9911 ± 0.0066 |
| LMA Body component (mean per-metric r) | 0.2146 |
| LMA Effort component (mean per-metric r) | 0.1547 |
| LMA Space component (mean per-metric r) | 0.1592 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.222 ± 0.248 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.248 ± 0.221 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.125 ± 0.314 |
| 4 | `f4_hands_to_head` *(Body)* | 0.264 ± 0.304 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.316 ± 0.349 |
| 6 | `f6_pelvis_height` *(Body)* | 0.223 ± 0.298 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.131 ± 0.357 |
| 8 | `f8_gait_distance` *(Body)* | 0.187 ± 0.377 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.123 ± 0.168 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.315 ± 0.198 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.227 ± 0.185 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.080 ± 0.048 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.165 ± 0.075 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.105 ± 0.072 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.068 ± 0.031 |
| 16 | `f30_distance_covered` *(Space)* | 0.115 ± 0.148 |
| 17 | `f31_area_covered` *(Space)* | 0.203 ± 0.146 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9953 | 0.372 | 0.144 | 0.155 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9917 | -0.014 | 0.052 | 0.248 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9975 | 0.416 | 0.201 | 0.092 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9804 | 0.125 | 0.110 | 0.050 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9788 | 0.122 | 0.251 | 0.280 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9901 | 0.590 | 0.231 | 0.202 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9968 | 0.113 | 0.308 | 0.430 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9948 | 0.178 | 0.108 | 0.073 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9986 | 0.082 | 0.077 | 0.060 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9875 | 0.162 | 0.064 | 0.002 |
