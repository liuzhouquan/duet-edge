# Inference config: `full_w3.0`

- Mode: **full**  (full / lead / music)
- CFG weights: w_music = **3.0**,  w_lead = **3.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 4.8816 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9916 ± 0.0060 |
| LMA Body component (mean per-metric r) | 0.2090 |
| LMA Effort component (mean per-metric r) | 0.1223 |
| LMA Space component (mean per-metric r) | 0.1009 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.200 ± 0.230 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.203 ± 0.260 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.139 ± 0.312 |
| 4 | `f4_hands_to_head` *(Body)* | 0.212 ± 0.318 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.314 ± 0.342 |
| 6 | `f6_pelvis_height` *(Body)* | 0.231 ± 0.292 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.192 ± 0.307 |
| 8 | `f8_gait_distance` *(Body)* | 0.181 ± 0.387 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.096 ± 0.138 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.256 ± 0.175 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.198 ± 0.162 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.061 ± 0.030 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.116 ± 0.056 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.073 ± 0.041 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.055 ± 0.040 |
| 16 | `f30_distance_covered` *(Space)* | 0.083 ± 0.136 |
| 17 | `f31_area_covered` *(Space)* | 0.119 ± 0.101 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9949 | 0.327 | 0.105 | 0.157 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9956 | 0.069 | 0.035 | 0.024 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9961 | 0.410 | 0.208 | 0.052 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9828 | 0.149 | 0.109 | 0.032 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9795 | 0.074 | 0.158 | 0.225 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9910 | 0.556 | 0.166 | 0.114 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9965 | 0.102 | 0.259 | 0.275 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9937 | 0.153 | 0.067 | 0.055 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9983 | 0.109 | 0.048 | 0.040 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9874 | 0.141 | 0.068 | 0.036 |
