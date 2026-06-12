# Inference config: `lead_w1.5`

- Mode: **lead**  (full / lead / music)
- CFG weights: w_music = **0.0**,  w_lead = **1.5**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 1.9972 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9897 ± 0.0099 |
| LMA Body component (mean per-metric r) | 0.2172 |
| LMA Effort component (mean per-metric r) | 0.1663 |
| LMA Space component (mean per-metric r) | 0.1910 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.259 ± 0.231 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.274 ± 0.239 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.115 ± 0.328 |
| 4 | `f4_hands_to_head` *(Body)* | 0.271 ± 0.341 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.280 ± 0.349 |
| 6 | `f6_pelvis_height` *(Body)* | 0.250 ± 0.296 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.129 ± 0.370 |
| 8 | `f8_gait_distance` *(Body)* | 0.160 ± 0.372 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.148 ± 0.147 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.322 ± 0.227 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.249 ± 0.226 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.075 ± 0.059 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.183 ± 0.115 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.122 ± 0.090 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.065 ± 0.047 |
| 16 | `f30_distance_covered` *(Space)* | 0.158 ± 0.122 |
| 17 | `f31_area_covered` *(Space)* | 0.224 ± 0.145 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9928 | 0.382 | 0.168 | 0.172 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9921 | -0.022 | 0.051 | 0.320 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9963 | 0.511 | 0.289 | 0.105 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9642 | 0.163 | 0.082 | 0.259 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9788 | 0.081 | 0.320 | 0.340 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9943 | 0.626 | 0.267 | 0.194 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9957 | 0.080 | 0.284 | 0.339 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9952 | 0.148 | 0.085 | 0.046 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9979 | 0.006 | 0.049 | 0.117 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9896 | 0.199 | 0.069 | 0.017 |
