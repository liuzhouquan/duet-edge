# Inference config: `lead_w1.0`

- Mode: **lead**  (full / lead / music)
- CFG weights: w_music = **0.0**,  w_lead = **1.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 1.9468 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9886 ± 0.0111 |
| LMA Body component (mean per-metric r) | 0.1929 |
| LMA Effort component (mean per-metric r) | 0.1608 |
| LMA Space component (mean per-metric r) | 0.1813 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.217 ± 0.230 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.237 ± 0.234 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.094 ± 0.330 |
| 4 | `f4_hands_to_head` *(Body)* | 0.229 ± 0.317 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.272 ± 0.332 |
| 6 | `f6_pelvis_height` *(Body)* | 0.217 ± 0.317 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.119 ± 0.365 |
| 8 | `f8_gait_distance` *(Body)* | 0.160 ± 0.337 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.142 ± 0.132 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.297 ± 0.237 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.228 ± 0.215 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.078 ± 0.063 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.202 ± 0.122 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.122 ± 0.087 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.056 ± 0.044 |
| 16 | `f30_distance_covered` *(Space)* | 0.145 ± 0.131 |
| 17 | `f31_area_covered` *(Space)* | 0.217 ± 0.135 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9901 | 0.344 | 0.163 | 0.208 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9900 | -0.037 | 0.055 | 0.293 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9966 | 0.447 | 0.258 | 0.124 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9575 | 0.106 | 0.066 | 0.098 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9824 | 0.082 | 0.326 | 0.302 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9933 | 0.597 | 0.296 | 0.227 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9963 | 0.103 | 0.266 | 0.326 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9958 | 0.124 | 0.071 | 0.035 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9943 | -0.013 | 0.052 | 0.149 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9894 | 0.176 | 0.057 | 0.050 |
