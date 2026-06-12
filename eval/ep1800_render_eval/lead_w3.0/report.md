# Inference config: `lead_w3.0`

- Mode: **lead**  (full / lead / music)
- CFG weights: w_music = **0.0**,  w_lead = **3.0**
- Checkpoint: `runs/train/exp9/weights/train-1800.pt`
- Eval data: val split (10 pairs, AIST++ ch01 sBM)

## Headline numbers (mean over 10 val pairs)

| Metric | Value |
|---|---:|
| PFC (Physical Foot Contact) ↓ | 3.6623 |
| LMA 51-dim Pearson r (paper-style) ↑ | 0.9920 ± 0.0065 |
| LMA Body component (mean per-metric r) | 0.2310 |
| LMA Effort component (mean per-metric r) | 0.1445 |
| LMA Space component (mean per-metric r) | 0.1958 |

Reference: GT human follower PFC = 1.2607, EDGE-paper solo PFC = 1.5363, EDGE solo reproduced on this val = 1.2984.

## 17-metric LMA Pearson r breakdown (mean ± std across pairs)

| # | Metric (component) | r mean ± std |
|---|---|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.305 ± 0.218 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.247 ± 0.266 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.114 ± 0.329 |
| 4 | `f4_hands_to_head` *(Body)* | 0.308 ± 0.336 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.307 ± 0.348 |
| 6 | `f6_pelvis_height` *(Body)* | 0.263 ± 0.266 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.145 ± 0.346 |
| 8 | `f8_gait_distance` *(Body)* | 0.159 ± 0.374 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.144 ± 0.131 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.286 ± 0.226 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.217 ± 0.207 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.073 ± 0.043 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.137 ± 0.087 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.085 ± 0.073 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.069 ± 0.035 |
| 16 | `f30_distance_covered` *(Space)* | 0.148 ± 0.116 |
| 17 | `f31_area_covered` *(Space)* | 0.244 ± 0.111 |

## Per-pair detail

| Pair (lead → follower) | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9951 | 0.373 | 0.129 | 0.221 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9956 | 0.025 | 0.041 | 0.268 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9953 | 0.533 | 0.271 | 0.164 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9810 | 0.242 | 0.078 | 0.173 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9783 | 0.077 | 0.256 | 0.331 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9943 | 0.615 | 0.214 | 0.242 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9966 | 0.097 | 0.269 | 0.336 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9948 | 0.131 | 0.103 | 0.137 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9986 | -0.005 | 0.015 | 0.051 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9904 | 0.222 | 0.069 | 0.036 |
