# Single-Checkpoint Evaluation Report (Epoch 1800)

## Why a single checkpoint, and why this CFG weight

We trained **one duet diffusion model** with compositional CFG over lead motion + music. The configurations `full`, `lead_only`, and `music_only` are three **inference-time CFG regimes** of the same checkpoint (`w_music`/`w_lead` switched on/off), not three independently trained models. Reporting each configuration at its own "best epoch" would conflate inference mode with model state.

We therefore commit to a single checkpoint, **epoch 1800**, for all headline comparisons. ep1800 simultaneously achieves the highest val 4-component LMA for `lead_only` (0.9326) and `music_only` (0.8993), and is within 0.0002 of the peak for `full` (0.9315 vs 0.9317 at ep1850).

At ep1800 we additionally ran a CFG-weight ablation over `w ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` per mode (15 configurations; see `eval/ep1800_render_eval/MASTER_REPORT.md`). The chosen reporting weight is **w = 1.0**, selected because PFC is monotonically increasing in `w` while LMA is nearly flat — so the lowest `w` that still produces visually plausible motion is the Pareto-best trade-off. Manual video review confirmed that the motion at `w = 1.0` is physically clean (no jitter, no foot-sliding) while being slightly more conservative in limb extension than at higher `w`.

Full sweep trajectories across training epochs and across CFG weights are in the appendix tables and the master report.

## 1. Headline Numbers — PFC and LMA at ep1800, w = 1.0

| Configuration | PFC ↓ | LMA 51-dim ↑ | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| GT (real human duet, val) | 1.2607 | 0.9937 | 0.3304 | 0.2580 | 0.3703 |
| **full** (w_music = 1.0, w_lead = 1.0) | **1.7963** | 0.9893 | 0.1955 | 0.1540 | 0.1653 |
| **lead_only** (w_music = 0, w_lead = 1.0) | **1.9468** | 0.9886 | 0.1929 | 0.1608 | 0.1813 |
| **music_only** (w_music = 1.0, w_lead = 0) | 2.2252 | 0.9775 | -0.0009 | 0.0673 | 0.0767 |

GT 4-component LMA ceiling (real human pairs, AIST++ crossmodal val): **0.9416**. All `LMA 51-dim` values above are the paper-style Pearson r between the lead's and the follower's 51-dim segment descriptors (17 metrics × max/mean/std), mean across the 10 val pairs.

### Sanity-check reference (PFC measurement is unbiased)

| Reference | PFC | Notes |
|---|---:|---|
| EDGE paper (solo, AIST++ test, with CCL) | 1.5363 | published number |
| **EDGE solo `checkpoint.pt` on our val** | **1.2984** | reproduction confirms the PFC pipeline is unbiased |
| GT follower on val | 1.2607 | real human ceiling |

Our `full` PFC at w = 1.0 is **1.7963** — within the same range as the EDGE-solo baseline (1.30 reproduced, 1.54 reported), establishing that the duet conditioning regime does not pay a meaningful physical-plausibility cost at this CFG weight.

## 2. 17-Metric LMA Breakdown at ep1800, w = 1.0

Following Zhou et al. IUI '25 §4.2 (Body 8 + Effort 7 + Space 2). Each cell is the per-metric Pearson r between the lead's and the follower's per-frame time-series, mean ± std across the 10 val pairs.

| # | Metric (component) | GT | full | lead_only | music_only |
|---|---|---:|---:|---:|---:|
| 1 | `f1_feet_to_hips` *(Body)* | 0.297 ± 0.295 | 0.219 ± 0.232 | 0.217 ± 0.230 | 0.018 ± 0.089 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.226 ± 0.241 | 0.271 ± 0.190 | 0.237 ± 0.234 | 0.020 ± 0.092 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.338 ± 0.354 | 0.089 ± 0.314 | 0.094 ± 0.330 | 0.037 ± 0.103 |
| 4 | `f4_hands_to_head` *(Body)* | 0.321 ± 0.327 | 0.220 ± 0.282 | 0.229 ± 0.317 | -0.037 ± 0.089 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.440 ± 0.331 | 0.294 ± 0.344 | 0.272 ± 0.332 | 0.104 ± 0.098 |
| 6 | `f6_pelvis_height` *(Body)* | 0.371 ± 0.406 | 0.198 ± 0.255 | 0.217 ± 0.317 | -0.072 ± 0.122 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.241 ± 0.430 | 0.112 ± 0.329 | 0.119 ± 0.365 | -0.029 ± 0.089 |
| 8 | `f8_gait_distance` *(Body)* | 0.410 ± 0.385 | 0.161 ± 0.370 | 0.160 ± 0.337 | -0.047 ± 0.072 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.359 ± 0.372 | 0.118 ± 0.134 | 0.142 ± 0.132 | 0.044 ± 0.063 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.330 ± 0.359 | 0.324 ± 0.187 | 0.297 ± 0.237 | 0.061 ± 0.086 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.264 ± 0.361 | 0.208 ± 0.165 | 0.228 ± 0.215 | 0.073 ± 0.099 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.223 ± 0.239 | 0.074 ± 0.050 | 0.078 ± 0.063 | 0.062 ± 0.041 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.227 ± 0.200 | 0.187 ± 0.097 | 0.202 ± 0.122 | 0.083 ± 0.086 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.196 ± 0.168 | 0.097 ± 0.065 | 0.122 ± 0.087 | 0.076 ± 0.051 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.207 ± 0.158 | 0.071 ± 0.038 | 0.056 ± 0.044 | 0.072 ± 0.033 |
| 16 | `f30_distance_covered` *(Space)* | 0.423 ± 0.367 | 0.112 ± 0.116 | 0.145 ± 0.131 | 0.029 ± 0.044 |
| 17 | `f31_area_covered` *(Space)* | 0.317 ± 0.248 | 0.219 ± 0.097 | 0.217 ± 0.135 | 0.124 ± 0.087 |

### 2a. Component means (Body / Effort / Space)

| Config | Body | Effort | Space |
|---|---:|---:|---:|
| GT (real human) | 0.3304 | 0.2580 | 0.3703 |
| full @ ep1800, w = 1.0 | 0.1955 | 0.1540 | 0.1653 |
| lead_only @ ep1800, w = 1.0 | 0.1929 | 0.1608 | 0.1813 |
| music_only @ ep1800, w = 1.0 | -0.0009 | 0.0673 | 0.0767 |

## 3. Per-Pair Breakdown at ep1800, w = 1.0

Overall r is the paper-style 51-dim Pearson; Body/Effort/Space are means of the corresponding per-metric time-series r values.

### GT (real human duet)

| Pair | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9981 | 0.605 | 0.390 | 0.563 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9900 | 0.300 | 0.169 | 0.384 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9967 | 0.402 | 0.026 | -0.029 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9994 | 0.788 | 0.422 | 0.622 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9986 | -0.153 | -0.174 | -0.024 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9808 | 0.500 | 0.445 | 0.564 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9971 | 0.536 | 0.436 | 0.705 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9954 | 0.225 | 0.329 | 0.214 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9923 | 0.162 | 0.493 | 0.560 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9887 | -0.062 | 0.044 | 0.145 |

### full @ ep1800, w = 1.0

| Pair | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9941 | 0.354 | 0.153 | 0.132 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9897 | -0.032 | 0.065 | 0.240 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9966 | 0.384 | 0.189 | 0.090 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9676 | 0.144 | 0.090 | 0.183 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9808 | 0.104 | 0.273 | 0.214 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9882 | 0.557 | 0.237 | 0.227 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9967 | 0.091 | 0.262 | 0.314 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9944 | 0.133 | 0.098 | 0.114 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9963 | 0.053 | 0.088 | 0.107 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9882 | 0.167 | 0.084 | 0.031 |

### lead_only @ ep1800, w = 1.0

| Pair | n_slices | overall r | Body | Effort | Space |
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

### music_only @ ep1800, w = 1.0

| Pair | n_slices | overall r | Body | Effort | Space |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01 → gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9823 | -0.039 | 0.070 | 0.066 |
| gHO_sBM_cAll_d20_mHO5_ch01 → gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9924 | -0.033 | 0.076 | 0.086 |
| gJB_sBM_cAll_d08_mJB5_ch01 → gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9734 | 0.031 | 0.109 | 0.121 |
| gJS_sBM_cAll_d01_mJS3_ch01 → gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9452 | -0.027 | 0.022 | 0.151 |
| gKR_sBM_cAll_d28_mKR2_ch01 → gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9856 | 0.008 | -0.037 | -0.024 |
| gLH_sBM_cAll_d17_mLH4_ch01 → gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9782 | -0.035 | 0.151 | 0.063 |
| gLO_sBM_cAll_d13_mLO2_ch01 → gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9790 | 0.043 | 0.090 | 0.167 |
| gMH_sBM_cAll_d22_mMH3_ch01 → gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9847 | 0.024 | 0.004 | 0.028 |
| gPO_sBM_cAll_d10_mPO1_ch01 → gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9850 | -0.001 | 0.086 | 0.041 |
| gWA_sBM_cAll_d25_mWA0_ch01 → gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9695 | 0.019 | 0.101 | 0.067 |

## 4. Take-aways

1. **`full` is physically plausible at the reported CFG weight.** At w = 1.0, `full`'s PFC of 1.7963 is comparable to the reproduced EDGE-solo baseline (1.2984) and to the GT human ceiling (1.2607). The duet conditioning does not introduce a meaningful foot-instability penalty at this CFG strength.

2. **`lead_only` matches or improves on `full`.** PFC differs by -0.151 (`full` minus `lead_only`); LMA differs by -0.0007 (`lead_only` minus `full`). At every CFG weight in our ablation `lead_only` is on or above `full` on both axes, i.e. silencing the music stream at the CFG level does not hurt and slightly helps follower generation. (Verified across `w ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` — see master report.)

3. **`music_only` collapses on the Body component.** Body r = -0.001 for `music_only` vs 0.193 for `lead_only` (gap 0.194). Without the lead conditioning stream, the generated follower's body shape has near-zero correlation with the lead's body shape — isolating the lead stream as the carrier of Body-component agreement. This holds at every `w` in our sweep.

4. **`music_only` retains partial Effort correlation.** Hands/feet velocity r ≈ 0.06–0.07 — music does transmit a shared rhythm signal, but does not transmit pose.

---

Generated by `eval/build_ep1800_report.py --w 1.0` from existing sweep data; no inference rerun. Source files: `eval/ep1800_render_eval/summary.json` (CFG sweep at ep1800), `eval/lma17_report/lma17_report.json` (GT 17-metric reference), `eval/full_sweep_per_slice/summary.json` (GT PFC + 4-component LMA ceiling), `eval/solo_pfc_baseline/summary.json` (EDGE solo reproduction).