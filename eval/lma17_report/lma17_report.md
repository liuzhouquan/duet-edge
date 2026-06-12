# 17-Metric LMA Evaluation Report

Following Zhou et al., *Real-Time Full-body Interaction with AI Dance Models* (IUI '25, Section 4.2). The 17 metrics comprise Body (8) + Effort (7) + Space (2). Shape (10 features) is excluded per the paper's stated methodology.

Each segment (one 5 s / 150-frame slice at 30 fps) is summarised by a 51-dim descriptor (17 metrics × {max, mean, std}). The headline similarity is the Pearson r between the lead's and follower's 51-dim descriptors. Per-metric scores are Pearson r between the two per-frame metric trajectories.

## 1. Headline 51-dim LMA Similarity

| Config | Epoch | n_pairs | r mean | r std |
|---|---:|---:|---:|---:|
| GT | None | 10 | 0.9937 | 0.0055 |
| full | 50 | 10 | 0.9193 | 0.0168 |
| full | 100 | 10 | 0.9336 | 0.0259 |
| full | 150 | 10 | 0.9164 | 0.0297 |
| full | 200 | 10 | 0.9273 | 0.0229 |
| full | 250 | 10 | 0.9524 | 0.0251 |
| full | 300 | 10 | 0.9577 | 0.0236 |
| full | 350 | 10 | 0.9767 | 0.0117 |
| full | 400 | 10 | 0.9807 | 0.0092 |
| full | 450 | 10 | 0.9841 | 0.0079 |
| full | 500 | 10 | 0.9842 | 0.0073 |
| full | 550 | 10 | 0.9859 | 0.0078 |
| full | 600 | 10 | 0.9872 | 0.0072 |
| full | 650 | 10 | 0.9876 | 0.0075 |
| full | 700 | 10 | 0.9879 | 0.0074 |
| full | 750 | 10 | 0.9885 | 0.0070 |
| full | 800 | 10 | 0.9888 | 0.0065 |
| full | 850 | 10 | 0.9894 | 0.0063 |
| full | 900 | 10 | 0.9902 | 0.0053 |
| full | 950 | 10 | 0.9896 | 0.0062 |
| full | 1000 | 10 | 0.9895 | 0.0065 |
| full | 1050 | 10 | 0.9915 | 0.0059 |
| full | 1100 | 10 | 0.9901 | 0.0062 |
| full | 1150 | 10 | 0.9916 | 0.0057 |
| full | 1200 | 10 | 0.9920 | 0.0059 |
| full | 1250 | 10 | 0.9921 | 0.0059 |
| full | 1300 | 10 | 0.9923 | 0.0054 |
| full | 1350 | 10 | 0.9923 | 0.0056 |
| full | 1400 | 10 | 0.9920 | 0.0059 |
| full | 1450 | 10 | 0.9919 | 0.0057 |
| full | 1500 | 10 | 0.9920 | 0.0061 |
| full | 1550 | 10 | 0.9920 | 0.0062 |
| full | 1600 | 10 | 0.9919 | 0.0064 |
| full | 1650 | 10 | 0.9915 | 0.0070 |
| full | 1700 | 10 | 0.9914 | 0.0070 |
| full | 1750 | 10 | 0.9913 | 0.0072 |
| full | 1800 | 10 | 0.9915 | 0.0069 |
| full | 1850 | 10 | 0.9916 | 0.0067 |
| full | 1900 | 10 | 0.9917 | 0.0066 |
| full | 1950 | 10 | 0.9914 | 0.0074 |
| full | 2000 | 10 | 0.9911 | 0.0080 |
| lead_only | 50 | 10 | 0.9038 | 0.0278 |
| lead_only | 100 | 10 | 0.9441 | 0.0159 |
| lead_only | 150 | 10 | 0.9275 | 0.0218 |
| lead_only | 200 | 10 | 0.9414 | 0.0225 |
| lead_only | 250 | 10 | 0.9454 | 0.0245 |
| lead_only | 300 | 10 | 0.9502 | 0.0211 |
| lead_only | 350 | 10 | 0.9697 | 0.0130 |
| lead_only | 400 | 10 | 0.9796 | 0.0096 |
| lead_only | 450 | 10 | 0.9837 | 0.0120 |
| lead_only | 500 | 10 | 0.9836 | 0.0125 |
| lead_only | 550 | 10 | 0.9842 | 0.0110 |
| lead_only | 600 | 10 | 0.9846 | 0.0104 |
| lead_only | 650 | 10 | 0.9857 | 0.0106 |
| lead_only | 700 | 10 | 0.9869 | 0.0105 |
| lead_only | 750 | 10 | 0.9876 | 0.0098 |
| lead_only | 800 | 10 | 0.9878 | 0.0095 |
| lead_only | 850 | 10 | 0.9887 | 0.0078 |
| lead_only | 900 | 10 | 0.9902 | 0.0065 |
| lead_only | 950 | 10 | 0.9908 | 0.0061 |
| lead_only | 1000 | 10 | 0.9912 | 0.0060 |
| lead_only | 1050 | 10 | 0.9914 | 0.0062 |
| lead_only | 1100 | 10 | 0.9914 | 0.0063 |
| lead_only | 1150 | 10 | 0.9915 | 0.0063 |
| lead_only | 1200 | 10 | 0.9911 | 0.0070 |
| lead_only | 1250 | 10 | 0.9909 | 0.0071 |
| lead_only | 1300 | 10 | 0.9915 | 0.0068 |
| lead_only | 1350 | 10 | 0.9915 | 0.0064 |
| lead_only | 1400 | 10 | 0.9912 | 0.0070 |
| lead_only | 1450 | 10 | 0.9915 | 0.0072 |
| lead_only | 1500 | 10 | 0.9916 | 0.0070 |
| lead_only | 1550 | 10 | 0.9914 | 0.0071 |
| lead_only | 1600 | 10 | 0.9911 | 0.0078 |
| lead_only | 1650 | 10 | 0.9910 | 0.0081 |
| lead_only | 1700 | 10 | 0.9911 | 0.0078 |
| lead_only | 1750 | 10 | 0.9911 | 0.0079 |
| lead_only | 1800 | 10 | 0.9911 | 0.0077 |
| lead_only | 1850 | 10 | 0.9911 | 0.0078 |
| lead_only | 1900 | 10 | 0.9912 | 0.0077 |
| lead_only | 1950 | 10 | 0.9911 | 0.0081 |
| lead_only | 2000 | 10 | 0.9911 | 0.0083 |
| music_only | 50 | 10 | 0.9468 | 0.0207 |
| music_only | 100 | 10 | 0.9412 | 0.0202 |
| music_only | 150 | 10 | 0.9228 | 0.0192 |
| music_only | 200 | 10 | 0.9356 | 0.0202 |
| music_only | 250 | 10 | 0.9681 | 0.0139 |
| music_only | 300 | 10 | 0.9420 | 0.0416 |
| music_only | 350 | 10 | 0.9361 | 0.0411 |
| music_only | 400 | 10 | 0.9368 | 0.0363 |
| music_only | 450 | 10 | 0.9416 | 0.0299 |
| music_only | 500 | 10 | 0.9434 | 0.0274 |
| music_only | 550 | 10 | 0.9468 | 0.0247 |
| music_only | 600 | 10 | 0.9519 | 0.0242 |
| music_only | 650 | 10 | 0.9556 | 0.0266 |
| music_only | 700 | 10 | 0.9583 | 0.0269 |
| music_only | 750 | 10 | 0.9605 | 0.0269 |
| music_only | 800 | 10 | 0.9638 | 0.0250 |
| music_only | 850 | 10 | 0.9654 | 0.0225 |
| music_only | 900 | 10 | 0.9706 | 0.0204 |
| music_only | 950 | 10 | 0.9715 | 0.0200 |
| music_only | 1000 | 10 | 0.9710 | 0.0186 |
| music_only | 1050 | 10 | 0.9726 | 0.0185 |
| music_only | 1100 | 10 | 0.9739 | 0.0186 |
| music_only | 1150 | 10 | 0.9756 | 0.0185 |
| music_only | 1200 | 10 | 0.9742 | 0.0177 |
| music_only | 1250 | 10 | 0.9740 | 0.0164 |
| music_only | 1300 | 10 | 0.9756 | 0.0147 |
| music_only | 1350 | 10 | 0.9774 | 0.0141 |
| music_only | 1400 | 10 | 0.9775 | 0.0142 |
| music_only | 1450 | 10 | 0.9786 | 0.0137 |
| music_only | 1500 | 10 | 0.9791 | 0.0136 |
| music_only | 1550 | 10 | 0.9787 | 0.0137 |
| music_only | 1600 | 10 | 0.9793 | 0.0143 |
| music_only | 1650 | 10 | 0.9792 | 0.0141 |
| music_only | 1700 | 10 | 0.9787 | 0.0142 |
| music_only | 1750 | 10 | 0.9789 | 0.0141 |
| music_only | 1800 | 10 | 0.9791 | 0.0131 |
| music_only | 1850 | 10 | 0.9779 | 0.0141 |
| music_only | 1900 | 10 | 0.9783 | 0.0139 |
| music_only | 1950 | 10 | 0.9793 | 0.0135 |
| music_only | 2000 | 10 | 0.9785 | 0.0136 |

## 2. Per-Metric Pearson r (mean ± std across val pairs)

To keep this section readable, the breakdown is shown only for GT and the *best* (highest headline r) checkpoint of each configuration. Full per-epoch numbers are in the companion JSON.

| # | Metric (component) | GT (real follower) | full @ ep1300 | lead_only @ ep1500 | music_only @ ep1950 |
|---|---|---|---|---|
| 1 | `f1_feet_to_hips` *(Body)* | 0.297 ± 0.295 | 0.238 ± 0.213 | 0.260 ± 0.216 | -0.018 ± 0.101 |
| 2 | `f2_hands_to_shoulders` *(Body)* | 0.226 ± 0.241 | 0.240 ± 0.258 | 0.257 ± 0.256 | 0.024 ± 0.098 |
| 3 | `f3_lhand_to_rhand` *(Body)* | 0.338 ± 0.354 | 0.101 ± 0.296 | 0.145 ± 0.319 | 0.057 ± 0.101 |
| 4 | `f4_hands_to_head` *(Body)* | 0.321 ± 0.327 | 0.216 ± 0.352 | 0.299 ± 0.355 | 0.018 ± 0.087 |
| 5 | `f5_hands_to_hips` *(Body)* | 0.440 ± 0.331 | 0.339 ± 0.316 | 0.346 ± 0.316 | 0.072 ± 0.113 |
| 6 | `f6_pelvis_height` *(Body)* | 0.371 ± 0.406 | 0.217 ± 0.332 | 0.262 ± 0.267 | -0.019 ± 0.178 |
| 7 | `f7_hipground_minus_feethip` *(Body)* | 0.241 ± 0.430 | 0.130 ± 0.385 | 0.128 ± 0.363 | -0.008 ± 0.130 |
| 8 | `f8_gait_distance` *(Body)* | 0.410 ± 0.385 | 0.210 ± 0.326 | 0.195 ± 0.386 | -0.026 ± 0.170 |
| 9 | `f11_pelvis_velocity` *(Effort)* | 0.359 ± 0.372 | 0.087 ± 0.164 | 0.136 ± 0.188 | 0.020 ± 0.045 |
| 10 | `f13_hands_velocity` *(Effort)* | 0.330 ± 0.359 | 0.289 ± 0.177 | 0.291 ± 0.214 | 0.113 ± 0.091 |
| 11 | `f14_feet_velocity` *(Effort)* | 0.264 ± 0.361 | 0.222 ± 0.206 | 0.234 ± 0.226 | 0.158 ± 0.080 |
| 12 | `f15_pelvis_acceleration` *(Effort)* | 0.223 ± 0.239 | 0.045 ± 0.045 | 0.064 ± 0.078 | 0.032 ± 0.053 |
| 13 | `f17_hands_acceleration` *(Effort)* | 0.227 ± 0.200 | 0.139 ± 0.071 | 0.138 ± 0.103 | 0.102 ± 0.069 |
| 14 | `f18_feet_acceleration` *(Effort)* | 0.196 ± 0.168 | 0.087 ± 0.077 | 0.105 ± 0.096 | 0.087 ± 0.043 |
| 15 | `f19_pelvis_jerk` *(Effort)* | 0.207 ± 0.158 | 0.038 ± 0.033 | 0.056 ± 0.065 | 0.046 ± 0.038 |
| 16 | `f30_distance_covered` *(Space)* | 0.423 ± 0.367 | 0.073 ± 0.149 | 0.142 ± 0.154 | 0.007 ± 0.033 |
| 17 | `f31_area_covered` *(Space)* | 0.317 ± 0.248 | 0.242 ± 0.101 | 0.257 ± 0.125 | 0.105 ± 0.072 |

### 2a. Component means (Body / Effort / Space)

| Config | Body | Effort | Space |
|---|---:|---:|---:|
| GT (real follower) | 0.3304 | 0.2580 | 0.3703 |
| full @ ep1300 | 0.2113 | 0.1296 | 0.1577 |
| lead_only @ ep1500 | 0.2367 | 0.1464 | 0.1992 |
| music_only @ ep1950 | 0.0125 | 0.0796 | 0.0561 |

## 3. Per-Pair Breakdown (best checkpoint of each config)

### GT (real follower)  (Epoch = None, overall r = 0.9937)

| Pair | n_slices | overall r | Body mean | Effort mean | Space mean |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01__gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9981 | 0.6051 | 0.3898 | 0.5631 |
| gHO_sBM_cAll_d20_mHO5_ch01__gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9900 | 0.3001 | 0.1694 | 0.3837 |
| gJB_sBM_cAll_d08_mJB5_ch01__gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9967 | 0.4022 | 0.0256 | -0.0291 |
| gJS_sBM_cAll_d01_mJS3_ch01__gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9994 | 0.7884 | 0.4219 | 0.6219 |
| gKR_sBM_cAll_d28_mKR2_ch01__gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9986 | -0.1533 | -0.1744 | -0.0243 |
| gLH_sBM_cAll_d17_mLH4_ch01__gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9808 | 0.4997 | 0.4446 | 0.5642 |
| gLO_sBM_cAll_d13_mLO2_ch01__gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9971 | 0.5358 | 0.4363 | 0.7050 |
| gMH_sBM_cAll_d22_mMH3_ch01__gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9954 | 0.2254 | 0.3292 | 0.2138 |
| gPO_sBM_cAll_d10_mPO1_ch01__gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9923 | 0.1620 | 0.4933 | 0.5596 |
| gWA_sBM_cAll_d25_mWA0_ch01__gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9887 | -0.0617 | 0.0443 | 0.1453 |

### full @ ep1300  (Epoch = 1300, overall r = 0.9923)

| Pair | n_slices | overall r | Body mean | Effort mean | Space mean |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01__gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9951 | 0.4691 | 0.1345 | 0.1726 |
| gHO_sBM_cAll_d20_mHO5_ch01__gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9931 | 0.0923 | 0.0629 | 0.2305 |
| gJB_sBM_cAll_d08_mJB5_ch01__gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9975 | 0.3420 | 0.1667 | 0.0495 |
| gJS_sBM_cAll_d01_mJS3_ch01__gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9790 | 0.0857 | 0.0687 | 0.1720 |
| gKR_sBM_cAll_d28_mKR2_ch01__gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9867 | 0.1024 | 0.2734 | 0.3488 |
| gLH_sBM_cAll_d17_mLH4_ch01__gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9939 | 0.6384 | 0.2142 | 0.0916 |
| gLO_sBM_cAll_d13_mLO2_ch01__gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9935 | 0.0916 | 0.2318 | 0.2830 |
| gMH_sBM_cAll_d22_mMH3_ch01__gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9938 | 0.1133 | 0.0313 | 0.0831 |
| gPO_sBM_cAll_d10_mPO1_ch01__gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9989 | 0.0247 | 0.0321 | 0.0423 |
| gWA_sBM_cAll_d25_mWA0_ch01__gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9917 | 0.1539 | 0.0802 | 0.1034 |

### lead_only @ ep1500  (Epoch = 1500, overall r = 0.9916)

| Pair | n_slices | overall r | Body mean | Effort mean | Space mean |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01__gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9946 | 0.4299 | 0.1747 | 0.1890 |
| gHO_sBM_cAll_d20_mHO5_ch01__gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9926 | 0.0243 | 0.0357 | 0.2781 |
| gJB_sBM_cAll_d08_mJB5_ch01__gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9965 | 0.5297 | 0.2058 | 0.1241 |
| gJS_sBM_cAll_d01_mJS3_ch01__gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9765 | 0.2589 | 0.0501 | 0.1246 |
| gKR_sBM_cAll_d28_mKR2_ch01__gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9803 | 0.1293 | 0.3422 | 0.4418 |
| gLH_sBM_cAll_d17_mLH4_ch01__gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9962 | 0.6217 | 0.2025 | 0.1729 |
| gLO_sBM_cAll_d13_mLO2_ch01__gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9966 | 0.0770 | 0.3035 | 0.3927 |
| gMH_sBM_cAll_d22_mMH3_ch01__gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9942 | 0.1198 | 0.0612 | 0.0913 |
| gPO_sBM_cAll_d10_mPO1_ch01__gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9984 | -0.0032 | 0.0393 | 0.0932 |
| gWA_sBM_cAll_d25_mWA0_ch01__gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9901 | 0.1797 | 0.0487 | 0.0848 |

### music_only @ ep1950  (Epoch = 1950, overall r = 0.9793)

| Pair | n_slices | overall r | Body mean | Effort mean | Space mean |
|---|---:|---:|---:|---:|---:|
| gBR_sBM_cAll_d04_mBR0_ch01__gBR_sBM_cAll_d05_mBR0_ch01 | 14 | 0.9870 | -0.0820 | 0.0995 | 0.0318 |
| gHO_sBM_cAll_d20_mHO5_ch01__gHO_sBM_cAll_d21_mHO5_ch01 | 5 | 0.9929 | 0.0761 | 0.0906 | 0.0049 |
| gJB_sBM_cAll_d08_mJB5_ch01__gJB_sBM_cAll_d09_mJB5_ch01 | 5 | 0.9753 | -0.0286 | 0.0849 | 0.1153 |
| gJS_sBM_cAll_d01_mJS3_ch01__gJS_sBM_cAll_d03_mJS3_ch01 | 8 | 0.9429 | 0.0161 | 0.0454 | 0.0835 |
| gKR_sBM_cAll_d28_mKR2_ch01__gKR_sBM_cAll_d30_mKR2_ch01 | 10 | 0.9797 | 0.0204 | -0.0099 | 0.0111 |
| gLH_sBM_cAll_d17_mLH4_ch01__gLH_sBM_cAll_d18_mLH4_ch01 | 7 | 0.9762 | -0.0227 | 0.1299 | 0.0899 |
| gLO_sBM_cAll_d13_mLO2_ch01__gLO_sBM_cAll_d15_mLO2_ch01 | 10 | 0.9786 | 0.0458 | 0.1112 | 0.0643 |
| gMH_sBM_cAll_d22_mMH3_ch01__gMH_sBM_cAll_d24_mMH3_ch01 | 8 | 0.9875 | 0.0060 | 0.0884 | 0.0075 |
| gPO_sBM_cAll_d10_mPO1_ch01__gPO_sBM_cAll_d11_mPO1_ch01 | 12 | 0.9916 | 0.0819 | 0.0813 | 0.1156 |
| gWA_sBM_cAll_d25_mWA0_ch01__gWA_sBM_cAll_d26_mWA0_ch01 | 14 | 0.9815 | 0.0118 | 0.0746 | 0.0373 |

---

Generated by `eval/run_lma17_sweep.py`. Raw numbers in the companion JSON.