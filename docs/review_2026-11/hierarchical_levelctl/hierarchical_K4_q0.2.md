## warm days: top 20% of the domain-mean T anomaly (threshold +0.36 sigma, 3693 days); years with most days: 2024 (301), 1998 (246), 1987 (217), 1983 (209), 2016 (195), 2023 (194), 2010 (191), 2015 (135)

### warm / eof: k-means on 40 EOFs (63% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 28 | +0.73 | +0.46 (0.002) | +0.47 | +0.33 (0.002) | +0.40 | +0.44 | +0.37 | 1998:68, 1983:66, 1987:64, 2010:59, 2015:59 |
| W1 | 26 | +0.74 | +0.37 (0.002) | +0.39 | +0.40 (0.002) | +0.37 | +0.36 | +0.33 | 2024:149, 2023:103, 1997:47, 1983:43, 2020:42 |
| W2 | 19 | +0.63 | +0.35 (0.002) | +0.35 | +0.30 (0.002) | +0.34 | +0.31 | +0.30 | 1998:60, 1983:53, 1987:46, 2016:44, 2024:43 |
| W3 | 28 | +0.61 | +0.25 (0.002) | +0.25 | +0.36 (0.002) | +0.41 | +0.20 | +0.39 | 1998:97, 1987:82, 2010:73, 2024:61, 2016:58 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| W0 | +0.73 | +0.12 (0.354) | +0.09 (0.600) | +0.03 (0.938) | +0.02 (0.980) |
| W1 | +0.74 | +0.13 (0.241) | +0.09 (0.644) | +0.14 (0.260) | +0.09 (0.567) |
| W2 | +0.63 | -0.02 (0.994) | +0.02 (0.992) | -0.16 (0.175) | -0.11 (0.441) |
| W3 | +0.61 | -0.26 (0.002) | -0.21 (0.022) | -0.06 (0.775) | -0.03 (0.942) |

### warm / msn: k-means on MSN a100-t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 29 | +0.73 | +0.44 (0.002) | +0.45 | +0.40 (0.002) | +0.40 | +0.42 | +0.36 | 1998:118, 2016:105, 2010:77, 2024:77, 1987:66 |
| W1 | 24 | +0.73 | +0.43 (0.002) | +0.44 | +0.37 (0.002) | +0.36 | +0.41 | +0.32 | 2024:122, 2023:76, 1983:61, 1988:50, 1998:48 |
| W2 | 23 | +0.61 | +0.26 (0.002) | +0.26 | +0.33 (0.002) | +0.42 | +0.20 | +0.39 | 1980:55, 2010:55, 1987:54, 2024:53, 1998:44 |
| W3 | 23 | +0.63 | +0.18 (0.050) | +0.18 | +0.23 (0.002) | +0.27 | +0.14 | +0.25 | 1987:58, 1983:52, 2024:49, 1997:45, 2023:45 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| W0 | +0.73 | +0.17 (0.078) | +0.07 (0.759) | +0.06 (0.889) | -0.02 (0.984) |
| W1 | +0.73 | +0.25 (0.002) | +0.13 (0.292) | +0.12 (0.382) | +0.03 (0.972) |
| W2 | +0.61 | -0.24 (0.002) | -0.12 (0.338) | +0.00 (1.000) | +0.09 (0.581) |
| W3 | +0.63 | -0.19 (0.048) | -0.09 (0.610) | -0.16 (0.113) | -0.07 (0.775) |

## cold days: bottom 20% of the domain-mean T anomaly (threshold -0.36 sigma, 3246 days); years with most days: 2000 (191), 1985 (156), 1989 (139), 1986 (137), 2008 (133), 2001 (124), 1984 (123), 1999 (115)

### cold / eof: k-means on 40 EOFs (67% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 16 | -0.84 | -0.06 (0.654) | -0.07 | -0.06 (0.696) | -0.04 | -0.06 | -0.03 | 2018:21, 1990:19, 2005:19, 1986:18, 2000:18 |
| C1 | 25 | -0.63 | -0.24 (0.002) | -0.24 | -0.19 (0.010) | -0.18 | -0.22 | -0.15 | 2000:45, 2022:40, 1985:38, 1984:34, 1989:34 |
| C2 | 34 | -0.54 | -0.32 (0.002) | -0.32 | -0.30 (0.002) | -0.32 | -0.29 | -0.28 | 2000:84, 1989:64, 2001:55, 1984:54, 1986:52 |
| C3 | 26 | -0.55 | -0.33 (0.002) | -0.33 | -0.24 (0.002) | -0.26 | -0.30 | -0.22 | 1985:59, 2008:48, 2000:44, 2011:43, 1999:38 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| C0 | -0.84 | +0.22 (0.002) | +0.13 (0.145) | +0.22 (0.008) | +0.05 (0.909) |
| C1 | -0.63 | +0.11 (0.318) | +0.10 (0.425) | +0.16 (0.099) | +0.12 (0.302) |
| C2 | -0.54 | -0.14 (0.095) | -0.07 (0.720) | -0.20 (0.024) | -0.08 (0.640) |
| C3 | -0.55 | -0.16 (0.024) | -0.13 (0.147) | -0.15 (0.125) | -0.07 (0.712) |

### cold / msn: k-means on MSN a100-t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 19 | -0.82 | -0.09 (0.362) | -0.09 | -0.06 (0.702) | -0.08 | -0.08 | -0.06 | 2018:27, 2022:27, 1992:20, 2000:20, 1990:19 |
| C1 | 26 | -0.59 | -0.19 (0.004) | -0.20 | -0.28 (0.004) | -0.26 | -0.16 | -0.24 | 1985:53, 1986:40, 2000:40, 1989:38, 1992:36 |
| C2 | 24 | -0.54 | -0.22 (0.002) | -0.23 | -0.26 (0.010) | -0.22 | -0.20 | -0.19 | 2000:90, 1985:67, 1984:55, 2001:54, 1999:48 |
| C3 | 31 | -0.56 | -0.35 (0.002) | -0.35 | -0.16 (0.085) | -0.22 | -0.32 | -0.17 | 2008:58, 2011:56, 1989:54, 2012:46, 2014:46 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| C0 | -0.82 | +0.22 (0.002) | +0.13 (0.135) | +0.18 (0.105) | +0.00 (1.000) |
| C1 | -0.59 | +0.17 (0.030) | +0.16 (0.048) | -0.03 (0.964) | -0.04 (0.960) |
| C2 | -0.54 | +0.02 (0.990) | +0.09 (0.390) | -0.05 (0.859) | +0.03 (0.984) |
| C3 | -0.56 | -0.31 (0.002) | -0.28 (0.002) | -0.09 (0.565) | +0.01 (1.000) |

