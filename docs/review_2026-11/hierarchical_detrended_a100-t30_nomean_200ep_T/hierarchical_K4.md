## warm days: top 20% of the domain-mean T anomaly (threshold +0.36 sigma, 3693 days); years with most days: 2024 (301), 1998 (246), 1987 (217), 1983 (209), 2016 (195), 2023 (194), 2010 (191), 2015 (135)

### warm / eof: k-means on 40 EOFs (66% var) of [T minus daily domain mean]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 22 | +0.70 | +0.39 (0.002) | +0.39 | +0.27 (0.002) | +0.33 | +0.36 | +0.29 | 1998:78, 2016:68, 2010:60, 2015:48, 1987:47 |
| W1 | 16 | +0.70 | +0.37 (0.002) | +0.38 | +0.39 (0.002) | +0.45 | +0.33 | +0.42 | 1998:77, 1983:63, 1987:60, 2024:59, 2010:38 |
| W2 | 25 | +0.74 | +0.37 (0.002) | +0.39 | +0.41 (0.002) | +0.38 | +0.36 | +0.34 | 2024:152, 2023:100, 1997:47, 1983:41, 2009:41 |
| W3 | 37 | +0.63 | +0.30 (0.002) | +0.31 | +0.33 (0.002) | +0.38 | +0.26 | +0.34 | 1987:84, 2024:72, 1998:71, 2019:67, 2010:61 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.03 (0.980) | -0.05 (0.889) |
| W1 | +0.11 (0.563) | +0.26 (0.018) |
| W2 | +0.15 (0.239) | +0.16 (0.203) |
| W3 | -0.23 (0.018) | -0.25 (0.026) |

### warm / msn: k-means on MSN a100-t30_nomean_200ep embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 25 | +0.75 | +0.43 (0.002) | +0.44 | +0.37 (0.002) | +0.36 | +0.41 | +0.32 | 2024:127, 2023:83, 1983:63, 1998:52, 1988:50 |
| W1 | 28 | +0.70 | +0.42 (0.002) | +0.43 | +0.39 (0.002) | +0.41 | +0.40 | +0.37 | 1998:118, 2016:106, 2010:85, 2024:71, 1987:63 |
| W2 | 24 | +0.62 | +0.27 (0.002) | +0.26 | +0.31 (0.002) | +0.38 | +0.22 | +0.36 | 1980:55, 2024:51, 1987:50, 2010:49, 1983:37 |
| W3 | 23 | +0.64 | +0.19 (0.042) | +0.19 | +0.24 (0.002) | +0.28 | +0.15 | +0.26 | 1987:61, 1983:52, 2023:52, 2024:52, 1997:40 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.24 (0.004) | +0.11 (0.457) |
| W1 | +0.14 (0.256) | +0.06 (0.841) |
| W2 | -0.22 (0.016) | -0.04 (0.938) |
| W3 | -0.18 (0.052) | -0.13 (0.296) |

## cold days: bottom 20% of the domain-mean T anomaly (threshold -0.36 sigma, 3246 days); years with most days: 2000 (191), 1985 (156), 1989 (139), 1986 (137), 2008 (133), 2001 (124), 1984 (123), 1999 (115)

### cold / eof: k-means on 40 EOFs (70% var) of [T minus daily domain mean]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 12 | -0.91 | -0.06 (0.759) | -0.06 | +0.01 (1.000) | +0.02 | -0.07 | +0.03 | 1990:19, 2022:17, 2003:14, 2005:14, 1987:13 |
| C1 | 19 | -0.60 | -0.23 (0.002) | -0.24 | -0.29 (0.002) | -0.30 | -0.19 | -0.27 | 1986:32, 1992:31, 2014:31, 1985:30, 2000:30 |
| C2 | 32 | -0.55 | -0.30 (0.002) | -0.31 | -0.24 (0.002) | -0.24 | -0.28 | -0.20 | 1985:74, 2000:59, 2008:56, 1986:52, 2011:50 |
| C3 | 38 | -0.58 | -0.34 (0.002) | -0.35 | -0.27 (0.002) | -0.30 | -0.31 | -0.26 | 2000:94, 1989:69, 1984:65, 2001:62, 1999:53 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.17 (0.022) | +0.24 (0.002) |
| C1 | +0.09 (0.491) | -0.08 (0.612) |
| C2 | -0.04 (0.907) | -0.00 (1.000) |
| C3 | -0.16 (0.032) | -0.15 (0.080) |

### cold / msn: k-means on MSN a100-t30_nomean_200ep embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 19 | -0.82 | -0.08 (0.451) | -0.08 | -0.05 (0.789) | -0.06 | -0.07 | -0.05 | 2018:26, 2022:25, 2000:21, 1992:20, 1999:19 |
| C1 | 25 | -0.60 | -0.18 (0.020) | -0.19 | -0.25 (0.010) | -0.24 | -0.16 | -0.21 | 1985:50, 2000:40, 1989:39, 1986:37, 1992:37 |
| C2 | 23 | -0.54 | -0.21 (0.002) | -0.23 | -0.27 (0.002) | -0.23 | -0.19 | -0.20 | 2000:84, 1985:69, 1984:54, 2001:51, 1999:45 |
| C3 | 33 | -0.55 | -0.36 (0.002) | -0.35 | -0.17 (0.056) | -0.24 | -0.33 | -0.19 | 2008:61, 2011:57, 1989:54, 2014:50, 1986:47 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.23 (0.002) | +0.20 (0.070) |
| C1 | +0.15 (0.062) | +0.02 (0.994) |
| C2 | +0.03 (0.978) | -0.08 (0.722) |
| C3 | -0.30 (0.002) | -0.13 (0.360) |

