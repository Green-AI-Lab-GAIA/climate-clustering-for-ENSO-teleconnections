## warm days: top 20% of the domain-mean T anomaly (threshold +0.36 sigma, 3693 days); years with most days: 2024 (301), 1998 (246), 1987 (217), 1983 (209), 2016 (195), 2023 (194), 2010 (191), 2015 (135)

### warm / eof: k-means on 40 EOFs (63% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 43 | +0.67 | +0.40 (0.002) | +0.41 | +0.38 (0.002) | +0.45 | +0.37 | +0.42 | 1998:127, 1987:107, 2010:105, 2016:90, 1983:82 |
| W1 | 32 | +0.72 | +0.39 (0.002) | +0.41 | +0.44 (0.002) | +0.41 | +0.38 | +0.38 | 2024:164, 2023:117, 1997:53, 1983:52, 2020:50 |
| W2 | 25 | +0.64 | +0.37 (0.002) | +0.37 | +0.31 (0.002) | +0.37 | +0.33 | +0.33 | 1998:90, 1983:75, 1987:75, 2016:62, 2024:57 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | -0.07 (0.628) | -0.02 (0.950) |
| W1 | +0.09 (0.529) | +0.13 (0.282) |
| W2 | -0.01 (0.988) | -0.13 (0.320) |

### warm / msn: k-means on MSN t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 28 | +0.75 | +0.45 (0.002) | +0.46 | +0.38 (0.002) | +0.38 | +0.43 | +0.34 | 2024:140, 2023:93, 1983:68, 1998:60, 1988:50 |
| W1 | 37 | +0.69 | +0.44 (0.002) | +0.45 | +0.39 (0.002) | +0.41 | +0.42 | +0.38 | 1998:137, 2016:113, 2010:100, 1987:85, 2024:81 |
| W2 | 35 | +0.62 | +0.18 (0.050) | +0.17 | +0.28 (0.002) | +0.34 | +0.12 | +0.32 | 1987:83, 2024:80, 1983:74, 1980:66, 2023:58 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.26 (0.002) | +0.12 (0.256) |
| W1 | +0.10 (0.406) | +0.00 (1.000) |
| W2 | -0.30 (0.002) | -0.10 (0.445) |

## cold days: bottom 20% of the domain-mean T anomaly (threshold -0.36 sigma, 3246 days); years with most days: 2000 (191), 1985 (156), 1989 (139), 1986 (137), 2008 (133), 2001 (124), 1984 (123), 1999 (115)

### cold / eof: k-means on 40 EOFs (67% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 20 | -0.84 | -0.08 (0.394) | -0.08 | -0.07 (0.431) | -0.05 | -0.07 | -0.04 | 2000:24, 2018:23, 2022:23, 1986:21, 1990:21 |
| C1 | 32 | -0.56 | -0.33 (0.002) | -0.34 | -0.26 (0.002) | -0.29 | -0.30 | -0.25 | 1985:65, 2000:64, 2008:53, 2011:51, 1986:49 |
| C2 | 49 | -0.56 | -0.34 (0.002) | -0.34 | -0.29 (0.002) | -0.30 | -0.31 | -0.26 | 2000:103, 1989:84, 1984:78, 1985:77, 2001:74 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.24 (0.002) | +0.25 (0.002) |
| C1 | -0.15 (0.016) | -0.17 (0.010) |
| C2 | -0.08 (0.334) | -0.08 (0.471) |

### cold / msn: k-means on MSN t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 26 | -0.78 | -0.10 (0.223) | -0.10 | -0.10 (0.312) | -0.11 | -0.09 | -0.09 | 2000:35, 2018:32, 2022:30, 1986:27, 1985:26 |
| C1 | 34 | -0.55 | -0.22 (0.002) | -0.23 | -0.30 (0.002) | -0.25 | -0.20 | -0.22 | 2000:104, 1985:99, 1984:66, 2001:63, 1999:58 |
| C2 | 39 | -0.56 | -0.37 (0.002) | -0.36 | -0.19 (0.032) | -0.26 | -0.33 | -0.21 | 2008:65, 2011:62, 1989:60, 2014:55, 2018:54 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.25 (0.002) | +0.18 (0.085) |
| C1 | +0.09 (0.425) | -0.08 (0.626) |
| C2 | -0.29 (0.002) | -0.10 (0.453) |

