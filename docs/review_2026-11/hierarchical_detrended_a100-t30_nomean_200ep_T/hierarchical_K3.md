## warm days: top 20% of the domain-mean T anomaly (threshold +0.36 sigma, 3693 days); years with most days: 2024 (301), 1998 (246), 1987 (217), 1983 (209), 2016 (195), 2023 (194), 2010 (191), 2015 (135)

### warm / eof: k-means on 40 EOFs (66% var) of [T minus daily domain mean]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 20 | +0.70 | +0.39 (0.002) | +0.39 | +0.37 (0.002) | +0.45 | +0.36 | +0.42 | 1998:95, 1983:78, 1987:69, 2024:66, 2010:43 |
| W1 | 36 | +0.71 | +0.39 (0.002) | +0.41 | +0.43 (0.002) | +0.40 | +0.37 | +0.36 | 2024:174, 2023:125, 1997:63, 2020:61, 2009:56 |
| W2 | 44 | +0.65 | +0.38 (0.002) | +0.38 | +0.33 (0.002) | +0.40 | +0.34 | +0.36 | 1998:122, 1987:108, 2016:108, 2010:102, 1980:78 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.11 (0.465) | +0.23 (0.054) |
| W1 | +0.06 (0.849) | +0.08 (0.680) |
| W2 | -0.12 (0.406) | -0.22 (0.076) |

### warm / msn: k-means on MSN a100-t30_nomean_200ep embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 30 | +0.75 | +0.45 (0.002) | +0.47 | +0.39 (0.002) | +0.38 | +0.44 | +0.34 | 2024:141, 2023:95, 1983:75, 1998:71, 2015:58 |
| W1 | 37 | +0.67 | +0.42 (0.002) | +0.43 | +0.38 (0.002) | +0.42 | +0.40 | +0.39 | 1998:129, 2016:112, 2010:101, 1987:86, 2024:74 |
| W2 | 33 | +0.63 | +0.18 (0.044) | +0.18 | +0.28 (0.002) | +0.34 | +0.13 | +0.31 | 2024:86, 1987:78, 1983:66, 2023:64, 1980:55 |

Composition among warm days (240 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.24 (0.004) | +0.11 (0.374) |
| W1 | +0.09 (0.529) | +0.01 (0.994) |
| W2 | -0.28 (0.002) | -0.10 (0.433) |

## cold days: bottom 20% of the domain-mean T anomaly (threshold -0.36 sigma, 3246 days); years with most days: 2000 (191), 1985 (156), 1989 (139), 1986 (137), 2008 (133), 2001 (124), 1984 (123), 1999 (115)

### cold / eof: k-means on 40 EOFs (70% var) of [T minus daily domain mean]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 18 | -0.83 | -0.08 (0.429) | -0.08 | -0.07 (0.451) | -0.06 | -0.07 | -0.05 | 2018:23, 2022:22, 1990:21, 2005:20, 1980:19 |
| C1 | 36 | -0.55 | -0.30 (0.002) | -0.31 | -0.25 (0.002) | -0.26 | -0.28 | -0.22 | 1985:81, 2008:63, 1986:62, 2000:58, 2011:53 |
| C2 | 46 | -0.58 | -0.35 (0.002) | -0.36 | -0.29 (0.002) | -0.32 | -0.32 | -0.27 | 2000:117, 1989:79, 2001:73, 1984:71, 1985:62 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.22 (0.002) | +0.21 (0.002) |
| C1 | -0.06 (0.594) | -0.04 (0.859) |
| C2 | -0.15 (0.022) | -0.16 (0.030) |

### cold / msn: k-means on MSN a100-t30_nomean_200ep embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 26 | -0.78 | -0.10 (0.199) | -0.10 | -0.10 (0.310) | -0.11 | -0.09 | -0.09 | 2000:36, 2022:32, 2018:31, 2021:28, 1986:27 |
| C1 | 34 | -0.55 | -0.21 (0.002) | -0.23 | -0.30 (0.002) | -0.25 | -0.19 | -0.21 | 2000:105, 1985:94, 1984:65, 2001:62, 1999:56 |
| C2 | 40 | -0.56 | -0.36 (0.002) | -0.36 | -0.19 (0.028) | -0.25 | -0.33 | -0.21 | 2008:72, 2011:62, 1989:61, 2014:56, 1986:55 |

Composition among cold days (248 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.24 (0.002) | +0.19 (0.066) |
| C1 | +0.11 (0.254) | -0.05 (0.763) |
| C2 | -0.30 (0.002) | -0.12 (0.328) |

