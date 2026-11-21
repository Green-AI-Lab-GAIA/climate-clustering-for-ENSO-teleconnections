## warm days: top 30% of the domain-mean T anomaly (threshold +0.21 sigma, 5378 days); years with most days: 2024 (328), 1998 (272), 1987 (256), 2010 (236), 1983 (230), 2016 (229), 2023 (224), 1980 (185)

### warm / eof: k-means on 40 EOFs (63% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 31 | +0.59 | +0.40 (0.002) | +0.40 | +0.34 (0.002) | +0.41 | +0.37 | +0.37 | 1980:85, 2010:85, 1998:80, 1987:77, 1983:67 |
| W1 | 25 | +0.62 | +0.39 (0.002) | +0.41 | +0.41 (0.002) | +0.39 | +0.37 | +0.35 | 2024:161, 2023:119, 2020:64, 1997:61, 1995:57 |
| W2 | 19 | +0.54 | +0.36 (0.002) | +0.37 | +0.30 (0.002) | +0.33 | +0.33 | +0.29 | 1998:68, 1987:64, 1983:62, 2024:53, 2016:52 |
| W3 | 25 | +0.47 | +0.22 (0.002) | +0.23 | +0.29 (0.002) | +0.32 | +0.18 | +0.29 | 1998:100, 1987:80, 2010:72, 2016:71, 2019:67 |

Composition among warm days (310 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | -0.01 (0.996) | +0.00 (1.000) |
| W1 | +0.15 (0.064) | +0.17 (0.076) |
| W2 | +0.06 (0.740) | -0.05 (0.779) |
| W3 | -0.20 (0.012) | -0.14 (0.151) |

### warm / msn: k-means on MSN t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 23 | +0.61 | +0.44 (0.002) | +0.45 | +0.35 (0.002) | +0.36 | +0.42 | +0.32 | 2024:130, 2023:82, 1983:65, 1988:65, 1998:60 |
| W1 | 29 | +0.60 | +0.41 (0.002) | +0.43 | +0.43 (0.002) | +0.40 | +0.40 | +0.37 | 2016:130, 1998:128, 2024:99, 2010:92, 1987:83 |
| W2 | 26 | +0.53 | +0.19 (0.038) | +0.19 | +0.25 (0.002) | +0.29 | +0.14 | +0.27 | 1983:70, 2024:70, 1987:68, 1997:65, 2023:60 |
| W3 | 21 | +0.47 | +0.17 (0.054) | +0.17 | +0.18 (0.052) | +0.26 | +0.13 | +0.24 | 2002:62, 1980:59, 2010:56, 1987:48, 1995:40 |

Composition among warm days (310 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| W0 | +0.25 (0.002) | +0.13 (0.252) |
| W1 | +0.13 (0.161) | +0.09 (0.541) |
| W2 | -0.13 (0.129) | -0.09 (0.559) |
| W3 | -0.26 (0.002) | -0.13 (0.233) |

## cold days: bottom 30% of the domain-mean T anomaly (threshold -0.23 sigma, 4863 days); years with most days: 2000 (249), 1985 (211), 1989 (201), 2008 (185), 1986 (183), 1999 (182), 1984 (180), 2001 (177)

### cold / eof: k-means on 40 EOFs (66% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 13 | -0.79 | -0.07 (0.608) | -0.08 | -0.02 (0.990) | -0.01 | -0.08 | +0.01 | 1990:23, 2013:23, 2000:22, 2004:22, 2003:21 |
| C1 | 19 | -0.47 | -0.29 (0.002) | -0.29 | -0.31 (0.002) | -0.35 | -0.25 | -0.32 | 2014:50, 1992:46, 2012:34, 2000:32, 2013:32 |
| C2 | 27 | -0.47 | -0.34 (0.002) | -0.35 | -0.26 (0.002) | -0.27 | -0.32 | -0.23 | 1985:84, 2000:65, 1999:62, 2008:59, 2011:58 |
| C3 | 40 | -0.46 | -0.37 (0.002) | -0.37 | -0.32 (0.002) | -0.34 | -0.34 | -0.30 | 2000:130, 1989:103, 1984:95, 2001:89, 1985:85 |

Composition among cold days (333 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.20 (0.002) | +0.28 (0.002) |
| C1 | +0.09 (0.392) | -0.10 (0.362) |
| C2 | -0.10 (0.239) | -0.00 (1.000) |
| C3 | -0.14 (0.056) | -0.18 (0.006) |

### cold / msn: k-means on MSN t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 23 | -0.67 | -0.11 (0.195) | -0.11 | -0.10 (0.404) | -0.12 | -0.09 | -0.10 | 2022:41, 2000:40, 2013:36, 2018:36, 1992:35 |
| C1 | 27 | -0.45 | -0.21 (0.002) | -0.23 | -0.30 (0.002) | -0.27 | -0.19 | -0.23 | 2000:124, 1985:100, 2001:86, 1999:79, 1984:75 |
| C2 | 21 | -0.46 | -0.28 (0.002) | -0.29 | -0.20 (0.022) | -0.23 | -0.26 | -0.19 | 1985:48, 2011:44, 1997:41, 2008:39, 1986:38 |
| C3 | 29 | -0.46 | -0.36 (0.002) | -0.36 | -0.22 (0.012) | -0.27 | -0.33 | -0.23 | 1989:79, 2008:70, 2011:66, 1984:61, 2012:58 |

Composition among cold days (333 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

| sub | share r ONI given TNA (p) | share r TNA given ONI (p) |
|---|---|---|
| C0 | +0.20 (0.002) | +0.18 (0.044) |
| C1 | +0.04 (0.905) | -0.09 (0.535) |
| C2 | -0.02 (0.978) | +0.03 (0.956) |
| C3 | -0.23 (0.002) | -0.13 (0.191) |

