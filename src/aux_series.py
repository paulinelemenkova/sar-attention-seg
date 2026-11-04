import glob
import re
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

IN, OUT = sys.argv[1], sys.argv[2]
with rasterio.open(f"{IN}/static_100m.tif") as d:
    elev100 = d.read(1).astype(float)
    land100 = d.read(4).astype(float)
e = elev100[5:5 + 469 * 5, 5:5 + 789 * 5].reshape(469, 5, 789, 5).mean(axis=(1, 3))
lf = land100[5:5 + 469 * 5, 5:5 + 789 * 5].reshape(469, 5, 789, 5).mean(axis=(1, 3))
BANDS = [(1500, 2000), (2000, 2500), (2500, 3000), (3000, 3500), (3500, 5200)]


def ffill_bfill(a):
    a = a.astype(np.int16)
    a[a == 255] = -1
    n = a.shape[0]
    out = a.copy()
    for t in range(1, n):
        m = out[t] < 0
        out[t][m] = out[t - 1][m]
    for t in range(n - 2, -1, -1):
        m = out[t] < 0
        out[t][m] = out[t + 1][m]
    return out


rows, brows, sca, sddmaps = [], [], {}, {}
for f in sorted(glob.glob(f"{IN}/modis_snow_*.tif")):
    year = int(re.findall(r"(\d{4})", f)[-1])
    with rasterio.open(f) as d:
        a = d.read()
    nd, ny, nx = a.shape
    inside = (a != 255).any(axis=0) & (lf >= 0.5)
    nodata_frac = (a[:, inside] == 255).mean()
    x = ffill_bfill(a[:, inside])
    doy = np.arange(1, nd + 1)
    jm = doy <= (91 if year % 4 else 92)
    seasonal = x[jm].sum(axis=0) >= 10
    scd = x.sum(axis=0)
    free = (x == 0).astype(np.int32)
    run8 = np.lib.stride_tricks.sliding_window_view(free, 8, axis=0).sum(axis=-1) == 8
    run8[:31] = False
    has = run8.any(axis=0)
    sdd = np.where(has, run8.argmax(axis=0) + 1, np.nan).astype(float)
    sdd[~seasonal] = np.nan
    el = e[inside]
    sca[year] = 100 * x.mean(axis=1)
    m = np.full((ny, nx), np.nan)
    m[inside] = sdd
    sddmaps[year] = m
    ok = np.isfinite(sdd) & (el >= 1500) & (sdd > 32)
    slope = stats.theilslopes(sdd[ok][::25], el[ok][::25])[0] * 100 if ok.sum() > 100 else np.nan
    rows.append({"year": year, "nodata_pct": 100 * nodata_frac, "seasonal_pct": 100 * seasonal.mean(),
                 "sca_peak_pct": sca[year].max(), "sca_peak_doy": int(sca[year].argmax() + 1),
                 "sca_1apr": sca[year][90], "sca_1may": sca[year][120],
                 "scd_mean_seasonal": scd[seasonal].mean(), "sdd_median": np.nanmedian(sdd),
                 "sdd_slope_d100m": slope})
    for lo, hi in BANDS:
        sel = (el >= lo) & (el < hi)
        brows.append({"year": year, "band": f"{lo}-{hi}", "sdd_median": np.nanmedian(sdd[sel]),
                      "scd_mean": scd[sel & seasonal].mean()})
T = pd.DataFrame(rows)
B = pd.DataFrame(brows)
T.to_csv(f"{OUT}/modis_years.csv", index=False, float_format="%.2f")
B.to_csv(f"{OUT}/modis_years_bands.csv", index=False, float_format="%.1f")
yrs = np.array(sorted(sca))
L = max(len(v) for v in sca.values())
S = np.full((len(yrs), L), np.nan)
for i, y in enumerate(yrs):
    S[i, :len(sca[y])] = sca[y]
np.savez_compressed(f"{OUT}/modis_years.npz", years=yrs, sca=S,
                    sdd=np.stack([sddmaps[y] for y in yrs]).astype(np.float32), elev=e.astype(np.float32),
                    bands_sdd=B.pivot(index="year", columns="band", values="sdd_median").to_numpy())
print(T.round(1).to_string())
print(B.pivot(index="year", columns="band", values="sdd_median").round(0).to_string())
for col in ["sdd_median", "scd_mean_seasonal", "sca_1apr"]:
    r = stats.theilslopes(T[col], T.year)
    k = stats.kendalltau(T.year, T[col])
    print(f"trend {col}: {10 * r[0]:+.1f} per decade, Kendall p = {k.pvalue:.2f}")
