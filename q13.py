import pandas as pd
import numpy as np
from pathlib import Path

INPUT_XLSX = "XYZ.xlsx"
OUTPUT_XLSX = "XYZ_with_Rf_Rg.xlsx"

WL_CANDIDATES = {"Wavelength (nm)", "Wavelength", "λ", "lambda", "波长", "波长(nm)"}
PREFERRED_SPD_COL = "光强"

from colour import SpectralDistribution, SpectralShape
from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018  # ✅ 0.4.6 正确入口

shape = SpectralShape(380, 780, 1)

df = pd.read_excel(INPUT_XLSX)

# 找波长列
wl_col = None
lower_map = {str(c).strip().lower(): c for c in df.columns}
for cand in WL_CANDIDATES:
    key = cand.lower()
    if key in lower_map:
        wl_col = lower_map[key]
        break
if wl_col is None:
    wl_col = df.columns[0]

# 找 SPD 列
if PREFERRED_SPD_COL in df.columns:
    spd_cols = [PREFERRED_SPD_COL]
else:
    spd_cols = [c for c in df.columns if c != wl_col and pd.api.types.is_numeric_dtype(df[c])]

if not spd_cols:
    raise ValueError("未找到任何 SPD 列")

# 清洗波长
wls = pd.to_numeric(df[wl_col], errors="coerce").to_numpy()
valid_rows = np.isfinite(wls)
df_valid = df.loc[valid_rows].copy()
wls = wls[valid_rows]

df_out = df.copy()
summary = []

for col in spd_cols:
    vals = pd.to_numeric(df_valid[col], errors="coerce").to_numpy()
    mask = np.isfinite(wls) & np.isfinite(vals)
    w = wls[mask]
    v = vals[mask]

    if len(w) < 10:
        summary.append({"SPD": col, "Rf": np.nan, "Rg": np.nan, "备注": "有效数据点太少"})
        df_out[f"Rf[{col}]"] = np.nan
        df_out[f"Rg[{col}]"] = np.nan
        continue

    sd = SpectralDistribution(dict(zip(w, v))).align(shape)

    spec = colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)

    Rf = float(spec.R_f)   # 保真度指数
    Rg = float(spec.R_g)   # 色域指数
    print(Rf)
    print(Rg)

    summary.append({"SPD": col, "Rf": Rf, "Rg": Rg})
    df_out[f"Rf[{col}]"] = Rf
    df_out[f"Rg[{col}]"] = Rg
"""
# 保存
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df_out.to_excel(writer, index=False, sheet_name="Data+RfRg")
    pd.DataFrame(summary).to_excel(writer, index=False, sheet_name="TM30-18_Summary")

print(f"完成，结果已保存到 {OUTPUT_XLSX}")
"""