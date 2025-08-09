import numpy as np
import pandas as pd
import colour
from scipy.optimize import minimize, LinearConstraint, Bounds
from colour import SpectralDistribution, SpectralShape
from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018

INPUT_XLSX = r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx"
SHEET_NAME = 0
COL_WL = "Wavelength (nm)"
COLS_CHANNELS = ["Blue", "Green", "Red", "Warm White", "Cold White"]
COL_V = "V(λ)"
COL_MEL = "melanopic"

# 正午日光目标
CCT_MIN, CCT_MAX = 5500.0, 6500.0   # = 6000 ± 500 K
RG_MIN, RG_MAX   = 95.0, 105.0
RF_FLOOR         = 88.0

# 初始权重
w_seed = np.array([0.000561, 0.050749, 0.000001, 0.541419, 0.407270], dtype=float)

#计算函数
def combine_intensity(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    """按权重线性叠加 5 个通道，得到 SPD 强度列（数值稳定下限裁剪）"""
    intens = np.zeros(len(df), dtype=float)
    for wi, col in zip(w, COLS_CHANNELS):
        #zip(w, COLS_CHANNELS) 把权重和列名配对(w[0], 'channel_red')
        intens += wi * pd.to_numeric(df[col], errors="coerce").to_numpy()#errors="coerce"如果某个值无法转换，就把它变成 NaN
    return np.clip(intens, 1e-9, None)
#np.clip(array, a_min, a_max)：将数组中的值限制在 [a_min, a_max] 范围内。


#计算cct函数
def calculate_cct(df_xy: pd.DataFrame) -> float:
    """
    用 colour 库函数计算 CCT（df 需含 'Wavelength (nm)' 与 'intensity'）
    """
    wavelengths = pd.to_numeric(df_xy["Wavelength (nm)"], errors="coerce").to_numpy()
    intensities = pd.to_numeric(df_xy["intensity"], errors="coerce").to_numpy()

    mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    wavelengths = wavelengths[mask]
    intensities = intensities[mask]

    if len(wavelengths) == 0 or np.all(intensities <= 0):
        return np.nan

    # 归一化强度对 CCT 无影响（只影响亮度）
    intensities = intensities / np.max(intensities)

    sd = SpectralDistribution(dict(zip(wavelengths, intensities)))
    XYZ = colour.sd_to_XYZ(sd)
    xy = colour.XYZ_to_xy(XYZ)
    cct = colour.xy_to_CCT(xy)
    try:
        return float(cct)
    except Exception:
        return float(cct[0])


#计算Rf，Rg函数 
def tm30_Rf_Rg(wavelengths_nm: np.ndarray, intensity: np.ndarray) -> tuple[float, float]:
    """TM-30 Rf、Rg（将 SPD 对齐到 380–780 nm，Δλ=1nm）"""
    shape = SpectralShape(380, 780, 1)
    sd = SpectralDistribution(dict(zip(wavelengths_nm, intensity))).align(shape)
    spec = colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)
    if hasattr(spec, "R_f"):
        return float(spec.R_f), float(spec.R_g)
    else: 
        return float(spec["R_f"]), float(spec["R_g"])


#计算mel_DER函数
def calc_mel_DER(df: pd.DataFrame, intensity: np.ndarray) -> float:
    """mel-DER = (∑ SPD*melanopic) / (∑ SPD*V(λ)) × 1.218"""
    mel = pd.to_numeric(df[COL_MEL], errors="coerce").to_numpy()
    V   = pd.to_numeric(df[COL_V],   errors="coerce").to_numpy()
    num = float(np.nansum(intensity * mel))
    den = float(np.nansum(intensity * V))
    if den <= 1e-12:
        return np.inf
    return num * 1.218 / den

#优化：最大化 Rf（正午日光模式）
def optimize_max_Rf_noon(w0: np.ndarray):
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    wavelengths = pd.to_numeric(df[COL_WL], errors="coerce").to_numpy()

    # 线性约束 & 边界：sum(w)=1, w_i >= 1e-6
    n = len(COLS_CHANNELS)
    lc   = LinearConstraint(np.ones((1, n)), [1.0], [1.0])
#LinearConstraint表示优化问题中的线性约束条件:形如lb≤A·x≤ub的约束,参数1：np.ones((1, n))：这是约束的系数矩阵A；参数2：约束的下界lb参数3：约束的上界 ub
#这里即为等式约束，即w1+w2...+wn=1    
    bnds = Bounds(np.full(n, 1e-6), np.ones(n))
#Bounds 用于为优化变量设置上下界约束，参数1：所有变量的下界数组，参数2：所有变量的上界数组；np.ones(n)：创建长度为n的数组
#np.full(n, 1e-6) 作用是创建一个长度（或元素总数）为 n 的数组，其中所有元素的值都被填充为 1e-6。
    
    # 非线性不等式约束（全部写成 >= 0）
    def cons_noon(w: np.ndarray) -> np.ndarray:
        intensity = combine_intensity(df, w)

        # CCT（库函数）
        df_tmp = df[[COL_WL]].copy()
        df_tmp["intensity"] = intensity
        CCT = calculate_cct(df_tmp)

        # 计算Rf函数tm30_Rf_Rg
        Rf, Rg = tm30_Rf_Rg(wavelengths, intensity)

        if not (np.isfinite(CCT) and np.isfinite(Rf) and np.isfinite(Rg)):
            #isfinite检查哪些元素是有限的,有限则返回true
            # 直接标成不可行
            return np.array([-1.0, -1.0, -1.0, -1.0, -1.0])

        return np.array([
            CCT - CCT_MIN,        # >=0  (CCT 下限)
            CCT_MAX - CCT,        # >=0  (CCT 上限)
            Rg  - RG_MIN,         # >=0
            RG_MAX - Rg,          # >=0
            Rf  - RF_FLOOR        # >=0
        ])

    nlc = {'type': 'ineq', 'fun': cons_noon}

    # 目标：最大化 Rf → 最小化 -Rf
    def neg_Rf(w: np.ndarray) -> float:
        intensity = combine_intensity(df, w)
        Rf, _ = tm30_Rf_Rg(wavelengths, intensity)
        return 1e6 if not np.isfinite(Rf) else -Rf
    #如果 Rf 有效 → 返回 -Rf；RF为无限，则返回1e6
    #初始点净化
    w0 = np.clip(np.asarray(w0, float), 1e-6, 1.0)
    w0 = w0 / w0.sum()

    res = minimize(
        neg_Rf, w0, method='SLSQP',
        bounds=bnds, constraints=[lc, nlc],
        options={'maxiter': 900, 'ftol': 1e-9, 'disp': True}
    )

    # 结果
    w_star = res.x
    intensity = combine_intensity(df, w_star)
    df_tmp = df[[COL_WL]].copy()
    df_tmp["intensity"] = intensity

    CCT = calculate_cct(df_tmp)
    Rf, Rg = tm30_Rf_Rg(wavelengths, intensity)
    mel_DER = calc_mel_DER(df, intensity)

    print("\n最大化Rf")
    print(f"success: {res.success}, message: {res.message}")
    for name, wi in zip(COLS_CHANNELS, w_star):
        print(f"{name:>10s}: {wi:.6f}")
    print(f"CCT   : {CCT:.2f} K  (目标 6000±500K)")
    print(f"Rf    : {Rf:.2f}    (≥88，越高越好)")
    print(f"Rg    : {Rg:.2f}    (95~105)")
    print(f"melDER: {mel_DER:.6f}")

    # 保存
    out = df[[COL_WL]].copy()
    out["intensity"] = intensity
    for name, wi in zip(COLS_CHANNELS, w_star):
        out[f"w*{name}"] = wi * pd.to_numeric(df[name], errors="coerce").to_numpy()
    meta = pd.DataFrame({
        "Metric": ["CCT[K]", "Rf", "Rg", "mel-DER"] + [f"w_{n}" for n in COLS_CHANNELS],
        "Value":  [CCT, Rf, Rg, mel_DER] + list(w_star),
    })
    with pd.ExcelWriter("noon_Rf_max_result.xlsx", engine="openpyxl") as ew:
        out.to_excel(ew, index=False, sheet_name="combined_spectrum")
        meta.to_excel(ew, index=False, sheet_name="metrics")

    return w_star, {'CCT': CCT, 'Rf': Rf, 'Rg': Rg, 'melDER': mel_DER}, res


if __name__ == "__main__":
    optimize_max_Rf_noon(w_seed)#w_seed：定义的初始权值
