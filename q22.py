import pandas as pd
import numpy as np
from colour import SpectralDistribution, SpectralShape
from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018
import colour
# -------------------------
# 1) 读取数据
# -------------------------
def load_data(file_path=r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx"):
    """
    需要包含列：
    'Wavelength (nm)', 'Blue','Green','Red','Warm White','Cold White',
    'L-cone-opic','M-cone-opic','S-cone-opic', 'melanopic', 'V(λ)'
    """
    df = pd.read_excel(file_path)
    # 排序并清理
    df = df.sort_values('Wavelength (nm)').reset_index(drop=True)
    return df

# -------------------------
# 2) 线性组合 5 路通道生成光源 SPD
# -------------------------
def combine_intensity(df, weights):
    w1, w2, w3, w4, w5 = weights
    df["intensity"] = (
        w1 * pd.to_numeric(df["Blue"], errors="coerce") +
        w2 * pd.to_numeric(df["Green"], errors="coerce") +
        w3 * pd.to_numeric(df["Red"], errors="coerce") +
        w4 * pd.to_numeric(df["Warm White"], errors="coerce") +
        w5 * pd.to_numeric(df["Cold White"], errors="coerce")
    )
    return df

# -------------------------
# 3) LMS→等效CMF，SPD→XYZ→CCT
# -------------------------
def calculate_cct(df):
    """
    从 DataFrame 读取波长和光强，计算 CCT。
    df: 包含 'Wavelength (nm)' 和 'intensity' 列的 DataFrame
    返回：CCT（单位：K）
    """
    # 提取波长和光强
    wavelengths = pd.to_numeric(df["Wavelength (nm)"], errors="coerce").to_numpy()
    intensities = pd.to_numeric(df["intensity"], errors="coerce").to_numpy()

    # 过滤无效数据
    mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    wavelengths = wavelengths[mask]
    intensities = intensities[mask]

    # 归一化光强
    intensities = intensities / np.max(intensities) if np.max(intensities) > 0 else intensities

    # 创建光谱数据
    spectrum = dict(zip(wavelengths, intensities))
    sd = SpectralDistribution(spectrum)

    # 计算 xy 色度坐标并转换为 CCT
    xy = colour.XYZ_to_xy(colour.sd_to_XYZ(sd))
    cct = colour.xy_to_CCT(xy)

    return cct

# -------------------------
# 4) 计算 TM-30 Rf / Rg
# -------------------------
def calculate_rf_rg(df):
    wavelengths = pd.to_numeric(df["Wavelength (nm)"], errors="coerce").to_numpy()
    intensities = pd.to_numeric(df["intensity"], errors="coerce").to_numpy()

    mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    w = wavelengths[mask]
    v = intensities[mask]

    if len(w) < 10:
        # 数据点太少
        return 0.0, 0.0

    # 对齐到 380-780nm, 1nm 步长
    shape = SpectralShape(380, 780, 1)
    sd = SpectralDistribution(dict(zip(w, v))).align(shape)
    spec = colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)
    Rf = float(spec.R_f) if np.isfinite(spec.R_f) else 0.0
    Rg = float(spec.R_g) if np.isfinite(spec.R_g) else 0.0
    return Rf, Rg

# -------------------------
# 5) 计算 melanopic DER
# -------------------------
def calculate_mel_der(df):
    """
    mel_DER = ( ∑ S(λ)*melanopic(λ) ) * 1.218 / ( ∑ S(λ)*V(λ) )
    这里 S(λ)=df['intensity']，melanopic 和 V(λ) 需在 df 中
    """
    S = pd.to_numeric(df['intensity'], errors='coerce').to_numpy()
    mel = pd.to_numeric(df['melanopic'], errors='coerce').to_numpy()
    Vlam = pd.to_numeric(df['V(λ)'], errors='coerce').to_numpy()

    num = np.nansum(S * mel)
    den = np.nansum(S * Vlam)

    if abs(den) < 1e-12:
        return 0.0
    return float(num * 1.218 / den)

# -------------------------
# 6) 主函数：输入权重 → 返回结果
# -------------------------
def calculate_parameters(file_path=r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx"):
    df = load_data(file_path)

    print("请依次输入权重 w1, w2, w3, w4, w5（对应 Blue, Green, Red, Warm White, Cold White）：")
    weights = []
    names = ["Blue", "Green", "Red", "Warm White", "Cold White"]
    for i in range(5):
        while True:
            try:
                w = float(input(f"输入 {names[i]} 的权重 w{i+1}（> 0）："))
                if w <= 0:
                    print("权重必须大于 0，请重新输入！")
                else:
                    weights.append(w)
                    break
            except ValueError:
                print("请输入有效的数字！")

    print(f"\n输入的权重: w1={weights[0]:.4f}, w2={weights[1]:.4f}, "
          f"w3={weights[2]:.4f}, w4={weights[3]:.4f}, w5={weights[4]:.4f}")
    print(f"权重和(不强制归一): {sum(weights):.4f}")

    # 1) 混光
    df = combine_intensity(df.copy(), weights)

    # 2) 计算指标
    CCT = calculate_cct(df)
    Rf, Rg = calculate_rf_rg(df)
    mel_der = calculate_mel_der(df)

    # 3) 输出
    print("\n计算结果：")
    print(f"CCT: {CCT:.2f} K" if np.isfinite(CCT) else "CCT: 计算失败 (NaN)")
    print(f"Rf: {Rf:.2f}")
    print(f"Rg: {Rg:.2f}")
    print(f"mel_DER: {mel_der:.4f}")

    return {
        "weights": weights,
        "CCT": CCT,
        "Rf": Rf,
        "Rg": Rg,
        "mel_DER": mel_der
    }

# -------------------------
# 入口
# -------------------------
if __name__ == "__main__":
    calculate_parameters(r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx")
