import numpy as np
import pandas as pd
import colour

def calculate_cct_from_excel(file_path):
    """
    读取波长和光强数据计算 CCT。
    返回CCT
    """
    
    df = pd.read_excel(file_path)
    
    wavelengths = df['Wavelength (nm)'].to_numpy()
    intensities = df['光强'].to_numpy()

    # 归一化光强
    intensities = intensities / np.max(intensities)

    # 创建光谱数据
    spectrum = np.column_stack((wavelengths, intensities))
    sd = colour.SpectralDistribution(dict(zip(wavelengths, intensities)))

    # 计算 xy 色度坐标和 CCT
    xy = colour.XYZ_to_xy(colour.sd_to_XYZ(sd))
    cct = colour.xy_to_CCT(xy)

    return cct

# 示例使用
file_path = r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx" 
cct = calculate_cct_from_excel(file_path)
print(f"相关色温 (CCT): {cct:.2f} K")
