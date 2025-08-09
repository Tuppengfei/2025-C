import pandas as pd
import numpy as np


file_path = r"C:\Users\31498\OneDrive\Desktop\uv.xlsx"

df = pd.read_excel(file_path)

required_columns = ['u', 'v', 'ut', 'vt']
if not all(col in df.columns for col in required_columns):
    raise ValueError("CSV文件中缺少必要的列 ('u', 'v', 'ut', 'vt')")

if df is not None:
    # 提取 u_s, v_s, u_t, v_t 列
    us = df['u']
    vs = df['v']
    ut = df['ut']
    vt = df['vt']

    # 计算两点之间的欧氏距离
    # distance = sqrt((ut - us)^2 + (vt - vs)^2)
    distance = np.sqrt((ut - us)**2 + (vt - vs)**2)

    # 确定符号：如果光源点在黑体轨迹上方(vs > vt)，则为正，否则为负
    # np.sign() 函数可以完美处理正、负和零的情况
    sign = np.sign(vs - vt)

    # 计算最终的Duv值
    df['Duv'] = sign * distance

    
    print("=" * 30)
    print(df[['u', 'v', 'ut', 'vt', 'Duv']].round(6))
    print("=" * 30)

    df.to_excel("Duv.xlsx")
    
