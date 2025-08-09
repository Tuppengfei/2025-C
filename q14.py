import pandas as pd
import numpy as np

df1 = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\q15.xlsx")

delt=1
df1['num']=df1['光强']*df1['melanopic']*delt
df1['den']=df1['光强']*df1["V(λ)"]*delt


num = df1['num'].sum()
den = df1["den"].sum()


if den == 0:
    print('分母为0，无法计算 mel_DER')
else:
    mel_DER = num * 1.218 / den
    print(mel_DER)

