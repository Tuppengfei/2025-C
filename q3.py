import pandas as pd
import numpy as np
from scipy.optimize import minimize

df1 = pd.read_excel(r"C:\Users\31498\OneDrive\Desktop\hs2025\C题\conbine.xlsx")
time_points = [f"{hour}点三十" for hour in range(5, 20)]

melDER = [1.038, 1.0349, 1.0316, 1.028, 1.0423, 1.0548, 1.066, 1.076, 1.067, 1.0568, 1.0454, 1.032, 1.0168, 0.978, 0.7542]

# 定义优化目标函数
def objective_function(weights, df, time_col, melDER_target):
    w1, w2, w3, w4, w5 = weights
    # 计算 mix1
    df["mix1"] = w1 * df["Blue"] + w2 * df["Green"] + w3 * df["Red"] + w4 * df["Warm White"] + w5 * df["Cold White"]
    total = (df[time_col] - df["mix1"]).abs().sum()
    
    delt = 1
    df['num'] = df['mix1'] * df['melanopic'] * delt
    df['den'] = df['mix1'] * df["V(λ)"] * delt
    num = df['num'].sum()
    den = df['den'].sum()
    if den == 0:
        return np.inf 
    mel_DER = num * 1.218 / den
    delta_mel = abs(mel_DER - melDER_target)

    #目标函数值和 mel_DER
    return 0.8 * delta_mel + 0.2 * total, mel_DER
# 存储权重结果、mel_DER 和 mix1 数据
results = []
mix1_dfs = []

# 对每个时间点进行优化
for i, time in enumerate(time_points):#enumerate生成i=0，time="5点三十"
    # 复制原始 DataFrame
    df_temp = df1.copy()
    
    # 定义约束条件：w1 + w2 + w3 + w4 + w5 = 1
    constraints = ({'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] + w[3] + w[4] - 1})#{'type': 'eq'} 表示这是一个等式约束
    #等式约束：f(x)=0;不等式约束：g(x)>0;eg:x1+x2=5=>lamda x : x[0]+x[1]-5;



    # 定义边界条件：每个权重在 [0, 1] 之间
    bounds = [(0, 1)] * 5
    initial_guess = [0.2, 0.2, 0.2, 0.2, 0.2]
    result = minimize(
        lambda w: objective_function(w,df_temp,time,melDER[i])[0],  # 仅优化目标函数
        #lambda w: ... 定义了一个以w为参数的匿名函数；[0]表示只取objective_function返回值的第一个元素
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    print(result)

    # 保存优化结果
    if result.success:
        # 计算优化后的 mix1 和 mel_DER
        df_temp["mix1"] = (result.x[0] * df_temp["Blue"] + #返回的结果：x: [ 3.829e-01  2.365e-02  5.826e-01  1.083e-02  4.607e-14]
                          result.x[1] * df_temp["Green"] + 
                          result.x[2] * df_temp["Red"] + 
                          result.x[3] * df_temp["Warm White"] + 
                          result.x[4] * df_temp["Cold White"])
        
        # 计算 mel_DER
        _, optimized_mel_DER = objective_function(result.x, df_temp, time, melDER[i])
        
        # 保存 mix1 列，命名为对应时间点
        mix1_dfs.append(df_temp[["mix1"]].rename(columns={"mix1": f"mix1_{time}"}))
        
        # 保存权重、mel_DER 和目标函数值
        results.append({
            '时间点': time,
            'w1': result.x[0],
            'w2': result.x[1],
            'w3': result.x[2],
            'w4': result.x[3],
            'w5': result.x[4],
            'mel_DER': optimized_mel_DER,
            '目标函数值': result.fun
        })
    else:
        print(f"{time} 优化失败: {result.message}")

# 将权重和 mel_DER 结果转换为 DataFrame
results_df = pd.DataFrame(results)
# 合并所有 mix1 列
mix1_combined = pd.concat(mix1_dfs, axis=1)

print(results_df)