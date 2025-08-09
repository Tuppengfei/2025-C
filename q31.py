import numpy as np
import pandas as pd
import colour
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from colour import SpectralDistribution, SpectralShape
from colour.quality.tm3018 import colour_fidelity_index_ANSIIESTM3018


BASIS_XLSX = r"C:\Users\31498\OneDrive\Desktop\conbine.xlsx"  
BASIS_SHEET = 0                             
EXTERNAL_XLSX = r"C:\Users\31498\OneDrive\Desktop\nihe.xlsx"  # 外界光谱
EXTERNAL_SHEET = 0


WL_COL = "Wavelength (nm)"
CHANNELS = ["Blue", "Green", "Red", "Warm White", "Cold White"]

# 如果外界光谱没有波长列，就用这个波段
DEFAULT_SHAPE = SpectralShape(380, 780, 1)

def sd_align(wavelengths_nm, intensities, shape=DEFAULT_SHAPE):
    sd = SpectralDistribution(dict(zip(wavelengths_nm, intensities)))
    return sd.align(shape)

def tm30_Rf_Rg_aligned(wavelengths_nm, intensities):
    sd = sd_align(wavelengths_nm, intensities)
    spec = colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)
    if hasattr(spec, "R_f"):
        return float(spec.R_f), float(spec.R_g)
    else:
        return float(spec["R_f"]), float(spec["R_g"])

def cct_from_sd(wavelengths_nm, intensities):
    # 归一化不影响 CCT
    intensities = np.asarray(intensities, float)
    if np.max(intensities) > 0:
        intensities = intensities / np.max(intensities)
    sd = SpectralDistribution(dict(zip(wavelengths_nm, intensities)))
    XYZ = colour.sd_to_XYZ(sd)
    xy = colour.XYZ_to_xy(XYZ)
    cct = colour.xy_to_CCT(xy)
    try:
        return float(cct)
    except Exception:
        return float(cct[0])

def combine_intensity(basis_df, w):
    intens = np.zeros(len(basis_df), dtype=float)
    for wi, col in zip(w, CHANNELS):
        intens += wi * pd.to_numeric(basis_df[col], errors="coerce").to_numpy()
    return np.clip(intens, 1e-12, None)

def load_basis(path, sheet=BASIS_SHEET):
    df = pd.read_excel(path, sheet_name=sheet)
    need = [WL_COL] + CHANNELS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"基底光谱缺少列：{miss}")
    wl = pd.to_numeric(df[WL_COL], errors="coerce").to_numpy()
    return df, wl

def load_external(path, sheet=EXTERNAL_SHEET, assume_shape=DEFAULT_SHAPE):
    df = pd.read_excel(path, sheet_name=sheet)
    # 外界光谱可能没有波长列，列名是时间。如果没有 WL_COL，就按默认 shape 构造波长
    if WL_COL in df.columns:
        wl = pd.to_numeric(df[WL_COL], errors="coerce").to_numpy()
        ext = df.drop(columns=[WL_COL])
    else:
        wl = assume_shape.wavelengths
        ext = df.copy()
        ext.index = range(len(wl))  # 保证行数一致
        if len(ext) != len(wl):
            raise ValueError(f"外界光谱行数 {len(ext)} 与默认波段长度 {len(wl)} 不一致。")
    return wl, ext

def align_basis_to_shape(basis_df, basis_wl, shape=DEFAULT_SHAPE):
    # 对每个通道按 shape 重采样
    aligned = {WL_COL: shape.wavelengths}
    for ch in CHANNELS:
        sd = sd_align(basis_wl, pd.to_numeric(basis_df[ch], errors="coerce").to_numpy(), shape)
        # SpectralDistribution.data is an OrderedDict-like; use values in the aligned grid order
        vals = np.array([sd[w] for w in shape.wavelengths])
        aligned[ch] = vals
    return pd.DataFrame(aligned)

def fit_weights(A, y, w0=None):
    """
    最小化 ||A w - y||^2, s.t. w>=0, sum(w)=1
    A: (N x 5) 基底矩阵（每列一个通道的 SPD）
    y: (N,) 外界光谱
    """
    n_ch = A.shape[1]
    if w0 is None:
        w0 = np.ones(n_ch) / n_ch
    w0 = np.clip(w0, 1e-9, 1.0)
    w0 = w0 / w0.sum()

    def obj(w):
        r = A @ w - y
        return float(np.dot(r, r))

    lc = LinearConstraint(np.ones((1, n_ch)), [1.0], [1.0])
    bnds = Bounds(np.zeros(n_ch), np.ones(n_ch))

    res = minimize(obj, w0, method='SLSQP', bounds=bnds, constraints=[lc],
                   options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
    return res.x, res

#主流程
def main():
    basis_df_raw, basis_wl = load_basis(BASIS_XLSX)
    # 基底按统一波段重采样（380–780, 1 nm）
    basis_df = align_basis_to_shape(basis_df_raw, basis_wl, DEFAULT_SHAPE)
    wl = basis_df[WL_COL].to_numpy()
    A = np.column_stack([basis_df[ch].to_numpy() for ch in CHANNELS])  # N x 5

    ext_wl, ext_df = load_external(EXTERNAL_XLSX)
    # 如果外界波长不是默认 shape，先重采样到默认 shape
    if not np.array_equal(ext_wl, DEFAULT_SHAPE.wavelengths):
        # 对每列时间做对齐
        ext_aligned = []
        for col in ext_df.columns:
            sd = sd_align(ext_wl, pd.to_numeric(ext_df[col], errors="coerce").to_numpy(), DEFAULT_SHAPE)
            ext_aligned.append([sd[w] for w in DEFAULT_SHAPE.wavelengths])
        Y = np.column_stack(ext_aligned)  # N x T
        time_labels = [str(c) for c in ext_df.columns]
    else:
        Y = ext_df.to_numpy()             # N x T
        time_labels = [str(c) for c in ext_df.columns]

    out_rows = []
    out_dir = Path("fits")
    out_dir.mkdir(exist_ok=True)

    for j, tlabel in enumerate(time_labels):
        y = np.asarray(Y[:, j], float)
        # 拟合权重
        w, res = fit_weights(A, y)
        y_fit = A @ w

        # 指标（外界与合成都算）
        CCT_ext = cct_from_sd(wl, y)
        Rf_ext, Rg_ext = tm30_Rf_Rg_aligned(wl, y)

        CCT_fit = cct_from_sd(wl, y_fit)
        Rf_fit, Rg_fit = tm30_Rf_Rg_aligned(wl, y_fit)

        out_rows.append({
            "time": tlabel,
            **{f"w_{ch}": w[i] for i, ch in enumerate(CHANNELS)},
            "CCT_ext": CCT_ext, "Rf_ext": Rf_ext, "Rg_ext": Rg_ext,
            "CCT_fit": CCT_fit, "Rf_fit": Rf_fit, "Rg_fit": Rg_fit,
            "LSQ_loss": res.fun
        })

        # 画图：一张图叠加 外界 vs 合成
        plt.figure()
        plt.plot(wl, y, label="External SPD")
        plt.plot(wl, y_fit, label="Fitted SPD", linestyle="--")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Relative SPD")
        plt.title(f"Spectrum Fit @ {tlabel}")
        plt.legend()
        #fig_path = out_dir / f"fit_{tlabel.replace(':','-').replace(' ','_')}.png"
        #plt.tight_layout()
        #plt.savefig(fig_path, dpi=160)
        #plt.close()
        plt.show()

    # 保存结果
    res_df = pd.DataFrame(out_rows)
    with pd.ExcelWriter("fit_results.xlsx", engine="openpyxl") as ew:
        res_df.to_excel(ew, index=False, sheet_name="weights_metrics")

    print(" Done. 输出：")
    print(" - 拟合权重与指标：fit_results.xlsx")
    print(" - 拟合图像：fits/fit_*.png")

if __name__ == "__main__":
    main()
