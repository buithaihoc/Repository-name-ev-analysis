# =====================================================================
# REGRESSION MODULE
# Xây dựng và đánh giá mô hình OLS cho EV adoption
# =====================================================================

import numpy as np
import pandas as pd
from scipy.stats import f, t


# ---------------------------------------------------------------------------
# Hằng số – danh sách biến
# ---------------------------------------------------------------------------
X_VARS_FULL = [
    "per_cap_income",
    "bachelor_attainment",
    "gasoline_price_per_gallon",
    "price_cents_per_kwh",
    "trucks_share",
    "incentives",
    "stations_per_100k_vehicles",
]

# Mô hình rút gọn 1: bỏ bachelor_attainment (VIF cao)
X_VARS_REDUCED = [v for v in X_VARS_FULL if v != "bachelor_attainment"]

# Mô hình rút gọn 2: bỏ thêm price_cents_per_kwh
X_VARS_REDUCED2 = [
    v for v in X_VARS_FULL
    if v not in {"bachelor_attainment", "price_cents_per_kwh"}
]

TARGET = "ev_share_pct"


# ---------------------------------------------------------------------------
# Core OLS
# ---------------------------------------------------------------------------
def run_ols(
    X_vars: list,
    df: pd.DataFrame,
    target: str = TARGET,
) -> dict:
    """
    Ước lượng mô hình OLS thủ công (không dùng statsmodels).

    Parameters
    ----------
    X_vars : list       Danh sách tên biến độc lập.
    df     : DataFrame  Dữ liệu.
    target : str        Tên biến phụ thuộc.

    Returns
    -------
    dict với các key:
        beta_hat, r_squared, sse, y_hat,
        sigma2_hat, sigma_hat, df_resid, n, k
    """
    y = df[target].values.reshape(-1, 1)
    X_raw = df[X_vars].values
    X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])  # thêm intercept

    # β̂ = (X'X)⁻¹ X'y
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    y_hat = X @ beta_hat
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    ss_resid = float(np.sum((y - y_hat) ** 2))
    r_squared = 1 - ss_resid / ss_total

    n, k = len(y), len(X_vars)
    df_resid = n - k - 1
    sigma2_hat = ss_resid / df_resid

    return {
        "beta_hat": beta_hat.flatten(),
        "r_squared": r_squared,
        "sse": ss_resid,
        "y_hat": y_hat.flatten(),
        "sigma2_hat": sigma2_hat,
        "sigma_hat": np.sqrt(sigma2_hat),
        "df_resid": df_resid,
        "n": n,
        "k": k,
        "X_vars": X_vars,
    }


# ---------------------------------------------------------------------------
# ANOVA & F-test
# ---------------------------------------------------------------------------
def anova_table(model: dict, df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Tạo bảng ANOVA và kiểm định F cho một kết quả OLS.

    Parameters
    ----------
    model : dict   Output từ run_ols().
    df    : DataFrame
    target: str

    Returns
    -------
    pd.DataFrame  Bảng ANOVA.
    """
    y = df[target].values
    SSE = model["sse"]
    SST = float(np.sum((y - np.mean(y)) ** 2))
    SSR = SST - SSE

    k = model["k"]
    df_reg = k
    df_res = model["df_resid"]
    MSR = SSR / df_reg
    MSE = SSE / df_res
    F_stat = MSR / MSE
    p_value = 1 - f.cdf(F_stat, dfn=df_reg, dfd=df_res)

    table = pd.DataFrame(
        {
            "Nguồn biến thiên": [
                "Hồi quy (Regression)",
                "Sai số (Residual)",
                "Tổng (Total)",
            ],
            "SS": [SSR, SSE, SST],
            "df": [df_reg, df_res, df_reg + df_res],
            "MS": [MSR, MSE, np.nan],
            "F": [F_stat, np.nan, np.nan],
            "P-value": [p_value, np.nan, np.nan],
        }
    )
    return table.round(4)


# ---------------------------------------------------------------------------
# T-test từng hệ số
# ---------------------------------------------------------------------------
def ttest_coefficients(
    model: dict,
    df: pd.DataFrame,
    target: str = TARGET,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Kiểm định t cho từng hệ số hồi quy trong mô hình.

    Parameters
    ----------
    model : dict   Output từ run_ols().
    df    : DataFrame
    target: str
    alpha : float  Mức ý nghĩa (mặc định 0.05).

    Returns
    -------
    pd.DataFrame  Bảng kết quả kiểm định t.
    """
    X_vars = model["X_vars"]
    X_raw = df[X_vars].values
    X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    XtX_inv = np.linalg.solve(X.T @ X, np.eye(X.shape[1]))

    df_t = model["df_resid"]
    t_crit = t.ppf(1 - alpha / 2, df=df_t)
    sigma2 = model["sigma2_hat"]
    betas = model["beta_hat"]

    col_names = ["Intercept"] + X_vars
    rows = []
    for j, name in enumerate(col_names):
        se_j = np.sqrt(sigma2 * XtX_inv[j, j])
        t0 = betas[j] / se_j
        p_val = 2 * (1 - t.cdf(abs(t0), df=df_t))
        reject = abs(t0) > t_crit
        rows.append(
            {
                "Biến": name,
                "Hệ số": round(betas[j], 6),
                "SE": round(se_j, 6),
                "t0": round(t0, 4),
                "p-value": round(p_val, 4),
                "Kết luận": "✓ Bác bỏ H₀" if reject else "✗ Không bác bỏ H₀",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# In kết quả tóm tắt
# ---------------------------------------------------------------------------
def print_model_summary(model: dict, label: str = "") -> None:
    """In tóm tắt kết quả OLS ra stdout."""
    header = f"=== KẾT QUẢ OLS{': ' + label if label else ''} ==="
    print("\n" + header)
    print(f"{'Intercept':<35} {model['beta_hat'][0]:>15.6f}")
    for i, var in enumerate(model["X_vars"], start=1):
        print(f"{var:<35} {model['beta_hat'][i]:>15.6f}")
    print("-" * 52)
    print(f"{'R²':<35} {model['r_squared']:>15.6f}")
    print(f"{'SSE':<35} {model['sse']:>15.4f}")
    print(f"{'σ² (phương sai ước lượng)':<35} {model['sigma2_hat']:>15.4f}")
    print(f"{'σ̂  (độ lệch chuẩn)':<35} {model['sigma_hat']:>15.4f}")
    print(f"{'Bậc tự do phần dư (df)':<35} {model['df_resid']:>15d}")


# ---------------------------------------------------------------------------
# Chạy độc lập
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_cleaning import load_and_clean

    df = load_and_clean()

    # --- Mô hình đầy đủ ---
    model_full = run_ols(X_VARS_FULL, df)
    print_model_summary(model_full, "Mô hình đầy đủ")

    # --- Mô hình rút gọn 1 ---
    model_r1 = run_ols(X_VARS_REDUCED, df)
    print_model_summary(model_r1, "Mô hình rút gọn 1 (bỏ bachelor_attainment)")

    print("\nBảng ANOVA (mô hình rút gọn 1):")
    print(anova_table(model_r1, df))

    print("\nKiểm định t từng hệ số (mô hình rút gọn 1):")
    print(ttest_coefficients(model_r1, df).to_string(index=False))

    # --- Mô hình rút gọn 2 ---
    model_r2 = run_ols(X_VARS_REDUCED2, df)
    print_model_summary(model_r2, "Mô hình rút gọn 2 (bỏ thêm price_cents_per_kwh)")

    print(
        f"\nSo sánh R²: Full={model_full['r_squared']:.4f} | "
        f"Reduced1={model_r1['r_squared']:.4f} | "
        f"Reduced2={model_r2['r_squared']:.4f}"
    )
