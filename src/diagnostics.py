# =====================================================================
# DIAGNOSTICS MODULE
# Biểu đồ khám phá, kiểm tra đa cộng tuyến và phân tích phần dư
# =====================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 100


# ---------------------------------------------------------------------------
# PHẦN 3: Đánh giá sơ bộ dữ liệu
# ---------------------------------------------------------------------------

def plot_ev_trend(df: pd.DataFrame) -> None:
    """
    Hình 3.1: Xu hướng EV Share trung bình theo thời gian (2019–2023).
    """
    trend = df.groupby("year")["ev_share_pct"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trend.index, trend["mean"], marker="o", linewidth=2,
            markersize=8, label="EV Share trung bình")
    ax.fill_between(
        trend.index,
        trend["mean"] - trend["std"],
        trend["mean"] + trend["std"],
        alpha=0.2,
        label="±1 độ lệch chuẩn",
    )
    ax.set_xlabel("Năm", fontsize=12, fontweight="bold")
    ax.set_ylabel("EV Share trung bình (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hình 3.1: Xu hướng EV Share trung bình theo thời gian (2019–2023)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_top10_states(df: pd.DataFrame, year: int = 2023) -> None:
    """
    Hình 3.2: Top 10 bang có tỷ lệ EV cao nhất trong `year`.
    """
    top10 = (
        df[df["year"] == year]
        .sort_values("ev_share_pct", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top10["state"], top10["ev_share_pct"], color="steelblue")
    ax.set_xlabel("EV Share (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bang", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Hình 3.2: Top 10 bang có tỷ lệ EV cao nhất năm {year}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%", va="center", fontsize=10)

    fig.tight_layout()
    plt.show()


def plot_charging_vs_ev(df: pd.DataFrame) -> None:
    """
    Hình 3.3: Mối quan hệ giữa hạ tầng sạc và EV Adoption.
    """
    x = df["stations_per_100k_vehicles"]
    y = df["ev_share_pct"]
    coef = np.polyfit(x, y, 1)
    trend = np.poly1d(coef)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.5, s=50, color="steelblue",
               edgecolors="black", linewidth=0.5)
    x_sorted = np.sort(x)
    ax.plot(
        x_sorted, trend(x_sorted),
        color="red", linestyle="--", linewidth=2,
        label=f"Xu hướng: y={coef[0]:.3f}x+{coef[1]:.3f}",
    )
    ax.set_xlabel("Trạm sạc / 100.000 phương tiện", fontsize=12, fontweight="bold")
    ax.set_ylabel("EV Share (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hình 3.3: Mối quan hệ giữa hạ tầng sạc và EV Adoption",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PHẦN 4: Phân tích định lượng – Tương quan & VIF
# ---------------------------------------------------------------------------

CORR_COLS = [
    "ev_share_pct",
    "stations_per_100k_vehicles",
    "per_cap_income",
    "gasoline_price_per_gallon",
    "price_cents_per_kwh",
    "incentives",
    "bachelor_attainment",
    "trucks_share",
]


def plot_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vẽ ma trận tương quan và trả về DataFrame tương quan.
    """
    corr_matrix = df[CORR_COLS].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        square=True,
        ax=ax,
    )
    ax.set_title(
        "Ma trận tương quan giữa các biến nghiên cứu",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    plt.show()

    # In cảnh báo
    c = corr_matrix.abs().unstack()
    high = c[(c < 1) & (c > 0.7)].drop_duplicates().sort_values(ascending=False)
    if len(high):
        print("\n⚠ Cặp biến có tương quan > 0.7:")
        print(high)
    else:
        print("\n✓ Không có cặp biến nào có tương quan > 0.7")

    return corr_matrix


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Tính VIF cho danh sách biến `features`.

    Parameters
    ----------
    df       : DataFrame
    features : list  Tên các biến cần kiểm tra.

    Returns
    -------
    pd.DataFrame  cột ['feature', 'VIF']
    """
    X = add_constant(df[features].dropna())
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i)
                for i in range(len(X.columns))],
    })
    return vif_df[vif_df["feature"] != "const"].reset_index(drop=True)


def plot_vif(vif_df: pd.DataFrame, title_suffix: str = "") -> None:
    """Vẽ biểu đồ thanh ngang cho VIF."""

    def _color(v):
        if v < 5:
            return "#2ecc71"
        elif v < 10:
            return "#f39c12"
        return "#e74c3c"

    colors = vif_df["VIF"].apply(_color)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(vif_df["feature"], vif_df["VIF"], color=colors)
    ax.axvline(10, color="red", linestyle="--", linewidth=1.5,
               label="Ngưỡng nguy hiểm (VIF=10)")
    ax.axvline(5, color="orange", linestyle="--", linewidth=1,
               label="Ngưỡng cảnh báo (VIF=5)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}", va="center", fontsize=10, fontweight="bold")

    title = f"Hệ số phóng đại phương sai (VIF){' – ' + title_suffix if title_suffix else ''}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hệ số VIF", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()


def pearson_correlation_test(
    df: pd.DataFrame,
    x_vars: list,
    target: str = "ev_share_pct",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Kiểm định hệ số tương quan Pearson giữa từng x_var và target.

    Returns
    -------
    pd.DataFrame  Bảng kết quả kiểm định.
    """
    y = df[target]
    n = len(df)
    df_free = n - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df_free)

    rows = []
    for var in x_vars:
        r, p_val = stats.pearsonr(df[var], y)
        T_stat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r ** 2)
        rows.append({
            "Biến": var,
            "r (Pearson)": round(r, 4),
            "T statistic": round(T_stat, 3),
            "df": df_free,
            "p-value": round(p_val, 5),
            "Kết luận": "Bác bỏ H0" if abs(T_stat) > t_crit else "Không bác bỏ H0",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PHẦN 6: Phân tích phần dư
# ---------------------------------------------------------------------------

def plot_residuals(residuals: np.ndarray, y_hat: np.ndarray) -> None:
    """
    Vẽ 3 biểu đồ chẩn đoán phần dư:
      1. Histogram + đường chuẩn lý thuyết
      2. Q-Q Plot
      3. Residuals vs Fitted Values
    """
    # --- 1. Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, stat="density", bins=30,
                 color="skyblue", edgecolor="black", alpha=0.7, ax=ax)
    mu, sd = residuals.mean(), residuals.std()
    xmin, xmax = ax.get_xlim()
    x_vals = np.linspace(xmin, xmax, 200)
    ax.plot(x_vals, norm.pdf(x_vals, mu, sd),
            color="red", linestyle="--", linewidth=2,
            label=f"N({mu:.2f}, {sd:.2f}²)")
    ax.set_title("Phân phối phần dư từ mô hình",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Phần dư (Residuals)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mật độ", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    # --- 2. Q-Q Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot của phần dư",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Lý thuyết (Theoretical Quantiles)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Mẫu (Sample Quantiles)",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    # --- 3. Residuals vs Fitted ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_hat, residuals, alpha=0.5, s=50,
               color="steelblue", edgecolors="black", linewidth=0.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="y=0")
    ax.set_xlabel("Giá trị dự đoán (Fitted Values)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Phần dư (Residuals)", fontsize=12, fontweight="bold")
    ax.set_title("Phần dư vs Giá trị dự đoán",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def residual_stats(residuals: np.ndarray) -> pd.DataFrame:
    """Trả về bảng thống kê mô tả phần dư."""
    return pd.DataFrame({
        "Thống kê": ["Trung bình", "Độ lệch chuẩn", "Min",
                     "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max"],
        "Giá trị": [
            residuals.mean(),
            residuals.std(),
            residuals.min(),
            np.percentile(residuals, 25),
            np.median(residuals),
            np.percentile(residuals, 75),
            residuals.max(),
        ],
    }).round(6)


# ---------------------------------------------------------------------------
# Chạy độc lập
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_cleaning import load_and_clean
    from src.regression import run_ols, X_VARS_FULL, X_VARS_REDUCED

    df = load_and_clean()

    # Đánh giá sơ bộ
    plot_ev_trend(df)
    plot_top10_states(df)
    plot_charging_vs_ev(df)

    # Tương quan
    corr_matrix = plot_correlation_matrix(df)

    # VIF (mô hình đầy đủ, dùng ln_income)
    df_vif = df.copy()
    df_vif["ln_income"] = np.log(df_vif["per_cap_income"])
    vif_features = [
        "stations_per_100k_vehicles", "ln_income",
        "gasoline_price_per_gallon", "price_cents_per_kwh",
        "trucks_share", "incentives",
    ]
    vif_df = compute_vif(df_vif, vif_features)
    print("\nBảng VIF:")
    print(vif_df.to_string(index=False))
    plot_vif(vif_df, "Mô hình sau khi loại bachelor_attainment")

    # Pearson
    pearson_df = pearson_correlation_test(df, X_VARS_FULL[:-1])
    print("\nKiểm định Pearson:")
    print(pearson_df.to_string(index=False))

    # Phần dư
    model_r1 = run_ols(X_VARS_REDUCED, df)
    residuals = df["ev_share_pct"].values - model_r1["y_hat"]
    plot_residuals(residuals, model_r1["y_hat"])
    print("\nThống kê phần dư:")
    print(residual_stats(residuals).to_string(index=False))
