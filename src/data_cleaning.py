# =====================================================================
# DATA CLEANING MODULE
# Chuẩn bị và làm sạch dữ liệu EV adoption tại Mỹ
# =====================================================================

import pandas as pd
import numpy as np


def load_and_clean(filepath: str = "data/EV_Data.csv") -> pd.DataFrame:
    """
    Đọc, làm sạch và chuẩn bị dữ liệu EV.

    Parameters
    ----------
    filepath : str
        Đường dẫn đến file CSV gốc.

    Returns
    -------
    df : pd.DataFrame
        DataFrame đã làm sạch, sẵn sàng cho phân tích.
    """

    # 1. Đọc dữ liệu
    df = pd.read_csv(filepath)
    print(f"Kích thước dữ liệu ban đầu: {df.shape}")

    # 2. Chuẩn hóa tên cột
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "pct", regex=False)
    )

    # 3. Loại cột index thừa
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")],
                 errors="ignore")

    # 4. Lọc giai đoạn nghiên cứu (2019–2023)
    df = df[(df["year"] >= 2019) & (df["year"] <= 2023)].copy()
    print(f"Sau khi lọc năm 2019–2023: {df.shape}")

    # 5. Giữ lại các biến cần thiết
    cols_needed = [
        "state", "year",
        "ev_share_pct",
        "stations", "total_vehicles",
        "per_cap_income",
        "bachelor_attainment",
        "gasoline_price_per_gallon",
        "price_cents_per_kwh",
        "trucks_share",
        "incentives",
    ]
    df = df[cols_needed].copy()

    # 6. Loại bỏ District of Columbia
    df = df[df["state"] != "district_of_columbia"].copy()
    print(f"Sau khi loại District of Columbia: {df.shape}")

    # 7. Tạo biến mật độ trạm sạc (stations per 100k vehicles)
    df["stations_per_100k_vehicles"] = (
        df["stations"] / df["total_vehicles"] * 100_000
    )

    # 8. Đổi đơn vị thu nhập: USD → nghìn USD
    df["per_cap_income"] = df["per_cap_income"] / 1_000

    # 9. Bỏ cột trung gian
    df = df.drop(columns=["stations", "total_vehicles"])

    # 10. Loại bỏ missing values
    df = df.dropna().copy()

    print(f"\n✓ Hoàn thành làm sạch dữ liệu")
    print(f"✓ Kích thước cuối cùng : {df.shape}")
    print(f"✓ Missing values       : {df.isna().sum().sum()}")

    return df


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trả về bảng thống kê mô tả cho các biến chính.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    vars_desc = [
        "ev_share_pct",
        "stations_per_100k_vehicles",
        "per_cap_income",
        "bachelor_attainment",
        "gasoline_price_per_gallon",
        "price_cents_per_kwh",
        "trucks_share",
        "incentives",
    ]

    desc = (
        df[vars_desc]
        .describe()
        .T[["mean", "std", "min", "max"]]
        .rename(columns={"mean": "Mean", "std": "Std", "min": "Min", "max": "Max"})
        .round(3)
    )
    desc.index.name = "Biến"
    return desc


# ---------------------------------------------------------------------------
# Chạy độc lập để kiểm tra
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_clean()
    print("\nBảng thống kê mô tả:")
    print(descriptive_stats(df))
    print("\nMẫu dữ liệu:")
    print(df.head())
