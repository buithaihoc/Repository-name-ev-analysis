# 🔋 EV Adoption Analysis – Các Yếu Tố Ảnh Hưởng đến Tỷ Lệ Chấp Nhận Xe Điện tại Mỹ

Phân tích định lượng các yếu tố kinh tế – xã hội tác động đến tỷ lệ xe điện (EV share) ở 50 bang tại Mỹ giai đoạn **2019–2023** bằng mô hình hồi quy OLS.

---

## 📁 Cấu trúc dự án

```
.
├── data/
│   └── EV_Data.csv               # Dữ liệu gốc
├── notebooks/
│   └── ev_analysis.ipynb         # Notebook phân tích đầy đủ
├── src/
│   ├── data_cleaning.py          # Làm sạch & chuẩn bị dữ liệu
│   ├── diagnostics.py            # Biểu đồ khám phá, tương quan, VIF, phần dư
│   └── regression.py             # Ước lượng OLS, ANOVA, t-test
└── README.md
```

---

## 📊 Biến nghiên cứu

| Biến | Mô tả |
|------|-------|
| `ev_share_pct` | Tỷ lệ xe điện (% tổng phương tiện) – **biến phụ thuộc** |
| `stations_per_100k_vehicles` | Mật độ trạm sạc trên 100.000 phương tiện |
| `per_cap_income` | Thu nhập bình quân đầu người (nghìn USD) |
| `bachelor_attainment` | Tỷ lệ dân số có bằng đại học trở lên (%) |
| `gasoline_price_per_gallon` | Giá xăng trung bình (USD/gallon) |
| `price_cents_per_kwh` | Giá điện trung bình (cents/kWh) |
| `trucks_share` | Tỷ trọng xe tải/bán tải trong tổng phương tiện (%) |
| `incentives` | Số lượng chính sách ưu đãi EV tại bang |

---

## ⚙️ Cài đặt

```bash
# Clone repo
git clone https://github.com/<your-username>/ev-adoption-analysis.git
cd ev-adoption-analysis

# Cài đặt dependencies
pip install -r requirements.txt
```

### Dependencies chính

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
statsmodels>=0.13
jupyter
```

---

## 🚀 Cách chạy

### Chạy notebook (khuyến nghị)

```bash
jupyter notebook notebooks/ev_analysis.ipynb
```

### Chạy từng module độc lập

```bash
# Làm sạch dữ liệu
python src/data_cleaning.py

# Kiểm tra tương quan, VIF và phần dư
python src/diagnostics.py

# Ước lượng và kiểm định mô hình OLS
python src/regression.py
```

---

## 🔬 Phương pháp

1. **Làm sạch dữ liệu** – chuẩn hóa tên biến, tạo biến mật độ trạm sạc, lọc giai đoạn nghiên cứu.
2. **Phân tích sơ bộ** – xu hướng theo thời gian, top 10 bang, biểu đồ phân tán.
3. **Kiểm tra đa cộng tuyến** – ma trận tương quan Pearson + chỉ số VIF.
4. **Xây dựng mô hình OLS** – ước lượng thủ công bằng ma trận `(X'X)⁻¹X'y`.
5. **Kiểm định** – kiểm định F tổng thể (ANOVA), kiểm định t từng hệ số.
6. **Phân tích phần dư** – histogram, Q-Q Plot, Residuals vs Fitted.

### Các mô hình

| Mô hình | Biến | R² |
|---------|------|----|
| Đầy đủ | 7 biến (bao gồm `bachelor_attainment`) | Xem notebook |
| Rút gọn 1 | 6 biến (bỏ `bachelor_attainment`) | Xem notebook |
| Rút gọn 2 | 5 biến (bỏ thêm `price_cents_per_kwh`) | Xem notebook |

---

## 📈 Kết quả nổi bật

- **Hạ tầng sạc** (`stations_per_100k_vehicles`) có tương quan mạnh nhất với EV adoption.
- **Thu nhập** và **giá xăng** tác động tích cực, **tỷ trọng xe tải** tác động âm.
- Mô hình rút gọn đạt R² tương đương mô hình đầy đủ, chứng tỏ các biến bị loại không mang thông tin bổ sung đáng kể.
- Kiểm định F: p-value < 0.001 → mô hình có ý nghĩa thống kê tổng thể.

---

## 📝 License

MIT License – xem file `LICENSE` để biết thêm chi tiết.
