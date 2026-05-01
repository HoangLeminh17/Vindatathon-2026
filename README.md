# Vindatathon 2026 - Giải pháp Toàn diện

Kho lưu trữ này chứa toàn bộ mã nguồn, phân tích và mô hình dự báo cho cuộc thi **Vindatathon 2026**. Giải pháp được cấu trúc thành 3 phần chính tương ứng với yêu cầu của đề bài: Giải trắc nghiệm (Phần 1), Khai phá dữ liệu EDA (Phần 2) và Dự báo chuỗi thời gian (Phần 3).

## 📂 Cấu trúc thư mục (Project Structure)

```text
Vindatathon-2026/
│
├── dataset/                  # Chứa toàn bộ các file dữ liệu gốc (.csv) do BTC cung cấp (Đã được Git ignore)
│
├── phan1.ipynb               # [Phần 1] Notebook chứa các đoạn code xử lý và tính toán đáp án trắc nghiệm
├── phan2.ipynb               # [Phần 2] Notebook phân tích Exploratory Data Analysis (EDA) & Trực quan hoá
├── phan3_v6.py               # [Phần 3] Pipeline mô hình dự báo doanh thu (Recursive Gradient Boosting)
│
├── submission.csv            # File kết quả dự báo cuối cùng cho tập test (Format chuẩn của Kaggle)
├── requirements.txt          # Danh sách các thư viện Python cần thiết
├── report_phan3.md           # Báo cáo kỹ thuật chi tiết cho Phần 3 (Nguồn văn bản)
├── vindatathon-report.pdf    # Báo cáo chính thức bằng định dạng NeurIPS
└── README.md                 # Tài liệu hướng dẫn sử dụng (File bạn đang đọc)
```

## ⚙️ Hướng dẫn cài đặt và Chạy lại kết quả (Reproducibility)

Để chạy lại toàn bộ kết quả của dự án, vui lòng làm theo các bước sau trên môi trường Python (Khuyến nghị sử dụng Python 3.9+).

### Bước 1: Chuẩn bị dữ liệu
1. Đảm bảo thư mục `dataset/` nằm cùng cấp với các file mã nguồn.
2. Tất cả các file CSV của cuộc thi (`sales.csv`, `products.csv`, `orders.csv`, v.v. cùng `sample_submission.csv`) phải được đặt trực tiếp bên trong thư mục `dataset/`.

### Bước 2: Cài đặt thư viện môi trường
Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh sau để cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy code Phần 1 & Phần 2
Cả hai phần này được viết bằng Jupyter Notebook để dễ dàng quan sát các bước phân tích:
1. Mở Jupyter Notebook / Jupyter Lab.
2. Chạy từ trên xuống dưới toàn bộ các cell trong `phan1.ipynb` để xem các truy vấn tính toán cho câu hỏi trắc nghiệm.
3. Chạy toàn bộ các cell trong `phan2.ipynb` để tạo ra các biểu đồ trực quan hoá và đọc các insight kinh doanh.

### Bước 4: Chạy mô hình Dự báo Doanh thu (Phần 3)
Chạy trực tiếp file script Python bằng lệnh sau trên Terminal:
```bash
python phan3_v6.py
```
**Luồng xử lý của hệ thống:**
1. Code sẽ tiến hành Cross-Validation theo thời gian (Shadow Validation) trên 548 ngày cuối của tập Train để in ra các metric kiểm chứng (R², MAE, RMSE).
2. Mô hình tiến hành huấn luyện đệ quy đa nhánh (Dual-branch Recursive Forecasting).
3. Script tự động xuất file dự đoán cuối cùng thành `submission.csv`.
4. Cuối cùng, script sẽ tự động in ra màn hình bảng Feature Importances phục vụ cho mục đích giải thích mô hình (Explainability).

*Lưu ý: Hệ thống đã thiết lập cố định Random Seed (random_state=42), đảm bảo kết quả sinh ra ở máy tính của Ban Giám Khảo sẽ giống 100% kết quả đã nộp.*

## 🏆 Phương pháp tiếp cận mô hình (Phần 3)
Giải pháp cho phần dự báo dài hạn (548 ngày) tập trung vào tính bền vững thay vì dùng các mô hình Deep Learning phức tạp dễ bị Overfitting:
- **Gradient Boosting Regressor (GBR)** kết hợp chiến lược mô hình hóa Đệ quy (Recursive Forecasting) phân nhánh kép.
- **Margin Preservation:** Dự báo tỷ lệ giá vốn (`COGS/Revenue`) để ngăn chặn tình trạng giá vốn cao hơn doanh thu.
- **Tuân thủ tuyệt đối quy định:** Không dùng dữ liệu ngoài (No external data), không lấy dữ liệu tương lai để làm mồi dự báo (No Leakage).
