# 🏆 CAPTCHA Reader VLM: Đấu Trường YOLO vs Donut vs TrOCR

Dự án này cung cấp một quy trình hoàn chỉnh (End-to-End) để tạo dữ liệu, huấn luyện và kiểm thử các mô hình AI đọc CAPTCHA bao gồm **YOLO**, **Donut**, và **TrOCR**.

---

## ⚙️ 1. Cài đặt môi trường (Installation)

Dự án sử dụng `uv` để quản lý môi trường và thư viện siêu tốc.

1. Cài đặt `uv` và đồng bộ hóa các thư viện cần thiết:
```bash
pip install uv -q
uv sync
```

2. **CẬP NHẬT QUAN TRỌNG CHO DONUT:** Bạn cần thay thế file `modeling_donut_swin.py` mặc định của thư viện `transformers` bằng file đã được tùy chỉnh ở thư mục gốc của dự án này.
   
   *Ví dụ (Copy file đè vào thư mục cài đặt của môi trường ảo):*
```bash
cp modeling_donut_swin.py D:\.vscode\testPMDRL\.venv\Lib\site-packages\transformers\models\donut\modeling_donut_swin.py
```
*(Hãy thay đổi đường dẫn `D:\...` cho phù hợp với máy của bạn).*

---

## 📥 2. Tải Dữ liệu & Trọng số (Data & Weights)

Vui lòng tải xuống các tài nguyên sau và giải nén trực tiếp vào thư mục gốc (Working Directory) của dự án:

* 🗂️ **Dataset:** [Tải tại đây](https://drive.google.com/file/d/1DeFltqICrB9WqTTz_ZX1QtAfxqjIGG_3/view?usp=sharing)
* 🏋️ **Weights (Trọng số đã train):** [Tải tại đây](https://drive.google.com/file/d/1z0CBiGQXsuGqSfY95p01H1RozrYASf8d/view?usp=sharing)

---

## 🚀 3. Chạy thử nghiệm (Inference / Testing)

Sau khi đã nạp đủ trọng số và dữ liệu vào thư mục gốc, bạn có thể khởi động giao diện để so sánh trực tiếp sức mạnh của 3 mô hình.

Khởi chạy ứng dụng UI:
```bash
uv run app.py
```
> **💡 Mẹo:** Sau quá trình huấn luyện (hoặc dùng pre-trained weights), bạn có thể kiểm tra hiệu suất của 3 mô hình với các ảnh CAPTCHA thực tế được lưu sẵn trong thư mục `real_captcha/`.

---

## 🔥 4. Huấn luyện Mô hình (Training)

Cấu hình huấn luyện đã được đồng bộ chuẩn xác. Bạn có thể train từng mô hình riêng lẻ bằng các lệnh sau:

* **Train TrOCR:**
```bash
uv run TrOCR_train.py
```
* **Train Donut:**
```bash
uv run Donut_train.py
```
* **Train YOLO:**
```bash
uv run YOLO.py
```

☁️ **Huấn luyện trực tuyến trên Google Colab:** Nếu bạn muốn tận dụng GPU miễn phí, hãy tải file `Captcha_Reader.ipynb` lên Colab và chạy theo thứ tự các ô lệnh (cells) đã được thiết lập sẵn.

---

## 🛠️ 5. Tự tạo Dataset mới (Generate Custom Dataset)

Dự án hỗ trợ công cụ tạo CAPTCHA tự động để làm phong phú dữ liệu huấn luyện.

1. Chỉnh sửa siêu dữ liệu (metadata) theo ý muốn tại file `config.py`.
2. Chạy file sample để xem thử một vài ảnh minh họa trước khi render hàng loạt:
```bash
uv run sample.py
```
3. Khởi tạo toàn bộ tập dữ liệu (kết quả sẽ được lưu tự động tại folder `captcha_dataset/`):
```bash
uv run generate_dataset.py
```