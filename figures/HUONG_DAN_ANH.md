# Hướng dẫn chèn ảnh vào slide LaTeX

## Bước 1 — Chuẩn bị thư mục

Thư mục `figures/` đã nằm cùng cấp với `presentation.tex`. Đặt ảnh vào đây.

## Bước 2 — Danh sách ảnh gợi ý

| File gợi ý | Nội dung | Cách lấy |
|------------|----------|----------|
| `02_logo.png` | Logo trường / ảnh nhóm | Tự chuẩn bị |
| `04_captcha_kho.png` | CAPTCHA khó đọc | Ảnh từ `real_captcha/` |
| `06_ba_mo_hinh.png` | Sơ đồ 3 mô hình | Vẽ PowerPoint / draw.io |
| `07_kien_truc.png` | Kiến trúc pipeline | Vẽ sơ đồ hoặc chụp từ README |
| `08_sinh_du_lieu.png` | Flow sinh dataset | Vẽ flowchart |
| `09_cau_truc_thu_muc.png` | Cây thư mục dataset | Snipping Tool / tree command |
| `10_captcha_sinh.png` | CAPTCHA sinh tổng hợp | Chạy `uv run sample.py`, chụp màn hình |
| `11_truoc_aug.png` | Ảnh trước augment | Từ `sample.py` |
| `11_sau_aug.png` | Ảnh sau augment | Từ `sample.py` |
| `12_yolo_bbox.png` | YOLO vẽ box | Chạy `uv run app.py`, upload ảnh |
| `13_donut_pipeline.png` | Pipeline Donut | Vẽ sơ đồ |
| `14_trocr.png` | Sơ đồ / ví dụ TrOCR | Vẽ hoặc screenshot |
| `15_wandb.png` | Biểu đồ train WandB | Screenshot wandb.ai |
| `16_gradio_ui.png` | Giao diện Gradio | Chụp màn hình `app.py` |
| `17_bang_ket_qua.png` | Bảng so sánh metrics | Excel → export PNG |
| `18a_input.png` | Ảnh CAPTCHA đầu vào | `real_captcha/` |
| `18b_yolo.png` | Kết quả YOLO có box | Từ Gradio |
| `22_qr_github.png` | QR code repo | Tạo tại qr-code-generator.com |

## Bước 3 — Chèn ảnh vào `presentation.tex`

Mỗi slide có **2 dòng** — một placeholder, một `\includegraphics` bị comment:

```latex
\placeholderimage{10cm}{Mô tả ảnh}
% \includegraphics[width=0.85\textwidth]{figures/07_kien_truc.png}
```

**Khi đã có ảnh**, sửa thành:

```latex
% \placeholderimage{10cm}{Mô tả ảnh}
\includegraphics[width=0.85\textwidth]{figures/07_kien_truc.png}
```

- `width=0.85\textwidth` — ảnh rộng 85% slide (chỉnh 0.5–1.0 tùy ý)
- `height=5cm` — dùng khi muốn cố định chiều cao

## Bước 4 — Biên dịch PDF

```bash
cd D:\LapTrinh\System\Captcha_Reader
pdflatex presentation.tex
pdflatex presentation.tex
```

Hoặc dùng **TeXstudio**, **Overleaf** (upload `presentation.tex` + thư mục `figures/`).

## Lưu ý

- Định dạng: **PNG** (ảnh có text/box) hoặc **JPG** (ảnh chụp màn hình)
- Tên file: không dấu, không khoảng trắng (dùng `_`)
- Nếu ảnh bị méo: thêm `keepaspectratio` → `\includegraphics[width=0.8\textwidth,keepaspectratio]{...}`
