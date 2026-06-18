import os

# ==============================================================================
# 1. CẤU HÌNH TỔNG QUAN (GENERAL SETUP)
# ==============================================================================
NUM_IMAGES = 10000      # 100k ảnh cho A100 
TRAIN_RATIO = 0.9        # Tỷ lệ Train/Val (90/10)

# ==============================================================================
# 2. CẤU HÌNH ĐƯỜNG DẪN (PATHS)
# ==============================================================================
DATASET_DIR = "captcha_dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "train", "OCR_labels")
TRAIN_LBL_YOLO_DIR = os.path.join(DATASET_DIR, "train", "labels") # Thư mục nhãn YOLO riêng cho Training
VAL_IMG_DIR = os.path.join(DATASET_DIR, "val", "images")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "val", "OCR_labels")
VAL_LBL_YOLO_DIR = os.path.join(DATASET_DIR, "val", "labels") # Thư mục nhãn YOLO riêng cho Validation
FONT_PATHS = ["arial.ttf"]

# ==============================================================================
# 3. CẤU HÌNH ẢNH & NHÃN (IMAGE & LABEL PARAMS)
# ==============================================================================
# Kích thước sân khấu (Nới IMG_W lên 280 nếu dùng 8 ký tự nhé)
IMG_W = 220
IMG_H = 60

# Bảng chữ cái
VOCAB = "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz"
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}
CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
# Số lượng ký tự sinh ra trên 1 ảnh
MIN_CHARS = 5
MAX_CHARS = 8

# ==============================================================================
# 4. CẤU HÌNH AUGMENTATION & DÍNH LẸO (DISTORTION & OVERLAP PARAMS)
# ==============================================================================
# Thuật toán bước nhảy (Tọa độ X) - Âm là dính lẹo, Dương là cách xa
STEP_MIN = -13
STEP_MAX = 8

# Cấu hình Font và Khung vẽ nháp
FONT_SIZE = 40
CHAR_CANVAS_SIZE = (100, 100) # Khung RGBA trung gian để vẽ từng chữ

# Biên độ màu sắc (Nền sáng, chữ tối)
BG_COLOR_MIN, BG_COLOR_MAX = 200, 255
TEXT_COLOR_MIN, TEXT_COLOR_MAX = 0, 100

# Biên độ biến dạng hình học cho từng ký tự
SCALE_MIN, SCALE_MAX = 0.7, 1.2     # Phóng to / Thu nhỏ
SHEAR_MIN, SHEAR_MAX = -0.3, 0.3    # Kéo xiên (Italic)
ANGLE_MIN, ANGLE_MAX = -35, 35      # Góc xoay (Độ)

# ==============================================================================
# 5. CẤU HÌNH BÓP MÉO LƯỢN SÓNG (WAVE DISTORTION & MORPHOLOGY)
# ==============================================================================
# Xác suất làm mỏng/đậm nét chữ (Erode/Dilate)
MORPH_PROB = 0.7
MORPH_KERNEL_MIN = 2
MORPH_KERNEL_MAX = 3

# Biên độ lượn sóng (Strength) - Càng to chữ càng bị bẻ cong gắt
DISTORT_STRENGTH_MIN = 3.0
DISTORT_STRENGTH_MAX = 6.0

# Bước sóng (Wavelength) - Càng nhỏ chữ càng xoắn quẩy như mì tôm
DISTORT_WAVELENGTH_MIN = 9.0
DISTORT_WAVELENGTH_MAX = 20.0
# ==============================================================================
# CẤU HÌNH ĐỘ DÀY NÉT CHỮ (BÉO HƠN ĐỂ CHỊU NHIỄU)
# ==============================================================================
STROKE_WIDTH_MIN = 1
STROKE_WIDTH_MAX = 2  # Để 2 là nét béo lắm rồi, để 3 có thể bị dính cục
FONT_PATHS = [
    "fonts/ARIAL.TTF",
    "fonts/CALIBRI.TTF",
    "fonts/CONSOLA.TTF",    # Đã sửa lại cho khớp
    "fonts/COUR.TTF",
    "fonts/GEORGIA.TTF",
    "fonts/SEGOEUI.TTF",
    "fonts/TAHOMA.TTF",
    "fonts/TIMES.TTF",
    "fonts/TREBUC.TTF",     # Đã sửa lại cho khớp
    "fonts/VERDANA.TTF"
]
# ==============================================================================
# 6. CẤU HÌNH CHỮ RỖNG (HOLLOW TEXT - YANDEX STYLE)
# ==============================================================================
# Xác suất một chữ cái bị "móc ruột" (Để 0.3 tức là trộn lẫn cả chữ đặc và chữ rỗng cho AI lú)
HOLLOW_PROB = 0.3 
HOLLOW_THICKNESS = 2  # Độ dày của cái viền chữ