import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import * 
class CaptchaGenerator:
    """Class chịu trách nhiệm duy nhất: Sinh ra 1 cặp (Ảnh, Nhãn YOLO) trên RAM"""
    
    def __init__(self):
        # Tải sẵn font vào bộ nhớ để tối ưu tốc độ khi gọi nhiều lần
        self.fonts = [ImageFont.truetype(f, FONT_SIZE) for f in FONT_PATHS]

    def _distort_single_char_cv2(self, pil_image):
        """Hàm private dùng nội bộ để làm méo 1 ký tự (Bản nâng cấp ADAPTIVE KERNEL)"""
        # ======================================================================
        # BƯỚC 1: ĐO LƯỜNG KÍCH THƯỚC THỰC TẾ CỦA CHỮ CÁI
        # ======================================================================
        bbox = pil_image.getbbox()
        if not bbox:
            return pil_image # Nếu ảnh rỗng thì bỏ qua luôn
            
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        min_dim = min(char_w, char_h)
        
        # ======================================================================
        # BƯỚC 2: TÍNH TOÁN ĐỘ DÀY (KERNEL) TỶ LỆ THUẬN VỚI KÍCH THƯỚC CHỮ
        # Búa (kernel) chỉ được to tối đa bằng 12% kích thước nhỏ nhất của chữ.
        # Và chốt hạ tuyệt đối không bao giờ vượt quá 3px để chống "cục ục ịch".
        # ======================================================================
        max_k = max(1, int(min_dim * 0.12))
        safe_max_kernel = min(max_k, 3) 

        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        is_hollow = random.random() < HOLLOW_PROB
        
        if is_hollow:
            max_safe_thick = max(1, int(min_dim * 0.15))
            k_size = min(HOLLOW_THICKNESS, max_safe_thick)
            k_size = min(k_size, 3) 
            k_size = max(1, k_size) 
            
            # =========================================================
            # BƯỚC ĐỆM: BƠM MÁU CHO NÉT THANH (PRE-DILATE)
            # =========================================================
            # Bơm béo toàn bộ phôi gốc lên 2x2 để đảm bảo không có nét nào mỏng dưới 2px
            kernel_base = np.ones((2, 2), np.uint8)
            base_img = cv2.dilate(open_cv_image, kernel_base, iterations=1)

            # Khởi tạo dao đục
            kernel_hollow = np.ones((k_size, k_size), np.uint8)
            
            # 2. Tạo Vỏ ngoài (Phình to) và Lõi trong (Ăn mòn) TỪ PHÔI ĐÃ LÀM BÉO
            outer = cv2.dilate(base_img, kernel_hollow, iterations=1)
            inner = cv2.erode(base_img, kernel_hollow, iterations=1)
            
            # 3. Rỗng = Vỏ - Lõi.
            open_cv_image = cv2.subtract(outer, inner)
            
            # =========================================================
            # BƯỚC HÀN GẮN: HẠ NGƯỠNG LƯỠI HÁI
            # =========================================================
            alpha_channel = open_cv_image[:, :, 3]
            # Hạ ngưỡng cắt từ 30 xuống 10 để "tha mạng" cho các pixel bị GridDistortion làm mờ nhẹ, 
            # giúp nét chữ nối liền nhau mà vẫn căng đét.
            _, solid_alpha = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
            open_cv_image[:, :, 3] = solid_alpha
        else:
            if random.random() < MORPH_PROB:
                # Bốc random từ 1 đến giới hạn an toàn vừa tính
                kernel_size = random.randint(1, safe_max_kernel)
                
                # Nếu bốc trúng số 1 thì bỏ qua vì matrix 1x1 chả có tác dụng gì, chỉ làm tốn CPU
                if kernel_size > 1:
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    if random.random() > 0.1: 
                        open_cv_image = cv2.dilate(open_cv_image, kernel, iterations=1)
                    else:
                        open_cv_image = cv2.erode(open_cv_image, kernel, iterations=1)
        
        # --- 1. TÁC ĐỘNG HÌNH THÁI HỌC (Làm đậm / Làm mảnh nét) ---
        if random.random() < MORPH_PROB:
            kernel_size = random.randint(MORPH_KERNEL_MIN, MORPH_KERNEL_MAX)
            # 50% dùng kernel dọc, 50% dùng kernel ngang
            if random.random() > 0.5:
                kernel = np.ones((kernel_size, 1), np.uint8)
            else:
                kernel = np.ones((1, kernel_size), np.uint8)
            
            # 50% nong to (dilate), 50% ăn mòn (erode)
            if random.random() > 0.5:
                open_cv_image = cv2.dilate(open_cv_image, kernel, iterations=1)
            else:
                open_cv_image = cv2.erode(open_cv_image, kernel, iterations=1)
        
        # --- 2. BIẾN DẠNG LƯỢN SÓNG (Sine Wave Mapping) ---
        h, w = open_cv_image.shape[:2]
        x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))

        strength_x = random.choice([-1, 1]) * random.uniform(DISTORT_STRENGTH_MIN, DISTORT_STRENGTH_MAX) 
        strength_y = random.choice([-1, 1]) * random.uniform(DISTORT_STRENGTH_MIN, DISTORT_STRENGTH_MAX)
        wavelength = random.uniform(DISTORT_WAVELENGTH_MIN, DISTORT_WAVELENGTH_MAX) 
        
        phase_x = random.uniform(0, 2 * np.pi)
        phase_y = random.uniform(0, 2 * np.pi)
        
        x_map = x_map + strength_x * np.sin(y_map / wavelength + phase_x)
        y_map = y_map + strength_y * np.cos(x_map / wavelength + phase_y)

        distorted_image = cv2.remap(
            open_cv_image, 
            x_map.astype(np.float32), 
            y_map.astype(np.float32), 
            interpolation=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )

        return Image.fromarray(cv2.cvtColor(distorted_image, cv2.COLOR_BGRA2RGBA))

    def generate_data(self):
        """
        Giao diện public: Sinh ra 1 dữ liệu hoàn chỉnh.
        Returns:
            base_image (PIL.Image): Ảnh CAPTCHA hệ RGB.
            labels_content (list of str): Danh sách nhãn chuẩn YOLO.
        """
        bg_color = (
            random.randint(BG_COLOR_MIN, BG_COLOR_MAX), 
            random.randint(BG_COLOR_MIN, BG_COLOR_MAX), 
            random.randint(BG_COLOR_MIN, BG_COLOR_MAX)
        )
        base_image = Image.new('RGB', (IMG_W, IMG_H), color=bg_color)
        num_chars = random.randint(MIN_CHARS, MAX_CHARS)
        chars = random.choices(VOCAB, k=num_chars)
        
        labels_content = []
        current_x = 5 

        for char in chars:
            font = random.choice(self.fonts)
            
            char_img = Image.new('RGBA', CHAR_CANVAS_SIZE, (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            
            text_color = (
                random.randint(TEXT_COLOR_MIN, TEXT_COLOR_MAX), 
                random.randint(TEXT_COLOR_MIN, TEXT_COLOR_MAX), 
                random.randint(TEXT_COLOR_MIN, TEXT_COLOR_MAX), 
                255
            )
            # --- ÁP DỤNG STROKE ĐỂ LÀM BÉO CHỮ ---
            stroke_w = random.randint(STROKE_WIDTH_MIN, STROKE_WIDTH_MAX)
            
            # Canh tọa độ vẽ vào giữa khung canvas, đắp thêm stroke cùng màu chữ
            char_draw.text(
                (30, 30), char, font=font, fill=text_color, 
                stroke_width=stroke_w, stroke_fill=text_color
            )

            scale_x = random.uniform(SCALE_MIN, SCALE_MAX) 
            scale_y = random.uniform(SCALE_MIN, SCALE_MAX)
            new_w, new_h = int(CHAR_CANVAS_SIZE[0] * scale_x), int(CHAR_CANVAS_SIZE[1] * scale_y)
            
            char_img = char_img.resize((new_w, new_h), resample=Image.BICUBIC)
            char_img = self._distort_single_char_cv2(char_img) 
            
            shear_factor = random.uniform(SHEAR_MIN, SHEAR_MAX) 
            char_img = char_img.transform((new_w, new_h), Image.AFFINE, (1, shear_factor, 0, 0, 1, 0), resample=Image.BICUBIC)
            
            angle = random.randint(ANGLE_MIN, ANGLE_MAX)
            rotated_char = char_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            
            temp_bbox = rotated_char.getbbox() 
            if not temp_bbox: continue
                
            cropped_char = rotated_char.crop(temp_bbox)
            cw, ch = cropped_char.size 
            
            max_y = max(0, IMG_H - ch)
            current_y = random.randint(0, max_y)
            
            if current_x + cw > IMG_W: break
                
            base_image.paste(cropped_char, (current_x, current_y), cropped_char)

            real_left, real_top = current_x, current_y
            x_center, y_center = real_left + cw / 2.0, real_top + ch / 2.0
            
            # Tính toán chuẩn hóa cơ bản
            norm_w = cw / IMG_W
            norm_h = ch / IMG_H
            norm_x = x_center / IMG_W
            norm_y = y_center / IMG_H
            
            # =========================================================
            # LƯỚI BẢO VỆ CHỐNG SAI SỐ LÀM TRÒN (FLOAT PRECISION MARGIN)
            # =========================================================
            margin = 0.002 # Vùng đệm 0.3 pixel để hấp thụ sai số do :.6f sinh ra
            
            norm_w = min(norm_w, 1.0 - 2 * margin)
            norm_h = min(norm_h, 1.0 - 2 * margin)
            
            # Khóa chặt tâm x, y không cho chạm sát mép tuyệt đối
            norm_x = np.clip(norm_x, norm_w / 2.0 + margin, 1.0 - norm_w / 2.0 - margin)
            norm_y = np.clip(norm_y, norm_h / 2.0 + margin, 1.0 - norm_h / 2.0 - margin)
            
            class_id = CHAR_TO_ID[char]
            labels_content.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            step = random.randint(STEP_MIN, STEP_MAX)
            current_x += cw + step
            current_x = max(current_x, 5)

        return base_image, labels_content