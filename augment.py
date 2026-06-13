import cv2
import numpy as np
import random
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# --- CÁC CLASS CUSTOM ĐƯỢC GIỮ NGUYÊN (SRP: Chỉ lo nhiệm vụ tạo hiệu ứng) ---
class RandomNoiseLines(ImageOnlyTransform):
    def __init__(self, num_lines=(1, 10), thickness=(1, 3), p=0.5):
        super().__init__(p=p)
        self.num_lines = num_lines
        self.thickness = thickness

    def apply(self, img, **params):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        n_lines = random.randint(self.num_lines[0], self.num_lines[1])
        for _ in range(n_lines):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            c = random.randint(50, 150) 
            thick = random.randint(self.thickness[0], self.thickness[1])
            cv2.line(img_copy, (x1, y1), (x2, y2), (c, c, c), thick)
        return img_copy

class RandomNoiseDots(ImageOnlyTransform):
    def __init__(self, num_dots=(35, 40), radius=(1, 3), p=0.5):
        super().__init__(p=p)
        self.num_dots = num_dots
        self.radius = radius

    def apply(self, img, **params):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        n_dots = random.randint(self.num_dots[0], self.num_dots[1])
        for _ in range(n_dots):
            x, y = random.randint(0, w), random.randint(0, h)
            r = random.randint(self.radius[0], self.radius[1])
            color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
            cv2.circle(img_copy, (x, y), r, color, -1)
        return img_copy

class RandomRegionBlur(ImageOnlyTransform):
    # Thêm blur_limit=(3, 5) vào đây để khống chế độ cận thị
    def __init__(self, num_regions=(1, 6), region_size=(30, 45), blur_limit=(3, 5), p=0.5):
        super().__init__(p=p)
        self.num_regions = num_regions
        self.region_size = region_size
        self.blur_limit = blur_limit

    def apply(self, img, **params):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        n_regions = random.randint(self.num_regions[0], self.num_regions[1])
        
        for _ in range(n_regions):
            rw = random.randint(self.region_size[0], min(self.region_size[1], w))
            rh = random.randint(self.region_size[0], min(self.region_size[1], h))
            
            x = random.randint(0, max(0, w - rw))
            y = random.randint(0, max(0, h - rh))
            
            # Cắt cái vùng ảnh đó ra
            roi = img_copy[y:y+rh, x:x+rw]
            
            # Bốc ngẫu nhiên độ mờ từ blur_limit
            k = random.randint(self.blur_limit[0], self.blur_limit[1])
            # Đảm bảo kernel size luôn là số lẻ (3, 5, 7...) vì OpenCV yêu cầu thế
            if k % 2 == 0:
                k += 1 
                
            # Đập độ mờ nhẹ nhàng vào cái vùng đó thôi
            img_copy[y:y+rh, x:x+rw] = cv2.blur(roi, (k, k))
            
        return img_copy

class StrikethroughLine(ImageOnlyTransform):
    # Thêm tham số num_lines=(min, max) vào đây
    def __init__(self, num_lines=(1, 5), thickness=(1, 2), p=0.5):
        super().__init__(p=p)
        self.num_lines = num_lines  # Lưu lại dải số lượng đường kẻ
        self.thickness = thickness  

    def apply(self, img, **params):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        # Bốc ngẫu nhiên xem lần này vẽ mấy đường
        n_lines = random.randint(self.num_lines[0], self.num_lines[1])
        
        for _ in range(n_lines):
            # Mỗi đường kẻ tự random vị trí y_start, y_end, màu sắc và độ dày riêng
            y_start = random.randint(h // 4, 3 * h // 4) 
            y_end = random.randint(h // 4, 3 * h // 4)
            c = random.randint(0, 80) # Giữ màu xám/đen
            
            thick = random.randint(self.thickness[0], self.thickness[1])
            cv2.line(img_copy, (0, y_start), (w, y_end), (c, c, c), thick)
            
        return img_copy
class RandomQuadInvert(ImageOnlyTransform):
    """Đảo màu cục bộ (Invert) trong một vùng tứ giác ngẫu nhiên"""
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]

        # 1. Bốc 2 điểm trên cạnh dài trên (y = 0) và sắp xếp tăng dần
        top_x = sorted([random.randint(0, w), random.randint(0, w)])
        
        # 2. Bốc 2 điểm trên cạnh dài dưới (y = h) và sắp xếp tăng dần
        bottom_x = sorted([random.randint(0, w), random.randint(0, w)])

        # 3. Tạo array 4 đỉnh tứ giác theo chiều kim đồng hồ
        pts = np.array([
            [top_x[0], 0],       # Điểm trên - trái
            [top_x[1], 0],       # Điểm trên - phải
            [bottom_x[1], h],    # Điểm dưới - phải (Phải vòng ngược lại)
            [bottom_x[0], h]     # Điểm dưới - trái
        ], np.int32)

        # 4. Tạo mặt nạ (Mask) đen xì, rồi tô trắng cái vùng tứ giác kia
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # 5. Tạo một bản copy đảo ngược toàn bộ màu của ảnh
        inverted_img = cv2.bitwise_not(img)

        # 6. Trộn ảnh: Nơi nào mask == 255 -> lấy ảnh Invert. Còn lại -> Lấy ảnh Gốc.
        # Phải expand_dims cho mask để nó từ 2D (h,w) lên 3D (h,w,1) khớp với ảnh màu
        mask_bool = mask[:, :, np.newaxis] == 255
        result = np.where(mask_bool, inverted_img, img)

        return result.astype(np.uint8)
# --- ĐÁP ỨNG SOLID: CLASS QUẢN LÝ PIPELINE & THỰC THI ---
class CaptchaAugmenter:
    """Class chịu trách nhiệm duy nhất cho việc cấu hình và thực thi Augmentation"""
    def __init__(self, bbox_format='yolo'):
        # Định nghĩa bộ luật/pipeline ngay khi khởi tạo
        self.transform = A.Compose([
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            
            # Quy hoạch gom nhóm để tránh ảnh nát bét (như anh em mình bàn)
            A.OneOf([
                RandomNoiseLines(num_lines=(2, 6), thickness=(1, 3), p=1.0),
                StrikethroughLine(num_lines=(1, 5), thickness=(1, 2), p=1.0),
                RandomRegionBlur(num_regions=(1, 6), region_size=(30, 45), p=1.0),
            ], p=0.7),
            
            A.OneOf([
                RandomQuadInvert(p=1.0),
                RandomNoiseDots(num_dots=(35, 40), radius=(2, 4), p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.6),

            A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
            A.Blur(blur_limit=2, p=0.2),
        ], bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels']))

    def augment(self, image: np.ndarray, bboxes: list, class_labels: list) -> tuple:
        """
        Hàm cốt lõi đáp ứng yêu cầu của Tú: Đầu vào là 1 ảnh + boxes -> Đầu ra là ảnh + boxes đã biến đổi.
        """
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
