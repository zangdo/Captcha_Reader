import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from config import *
from gen_image_label import CaptchaGenerator
from augment import CaptchaAugmenter 

class DatasetBuilder:
    """Class chịu trách nhiệm thao tác I/O: Tạo folder, chạy vòng lặp và lưu file"""
    
    def __init__(self):
        self.generator = CaptchaGenerator() # Khởi tạo lõi sinh data gốc
        self.val_augmenter = CaptchaAugmenter() # Khởi tạo bộ bóp méo cố định cho tập Val
        self._setup_directories()

    def _setup_directories(self):
        """Khởi tạo cây thư mục"""
        for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
            os.makedirs(d, exist_ok=True)

    def _apply_val_augmentation(self, img, labels):
        """
        Hàm nội bộ để "hành hạ" ảnh Validation trước khi lưu xuống ổ cứng.
        Đảm bảo ảnh Val bám sát phân phối nhiễu thật của hệ thống.
        """
        # 1. Chuyển PIL Image sang Numpy Array (RGB) để nạp cho Albumentations
        img_np = np.array(img.convert("RGB"))
        
        # 2. Phân tách file nhãn YOLO gốc để lấy bboxes và class_labels truyền cho Augmenter
        bboxes = []
        class_labels = []
        for label in labels:
            parts = label.split()
            class_id = int(parts[0])
            box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            bboxes.append(box)
            class_labels.append(class_id)
        
        try:
            # 3. Thực thi bóp méo qua hộp đen CaptchaAugmenter của cậu
            aug_img_np, aug_bboxes, aug_classes = self.val_augmenter.augment(
                image=img_np, bboxes=bboxes, class_labels=class_labels
            )
            
            # 4. Chuyển ma trận ảnh đã biến đổi ngược lại về định dạng PIL
            final_img = Image.fromarray(aug_img_np)
            
            # 5. Cập nhật lại nhãn tọa độ mới (Đề phòng trường hợp sau này cậu vẫn cần dùng box)
            final_labels = []
            for box, cls in zip(aug_bboxes, aug_classes):
                final_labels.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
                
            return final_img, final_labels
            
        except Exception as e:
            # Phòng trường hợp Albumentations loại bỏ box do bóp méo quá đà làm lỗi hệ thống,
            # Ta sẽ fallback trả về ảnh phôi gốc (vốn dĩ cũng đã có nhiễu sẵn của hàm generator)
            return img, labels

    def _save_data(self, img, labels, index, split_type):
        """Hàm nội bộ để ghi file xuống ổ cứng"""
        if split_type == 'train':
            out_img_dir, out_lbl_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
        else:
            out_img_dir, out_lbl_dir = VAL_IMG_DIR, VAL_LBL_DIR

        # Ép đuôi ảnh về .png để bảo toàn chất lượng pixel sau khi bóp méo (Hạn chế nhiễu nén của .jpg)
        img_name = f"captcha_{index:06d}.png"
        txt_name = f"captcha_{index:06d}.txt"
        
        # Lưu ảnh
        img.save(os.path.join(out_img_dir, img_name))
        
        # Lưu file nhãn
        with open(os.path.join(out_lbl_dir, txt_name), "w") as f:
            f.write("\n".join(labels))

    def build(self):
        """Thực thi toàn bộ quy trình xây dựng Dataset"""
        num_train = int(NUM_IMAGES * TRAIN_RATIO)
        num_val = NUM_IMAGES - num_train
        
        print(f"🚀 Khởi động xưởng sản xuất Data: Tổng {NUM_IMAGES} ảnh")
        print(f"📦 Phân bổ: {num_train} ảnh Train, {num_val} ảnh Validation")
        print(f"📁 Đang lưu tại: {os.path.abspath(DATASET_DIR)}")
        
        print("\n🔥 Đang sinh tập TRAIN (Lưu ảnh phôi gốc - Sẽ bổ sung nhiễu ngẫu nhiên lúc train)...")
        for i in tqdm(range(num_train), desc="Train Data", unit="img"):
            img, labels = self.generator.generate_data()
            self._save_data(img, labels, i, split_type='train')
            
        print("\n❄️ Đang sinh tập VALIDATION (Đã nhúng sẵn nhiễu cố định vào ổ cứng)...")
        for i in tqdm(range(num_train, NUM_IMAGES), desc="Val Data", unit="img"):
            # 1. Sinh ảnh phôi
            img, labels = self.generator.generate_data()
            
            # 2. Tiến hành "nướng" augment tĩnh cố định cho tập Val luôn
            img_val, labels_val = self._apply_val_augmentation(img, labels)
            
            # 3. Lưu xuống đĩa
            self._save_data(img_val, labels_val, i, split_type='val')

        print("\n✅ XONG! Tập dữ liệu chuẩn SOLID bọc sẵn Augment tập Val đã sẵn sàng!")

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()