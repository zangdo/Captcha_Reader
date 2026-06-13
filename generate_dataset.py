import os
from tqdm import tqdm
from config import *
from gen_image_label import CaptchaGenerator

class DatasetBuilder:
    """Class chịu trách nhiệm thao tác I/O: Tạo folder, chạy vòng lặp và lưu file"""
    
    def __init__(self):
        self.generator = CaptchaGenerator() # Khởi tạo lõi sinh data
        self._setup_directories()

    def _setup_directories(self):
        """Khởi tạo cây thư mục"""
        for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
            os.makedirs(d, exist_ok=True)

    def _save_data(self, img, labels, index, split_type):
        """Hàm nội bộ để ghi file xuống ổ cứng"""
        if split_type == 'train':
            out_img_dir, out_lbl_dir = TRAIN_IMG_DIR, TRAIN_LBL_DIR
        else:
            out_img_dir, out_lbl_dir = VAL_IMG_DIR, VAL_LBL_DIR

        img_name = f"captcha_{index:06d}.jpg"
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
        
        print("\n🔥 Đang sinh tập TRAIN...")
        for i in tqdm(range(num_train), desc="Train Data", unit="img"):
            # 1. Gọi lõi sinh data
            img, labels = self.generator.generate_data()
            # 2. Giao cho hàm I/O lưu trữ
            self._save_data(img, labels, i, split_type='train')
            
        print("\n❄️ Đang sinh tập VALIDATION...")
        for i in tqdm(range(num_train, NUM_IMAGES), desc="Val Data", unit="img"):
            img, labels = self.generator.generate_data()
            self._save_data(img, labels, i, split_type='val')

        print("\n✅ XONG! Tập dữ liệu chuẩn SOLID đã sẵn sàng!")

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()