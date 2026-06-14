import os
import wandb
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from augment import CaptchaAugmenter
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
)

# Import dictionary từ file config của cậu để dịch ngược ID về Chữ cái
from config import CHAR_TO_ID
# Tạo từ điển dịch ngược: {0: 'a', 1: 'b', ...}
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}

# ==============================================================================
# 1. TRẠM THÔNG DỊCH: ĐỌC DỮ LIỆU YOLO -> BIẾN THÀNH TEXT CHO DONUT
# ==============================================================================
class CaptchaDonutDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, augmenter=None, max_length=12):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.augmenter = augmenter  # <--- Lưu lại bộ Augmenter
        self.max_length = max_length
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(self.label_dir, txt_name)

        # Khởi tạo sẵn để nhặt data từ file YOLO
        text_label = ""
        bboxes = []
        class_labels = []

        # 1. Đọc file YOLO để lấy TOÀN BỘ ký tự và tọa độ
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    class_id = int(parts[0])
                    
                    # Gom tọa độ lại cho thằng Augmenter nó khỏi chửi
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    
                    text_label += ID_TO_CHAR[class_id].upper() # Ép hoa như cậu muốn
                    bboxes.append(box)
                    class_labels.append(class_id)

        # 2. Đọc ảnh và chuyển sang Numpy Array
        image = np.array(Image.open(img_path).convert("RGB"))

        # 3. AUGMENT ON-THE-FLY TRƯỚC KHI VÀO GPU
        if self.augmenter:
            # Nhét đủ 3 tham số vào, nhưng lúc nhận về chỉ "nhặt" đúng cái index [0] (là ảnh)
            # Hai dấu _ dùng để vứt bỏ bboxes và class_labels đã bị biến đổi
            image, _, _ = self.augmenter.augment(image=image, bboxes=bboxes, class_labels=class_labels)

        # ... (Đoạn dưới chuyển pixel_values và labels cho Donut giữ nguyên như cũ) ...
        sequence = f"<s_captcha>{text_label}</s_captcha>"
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

# ==============================================================================
# 2. HÀM TÍNH TOÁN EXACT MATCH (IN RA CHO SƯỚNG MẮT)
# ==============================================================================
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions.argmax(axis=-1) # Lấy token có xác suất cao nhất

    # Đưa các token -100 về lại pad_token để decode không bị lỗi
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    # Dọn dẹp khoảng trắng dư thừa và so sánh
    exact_matches = 0
    for p, l in zip(pred_str, label_str):
        # Cắt bỏ cái tag <s_captcha> lúc in ra
        p_clean = p.replace("<s_captcha>", "").strip()
        l_clean = l.replace("<s_captcha>", "").strip()
        if p_clean == l_clean:
            exact_matches += 1

    return {"exact_match": exact_matches / len(label_str)}

# ==============================================================================
# 3. HỆ THỐNG HUẤN LUYỆN CHÍNH (MAIN TIER)
# ==============================================================================
if __name__ == "__main__":
    # ==============================================================================
    # 1. KÍCH HOẠT CẤU HÌNH CỦA WANDB (MỐC THỜI GIAN: 2026)
    # ==============================================================================
    # Bật tính năng log model lên hệ thống Artifacts của wandb khi kết thúc (end)
    os.environ["WANDB_LOG_MODEL"] = "end" 
    os.environ["WANDB_PROJECT"] = "donut-captcha-vlm"
    
    print("🚀 Khởi động nạp mô hình Donut...")
    
    # Load mô hình pre-trained
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Ép kích thước ảnh của mô hình về chuẩn CAPTCHA (60x220)
    processor.image_processor.size = {"height": 60, "width": 220}
    processor.image_processor.do_align_long_axis = False

    # Thêm token báo hiệu tác vụ giải CAPTCHA
    processor.tokenizer.add_tokens(["<s_captcha>", "</s_captcha>"])
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Khai báo token đặc biệt cho Model
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_captcha>"])[0]

    print("📦 Đang chuẩn bị Dataset...")
    # Khởi tạo bộ bóp méo hình ảnh
    aug_module = CaptchaAugmenter()
    
    # Tập Train: Bật bóp méo on-the-fly để AI luyện công
    train_dataset = CaptchaDonutDataset("captcha_dataset/train/images", "captcha_dataset/train/labels", processor, augmenter=aug_module)
    
    # TẬP VAL: SỬA LỖI - ĐỂ NONE ĐỂ GIỮ ẢNH SẠCH LÚC ĐÁNH GIÁ ĐỘ BÁ ĐẠO
    val_dataset = CaptchaDonutDataset("captcha_dataset/val/images", "captcha_dataset/val/labels", processor, augmenter=None)

    print(f"✅ Đã tải: {len(train_dataset)} ảnh Train | {len(val_dataset)} ảnh Val")

    # ==============================================================================
    # 2. CẤU HÌNH TRAINING ARGUMENTS CHO A100 + WANDB
    # ==============================================================================
    training_args = TrainingArguments(
        output_dir="/content/drive/MyDrive/Captcha_Reader/donut_captcha_tmp", # Vẫn phải giữ để lưu checkpoint tạm phục vụ Early Stopping
        per_device_train_batch_size=128,   
        per_device_eval_batch_size=128,
        dataloader_num_workers=8,          
        bf16=True,                         
        learning_rate=2e-5,                
        num_train_epochs=100,              
        
        # Đổi thành "wandb" để kích hoạt vẽ biểu đồ và đẩy trọng số lên mây tự động
        report_to="wandb", 
        
        # Cơ chế Checkpoint & Early Stopping (Bắt buộc lưu local tạm thời)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,           
        
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("🔥 Bắt đầu quá trình nướng GPU và đẩy Log lên Weights & Biases...")
    trainer.train()
    
    # Kết thúc phiên chạy wandb mượt mà
    wandb.finish()
    print("🎉 Hoàn tất! Bạn có thể vào thẳng wandb.ai để tải Trọng số bản ngon nhất trong mục Artifacts!")