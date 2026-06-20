import os
from sample_test_TrOCR import SampleTestCallback # Nhớ sửa file sample_test tương thích với TrOCR nhé
import evaluate
from utils import clean_text
import wandb
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from augment import CaptchaAugmenter
import torch
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    EarlyStoppingCallback,
    default_data_collator
)

# Import dictionary từ file config của cậu để dịch ngược ID về Chữ cái
from config import CHAR_TO_ID
# Tạo từ điển dịch ngược: {0: 'a', 1: 'b', ...}
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}

# ==============================================================================
# 1. TRẠM THÔNG DỊCH: ĐỌC DỮ LIỆU YOLO -> BIẾN THÀNH TEXT CHO TrOCR
# ==============================================================================
cer_metric = evaluate.load("cer")

class CaptchaTrOCRDataset(Dataset):
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
                # TrOCR không cần bận tâm thứ tự tọa độ X, nhưng ta vẫn lấy bboxes để nhét vào Augmenter
                for line in f.readlines():
                    parts = line.split()
                    class_id = int(float(parts[0]))
                    box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    
                    text_label += ID_TO_CHAR[class_id].upper() # Ép hoa như cậu muốn
                    bboxes.append(box)
                    class_labels.append(class_id)

        # 2. Đọc ảnh và chuyển sang Numpy Array
        image = np.array(Image.open(img_path).convert("RGB"))

        # 3. AUGMENT ON-THE-FLY TRƯỚC KHI VÀO GPU
        if self.augmenter:
            try:
                # Nhét đủ 3 tham số vào, lấy lại ảnh đã bị cào xước/nhiễu
                image, _, _ = self.augmenter.augment(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                pass # Bỏ qua ảnh lỗi augment để tránh sập epoch

        # 4. CHUYỂN HÓA DỮ LIỆU CHO TrOCR
        # - Mắt (Vision): Ép ảnh thành ma trận Pixel Values
        # Chuyển numpy array về lại PIL Image vì processor của TrOCR thích PIL hơn
        pil_image = Image.fromarray(image)
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.squeeze()

        # - Não (Language): Khác biệt lớn -> TrOCR KHÔNG CẦN token <s_captcha>
        labels = self.processor.tokenizer(
            text_label,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze()

        # Bịp PyTorch: Đổi pad_token thành -100 để hàm Loss bỏ qua các ký tự đệm rỗng
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

# ==============================================================================
# 2. HÀM TÍNH TOÁN EXACT MATCH VÀ CER (GỌN HƠN DONUT NHIỀU)
# ==============================================================================
def compute_metrics(pred, processor):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions # Seq2SeqTrainer tự generate ra ID, không phải là logits nữa!

    # Bỏ qua token -100 lúc nãy
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    # Dịch ngược ID thành chữ người đọc được
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    p_clean_list = []
    l_clean_list = []
    exact_matches = 0
    
    for p, l in zip(pred_str, label_str):
        # Vẫn gọi hàm clean_text của cậu để đảm bảo text sạch sẽ
        p_clean = clean_text(p)
        l_clean = clean_text(l)
        
        p_clean_list.append(p_clean)
        l_clean_list.append(l_clean)
        
        if p_clean == l_clean:
            exact_matches += 1

    cer_score = cer_metric.compute(predictions=p_clean_list, references=l_clean_list)

    return {
        "exact_match": exact_matches / len(label_str) if len(label_str) > 0 else 0,
        "cer": cer_score
    }

# ==============================================================================
# 3. HỆ THỐNG HUẤN LUYỆN CHÍNH (MAIN TIER)
# ==============================================================================
if __name__ == "__main__":
    # ==============================================================================
    # 1. KÍCH HOẠT CẤU HÌNH CỦA WANDB (MỐC THỜI GIAN: 2026)
    # ==============================================================================
    os.environ["WANDB_LOG_MODEL"] = "end" 
    os.environ["WANDB_PROJECT"] = "trocr-captcha-vlm"
    
    print("🚀 Khởi động nạp mô hình TrOCR (Vision-Language Model)...")
    
    # Load mô hình pre-trained bản chuyên in ấn
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    # Cấu hình "Não" của TrOCR (Tương tự như Donut nhưng set bằng tham số mặc định của RoBERTa)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Các thông số chuyên dụng để sinh chuỗi (Generation)
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 12 
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    print("📦 Đang chuẩn bị Dataset...")
    aug_module = CaptchaAugmenter()
    
    # Tập Train: Bật bóp méo on-the-fly
    train_dataset = CaptchaTrOCRDataset("captcha_dataset/train/images", "captcha_dataset/train/OCR_labels", processor, augmenter=aug_module)
    
    # Tập Val: Để None giữ ảnh sạch
    val_dataset = CaptchaTrOCRDataset("captcha_dataset/val/images", "captcha_dataset/val/OCR_labels", processor, augmenter=None)

    print(f"✅ Đã tải: {len(train_dataset)} ảnh Train | {len(val_dataset)} ảnh Val")

    # ==============================================================================
    # 2. CẤU HÌNH TRAINING ARGUMENTS CHO A100 + WANDB
    # ==============================================================================
    # Lưu ý: Chuyển sang dùng Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_trocr_checkpoints", 
        per_device_train_batch_size=64,   
        per_device_eval_batch_size=64,
        dataloader_num_workers=8,          
        bf16=True,                         
        learning_rate=2e-5,                
        num_train_epochs=100,              
        
        predict_with_generate=True, # <--- QUAN TRỌNG: Kích hoạt chế độ sinh Text lúc Validate
        
        report_to="wandb", 
        
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="cer", # Đổi mốc tracking qua CER thay vì Loss cho thực dụng
        greater_is_better=False,           
        
        logging_steps=10,
    )

    # Dùng Seq2SeqTrainer thay vì Trainer thường
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.image_processor, # Bắt buộc truyền tokenizer vào đây
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics = lambda pred: compute_metrics(pred, processor),
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            
            SampleTestCallback(
                processor=processor, 
                eval_dataset=val_dataset, 
                num_samples=10,
                output_dir="/content/drive/MyDrive/Captcha_Reader/sample_tests" 
            )
        ]
    )

    print("🔥 Bắt đầu quá trình nướng GPU và đẩy Log lên Weights & Biases (TrOCR Version)...")
    trainer.train()
    
    wandb.finish()
    print("🎉 Hoàn tất! TrOCR đã luyện thành công!")