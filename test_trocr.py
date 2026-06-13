import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessor, LogitsProcessorList
from augment import get_train_transforms

# VOCAB chuẩn
VOCAB = "23456789ABCDEFGHJKLMNPRSTUVWXYZ"
id_to_char = {idx: char for idx, char in enumerate(VOCAB)}

# --- CLASS ÉP VOCAB (LOẠI BỎ KÝ TỰ RÁC) ---
class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_chars):
        self.allowed_ids = set([tokenizer.eos_token_id])
        for char in allowed_chars:
            token_ids = tokenizer(char, add_special_tokens=False).input_ids
            self.allowed_ids.update(token_ids)
        self.allowed_ids = list(self.allowed_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, -float('inf'))
        mask[:, self.allowed_ids] = scores[:, self.allowed_ids]
        return mask

def get_ground_truth(txt_path):
    if not os.path.exists(txt_path):
        return "Unknown"
    with open(txt_path, "r") as f:
        lines = f.readlines()
        parsed = [{'class_id': int(p.split()[0]), 'x_center': float(p.split()[1])} for p in lines]
        parsed.sort(key=lambda x: x['x_center'])
        return "".join([id_to_char[item['class_id']] for item in parsed])

def main():
    print("⏳ Đang khởi động A6000 & Load TrOCR...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)
    
    logits_processor = LogitsProcessorList([
        RestrictVocabLogitsProcessor(processor.tokenizer, VOCAB)
    ])
    transform = get_train_transforms()

    img_dir, lbl_dir = "images", "labels"
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])[:10]

    # Lưu kết quả để vẽ Plot
    plot_data = []

    print("🔍 Đang cho AI dự đoán...")
    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        img_path, txt_path = os.path.join(img_dir, img_name), os.path.join(lbl_dir, base_name + ".txt")
        
        gt_text = get_ground_truth(txt_path)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Băm ảnh qua lò Augment
        augmented = transform(image=image, bboxes=[[0.1, 0.1, 0.2, 0.2]], class_labels=[0])
        aug_img_np = augmented['image']

        pixel_values = processor(images=Image.fromarray(aug_img_np), return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, 
                max_new_tokens=6, 
                logits_processor=logits_processor
            )
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(" ", "").upper()

        plot_data.append({
            'image': aug_img_np,
            'gt': gt_text,
            'pred': pred_text
        })

    # --- TẠO CỬA SỔ HIỂN THỊ (DASHBOARD) ---
    print("🎨 Đang render cửa sổ kết quả...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()

    for i, data in enumerate(plot_data):
        axes[i].imshow(data['image'])
        
        # Đánh giá đúng/sai để tô màu tiêu đề
        is_correct = (data['gt'] == data['pred'])
        title_color = 'green' if is_correct else 'red'
        status = "✅" if is_correct else "❌"
        
        title_text = f"GT: {data['gt']}\nPred: {data['pred']} {status}"
        
        axes[i].set_title(title_text, color=title_color, fontsize=14, weight='bold', pad=10)
        axes[i].axis('off')

    plt.suptitle("KIỂM TRA BẢN LĨNH TrOCR TRƯỚC CAPTCHA 'NHÀ LÀM'", fontsize=20, weight='heavy', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()