import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LogitsProcessor, LogitsProcessorList

# --- IMPORT CLASS CỦA TÚ ---
# Giả sử class chứa hàm generate_data tên là CaptchaGenerator
from gen_image_label import CaptchaGenerator 

# Cấu hình VOCAB và ID
from config import VOCAB
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}
id_to_char = {idx: char for idx, char in enumerate(VOCAB)}

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

def main():
    print("⏳ Đang khởi động AI & Load TrOCR...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)
    
    logits_processor = LogitsProcessorList([
        RestrictVocabLogitsProcessor(processor.tokenizer, VOCAB)
    ])

    # Khởi tạo Generator
    generator = CaptchaGenerator()
    
    plot_data = []
    print("🔍 Đang sinh dữ liệu và dự đoán...")
    
    for _ in range(10):
        # 1. Sinh ảnh và nhãn từ hàm của Tú
        img_pil, labels_content = generator.generate_data()
        
        # 2. Giải mã nhãn YOLO (để lấy GT text)
        parsed_gt = []
        for line in labels_content:
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            parsed_gt.append({'char': id_to_char[class_id], 'x': x_center})
        
        parsed_gt.sort(key=lambda x: x['x'])
        gt_text = "".join([item['char'] for item in parsed_gt])

        # 3. Predict với TrOCR
        pixel_values = processor(images=img_pil, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values, 
                max_new_tokens=8, 
                logits_processor=logits_processor
            )
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(" ", "").upper()

        plot_data.append({
            'image': img_pil,
            'gt': gt_text,
            'pred': pred_text
        })

    # --- TẠO DASHBOARD ---
    print("🎨 Đang render kết quả...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()

    for i, data in enumerate(plot_data):
        axes[i].imshow(data['image'])
        
        is_correct = (data['gt'] == data['pred'])
        title_color = 'green' if is_correct else 'red'
        status = "✅" if is_correct else "❌"
        
        title_text = f"GT: {data['gt']}\nPred: {data['pred']} {status}"
        axes[i].set_title(title_text, color=title_color, fontsize=12, weight='bold', pad=10)
        axes[i].axis('off')

    plt.suptitle("KIỂM TRA BẢN LĨNH TrOCR VỚI CAPTCHA TỰ SINH", fontsize=18, weight='heavy', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()