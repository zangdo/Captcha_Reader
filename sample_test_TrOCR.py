from transformers import TrainerCallback
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import clean_text
import wandb
import os

class SampleTestCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, num_samples=10, output_dir="/content/drive/MyDrive/Captcha_Reader"):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_save(self, args, state, control, model, **kwargs):
        print(f"\n[Callback] 📸 Đang chụp ảnh nghiệm thu Epoch {state.epoch:.2f}...")
        
        model.eval()
        device = model.device
        
        dataset_len = len(self.eval_dataset)
        indices = random.sample(range(dataset_len), min(self.num_samples, dataset_len))
        
        # Chuẩn bị bảng WandB
        wandb_table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Status"])
        
        # Chuẩn bị khung Matplotlib
        fig, axes = plt.subplots(nrows=self.num_samples, ncols=1, figsize=(6, 2 * self.num_samples))
        if self.num_samples == 1:
            axes = [axes]
            
        correct_count = 0
        
        for i, idx in enumerate(indices):
            sample = self.eval_dataset[idx]
            
            # 1. Trích xuất Tensor ảnh và Nhãn thật
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            
            label_ids = sample['labels'].clone()
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            
            # Đổi skip_special_tokens=True vì TrOCR không cần giữ tag nào cả
            raw_true = self.processor.decode(label_ids, skip_special_tokens=True)
            true_label = clean_text(raw_true) 

            # 2. Ép model tự sinh chữ (Inference của TrOCR)
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_length=12,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    use_cache=True,
                    num_beams=4, # TrOCR chuộng Beam Search = 4 để tìm text mượt hơn
                    early_stopping=True, # Dừng sớm khi đã sinh đủ chữ
                    return_dict_in_generate=True,
                )
            
            # Giải mã nhãn dự đoán
            raw_pred = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)
            pred_label = clean_text(raw_pred)
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct_count += 1
            status = "✅" if is_correct else "❌"

            # ==========================================
            # 3. KHÔI PHỤC ẢNH ĐỂ VẼ (Denormalization)
            # ==========================================
            img_tensor = pixel_values.squeeze(0).cpu()
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) # TrOCR xài chuẩn Mean 0.5, Std 0.5 thay vì ImageNet!
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_numpy = img_tensor.permute(1, 2, 0).numpy()
            img_numpy = np.clip(img_numpy, 0, 1) 
            
            # 4. Ghi lên WandB Table
            wandb_table.add_data(wandb.Image(img_numpy), true_label, pred_label, status)
            
            # 5. Vẽ lên file Matplotlib
            ax = axes[i]
            ax.imshow(img_numpy)
            ax.axis("off")
            color = "green" if is_correct else "red"
            ax.set_title(f"GT: {true_label} | Pred: {pred_label}", color=color, fontsize=12, fontweight='bold')

        if wandb.run is not None:
            wandb.log({f"Evaluation_Epoch_{state.epoch:.2f}": wandb_table}, step=state.global_step)
            
        plt.tight_layout()
        img_path = os.path.join(self.output_dir, f"visual_test_epoch_{state.epoch:.2f}.png")
        plt.savefig(img_path)
        plt.close(fig)
        
        print(f"🎯 Tổng kết trực quan: Đúng {correct_count}/{self.num_samples}")
        print(f"💾 Đã lưu file ảnh báo cáo tại: {img_path}")
        print(f"🚀 Đã bắn dữ liệu lên bảng điều khiển WandB!")
        
        model.train()