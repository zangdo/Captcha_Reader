import os
import random
import cv2
import numpy as np
import torch
import wandb
from ultralytics import YOLO
from config import CHARSET, VOCAB

# ==============================================================================
# 1. CLASS VẼ BOX CỦA TÚ (Giữ nguyên tinh hoa, chỉ sửa tên biến xíu cho khớp)
# ==============================================================================
class BBoxMarker:
    """Class chịu trách nhiệm duy nhất: Vẽ Bounding Box lên ảnh từ dữ liệu trong RAM"""
    def __init__(self):
        # Tạo mapping ngược từ ID ra Ký tự
        self.id_to_char = {idx: char for idx, char in enumerate(VOCAB)}

    def draw_on_ram(self, image_np: np.ndarray, yolo_labels: list) -> np.ndarray:
        """Nhận ảnh Numpy (BGR) và List nhãn trực tiếp từ RAM"""
        img = image_np.copy()
        h, w = img.shape[:2]

        for label_str in yolo_labels:
            parts = label_str.strip().split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, norm_w, norm_h = map(float, parts[1:])

            # Giải mã định dạng YOLO (0-1) về tọa độ Pixel tuyệt đối
            box_w, box_h = int(norm_w * w), int(norm_h * h)
            x_min = int((x_center * w) - (box_w / 2))
            y_min = int((y_center * h) - (box_h / 2))
            x_max = x_min + box_w
            y_max = y_min + box_h

            char_label = self.id_to_char.get(class_id, "?")

            # 1. Vẽ khung bao quanh chữ
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            
            # 2. Vẽ nền đặc bên trên
            (text_w, text_h), _ = cv2.getTextSize(char_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x_min, max(0, y_min - text_h - 4)), (x_min + text_w + 2, y_min), (0, 255, 0), -1)
            
            # 3. In chữ màu đen đè lên nền
            cv2.putText(img, char_label, (x_min + 1, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return img

# ==============================================================================
# 2. HOOK TRỰC QUAN HÓA (Phiên bản YOLOv8)
# ==============================================================================
class YoloVisualLogger:
    def __init__(self, val_dir, num_samples=10):
        self.val_dir = val_dir
        self.num_samples = num_samples
        self.marker = BBoxMarker() # Gọi bộ vẽ của Tú

    def decode_gt_string(self, txt_path):
        """Hàm đọc file txt để lấy chuỗi Ground Truth thật"""
        if not os.path.exists(txt_path): return ""
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        parsed_boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parsed_boxes.append((float(parts[1]), int(parts[0])))
        parsed_boxes.sort(key=lambda item: item[0])
        return "".join([CHARSET[item[1]].upper() for item in parsed_boxes])

    def on_fit_epoch_end(self, trainer):
        if trainer.epoch < 0: return
        
        print(f"\n[Visual Callback] 📸 Đang chụp {self.num_samples} ảnh nghiệm thu đẩy lên WandB cho Epoch {trainer.epoch}...")
        
        all_images = [os.path.join(self.val_dir, f) for f in os.listdir(self.val_dir) if f.endswith('.png')]
        if not all_images: return
        sample_paths = random.sample(all_images, min(self.num_samples, len(all_images)))
        
        wandb_table = wandb.Table(columns=["Image (w/ Bbox)", "Ground Truth", "Prediction", "Status"])
        correct_count = 0
        
        # 💥 KHÓA BẢO VỆ GRADIENT CHO VISUAL LOGGER
        prev_grad_state = torch.is_grad_enabled()
        
        try:
            with torch.no_grad():
                eval_model = YOLO(trainer.last)
                
                for img_path in sample_paths:
                    txt_path = img_path.replace("images", "labels").replace(".png", ".txt")
                    true_label = self.decode_gt_string(txt_path)
                    
                    res = eval_model(img_path, verbose=False)[0]
                    
                    pseudo_yolo_labels = []
                    pred_boxes_for_string = []
                    
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        nx, ny, nw, nh = box.xywhn[0].tolist() 
                        pseudo_yolo_labels.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
                        
                        x_center = float(box.xywh[0][0])
                        pred_boxes_for_string.append((x_center, cls_id))
                        
                    pred_boxes_for_string.sort(key=lambda item: item[0])
                    pred_label = "".join([CHARSET[item[1]].upper() for item in pred_boxes_for_string])
                    
                    is_correct = (pred_label == true_label)
                    if is_correct: correct_count += 1
                    status = "✅" if is_correct else "❌"

                    img_bgr = res.orig_img.copy() 
                    marked_img_bgr = self.marker.draw_on_ram(img_bgr, pseudo_yolo_labels)
                    marked_img_rgb = cv2.cvtColor(marked_img_bgr, cv2.COLOR_BGR2RGB)

                    wandb_table.add_data(wandb.Image(marked_img_rgb), true_label, pred_label, status)

                if wandb.run is not None:
                    wandb.log({f"Visual_Inspection/Epoch_{trainer.epoch}": wandb_table}, step=trainer.epoch)
                
                print(f"🎯 Trực quan hóa BBox: Đúng {correct_count}/{self.num_samples}. Đã đẩy lên mây!")
                
        finally:
            # 💥 TRẢ LẠI HIỆN TRƯỜNG
            torch.set_grad_enabled(prev_grad_state)
            trainer.model.train()