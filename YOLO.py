import sys
from types import ModuleType
import importlib.machinery
import torch
# 1. Tạo một Module rỗng giả lập torchaudio
mock_torchaudio = ModuleType('torchaudio')

# 2. Đắp thêm __spec__ giả để lừa importlib.util.find_spec
mock_torchaudio.__spec__ = importlib.machinery.ModuleSpec(
    name='torchaudio', 
    loader=None
)
# 3. Kính thưa Python, đây là module torchaudio "xịn"
sys.modules['torchaudio'] = mock_torchaudio
import numpy as np
import evaluate
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances
from ultralytics import YOLO
from augment import CaptchaAugmenter # Import class nhiễu của cậu vào
import wandb
import os
from config import CHARSET, VAL_IMG_DIR
from sample_test_YOLO import YoloVisualLogger # Import class vẽ box của Tú vào đây  
cer_metric = evaluate.load("cer")
# ==============================================================================
# 1. GHI ĐÈ DATASET: CHÈN AUGMENTER ON-THE-FLY
# ==============================================================================
def decode_yolo_txt_label(txt_path):
    """Đọc file nhãn gốc, sắp xếp theo tọa độ X từ trái qua phải để hoàn thiện chữ gốc"""
    if not os.path.exists(txt_path):
        return ""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    parsed_boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            parsed_boxes.append((x_center, class_id))
            
    # Sắp xếp ký tự từ trái qua phải theo tâm X
    parsed_boxes.sort(key=lambda item: item[0])
    return "".join([CHARSET[item[1]].upper() for item in parsed_boxes])


# ==============================================================================
# CUSTOM CALLBACK: CHỐT CHẶN TÍNH CER & EXACT MATCH CHO YOLO
# ==============================================================================
def on_fit_epoch_end(trainer):
    if trainer.epoch < 0: return
    
    # 💥 BƯỚC 1: LƯU LẠI TRẠNG THÁI GRADIENT CỦA EPOCH HIỆN TẠI
    prev_grad_state = torch.is_grad_enabled()
    
    try:
        # 2. ÉP KHU VỰC NÀY VÀO VÙNG CÁCH LY (Không sinh đạo hàm)
        with torch.no_grad():
            # Khởi tạo an toàn, không sợ YOLO làm loạn Global State nữa
            eval_model = YOLO(trainer.last)
            
            val_images = [os.path.join(VAL_IMG_DIR, f) for f in os.listdir(VAL_IMG_DIR) if f.endswith('.png')]
            p_clean_list, l_clean_list = [], []
            exact_matches = 0
            
            print(f"\n[Custom Metrics] 📝 Đang tính toán CER & Exact Match cho Epoch {trainer.epoch}...")
            
            for img_path in val_images:
                txt_path = img_path.replace("images", "labels").replace(".png", ".txt")
                gt_string = decode_yolo_txt_label(txt_path)
                if not gt_string: continue
                    
                # Gọi thẳng hàm predict bậc cao, không cần quan tâm NMS hay Scale Box nữa
                res = eval_model(img_path, verbose=False)[0]
                
                pred_boxes = []
                for box in res.boxes:
                    x_center = float(box.xywh[0][0])
                    class_id = int(box.cls[0])
                    pred_boxes.append((x_center, class_id))
                    
                pred_boxes.sort(key=lambda item: item[0])
                pred_string = "".join([CHARSET[item[1]].upper() for item in pred_boxes])
                
                p_clean_list.append(pred_string)
                l_clean_list.append(gt_string)
                if pred_string == gt_string: exact_matches += 1

            total_loss = sum(trainer.loss_items) if hasattr(trainer, 'loss_items') else 0
            current_lr = trainer.optimizer.param_groups[0]['lr']
            gnorm = getattr(trainer, 'grad_norm', 0)
            
            if len(l_clean_list) > 0:
                epoch_cer = cer_metric.compute(predictions=p_clean_list, references=l_clean_list)
                epoch_em = exact_matches / len(l_clean_list)
                map50 = trainer.metrics.get('metrics/mAP50(B)', 0.0) 
                map50_95 = trainer.metrics.get('metrics/mAP50-95(B)', 0.0)
                if wandb.run is not None:
                    wandb.log({
                        "metrics/captcha_cer": epoch_cer,
                        "metrics/captcha_exact_match": epoch_em,
                        "metrics/mAP50": map50,            # Bắn mAP50 lên
                        "metrics/mAP50_95": map50_95,      # Bắn mAP50-95 lên
                        "train/loss": total_loss,
                        "train/learning_rate": current_lr,
                        "train/grad_norm": gnorm
                    }, step=trainer.epoch)
                    
                print(f"[Custom Metrics] 🎯 Kết quả Epoch {trainer.epoch} -> CER: {epoch_cer:.4f} | EM: {epoch_em:.4f}")

    finally:
        # 3. 💥 BƯỚC CỨU MẠNG: Dọn dẹp chiến trường và hoàn trả nguyên vẹn trạng thái gốc cho PyTorch
        torch.set_grad_enabled(prev_grad_state)
        trainer.model.train()

class CustomCaptchaDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Chỉ kích hoạt bóp méo nếu đang ở chế độ Train (augment=True)
        self.my_augmenter = CaptchaAugmenter() if self.augment else None

    def get_image_and_label(self, index):
        """Hàm này móc thẳng vào ruột YOLO ngay sau khi nó đọc ảnh từ ổ cứng"""
        
        # Gọi hàm gốc của YOLO để bốc ảnh và tọa độ lên
        data = super().get_image_and_label(index)
        
        # Nếu đang train thì lôi ra hành hạ
        if self.my_augmenter:
            img_bgr = data['img'] 
            img_rgb = img_bgr[:, :, ::-1] 
            bboxes = data['instances'].bboxes.tolist() 
            classes = data['cls'].flatten().tolist()
            
            try:
                # 💥 Vã Augment On-the-fly của Tú vào đây!
                aug_img_rgb, aug_bboxes, aug_classes = self.my_augmenter.augment(
                    image=img_rgb, bboxes=bboxes, class_labels=classes
                )
                
                # 1. Cập nhật lại ảnh và nhãn Class
                data['img'] = aug_img_rgb[:, :, ::-1] 
                data['cls'] = np.array(aug_classes, dtype=np.float32).reshape(-1, 1)
                if len(aug_bboxes) == 0:
                    aug_bboxes_np = np.zeros((0, 4), dtype=np.float32)
                else:
                    aug_bboxes_np = np.array(aug_bboxes, dtype=np.float32)
                orig_segments = data['instances'].segments
                orig_keypoints = data['instances'].keypoints
                data['instances'] = Instances(
                    bboxes=aug_bboxes_np, 
                    segments=orig_segments, 
                    keypoints=orig_keypoints, 
                    bbox_format="xywh", 
                    normalized=True
                )
                
            except Exception as e:
                # Nếu lỗi thì in ra log để biết đường mò, không pass im lặng nữa
                print(f"⚠️ Bỏ qua Augment ở ảnh index {index} do lỗi: {e}")
                pass

        return data

# ==============================================================================
# 2. GHI ĐÈ TRAINER: ÉP YOLO DÙNG DATASET MỚI TẠO
# ==============================================================================
class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """Hàm này bị gọi khi YOLO chuẩn bị dữ liệu train/val"""
        
        # Bắt chước y hệt cấu hình hàm gốc của tác giả Ultralytics
        return CustomCaptchaDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train", # Chỉ bật augment lúc train
            hyp=self.args,
            rect=self.args.rect or not mode == "train",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max() if self.model else 32),
            pad=0.0 if mode == "train" else 0.5,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

# ==============================================================================
# 3. KÍCH HOẠT TRAINING BẰNG CUSTOM TRAINER
# ==============================================================================
if __name__ == "__main__":
    # Thay vì dùng lệnh `model = YOLO(...); model.train(...)` truyền thống,
    # Cậu gọi thẳng thằng CustomTrainer và ném cấu hình (overrides) cho nó!
    wandb.init(
        project="Captcha_YOLOv11x",      # Tên Project trên Dashboard
        name="YOLOv11x_A100_Run1",       # Tên phiên chạy (cậu tự đặt cho ngầu)
        config={                       # Lưu lại cấu hình để sau này dễ xem lại
            "architecture": "YOLOv11x",
            "imgsz": 320,
            "batch_size": 128,
            "epochs": 150
        }
    )
    visual_logger = YoloVisualLogger(val_dir=VAL_IMG_DIR, num_samples=10)
    trainer = CustomTrainer(overrides={
        "model": "yolo11x.pt",         # Load tạ gốc của YOLOv8n
        "data": "dataset.yaml",        # File config trỏ đến thư mục train/val
        "epochs": 150,
        "imgsz": 320,
        "batch": 128,
        "patience": 30,
        "project": "Captcha_YOLOv11x",
        
        # CHỐT AN TOÀN: Vẫn phải tắt đống augment phá game mặc định của YOLO
        "mosaic": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "mixup": 0.0
    })
    
    # Nhớ thêm cái hook vẽ CER/ExactMatch lên WandB anh em mình viết ban nãy vào đây
    trainer.add_callback("on_fit_epoch_end", on_fit_epoch_end) 
    trainer.add_callback("on_fit_epoch_end", visual_logger.on_fit_epoch_end) # Thêm callback vẽ box của Tú vào đây
    # Nổ máy! Lúc này Data chạy ngầm qua cái CaptchaAugmenter của cậu
    trainer.train()