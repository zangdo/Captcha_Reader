import re
import cv2
import torch
import numpy as np
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.data.augment import LetterBox
def clean_text(text: str) -> str:
    """
    Làm sạch chuỗi sinh ra từ mô hình:
    1. Tư duy cắt kẹp chả: Chỉ lấy phần trước </s_captcha> đầu tiên (chống ảo giác sinh chữ thừa).
    2. Quét RegEx dọn sạch mọi special tokens (<s_captcha>, <pad>, <unk>,...).
    """
    if "</s_captcha>" in text:
        text = text.split("</s_captcha>")[0]
    
    # Quét dọn mọi tag bọc trong < >
    clean = re.sub(r'<[^>]+>', '', text).strip()
    
    return clean
def run_safe_inference(trainer, img_path):
    """Hút data trực tiếp qua model RAM, không chạm ổ cứng, không phá PyTorch Graph"""
    device = next(trainer.model.parameters()).device
    trainer.model.eval() # Đưa model về trạng thái suy luận
    
    img0 = cv2.imread(img_path)
    # 1. Tiền xử lý ảnh (Letterbox) y hệt quy trình của Ultralytics
    img = LetterBox(new_shape=trainer.args.imgsz, auto=False, stride=32)(image=img0)
    img = img.transpose((2, 0, 1))[::-1]  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0) # Bơm thêm batch_size = 1
    
    # 2. Suy luận cách ly (Không sinh đạo hàm)
    with torch.no_grad():
        preds = trainer.model(img_tensor)
        # Lọc nhiễu bằng Non-Maximum Suppression
        det = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)[0]
        
        # Ánh xạ tọa độ từ khung Tensor 320x320 về lại tỷ lệ ảnh gốc (220x60)
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            
    return det, img0