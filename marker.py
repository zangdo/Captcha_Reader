import cv2
import numpy as np
from config import *

class BBoxMarker:
    """Class chịu trách nhiệm duy nhất: Vẽ Bounding Box lên ảnh từ dữ liệu trong RAM"""
    
    def __init__(self):
        # Tạo mapping ngược từ ID (0, 1, 2...) ra Ký tự (2, 3, A, B...)
        self.id_to_char = {idx: char for idx, char in enumerate(VOCAB)}

    def draw_on_ram(self, image_np: np.ndarray, yolo_labels: list) -> np.ndarray:
        """
        Nhận ảnh Numpy và List nhãn trực tiếp từ RAM, không đụng chạm ổ cứng.
        """
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

            # Lấy tên ký tự thật
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
# MAIN TEST SCRIPT: KẾT NỐI TRỰC TIẾP TỪ LÕI SINH DATA TRÊN RAM
# ==============================================================================
if __name__ == "__main__":
    from gen_image_label import CaptchaGenerator
    
    # Khởi tạo 2 module độc lập
    generator = CaptchaGenerator()
    marker = BBoxMarker()
    
    print("Đang test quy trình: Sinh Data (RAM) -> Vẽ BBox (RAM) -> Hiển thị...")
    
    # Sinh 3 ảnh liên tục để test mà không tốn 1 byte ổ cứng nào
    for i in range(3):
        # 1. Sinh Data thẳng trên RAM
        pil_img, labels = generator.generate_data()
        
        # 2. Convert PIL Image (RGB) sang Numpy (BGR) cho OpenCV
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 3. Quăng thẳng vào Marker để vẽ
        result_img = marker.draw_on_ram(cv2_img, labels)
        
        # Phóng to ảnh lên x2 cho dễ soi
        h, w = result_img.shape[:2]
        zoomed_img = cv2.resize(result_img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        
        # Bấm phím bất kỳ để xem ảnh tiếp theo, ESC để thoát
        cv2.imshow(f"Test RAM-to-RAM #{i+1}", zoomed_img)
        key = cv2.waitKey(0)
        if key == 27: 
            break
            
    cv2.destroyAllWindows()