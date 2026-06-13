import matplotlib.pyplot as plt
import numpy as np
import cv2
# Nhập đội hình "Siêu anh hùng" chuẩn SOLID của cậu vào đây
from gen_image_label import CaptchaGenerator
from augment import CaptchaAugmenter
from marker import BBoxMarker

def run_pipeline_test(num_samples=10):
    print("🚀 Khởi động dây chuyền Test Pipeline RAM-to-RAM...")
    
    # Khởi tạo các module (Mỗi thằng chỉ lo đúng việc của nó)
    generator = CaptchaGenerator()
    augmenter = CaptchaAugmenter(bbox_format='yolo')
    marker = BBoxMarker()

    for i in range(num_samples):
        print(f"Đang xử lý ảnh {i+1}/{num_samples}...")
        
        # ---------------------------------------------------------
        # BƯỚC 1: SINH DATA GỐC (RAM)
        # ---------------------------------------------------------
        pil_img, original_labels_str = generator.generate_data()
        
        # Chuyển PIL Image (RGB) sang Numpy Array (RGB) để dùng cho Albumentations
        img_rgb = np.array(pil_img)
        
        # ---------------------------------------------------------
        # BƯỚC 2: BÓC TÁCH LABEL CHO AUGMENTER
        # ---------------------------------------------------------
        # Albumentations cần bboxes và class_labels tách riêng dạng list số thực
        bboxes = []
        class_labels = []
        for lbl in original_labels_str:
            parts = lbl.split()
            class_labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])

        # ---------------------------------------------------------
        # BƯỚC 3: ÁP DỤNG AUGMENTATION HỦY DIỆT
        # ---------------------------------------------------------
        aug_img_rgb, aug_bboxes, aug_labels = augmenter.augment(img_rgb, bboxes, class_labels)

        # ---------------------------------------------------------
        # BƯỚC 4: ĐÓNG GÓI LẠI LABEL CHO MARKER
        # ---------------------------------------------------------
        # Marker đang chờ một list chuỗi string chuẩn YOLO
        aug_labels_str = []
        for bbox, cls_id in zip(aug_bboxes, aug_labels):
            # ÉP KIỂU int(cls_id) NGAY TẠI ĐÂY
            aug_labels_str.append(f"{int(cls_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        # ---------------------------------------------------------
        # BƯỚC 5: ĐÓNG DẤU BOUNDING BOX
        # ---------------------------------------------------------
        # Chuyển hệ màu từ RGB về BGR để OpenCV vẽ và hiển thị cho chuẩn
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)

        # Gọi Marker vẽ lên 2 ảnh
        marked_original = marker.draw_on_ram(img_bgr, original_labels_str)
        marked_augmented = marker.draw_on_ram(aug_img_bgr, aug_labels_str)

        # ---------------------------------------------------------
        # BƯỚC 6: TRÌNH DIỄN GIAO DIỆN SÓNG ĐÔI
        # ---------------------------------------------------------
        # Phóng to x2 cho dễ soi
        h, w = marked_original.shape[:2]
        zoomed_orig = cv2.resize(marked_original, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        zoomed_aug = cv2.resize(marked_augmented, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)

        # Nối 2 ảnh lại với nhau theo chiều ngang
        divider = np.ones((h * 2, 5, 3), dtype=np.uint8) * 255
        combined_img = np.hstack((zoomed_orig, divider, zoomed_aug))

        # Đổi lại hệ màu từ BGR (của OpenCV) sang RGB để Matplotlib không bị ngược màu (xanh thành đỏ)
        combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

        # Hiển thị xịn xò
        plt.figure(figsize=(15, 5))
        plt.imshow(combined_rgb)
        plt.title(f"Test #{i+1}: Original vs Augmented (Tắt cửa sổ này để gen ảnh tiếp theo)")
        plt.axis('off')
        plt.show()
    print("✅ Hoàn thành 10 lượt test!")

if __name__ == "__main__":
    run_pipeline_test(10)