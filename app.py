import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, DonutProcessor
from utils import clean_text
# Lưu ý: Cần import thêm Donut nếu cậu có dùng, ở đây tớ mock tạm Donut vì chưa có tạ trong ảnh

# ==============================================================================
# 1. KHỞI TẠO TẠ (LOAD MODELS LÊN RAM 1 LẦN DUY NHẤT LÚC KHỞI ĐỘNG)
# ==============================================================================
print("⏳ Đang nạp tạ các đấu sĩ lên võ đài...")
device = torch.device("cpu") # Máy cá nhân cứ chạy CPU cho lành

# 🥊 Đấu sĩ 1: YOLO11x
# Đường dẫn dựa trên ảnh chụp của cậu
YOLO_WEIGHT_PATH = "./Weights/best_yolo11x.pt" 
yolo_model = YOLO(YOLO_WEIGHT_PATH)
# Hardcode VOCAB (Bộ chữ cái của cậu lúc train YOLO, nhớ sửa lại cho đúng nếu khác)
VOCAB = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
id_to_char = {idx: char for idx, char in enumerate(VOCAB)}

# 🥊 Đấu sĩ 2: TrOCR
TROCR_WEIGHT_PATH = "./Weights/best_trocr_model"
trocr_processor = TrOCRProcessor.from_pretrained(TROCR_WEIGHT_PATH)
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_WEIGHT_PATH).to(device)
trocr_model.eval()

# 🥊 Đấu sĩ 3: Donut 
DONUT_WEIGHT_PATH = "./Weights/best_donut_model"
print("🥊 Đang nạp Đấu sĩ Donut (Cấu hình đồng bộ)...")

donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
donut_model = VisionEncoderDecoderModel.from_pretrained(DONUT_WEIGHT_PATH)
    # Tắt xuất Cache của toàn bộ model
donut_model.config.use_cache = False

# Chắc cú thì tắt luôn cả trong cấu hình của thằng Não (BART Decoder)
donut_model.config.decoder.use_cache = False
# Ép kích thước ảnh của mô hình về chuẩn CAPTCHA (60x220)
donut_processor.image_processor.size = {"height": 128, "width": 384}
donut_processor.image_processor.do_align_long_axis = False

# Thêm token báo hiệu tác vụ giải CAPTCHA
donut_processor.tokenizer.add_tokens(["<s_captcha>", "</s_captcha>"])
donut_model.decoder.resize_token_embeddings(len(donut_processor.tokenizer))

# Khai báo token đặc biệt cho Model
donut_model.config.pad_token_id = donut_processor.tokenizer.pad_token_id
donut_model.config.decoder_start_token_id = donut_processor.tokenizer.convert_tokens_to_ids(["<s_captcha>"])[0]
captcha_eos_id = donut_processor.tokenizer.convert_tokens_to_ids(["</s_captcha>"])[0]
donut_model.config.eos_token_id = captcha_eos_id
donut_model.config.decoder.eos_token_id = captcha_eos_id
donut_model.to(device)
donut_model.eval()
# Thêm dòng này ngay sau khi add_special_tokens trong app.py
print(f"Token ID của <s_captcha>: {donut_processor.tokenizer.convert_tokens_to_ids('<s_captcha>')}")
print("✅ Donut đã đồng bộ cấu hình thành công!")

print("✅ Đã nạp tạ xong! Sẵn sàng chiến đấu.")
def run_yolo(img_pil: Image.Image):
    """
    Kế thừa tinh hoa từ BBoxMarker của Tú: Vẽ Box và gom text từ trái qua phải.
    Không cần Ground Truth.
    """
    # 1. Chạy YOLO nặn ra Bounding Boxes
    results = yolo_model(img_pil, verbose=False)[0]
    
    # 2. Xử lý ảnh: Chuyển PIL sang Numpy BGR để cv2 dễ vẽ
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    pred_boxes_for_string = []
    
    # 3. Duyệt qua từng Box YOLO phát hiện được
    for box in results.boxes:
        cls_id = int(box.cls[0])
        char_label = id_to_char.get(cls_id, "?")
        
        # Lấy tọa độ chuẩn hóa xywhn để vẽ theo phong cách của cậu
        nx, ny, nw, nh = box.xywhn[0].tolist() 
        
        # Tính toán tọa độ Pixel tuyệt đối
        box_w, box_h = int(nw * w), int(nh * h)
        x_center_px, y_center_px = int(nx * w), int(ny * h)
        x_min = int(x_center_px - (box_w / 2))
        y_min = int(y_center_px - (box_h / 2))
        x_max = x_min + box_w
        y_max = y_min + box_h
        
        # --- VẼ HỘP TRỰC TIẾP ---
        # Vẽ viền
        cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        # Vẽ nền xanh đặc cho label
        (text_w, text_h), _ = cv2.getTextSize(char_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x_min, max(0, y_min - text_h - 4)), (x_min + text_w + 2, y_min), (0, 255, 0), -1)
        # In chữ đen lên nền
        cv2.putText(img_bgr, char_label, (x_min + 1, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Lưu lại x_center để sắp xếp chữ từ trái qua phải
        pred_boxes_for_string.append((x_center_px, char_label))
        
    # 4. Gom chữ: Sắp xếp các ký tự theo tọa độ x_center tăng dần (trái -> phải)
    pred_boxes_for_string.sort(key=lambda item: item[0])
    final_text = "".join([item[1] for item in pred_boxes_for_string])
    
    # 5. Đóng gói trả về UI: Chuyển lại sang PIL
    drawn_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(drawn_img_rgb), final_text


def run_trocr(img_pil: Image.Image):
    """
    Nhận ảnh và đọc trực tiếp ra chuỗi bằng TrOCR
    """
    # TrOCRProcessor yêu cầu đầu vào RGB
    img_rgb = img_pil.convert("RGB")
    pixel_values = trocr_processor(img_rgb, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = trocr_model.generate(
            pixel_values,
            max_length=12,
            num_beams=4,
            early_stopping=True
        )
        
    predicted_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text


def run_donut(img_pil: Image.Image):
    """
    Nhận ảnh và đọc chữ bằng Donut (Sử dụng Prompt đặc biệt)
    """
    img_rgb = img_pil.convert("RGB")
    pixel_values = donut_processor(img_rgb, return_tensors="pt").pixel_values.to(device)
    
    # 💥 ĐẶC SẢN CỦA DONUT: Phải mớm Prompt <s_captcha> vào não nó trước
    task_prompt = "<s_captcha>"
    decoder_input_ids = donut_processor.tokenizer(
        task_prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids.to(device)
    
    with torch.no_grad():
        generated_ids = donut_model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=12,
            num_beams=4,
            early_stopping=True
        )
        
    # QUAN TRỌNG: Để skip_special_tokens=False để giữ lại thẻ </s_captcha> phục vụ cho việc cắt kẹp chả
    predicted_text = donut_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Cho qua máy lọc rác
    final_text = clean_text(predicted_text)
    
    return final_text

def process_captcha(input_img):
    if input_img is None:
        return None, "⚠️ Vui lòng paste/upload ảnh!", "⚠️ Trống", "⚠️ Trống"
    w, h = input_img.size
    new_h = int(h * (220 / w))
    input_img_rescaled = input_img.resize((220, new_h))
    yolo_img, yolo_text = run_yolo(input_img_rescaled)
    donut_text = run_donut(input_img_rescaled)
    trocr_text = run_trocr(input_img_rescaled)

    return yolo_img, yolo_text, donut_text, trocr_text

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN XẾP TẦNG (TỐI ƯU ẢNH NGANG)
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🏆 Đấu Trường CAPTCHA: YOLO vs Donut vs TrOCR</h1>")
    
    # --- KHU VỰC ĐẦU VÀO ---
    with gr.Group():
        gr.Markdown("### Đầu vào (Input)")
        # Khung ảnh đầu vào chiếm full chiều ngang, fix cứng chiều cao 200px để không bị lấn chiếm màn hình
        input_image = gr.Image(label="Stage: Paste hoặc Kéo thả ảnh vào đây", type="pil", height=200)
        run_btn = gr.Button("RUN PREDICT", variant="primary", size="lg")
            
    # Thay gr.Divider() bằng dòng Markdown này cho nó chuyên nghiệp:
    gr.Markdown("---")
    gr.Markdown("### Kết Quả Phân Tích")
    
    # --- HÀNG 1: YOLO (Cần không gian rộng nhất để vẽ Box) ---
    with gr.Group():
        gr.Markdown("<h3 style='color: #ff7f0e;'>1. YOLO11x Detection</h3>")
        with gr.Row():
            # Chia tỷ lệ 7:3 (Ảnh chiếm 7 phần, text chiếm 3 phần)
            yolo_output_img = gr.Image(label="Bản đồ Bounding Box", type="pil", height=250, interactive=False, scale=7)
            yolo_output_text = gr.Textbox(label="Kết quả đọc được", lines=2, scale=3)
            
    # --- HÀNG 2: DONUT ---
    with gr.Group():
        gr.Markdown("<h3 style='color: #2ca02c;'>2. Donut Document UI</h3>")
        donut_output_text = gr.Textbox(label="Kết quả đọc được", lines=1)
            
    # --- HÀNG 3: TROCR ---
    with gr.Group():
        gr.Markdown("<h3 style='color: #1f77b4;'>3. Microsoft TrOCR</h3>")
        trocr_output_text = gr.Textbox(label="Kết quả đọc được", lines=1)

    # Nối sự kiện nút bấm
    run_btn.click(
        fn=process_captcha,
        inputs=[input_image],
        outputs=[yolo_output_img, yolo_output_text, donut_output_text, trocr_output_text]
    )

if __name__ == "__main__":
    print("Đang khởi động Giao diện Đấu trường...")

    demo.launch(theme=gr.themes.Soft(), inbrowser=True)