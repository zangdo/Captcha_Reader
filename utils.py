import re
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