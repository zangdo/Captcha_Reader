"""
Sinh ảnh minh họa kiến trúc cho báo cáo / slide.
Chạy: uv run visualize_architecture.py
Output: figures/arch_*.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Màu chủ đạo
C = {
    "input": "#E8F4FD",
    "process": "#FFF3E0",
    "model_yolo": "#FFE0B2",
    "model_donut": "#C8E6C9",
    "model_trocr": "#BBDEFB",
    "encoder": "#90CAF9",
    "decoder": "#A5D6A7",
    "attention": "#FFCC80",
    "ffn": "#CE93D8",
    "norm": "#F8BBD0",
    "output": "#E1BEE7",
    "arrow": "#455A64",
    "text": "#212121",
    "residual": "#78909C",
}


def _box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#37474F",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C["text"],
        weight=weight,
        zorder=3,
        wrap=True,
    )
    return x + w / 2, y, x + w / 2, y + h


def _arrow(ax, x1, y1, x2, y2, label=None):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.4,
        color=C["arrow"],
        connectionstyle="arc3,rad=0.0",
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, ha="center", fontsize=7, color=C["arrow"])


def _residual(ax, x_left, x_right, y, label="Skip"):
    ax.plot([x_left, x_right], [y, y], linestyle="--", color=C["residual"], linewidth=1.2, zorder=1)
    ax.text((x_left + x_right) / 2, y + 0.12, label, ha="center", fontsize=7, color=C["residual"])


def _setup_ax(ax, title: str):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    _setup_ax(ax, "Pipeline tổng thể — Captcha Reader VLM")

    # Hàng trên: luồng chính
    _box(ax, 0.2, 6.0, 1.4, 1.0, "config.py", C["process"], fontsize=9)
    _box(ax, 2.0, 6.0, 2.0, 1.0, "CaptchaGenerator\n+ Augmenter", C["process"], fontsize=8)
    _box(ax, 4.5, 6.0, 1.6, 1.0, "captcha_\ndataset", C["input"], fontsize=8)
    _box(ax, 6.5, 6.0, 2.4, 1.0, "Train scripts\n(YOLO/Donut/TrOCR)", C["process"], fontsize=8)
    _box(ax, 9.3, 6.0, 1.5, 1.0, "Weights/", C["output"], fontsize=9)

    _arrow(ax, 1.6, 6.5, 2.0, 6.5)
    _arrow(ax, 4.0, 6.5, 4.5, 6.5)
    _arrow(ax, 6.1, 6.5, 6.5, 6.5)
    _arrow(ax, 8.9, 6.5, 9.3, 6.5)

    # Hàng dưới: inference
    _box(ax, 0.2, 3.5, 1.4, 1.0, "Ảnh\nCAPTCHA", C["input"], fontsize=9)
    _box(ax, 2.3, 4.3, 1.5, 0.8, "YOLO11x", C["model_yolo"], fontsize=9)
    _box(ax, 2.3, 3.2, 1.5, 0.8, "Donut", C["model_donut"], fontsize=9)
    _box(ax, 2.3, 2.1, 1.5, 0.8, "TrOCR", C["model_trocr"], fontsize=9)
    _box(ax, 4.5, 3.0, 2.0, 1.2, "app.py\n(Gradio Demo)", C["output"], fontsize=9, bold=True)
    _box(ax, 7.0, 3.0, 2.5, 1.2, "Kết quả:\nBox + 3 chuỗi text", C["output"], fontsize=9)

    _arrow(ax, 1.6, 4.0, 2.3, 4.7)
    _arrow(ax, 1.6, 4.0, 2.3, 3.6)
    _arrow(ax, 1.6, 4.0, 2.3, 2.5)
    _arrow(ax, 3.8, 4.7, 4.5, 3.8)
    _arrow(ax, 3.8, 3.6, 4.5, 3.6)
    _arrow(ax, 3.8, 2.5, 4.5, 3.4)
    _arrow(ax, 6.5, 3.6, 7.0, 3.6)

    ax.text(5.0, 1.0, "Metrics: CER, Exact Match, mAP (YOLO) | Tracking: Weights & Biases", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_transformer_block():
    fig, ax = plt.subplots(figsize=(6, 8))
    _setup_ax(ax, "Khối Transformer Encoder (minh họa)")

    x, w, h = 2.0, 6.0, 0.9
    y = 8.5
    _box(ax, x, y, w, 0.6, "Input Embedding + Positional Encoding", C["input"], fontsize=9)

    layers = [
        ("Multi-Head Self-Attention", C["attention"]),
        ("Add & Layer Norm", C["norm"]),
        ("Feed-Forward Network (FFN)", C["ffn"]),
        ("Add & Layer Norm", C["norm"]),
    ]
    y = 7.2
    for i, (name, col) in enumerate(layers):
        _box(ax, x, y, w, h, name, col, fontsize=10)
        if i == 0:
            _residual(ax, x - 0.5, x + w + 0.5, y + h + 0.35, "Residual")
        if i == 2:
            _residual(ax, x - 0.5, x + w + 0.5, y + h + 0.35, "Residual")
        if i < len(layers) - 1:
            _arrow(ax, x + w / 2, y, x + w / 2, y - 0.25)
        y -= 1.35

    _arrow(ax, x + w / 2, 8.5, x + w / 2, 8.05)
    _box(ax, x, 1.8, w, 0.8, "Output Hidden States", C["output"], fontsize=10, bold=True)
    _arrow(ax, x + w / 2, 2.55, x + w / 2, 2.6)

    ax.text(5.0, 0.6, "× N layers (lặp khối trên)", ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_transformer_block.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_yolo():
    fig, ax = plt.subplots(figsize=(10, 5))
    _setup_ax(ax, "YOLO11x — Phát hiện từng ký tự CAPTCHA")

    _box(ax, 0.4, 4.0, 1.6, 1.0, "Ảnh CAPTCHA\n220×60", C["input"], fontsize=9)
    _box(ax, 2.5, 3.7, 2.0, 1.6, "Backbone\n(CSP + C2f)", C["encoder"], fontsize=9)
    _box(ax, 5.0, 3.7, 2.0, 1.6, "Neck\n(FPN / PAN)", C["process"], fontsize=9)
    _box(ax, 7.5, 3.7, 2.0, 1.6, "Detection Head\n32 classes", C["model_yolo"], fontsize=9, bold=True)

    _arrow(ax, 2.0, 4.5, 2.5, 4.5)
    _arrow(ax, 4.5, 4.5, 5.0, 4.5)
    _arrow(ax, 7.0, 4.5, 7.5, 4.5)

    _box(ax, 2.0, 1.5, 2.2, 1.0, "BBox + Class\nmỗi ký tự", C["output"], fontsize=9)
    _box(ax, 5.0, 1.5, 2.2, 1.0, "Sort theo\ntọa độ x", C["process"], fontsize=9)
    _box(ax, 7.8, 1.5, 1.8, 1.0, "Chuỗi\ntext", C["output"], fontsize=10, bold=True)

    _arrow(ax, 8.5, 3.7, 8.5, 2.5)
    _arrow(ax, 7.8, 2.0, 7.0, 2.0)
    _arrow(ax, 5.0, 2.0, 4.2, 2.0)
    _arrow(ax, 2.0, 2.0, 3.1, 2.0)

    ax.text(5.0, 0.5, "Train: CustomCaptchaDataset + CaptchaAugmenter on-the-fly", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_yolo.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut():
    fig, ax = plt.subplots(figsize=(11, 6))
    _setup_ax(ax, "Donut — Vision-Language Model cho CAPTCHA")

    _box(ax, 0.3, 4.2, 1.5, 1.0, "Ảnh\n128×384", C["input"], fontsize=9)

    # Encoder stack
    enc_x = 2.2
    _box(ax, enc_x, 5.0, 2.4, 0.7, "Patch Embedding", C["encoder"], fontsize=8)
    _box(ax, enc_x, 3.8, 2.4, 1.0, "Swin Transformer\nEncoder", C["encoder"], fontsize=9, bold=True)
    _box(ax, enc_x, 2.5, 2.4, 0.9, "Swin Block × L\n(Attn + FFN + Residual)", C["attention"], fontsize=8)

    _box(ax, 5.2, 3.8, 2.6, 1.6, "BART Decoder\n(Autoregressive)", C["decoder"], fontsize=9, bold=True)
    _box(ax, 5.4, 2.3, 2.2, 0.9, "Cross-Attention\n+ Self-Attention", C["attention"], fontsize=8)

    _box(ax, 8.3, 4.5, 1.5, 0.8, "Prompt\n<s_captcha>", C["process"], fontsize=8)
    _box(ax, 8.3, 3.2, 1.5, 0.9, "Token\nPredictor", C["model_donut"], fontsize=9)
    _box(ax, 8.1, 1.5, 1.9, 1.0, "TEXT\n</s_captcha>", C["output"], fontsize=10, bold=True)

    _arrow(ax, 1.8, 4.7, 2.2, 4.5)
    _arrow(ax, 3.4, 3.8, 3.4, 3.4)
    _arrow(ax, 4.6, 4.5, 5.2, 4.5)
    _arrow(ax, 7.8, 4.5, 8.3, 4.9)
    _arrow(ax, 7.8, 4.2, 8.3, 3.65)
    _arrow(ax, 9.05, 3.2, 9.05, 2.5)

    _residual(ax, enc_x - 0.3, enc_x + 2.7, 2.0, "Residual trong Swin Block")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_trocr():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("TrOCR — Chi tiết ViT Encoder + RoBERTa Decoder (trocr-base-printed)",
                 fontsize=14, fontweight="bold", pad=14)

    # --- Input ---
    _box(ax, 0.4, 4.5, 1.4, 1.0, "pixel_values\n(anh CAPTCHA)", C["input"], fontsize=8)

    # =========================================================================
    # COT TRAI: ViT ENCODER
    # =========================================================================
    ax.text(3.5, 9.3, "ViT Encoder (12 layers)", ha="center", fontsize=11, fontweight="bold", color="#1565C0")

    vit_x = 1.8
    vit_w = 3.4
    vit_steps = [
        (8.0, "Patch Embeddings\nConv2d (patch → vector)", C["encoder"], 8),
        (7.0, "+ Position Embeddings", C["process"], 8),
        (5.9, "ViT Encoder Layer  (x12)", C["encoder"], 9, True),
        (4.7, "  LayerNorm", C["norm"], 7),
        (4.0, "  Multi-Head Self-Attention", C["attention"], 7),
        (3.3, "  Residual #1", C["norm"], 7),
        (2.6, "  LayerNorm", C["norm"], 7),
        (1.9, "  MLP (FFN: Linear → GELU → Linear)", C["ffn"], 7),
        (1.2, "  Residual #2", C["norm"], 7),
        (0.3, "encoder_hidden_states\n(B, seq_img, 768)", C["output"], 8, True),
    ]

    prev_bottom = None
    for item in vit_steps:
        y, text, col = item[0], item[1], item[2]
        fs = item[3]
        bold = item[4] if len(item) > 4 else False
        h = 0.55 if not bold else 0.65
        if "x12" in text or "encoder_hidden" in text:
            h = 0.7
        _box(ax, vit_x, y, vit_w, h, text, col, fontsize=fs, bold=bold)
        if prev_bottom is not None:
            _arrow(ax, vit_x + vit_w / 2, prev_bottom, vit_x + vit_w / 2, y + h)
        prev_bottom = y

    _residual(ax, vit_x - 0.35, vit_x + vit_w + 0.35, 3.6, "skip")
    _residual(ax, vit_x - 0.35, vit_x + vit_w + 0.35, 1.55, "skip")

    # =========================================================================
    # COT PHAI: RoBERTa DECODER
    # =========================================================================
    ax.text(10.5, 9.3, "RoBERTa Decoder (6 layers)", ha="center", fontsize=11, fontweight="bold", color="#2E7D32")

    dec_x = 8.8
    dec_w = 3.4
    dec_steps = [
        (8.0, "Token Embeddings\n+ Position Embeddings", C["decoder"], 8),
        (7.0, "decoder_input_ids\n(cls_token start)", C["process"], 8),
        (5.9, "RoBERTa Decoder Layer  (x6)", C["decoder"], 9, True),
        (4.7, "  LayerNorm", C["norm"], 7),
        (4.0, "  Masked Self-Attention\n(causal — nhin qua khu)", C["attention"], 7),
        (3.3, "  Residual #1", C["norm"], 7),
        (2.6, "  LayerNorm", C["norm"], 7),
        (1.9, "  Cross-Attention\n(Query=text, K/V=encoder)", C["attention"], 7, True),
        (1.2, "  Residual #2 + FFN + Residual #3", C["ffn"], 7),
        (0.3, "LM Head (Linear)\n→ logits tung ky tu", C["output"], 8, True),
    ]

    prev_bottom = None
    for item in dec_steps:
        y, text, col = item[0], item[1], item[2]
        fs = item[3]
        bold = item[4] if len(item) > 4 else False
        h = 0.55
        if "x6" in text or "Cross-Attention" in text or "LM Head" in text:
            h = 0.7 if "Cross" not in text else 0.75
        if "x6" in text:
            h = 0.65
        _box(ax, dec_x, y, dec_w, h, text, col, fontsize=fs, bold=bold)
        if prev_bottom is not None:
            _arrow(ax, dec_x + dec_w / 2, prev_bottom, dec_x + dec_w / 2, y + h)
        prev_bottom = y

    # Cross-attention: encoder hidden states -> decoder K, V
    _arrow(ax, vit_x + vit_w, 0.65, dec_x, 2.25, label="K, V")

    # Input chi vao ViT Encoder
    _arrow(ax, 1.8, 5.0, vit_x, 8.35)

    # Generation
    _box(ax, 11.0, 4.6, 2.4, 1.0, "Beam Search\nnum_beams=4\nmax_length=12", C["process"], fontsize=8)
    _box(ax, 11.0, 3.2, 2.4, 1.0, "Chuoi ky tu\n(CAPTCHA text)", C["output"], fontsize=9, bold=True)
    _arrow(ax, dec_x + dec_w / 2, 0.3, 11.0, 5.1)
    _arrow(ax, 12.2, 4.6, 12.2, 4.2)

    ax.text(7.0, -0.5,
            "VisionEncoderDecoderModel | Train: Seq2SeqTrainer, predict_with_generate=True | Khong can prompt dac biet",
            ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_trocr.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_comparison():
    fig, ax = plt.subplots(figsize=(12, 5))
    _setup_ax(ax, "So sánh ba hướng tiếp cận đọc CAPTCHA")

    cols = [
        ("YOLO11x", C["model_yolo"], [
            "1. Detect bbox từng ký tự",
            "2. 32 class (CHARSET)",
            "3. Sort trái → phải",
            "4. Ghép chuỗi",
        ]),
        ("Donut VLM", C["model_donut"], [
            "1. Swin Encoder",
            "2. Prompt <s_captcha>",
            "3. BART Decoder sinh text",
            "4. clean_text()",
        ]),
        ("TrOCR", C["model_trocr"], [
            "1. ViT Encoder",
            "2. RoBERTa Decoder",
            "3. Beam search",
            "4. Text end-to-end",
        ]),
    ]

    for i, (title, color, lines) in enumerate(cols):
        x = 0.5 + i * 3.3
        _box(ax, x, 6.5, 3.0, 0.8, title, color, fontsize=11, bold=True)
        _box(ax, x, 1.5, 3.0, 4.7, "\n".join(lines), "#FAFAFA", fontsize=9)
        _arrow(ax, x + 1.5, 6.5, x + 1.5, 6.2)

    _box(ax, 0.5, 0.3, 9.5, 0.9, "Input chung: ảnh CAPTCHA 220×60  →  Metrics: CER, Exact Match", C["input"], fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "arch_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# =============================================================================
# DonutSwin — từ modeling_donut_swin.py
# Config donut-base: depths=[2,2,14,2], embed_dim=128, heads=[4,8,16,32]
# =============================================================================

def draw_donut_swin_overview():
    """DonutSwinModel: pixel_values -> Embeddings -> Encoder -> output"""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    _setup_ax(ax, "DonutSwinModel — Tổng quan (modeling_donut_swin.py)")

    _box(ax, 0.2, 4.2, 1.5, 1.0, "pixel_values\n(B,C,H,W)", C["input"], fontsize=8)
    _box(ax, 2.2, 3.8, 2.2, 1.8, "DonutSwinEmbeddings\n• PatchEmbeddings (Conv2d)\n• LayerNorm + Dropout", C["encoder"], fontsize=8)
    _box(ax, 5.0, 3.5, 3.2, 2.4, "DonutSwinEncoder\n4 × DonutSwinStage\n(depths: 2,2,14,2)", C["encoder"], fontsize=8, bold=True)
    _box(ax, 8.6, 4.0, 2.0, 1.4, "last_hidden_state\n(B, seq, C)", C["output"], fontsize=9, bold=True)

    _arrow(ax, 1.7, 4.7, 2.2, 4.7)
    _arrow(ax, 4.4, 4.7, 5.0, 4.7)
    _arrow(ax, 8.2, 4.7, 8.6, 4.7)

    # Class hierarchy footer
    classes = [
        "DonutSwinModel",
        "├── DonutSwinEmbeddings → DonutSwinPatchEmbeddings",
        "├── DonutSwinEncoder → DonutSwinStage × 4",
        "│     ├── DonutSwinLayer × depth",
        "│     └── DonutSwinPatchMerging (stage 0–2)",
        "└── (khong LayerNorm cuoi — ban Donut patch)",
    ]
    ax.text(0.3, 2.8, "\n".join(classes), fontsize=8, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_embeddings():
    fig, ax = plt.subplots(figsize=(11, 5))
    _setup_ax(ax, "DonutSwinEmbeddings + DonutSwinPatchEmbeddings")

    _box(ax, 0.3, 4.0, 1.6, 1.0, "pixel_values", C["input"], fontsize=9)
    _box(ax, 2.3, 4.5, 2.0, 0.7, "maybe_pad()", C["process"], fontsize=8)
    _box(ax, 2.3, 3.3, 2.0, 0.9, "Conv2d projection\n(kernel=stride=patch)", C["encoder"], fontsize=8)
    _box(ax, 4.8, 3.8, 1.8, 1.2, "flatten + transpose\n(B, H×W, C)", C["process"], fontsize=8)
    _box(ax, 7.1, 4.2, 1.5, 0.8, "LayerNorm", C["norm"], fontsize=9)
    _box(ax, 7.1, 3.2, 1.5, 0.8, "Dropout", C["process"], fontsize=9)
    _box(ax, 9.0, 3.7, 1.8, 1.2, "hidden_states\n+ dimensions", C["output"], fontsize=9, bold=True)

    _arrow(ax, 1.9, 4.5, 2.3, 4.85)
    _arrow(ax, 3.3, 4.5, 3.3, 4.2)
    _arrow(ax, 4.3, 3.8, 4.8, 4.4)
    _arrow(ax, 6.6, 4.4, 7.1, 4.6)
    _arrow(ax, 7.85, 4.2, 7.85, 4.0)
    _arrow(ax, 8.6, 3.7, 9.0, 4.3)

    ax.text(5.0, 1.8, "output_dimensions = (H/patch, W/patch)  →  truyền xuống Encoder", ha="center", fontsize=8, style="italic")
    ax.text(5.0, 0.8, "Tuỳ chọn: position_embeddings, mask_token (bool_masked_pos)", ha="center", fontsize=8, color="#616161")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_embeddings.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_encoder_stages():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    _setup_ax(ax, "DonutSwinEncoder — 4 Stage (dim tăng gấp đôi mỗi stage)")

    stages = [
        ("Stage 0", "dim=128", "2 blocks", "heads=4", True),
        ("Stage 1", "dim=256", "2 blocks", "heads=8", True),
        ("Stage 2", "dim=512", "14 blocks", "heads=16", True),
        ("Stage 3", "dim=1024", "2 blocks", "heads=32", False),
    ]
    x = 0.3
    for i, (name, dim, blocks, heads, has_merge) in enumerate(stages):
        _box(ax, x, 4.5, 2.4, 1.5, f"{name}\n{dim}\n{blocks}\n{heads}", C["encoder"], fontsize=8, bold=(i == 2))
        if has_merge:
            _box(ax, x + 0.5, 3.0, 1.4, 0.9, "Patch\nMerging", C["attention"], fontsize=8)
            _arrow(ax, x + 1.2, 4.5, x + 1.2, 3.9)
        if i < 3:
            _arrow(ax, x + 2.4, 5.0, x + 2.9, 5.0)
        x += 2.9

    ax.text(6.5, 2.0, "Patch Merging: gộp 2×2 patch → Linear(4C → 2C), giảm H×W một nửa", ha="center", fontsize=8, style="italic")
    ax.text(6.5, 1.0, "drop_path_rate tăng dần theo tổng số layer (DonutDropPath)", ha="center", fontsize=8, color="#616161")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_encoder_stages.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_stage():
    fig, ax = plt.subplots(figsize=(10, 5))
    _setup_ax(ax, "DonutSwinStage — Khối trong mỗi Stage")

    _box(ax, 0.3, 4.5, 1.5, 0.9, "Input\nhidden_states", C["input"], fontsize=8)
    _box(ax, 2.2, 5.2, 2.2, 0.75, "DonutSwinLayer #0\nshift_size = 0 (W-MSA)", C["encoder"], fontsize=7)
    _box(ax, 2.2, 4.2, 2.2, 0.75, "DonutSwinLayer #1\nshift_size = W/2 (SW-MSA)", C["encoder"], fontsize=7)
    _box(ax, 2.2, 3.0, 2.2, 0.75, "...\n× depth blocks", C["process"], fontsize=8)
    _box(ax, 5.0, 3.8, 2.4, 1.6, "DonutSwinPatchMerging\n• concat 2×2 patches\n• LayerNorm(4C)\n• Linear(4C→2C)", C["attention"], fontsize=8)
    _box(ax, 8.0, 4.0, 1.8, 1.2, "Output stage\n(dim ×2, H/2, W/2)", C["output"], fontsize=8, bold=True)

    _arrow(ax, 1.8, 4.9, 2.2, 5.55)
    _arrow(ax, 3.3, 5.2, 3.3, 4.95)
    _arrow(ax, 3.3, 4.2, 3.3, 3.75)
    _arrow(ax, 4.4, 3.5, 5.0, 4.2)
    _arrow(ax, 7.4, 4.6, 8.0, 4.6)

    ax.text(5.0, 1.5, "Layer chẵn: W-MSA  |  Layer lẻ: Shifted Window MSA (SW-MSA)", ha="center", fontsize=8, style="italic")
    ax.text(5.0, 0.6, "Stage cuối (Stage 3): downsample = None", ha="center", fontsize=8, color="#616161")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_stage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_layer():
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("DonutSwinLayer — Chi tiết forward()", fontsize=13, fontweight="bold", pad=10)

    x, w = 1.8, 6.4
    steps = [
        (10.5, "Input shortcut", C["input"], 8, False),
        (9.5, "LayerNorm (before)", C["norm"], 9, False),
        (8.5, "Reshape → (B,H,W,C) + pad", C["process"], 8, False),
        (7.5, "Cyclic Shift (nếu SW-MSA)", C["process"], 8, False),
        (6.5, "window_partition()", C["process"], 8, False),
        (5.4, "DonutSwinAttention\n(Window MHA + Rel. Pos. Bias)", C["attention"], 9, True),
        (4.3, "window_reverse() + reverse shift", C["process"], 8, False),
        (3.2, "DropPath + Residual #1\nshortcut + attn_out", C["norm"], 9, True),
        (2.1, "LayerNorm (after)", C["norm"], 9, False),
        (1.0, "FFN: Intermediate → Output\n(MLP ratio × dim)", C["ffn"], 9, False),
        (-0.1, "Residual #2\nhidden + ffn_out", C["output"], 9, True),
    ]

    prev_bottom = None
    for y, text, col, fs, bold in steps:
        h = 0.95 if "\n" in text else 0.75
        _box(ax, x, y, w, h, text, col, fontsize=fs, bold=bold)
        if prev_bottom is not None:
            _arrow(ax, x + w / 2, prev_bottom, x + w / 2, y + h)
        prev_bottom = y

    _residual(ax, 0.5, 8.2, 3.55, "Residual #1")
    _residual(ax, 0.5, 8.2, 0.35, "Residual #2")

    ax.text(5.0, -0.8, "get_attn_mask() tạo mask cho SW-MSA khi shift_size > 0", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_layer.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_attention():
    fig, ax = plt.subplots(figsize=(11, 6))
    _setup_ax(ax, "DonutSwinSelfAttention — Window Multi-Head Attention")

    _box(ax, 0.3, 4.5, 1.8, 1.0, "hidden_states\n(trong window)", C["input"], fontsize=8)
    _box(ax, 2.5, 5.0, 1.4, 0.7, "Linear Q", C["attention"], fontsize=8)
    _box(ax, 2.5, 4.0, 1.4, 0.7, "Linear K", C["attention"], fontsize=8)
    _box(ax, 2.5, 3.0, 1.4, 0.7, "Linear V", C["attention"], fontsize=8)
    _box(ax, 4.3, 4.0, 2.0, 1.4, "QK^T / sqrt(d)\n+ Relative Position Bias", C["attention"], fontsize=8, bold=True)
    _box(ax, 6.7, 4.2, 1.5, 1.0, "Softmax\n+ Dropout", C["process"], fontsize=8)
    _box(ax, 8.5, 4.2, 1.5, 1.0, "Attn x V\n-> context", C["output"], fontsize=9, bold=True)

    _arrow(ax, 2.1, 5.0, 2.5, 5.35)
    _arrow(ax, 2.1, 4.7, 2.5, 4.35)
    _arrow(ax, 2.1, 4.3, 2.5, 3.35)
    _arrow(ax, 3.9, 4.7, 4.3, 4.7)
    _arrow(ax, 6.3, 4.7, 6.7, 4.7)
    _arrow(ax, 8.2, 4.7, 8.5, 4.7)

    _box(ax, 4.0, 1.5, 5.5, 1.5,
         "PATCH (du an): Neu actual_seq_len != window^2\n"
         "-> interpolate relative_position_bias\n"
         "(bicubic) — ho tro anh CAPTCHA nho",
         "#FFF9C4", fontsize=8)

    ax.text(5.0, 0.5, "DonutSwinAttention = SelfAttention + SelfOutput (dense, dropout)", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_attention.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_donut_swin_patch_merging():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _setup_ax(ax, "DonutSwinPatchMerging — Giam resolution, tang channels")

    _box(ax, 0.3, 4.0, 1.6, 1.0, "Feature map\nH x W x C", C["input"], fontsize=8)
    _box(ax, 2.3, 4.3, 1.8, 0.7, "maybe_pad()", C["process"], fontsize=8)
    _box(ax, 2.3, 3.2, 1.8, 0.9, "Concat 2x2\npatches -> 4C", C["encoder"], fontsize=8)
    _box(ax, 4.5, 3.8, 1.5, 1.0, "LayerNorm\n(4xdim)", C["norm"], fontsize=8)
    _box(ax, 6.4, 3.8, 2.0, 1.0, "Linear reduction\n4C -> 2C", C["attention"], fontsize=8, bold=True)
    _box(ax, 8.8, 3.8, 1.5, 1.0, "H/2 x W/2\nx 2C", C["output"], fontsize=9, bold=True)

    _arrow(ax, 1.9, 4.5, 2.3, 4.65)
    _arrow(ax, 3.2, 4.3, 3.2, 4.1)
    _arrow(ax, 4.1, 3.7, 4.5, 4.3)
    _arrow(ax, 6.0, 4.3, 6.4, 4.3)
    _arrow(ax, 8.4, 4.3, 8.8, 4.3)

    ax.text(5.0, 1.5, "Sau merging: input_dimensions cap nhat (H/2, W/2) cho stage tiep theo", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_donut_swin_patch_merging.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


    return out


# =============================================================================
# Pipeline sinh data — gen_image_label.py + augment.py + generate_dataset.py
# =============================================================================

def draw_data_gen_pipeline():
    """CaptchaGenerator.generate_data() — luong chinh"""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Pipeline sinh CAPTCHA — CaptchaGenerator.generate_data()",
                 fontsize=14, fontweight="bold", pad=12)

    # --- Khoi tao ---
    ax.text(2.5, 10.2, "Khoi tao (config.py)", fontsize=10, fontweight="bold", color="#37474F")
    _box(ax, 0.3, 9.0, 2.2, 0.8, "Image.new RGB\n220x60, nen sang", C["input"], fontsize=7)
    _box(ax, 0.3, 7.9, 2.2, 0.8, "random 5-8 ky tu\ntu VOCAB (56)", C["process"], fontsize=7)
    _box(ax, 0.3, 6.8, 2.2, 0.8, "10 font TTF\ntai san trong RAM", C["process"], fontsize=7)

    # --- Vong lap tung ky tu ---
    ax.text(7.5, 10.2, "Vong lap tung ky tu", fontsize=10, fontweight="bold", color="#37474F")
    char_steps = [
        (9.0, "Chon font ngau nhien", C["process"]),
        (8.1, "Ve chu len canvas RGBA 100x100\nstroke_width 1-2, mau toi", C["encoder"]),
        (7.2, "Scale x,y: 0.7 - 1.2", C["process"]),
        (6.3, "_distort_single_char_cv2()\n(xem arch_char_distort.png)", C["attention"], True),
        (5.4, "Shear (xiên): -0.3 .. 0.3", C["process"]),
        (4.5, "Rotate: -35° .. +35°", C["process"]),
        (3.6, "Crop bbox + paste len anh nen\nvi tri y ngau nhien", C["encoder"]),
        (2.7, "Tinh bbox YOLO chuan hoa\nclass x y w h + margin 0.002", C["output"], True),
        (1.8, "step = random(-13..8)\ncurrent_x += cw + step", C["process"]),
    ]
    cx, cw = 4.0, 6.5
    prev = None
    for y, text, col, *bold in char_steps:
        b = bold[0] if bold else False
        h = 0.65 if "\n" not in text else 0.8
        _box(ax, cx, y, cw, h, text, col, fontsize=7, bold=b)
        if prev is not None:
            _arrow(ax, cx + cw / 2, prev, cx + cw / 2, y + h)
        prev = y

    _arrow(ax, 2.5, 7.2, 4.0, 9.35)
    ax.text(cx + cw / 2, 1.0, "Het anh (current_x + cw > 220) -> break", ha="center", fontsize=7, style="italic")

    # --- Output ---
    ax.text(12.5, 10.2, "Output", fontsize=10, fontweight="bold", color="#37474F")
    _box(ax, 11.2, 7.5, 3.2, 1.2, "base_image (PIL RGB)\n220 x 60", C["output"], fontsize=8, bold=True)
    _box(ax, 11.2, 5.8, 3.2, 1.4, "labels_content[]\nYOLO: class cx cy w h\n(OCR_labels, 56 class)", C["output"], fontsize=7, bold=True)
    _arrow(ax, 10.5, 3.1, 11.2, 6.5)
    _arrow(ax, 10.5, 3.1, 11.2, 6.0)

    # --- DatasetBuilder ---
    ax.text(12.5, 4.5, "DatasetBuilder.build()", fontsize=10, fontweight="bold", color="#37474F")
    _box(ax, 11.0, 2.8, 3.5, 1.0, "TRAIN (9000): luu anh phoi goc\naugment on-the-fly luc train", C["model_yolo"], fontsize=7)
    _box(ax, 11.0, 1.4, 3.5, 1.0, "VAL (1000): CaptchaAugmenter\ntruoc khi luu xuong dia", C["model_donut"], fontsize=7)
    _arrow(ax, 12.75, 5.8, 12.75, 3.8)
    _arrow(ax, 12.75, 5.8, 12.75, 2.4)

    fig.tight_layout()
    out = FIG_DIR / "arch_data_gen_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_char_distort_pipeline():
    """_distort_single_char_cv2() — méo từng ký tự"""
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_title("_distort_single_char_cv2() — Méo từng ký tự (gen_image_label.py)",
                 fontsize=13, fontweight="bold", pad=10)

    x, w = 1.5, 7.0
    steps = [
        (11.5, "Input: char RGBA (PIL)", C["input"], 8, False),
        (10.6, "Do kich thuoc chu -> tinh safe_max_kernel (max 3px)", C["process"], 7, False),
        (9.5, "Chuyen PIL -> OpenCV BGRA", C["process"], 8, False),
        (8.4, "Nhanh A: HOLLOW (p=30%)\npre-dilate -> outer-inner\nalpha threshold=10", C["attention"], 8, True),
        (7.2, "Nhanh B: MORPH (p=70%)\ndilate hoac erode\nkernel thich ung", C["attention"], 8, False),
        (6.1, "Morph them: kernel doc/ngang\ndilate hoac erode (p=70%)", C["ffn"], 7, False),
        (4.9, "Wave Distortion (luon ap dung)\nx' = x + sx*sin(y/λ+φx)\ny' = y + sy*cos(x/λ+φy)", C["attention"], 8, True),
        (3.7, "cv2.remap (INTER_CUBIC)\nborder transparent", C["process"], 8, False),
        (2.5, "Output: PIL RGBA da méo", C["output"], 9, True),
    ]
    prev = None
    for y, text, col, fs, bold in steps:
        h = 0.85 if "\n" in text else 0.6
        _box(ax, x, y, w, h, text, col, fontsize=fs, bold=bold)
        if prev is not None:
            _arrow(ax, x + w / 2, prev, x + w / 2, y + h)
        prev = y

    # Branch labels
    ax.text(0.5, 8.0, "30%", fontsize=8, color="#E65100", fontweight="bold")
    ax.text(0.5, 7.2, "70%", fontsize=8, color="#6A1B9A", fontweight="bold")

    ax.text(5.0, 1.2, "s in [3,6]  |  λ in [9,20]  |  HOLLOW_PROB=0.3  |  MORPH_PROB=0.7", ha="center", fontsize=7, style="italic")
    ax.text(5.0, 0.4, "Sau do: shear -> rotate -> crop -> paste vao anh CAPTCHA", ha="center", fontsize=7, color="#616161")

    fig.tight_layout()
    out = FIG_DIR / "arch_char_distort_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_augment_pipeline():
    """CaptchaAugmenter — pipeline lam nhieu + luong train/val"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("Pipeline lam nhieu — CaptchaAugmenter.augment() (augment.py)",
                 fontsize=14, fontweight="bold", pad=12)

    # Input
    _box(ax, 0.3, 5.0, 2.0, 1.4, "Input\nimage (numpy RGB)\nbboxes YOLO\nclass_labels", C["input"], fontsize=8)

    # Pipeline steps vertical in center
    px, pw = 3.2, 5.5
    aug_steps = [
        (8.5, "1. GridDistortion\nnum_steps=5, limit=0.2\np=0.3", C["attention"]),
        (7.2, "2. OneOf (p=0.7)", C["process"], True),
        (6.0, "   RandomNoiseLines (2-6 duong)", C["encoder"]),
        (5.2, "   StrikethroughLine (1-5 gach ngang)", C["encoder"]),
        (4.4, "   RandomRegionBlur (1-6 vung)", C["encoder"]),
        (3.2, "3. OneOf (p=0.6)", C["process"], True),
        (2.0, "   RandomQuadInvert (dao mau vung)", C["ffn"]),
        (1.2, "   RandomNoiseDots (35-40 cham)", C["ffn"]),
        (0.4, "   GaussNoise", C["ffn"]),
    ]
    prev = None
    for item in aug_steps:
        y, text, col = item[0], item[1], item[2]
        bold = item[3] if len(item) > 3 else False
        h = 0.55 if not text.startswith("   ") else 0.45
        if "OneOf" in text or "GridDistortion" in text:
            h = 0.65
        _box(ax, px, y, pw, h, text, col, fontsize=7, bold=bold)
        if prev is not None:
            _arrow(ax, px + pw / 2, prev, px + pw / 2, y + h)
        prev = y

    _box(ax, px, -0.5, pw, 0.65, "4. ColorJitter (p=0.5)\n5. Blur blur_limit=2 (p=0.2)", C["norm"], fontsize=7)
    _arrow(ax, px + pw / 2, 0.4, px + pw / 2, 0.15)

    # Bbox sync
    _box(ax, 9.2, 4.0, 4.2, 2.0,
         "BboxParams(format='yolo')\nAlbumentations dong bo\nbbox khi bien doi anh\n\nFallback: neu loi -> giu anh goc",
         "#FFF9C4", fontsize=8)

    # Output
    _box(ax, 9.2, 1.5, 4.2, 1.8, "Output\naug_image (numpy)\naug_bboxes\naug_class_labels", C["output"], fontsize=8, bold=True)

    _arrow(ax, 2.3, 5.7, px, 8.85)
    _arrow(ax, px + pw, 4.0, 9.2, 5.0)
    _arrow(ax, px + pw / 2, -0.5, 9.2, 2.4)

    # When used
    ax.text(12.0, 9.5, "Khi nao dung?", fontsize=10, fontweight="bold")
    uses = [
        ("VAL: generate_dataset.py\n_apply_val_augmentation()\nluu co dinh truoc khi ghi dia", C["model_donut"]),
        ("TRAIN: on-the-fly trong\nYOLO / Donut / TrOCR Dataset\nmoi epoch khac nhau", C["model_yolo"]),
        ("Debug: sample.py\nxem truoc truoc/sau", C["model_trocr"]),
    ]
    uy = 8.8
    for text, col in uses:
        _box(ax, 9.2, uy, 4.2, 0.85, text, col, fontsize=7)
        uy -= 1.05

    fig.tight_layout()
    out = FIG_DIR / "arch_augment_pipeline.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def draw_data_full_overview():
    """Tong quan end-to-end: config -> gen -> augment -> luu -> train"""
    fig, ax = plt.subplots(figsize=(15, 5))
    _setup_ax(ax, "Tong quan End-to-End: Sinh data + Lam nhieu + Luu + Train")

    boxes = [
        (0.2, 3.8, 1.3, 1.0, "config.py", C["process"]),
        (1.8, 3.8, 1.8, 1.0, "CaptchaGenerator\ngenerate_data()", C["encoder"]),
        (3.9, 4.5, 1.6, 0.7, "TRAIN\nanh phoi", C["model_yolo"]),
        (3.9, 3.0, 1.6, 0.7, "VAL\n+ Augment", C["model_donut"]),
        (5.8, 3.8, 1.6, 1.0, "captcha_\ndataset/", C["input"]),
        (7.7, 3.8, 1.8, 1.0, "build_yolo_\nlabels()", C["process"]),
        (9.8, 4.3, 1.4, 0.7, "YOLO\naugment OT", C["model_yolo"]),
        (9.8, 3.3, 1.4, 0.7, "Donut\naugment OT", C["model_donut"]),
        (9.8, 2.3, 1.4, 0.7, "TrOCR\naugment OT", C["model_trocr"]),
        (11.5, 3.8, 1.5, 1.0, "Train\n3 model", C["output"]),
    ]
    for x, y, w, h, t, c in boxes:
        _box(ax, x, y, w, h, t, c, fontsize=7)

    arrows = [
        (1.5, 4.3, 1.8, 4.3), (3.6, 4.3, 3.9, 4.85), (3.6, 4.3, 3.9, 3.35),
        (5.5, 4.85, 5.8, 4.5), (5.5, 3.35, 5.8, 4.1),
        (7.4, 4.3, 7.7, 4.3), (9.5, 4.3, 9.8, 4.65), (9.5, 4.3, 9.8, 3.65),
        (9.5, 4.3, 9.8, 2.65), (11.2, 4.3, 11.5, 4.3),
    ]
    for a in arrows:
        _arrow(ax, *a)

    ax.text(7.5, 1.8, "OT = on-the-fly (CaptchaAugmenter)  |  OCR_labels (56 cls) -> labels/ (32 cls YOLO)", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = FIG_DIR / "arch_data_full_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    tasks = [
        draw_pipeline,
        draw_transformer_block,
        draw_yolo,
        draw_donut,
        draw_trocr,
        draw_comparison,
        draw_donut_swin_overview,
        draw_donut_swin_embeddings,
        draw_donut_swin_encoder_stages,
        draw_donut_swin_stage,
        draw_donut_swin_layer,
        draw_donut_swin_attention,
        draw_donut_swin_patch_merging,
        draw_data_gen_pipeline,
        draw_char_distort_pipeline,
        draw_augment_pipeline,
        draw_data_full_overview,
    ]
    print("Dang tao anh kien truc...")
    for fn in tasks:
        path = fn()
        print(f"  OK: {path}")
    print(f"\nHoan tat! Xem trong: {FIG_DIR}")


if __name__ == "__main__":
    main()
