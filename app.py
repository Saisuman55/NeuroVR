"""
BrainTumor AI — Gradio Interface for Hugging Face Spaces (free tier).

Runs the full inference pipeline (EfficientNet-B4 + U-Net) and displays
5 output panels: Original, Binary Mask, Green Overlay, Contour, Heatmap.
"""

import os
import subprocess
from PIL import Image

import gradio as gr

# ─── Model weight download at startup ────────────────────────────────────────
import download_models  # noqa: F401 — runs download logic on import


# ─── Paths ───────────────────────────────────────────────────────────────────
PRED_DIR   = os.path.join("outputs", "predictions")
UPLOAD_DIR = os.path.join("outputs", "uploads")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _read_img(filename: str) -> Image.Image | None:
    path = os.path.join(PRED_DIR, filename)
    if os.path.exists(path):
        return Image.open(path).copy()
    return None


# ─── Core inference function ─────────────────────────────────────────────────
def run_inference(image: Image.Image):
    """
    Accepts a PIL image from Gradio, runs the ML pipeline,
    returns 5 result images + a text summary.
    """
    if image is None:
        return [None] * 5 + ["⚠️ Please upload an MRI image first."]

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,   exist_ok=True)

    # Save uploaded image to temp file
    tmp_path = os.path.join(UPLOAD_DIR, "temp_inference.jpg")
    image.save(tmp_path, format="JPEG", quality=95)

    # Run inference script
    try:
        result = subprocess.run(
            ["python3", "src/inference.py", "--image", tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            return [None] * 5 + [f"❌ Inference failed:\n\n```\n{stderr}\n```"]

    except subprocess.TimeoutExpired:
        return [None] * 5 + ["⏱️ Inference timed out (>120s). Model may not be loaded yet."]
    except Exception as e:
        return [None] * 5 + [f"❌ Error: {e}"]

    # Parse classification output
    pred_class  = "Unknown"
    confidence  = 0.0

    for line in stdout.split("\n"):
        if "Predicted class:" in line:
            parts = line.split("Predicted class:")[1].strip()
            pred_class = parts.split("(")[0].strip().upper()
            try:
                confidence = float(parts.split("confidence:")[1].replace(")", "").strip())
            except Exception:
                pass

    # Build severity indicator
    if pred_class == "NOTUMOR":
        severity = "🟢 CLEAR — No tumor detected"
    elif confidence > 0.95:
        severity = "🔴 CRITICAL — High confidence tumor"
    else:
        severity = "🟠 HIGH — Tumor likely, confirm with specialist"

    summary = (
        f"## 🧠 Diagnosis Result\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Primary Finding** | `{pred_class}` |\n"
        f"| **Confidence** | `{confidence * 100:.1f}%` |\n"
        f"| **Severity** | {severity} |\n"
        f"| **Model** | EfficientNet-B4 + U-Net/ResNet34 |\n\n"
        f"*⚠️ This is an automated AI result. Not for clinical diagnosis.*"
    )

    # Load the 5 output images
    original = _read_img("original.png")
    mask     = _read_img("binary_mask.png")
    overlay  = _read_img("green_overlay.png")
    contour  = _read_img("contour.png")
    heatmap  = _read_img("heatmap.png")

    return original, mask, overlay, contour, heatmap, summary


# ─── Gradio UI ───────────────────────────────────────────────────────────────
CSS = """
#title    { text-align: center; }
#subtitle { text-align: center; color: #94a3b8; margin-bottom: 1rem; }
footer    { display: none !important; }
"""

with gr.Blocks(
    title="🧠 NeuroVR — BrainTumor AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
    css=CSS,
) as demo:

    gr.Markdown("# 🧠 NeuroVR — BrainTumor AI", elem_id="title")
    gr.Markdown(
        "Upload a brain MRI scan · Get instant tumor classification & pixel-level segmentation",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(
                type="pil",
                label="📤 Upload MRI Scan",
                sources=["upload"],
                height=300,
            )
            run_btn = gr.Button("🔬 Run Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            result_text = gr.Markdown(
                "*Upload an MRI scan and click **Run Analysis** to see results.*"
            )

    gr.Markdown("---")
    gr.Markdown("### 📊 Segmentation Output Gallery")

    with gr.Row():
        out_original = gr.Image(label="Original MRI",         show_label=True, height=220)
        out_mask     = gr.Image(label="Binary Mask",          show_label=True, height=220)
        out_overlay  = gr.Image(label="Green Overlay",        show_label=True, height=220)
        out_contour  = gr.Image(label="Contour",              show_label=True, height=220)
        out_heatmap  = gr.Image(label="Probability Heatmap",  show_label=True, height=220)

    run_btn.click(
        fn=run_inference,
        inputs=[inp],
        outputs=[out_original, out_mask, out_overlay, out_contour, out_heatmap, result_text],
        show_progress="full",
    )

    gr.Markdown(
        "---\n"
        "**Models**: EfficientNet-B4 (classification) · U-Net + ResNet34 (segmentation)  \n"
        "**Dataset**: Brain Tumor MRI (Kaggle) · LGG MRI Segmentation (Kaggle)  \n"
        "⚠️ *For educational purposes only — not a clinical diagnostic tool.*"
    )


# ─── Launch ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # HF Spaces sets PORT env var; locally defaults to 7860
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
