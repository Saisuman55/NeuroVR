"""
BrainTumor AI — Gradio Interface for Hugging Face Spaces (ZeroGPU free tier).

Imports inference code directly (no subprocess) so @spaces.GPU GPU allocation
is shared with the model code running in the same process.
"""

import os
import sys

from PIL import Image

# ─── Patch 1: gradio_client bool-schema bug ──────────────────────────────────
try:
    import gradio_client.utils as _gc_utils
    _orig_gc = _gc_utils._json_schema_to_python_type

    def _patched_gc(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_gc(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_gc
    print("[patch1] gradio_client bool-schema bug patched ✔")
except Exception as _e:
    print(f"[patch1] skipped: {_e}")


# ─── Patch 2: starlette TemplateResponse API break ───────────────────────────
try:
    import starlette.templating as _st
    _orig_st = _st.Jinja2Templates.TemplateResponse

    def _patched_st(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = dict(args[1]) if len(args) > 1 else kwargs.pop("context", {})
            request = context.pop("request", None)
            if request is not None:
                return _orig_st(self, request, name, context, *args[2:], **kwargs)
        return _orig_st(self, *args, **kwargs)

    _st.Jinja2Templates.TemplateResponse = _patched_st
    print("[patch2] starlette TemplateResponse API break patched ✔")
except Exception as _e:
    print(f"[patch2] skipped: {_e}")


import spaces          # noqa: E402
import gradio as gr   # noqa: E402

# ─── Add src/ to path so we can import inference directly ─────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ─── Model weight download at startup ────────────────────────────────────────
import download_models  # noqa: F401 — triggers ensure_models() automatically

# ─── Paths ───────────────────────────────────────────────────────────────────
PRED_DIR   = os.path.join(_BASE_DIR, "outputs", "predictions")
UPLOAD_DIR = os.path.join(_BASE_DIR, "outputs", "uploads")
CONFIG     = os.path.join(_BASE_DIR, "config.yaml")


def _read_img(filename: str):
    path = os.path.join(PRED_DIR, filename)
    if os.path.exists(path):
        return Image.open(path).copy()
    return None


# ─── Core inference (runs inside the @spaces.GPU GPU context) ─────────────────
@spaces.GPU
def run_inference(image):
    """Direct import of src/inference.py — no subprocess, GPU context preserved."""
    if image is None:
        return [None] * 5 + ["⚠️ Please upload an MRI image first."]

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,   exist_ok=True)

    # Save uploaded PIL image to disk
    tmp_path = os.path.join(UPLOAD_DIR, "temp_inference.jpg")
    image.save(tmp_path, format="JPEG", quality=95)

    try:
        # Import and call inference directly in this process (GPU context is shared)
        from inference import run_inference as _infer
        _infer(tmp_path, CONFIG)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return [None] * 5 + [f"❌ Inference error:\n\n```\n{tb}\n```"]

    # ─── Parse classification output from saved files ─────────────────────────
    pred_class = "Unknown"
    confidence = 0.0
    result_txt = os.path.join(PRED_DIR, "inference_result.txt")
    if os.path.exists(result_txt):
        with open(result_txt) as f:
            for line in f:
                if "Predicted class:" in line:
                    parts = line.split("Predicted class:")[1].strip()
                    pred_class = parts.split("(")[0].strip().upper()
                    try:
                        confidence = float(
                            parts.split("confidence:")[1].replace(")", "").strip()
                        )
                    except Exception:
                        pass

    # Fallback: read stdout printed by inference (captured via print redirect)
    # The run_inference function prints: "Predicted class: glioma (confidence: 0.9932)"
    # We rely on the files saved to PRED_DIR for classification + segmentation results.

    if pred_class == "NOTUMOR":
        severity = "🟢 CLEAR — No tumor detected"
    elif confidence > 0.95:
        severity = "🔴 CRITICAL — High confidence tumor"
    elif confidence > 0.0:
        severity = "🟠 HIGH — Tumor likely, confirm with specialist"
    else:
        severity = "🔵 COMPLETE — See segmentation maps below"

    summary = (
        f"## 🧠 Diagnosis Result\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Primary Finding** | `{pred_class}` |\n"
        f"| **Confidence** | `{confidence * 100:.1f}%` |\n"
        f"| **Severity** | {severity} |\n"
        f"| **Model** | EfficientNet-B4 + U-Net/ResNet34 |\n\n"
        f"*⚠️ AI result only — not for clinical diagnosis.*"
    )

    return (
        _read_img("original.png"),
        _read_img("binary_mask.png"),
        _read_img("green_overlay.png"),
        _read_img("contour.png"),
        _read_img("heatmap.png"),
        summary,
    )


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
        "Upload a brain MRI scan · Get instant tumor classification & segmentation",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="📤 Upload MRI Scan",
                           sources=["upload"], height=300)
            run_btn = gr.Button("🔬 Run Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            result_text = gr.Markdown(
                "*Upload an MRI scan and click **Run Analysis** to see results.*"
            )

    gr.Markdown("---")
    gr.Markdown("### 📊 Segmentation Output Gallery")

    with gr.Row():
        out_original = gr.Image(label="Original MRI",        show_label=True, height=220)
        out_mask     = gr.Image(label="Binary Mask",         show_label=True, height=220)
        out_overlay  = gr.Image(label="Green Overlay",       show_label=True, height=220)
        out_contour  = gr.Image(label="Contour",             show_label=True, height=220)
        out_heatmap  = gr.Image(label="Probability Heatmap", show_label=True, height=220)

    run_btn.click(
        fn=run_inference,
        inputs=[inp],
        outputs=[out_original, out_mask, out_overlay, out_contour, out_heatmap, result_text],
        show_progress="full",
    )

    gr.Markdown(
        "---\n"
        "**Models**: EfficientNet-B4 (classification) · U-Net + ResNet34 (segmentation)  \n"
        "⚠️ *For educational purposes only — not a clinical diagnostic tool.*"
    )


# ─── Launch ──────────────────────────────────────────────────────────────────
demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
