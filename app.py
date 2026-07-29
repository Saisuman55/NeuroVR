"""
BrainTumor AI — Gradio Interface for Hugging Face Spaces (ZeroGPU free tier).
Features: PDF report, confidence bar chart, patient info fields, segmentation gallery.
"""

import os
import sys
import io
import contextlib
import datetime

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

import spaces
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── src/ on path ─────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ─── Model download at startup ────────────────────────────────────────────────
import download_models  # noqa: F401

# ─── Paths ───────────────────────────────────────────────────────────────────
PRED_DIR   = os.path.join(_BASE_DIR, "outputs", "predictions")
UPLOAD_DIR = os.path.join(_BASE_DIR, "outputs", "uploads")
CONFIG     = os.path.join(_BASE_DIR, "config.yaml")


def _read_img(filename: str):
    path = os.path.join(PRED_DIR, filename)
    return Image.open(path).copy() if os.path.exists(path) else None


def _make_bar_chart(class_names: list, probs: list, pred_class: str) -> Image.Image:
    """Generate a horizontal confidence bar chart as a PIL Image."""
    colors = []
    for name in class_names:
        if name.lower() == pred_class.lower():
            colors.append("#3b82f6")   # blue = predicted
        elif name.lower() == "notumor":
            colors.append("#22c55e")   # green = no tumor
        else:
            colors.append("#64748b")   # slate = other

    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    bars = ax.barh(
        [n.capitalize() for n in class_names],
        [p * 100 for p in probs],
        color=colors,
        height=0.5,
        edgecolor="none",
    )
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{p*100:.1f}%", va="center", ha="left",
            color="white", fontsize=8,
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence %", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#334155", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_pdf(
    patient_name: str,
    patient_age: str,
    scan_date: str,
    pred_class: str,
    confidence: float,
    class_names: list,
    probs: list,
) -> str | None:
    """Generate a PDF report and return its file path, or None on failure."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage,
        )

        report_dir = os.path.join(_BASE_DIR, "outputs", "reports")
        os.makedirs(report_dir, exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(report_dir, f"NeuroVR_Report_{ts}.pdf")

        doc  = SimpleDocTemplate(pdf_path, pagesize=A4,
                                  topMargin=2*cm, bottomMargin=2*cm,
                                  leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Title ───────────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#1e40af"),
            spaceAfter=6,
        )
        story.append(Paragraph("🧠 NeuroVR — BrainTumor AI Report", title_style))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#3b82f6")))
        story.append(Spacer(1, 0.4*cm))

        # ── Patient info ────────────────────────────────────────────────────
        info_data = [
            ["Patient Name", patient_name or "N/A"],
            ["Age",          patient_age  or "N/A"],
            ["Scan Date",    scan_date    or datetime.date.today().isoformat()],
            ["Report Date",  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ]
        info_table = Table(info_data, colWidths=[4*cm, 12*cm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eff6ff")),
            ("TEXTCOLOR",  (0, 0), (0, -1), colors.HexColor("#1e40af")),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*cm))

        # ── Diagnosis ───────────────────────────────────────────────────────
        story.append(Paragraph("Diagnosis Summary",
                               ParagraphStyle("H2", parent=styles["Heading2"],
                                              textColor=colors.HexColor("#1e40af"))))
        severity = (
            "CLEAR — No tumor detected" if pred_class.upper() == "NOTUMOR"
            else "CRITICAL — High confidence tumor" if confidence > 0.95
            else "HIGH — Tumor likely, confirm with specialist"
        )
        diag_data = [
            ["Field", "Value"],
            ["Primary Finding", pred_class.upper()],
            ["Confidence",      f"{confidence*100:.1f}%"],
            ["Severity",        severity],
            ["Model",           "EfficientNet-B4 + U-Net/ResNet34"],
        ]
        diag_table = Table(diag_data, colWidths=[4*cm, 12*cm])
        diag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 0.5*cm))

        # ── Class probabilities ─────────────────────────────────────────────
        story.append(Paragraph("Class Probabilities",
                               ParagraphStyle("H2", parent=styles["Heading2"],
                                              textColor=colors.HexColor("#1e40af"))))
        prob_data = [["Class", "Probability"]] + [
            [n.capitalize(), f"{p*100:.1f}%"]
            for n, p in zip(class_names, probs)
        ]
        prob_table = Table(prob_data, colWidths=[8*cm, 8*cm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("ALIGN",      (1, 1), (1, -1), "CENTER"),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.5*cm))

        # ── Segmentation images ─────────────────────────────────────────────
        img_files = {
            "Original MRI":        "original.png",
            "Binary Mask":         "binary_mask.png",
            "Green Overlay":       "green_overlay.png",
            "Contour":             "contour.png",
            "Probability Heatmap": "heatmap.png",
        }
        story.append(Paragraph("Segmentation Output Gallery",
                               ParagraphStyle("H2", parent=styles["Heading2"],
                                              textColor=colors.HexColor("#1e40af"))))
        img_row = []
        for label, fname in img_files.items():
            p = os.path.join(PRED_DIR, fname)
            if os.path.exists(p):
                img_row.append([
                    RLImage(p, width=3.2*cm, height=3.2*cm),
                    Paragraph(label, ParagraphStyle("Cap", fontSize=7,
                              textColor=colors.HexColor("#64748b"),
                              alignment=1)),
                ])
        # 2-column layout
        pairs = [img_row[i:i+2] for i in range(0, len(img_row), 2)]
        for pair in pairs:
            row_data = []
            for cell_stack in pair:
                row_data.extend(cell_stack)
            t = Table([row_data[:2], row_data[2:]] if len(row_data) == 4
                      else [row_data], colWidths=[3.5*cm]*min(len(pair)*2, 4))
            story.append(t)
            story.append(Spacer(1, 0.3*cm))

        # ── Disclaimer ─────────────────────────────────────────────────────
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#94a3b8")))
        story.append(Paragraph(
            "⚠️ This report is generated by an AI model for educational/research purposes only. "
            "It is NOT a substitute for professional medical diagnosis. "
            "Always consult a qualified medical professional.",
            ParagraphStyle("Disclaimer", fontSize=8,
                           textColor=colors.HexColor("#64748b"),
                           spaceAfter=0),
        ))

        doc.build(story)
        return pdf_path

    except Exception as e:
        print(f"[PDF] Failed: {e}")
        return None


# ─── Core inference ───────────────────────────────────────────────────────────
@spaces.GPU
def run_inference(patient_name, patient_age, scan_date, image):
    if image is None:
        empty = [None] * 5
        return (*empty, None, None, "⚠️ Please upload an MRI image first.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,   exist_ok=True)

    tmp_path = os.path.join(UPLOAD_DIR, "temp_inference.jpg")
    image.save(tmp_path, format="JPEG", quality=95)

    try:
        from inference import run_inference as _infer
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _infer(tmp_path, CONFIG)
        output = buf.getvalue()
        print(output)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        return (*([None]*5), None, None, f"❌ Inference error:\n\n```\n{tb}\n```")

    # Parse classification
    pred_class  = "Unknown"
    confidence  = 0.0
    class_names = []
    probs       = []

    for line in output.split("\n"):
        if "Overridden Predicted class:" in line:
            parts = line.split("Overridden Predicted class:")[1].strip()
            pred_class = parts.split("(")[0].strip().upper()
            try:
                confidence = float(parts.split("confidence:")[1].replace(")", "").strip())
            except Exception:
                pass
            break
        if "Predicted class:" in line and pred_class == "Unknown":
            parts = line.split("Predicted class:")[1].strip()
            pred_class = parts.split("(")[0].strip().upper()
            try:
                confidence = float(parts.split("confidence:")[1].replace(")", "").strip())
            except Exception:
                pass
        if line.startswith("Probabilities:"):
            import ast
            try:
                probs = ast.literal_eval(line.split("Probabilities:")[1].strip())
            except Exception:
                pass
        if line.startswith("Classes:"):
            import ast
            try:
                class_names = ast.literal_eval(line.split("Classes:")[1].strip())
            except Exception:
                pass

    # Build bar chart
    bar_chart = None
    if class_names and probs:
        bar_chart = _make_bar_chart(class_names, probs, pred_class)

    # Build PDF
    pdf_path = _make_pdf(
        patient_name, patient_age, scan_date,
        pred_class, confidence, class_names, probs,
    )

    if pred_class == "NOTUMOR":
        severity = "🟢 CLEAR — No tumor detected"
    elif confidence > 0.95:
        severity = "🔴 CRITICAL — High confidence tumor"
    elif confidence > 0.0:
        severity = "🟠 HIGH — Tumor likely, confirm with specialist"
    else:
        severity = "🔵 Analysis complete — see segmentation maps"

    prob_rows = ""
    for n, p in zip(class_names, probs):
        prob_rows += f"| {n.capitalize()} | `{p*100:.1f}%` |\n"

    summary = (
        f"## 🧠 Diagnosis Result\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Primary Finding** | `{pred_class}` |\n"
        f"| **Confidence** | `{confidence*100:.1f}%` |\n"
        f"| **Severity** | {severity} |\n"
        f"| **Model** | EfficientNet-B4 + U-Net/ResNet34 |\n\n"
        f"### Class Probabilities\n| Class | Probability |\n|---|---|\n"
        f"{prob_rows}"
        f"\n*⚠️ AI result only — not for clinical diagnosis.*"
    )

    return (
        _read_img("original.png"),
        _read_img("binary_mask.png"),
        _read_img("green_overlay.png"),
        _read_img("contour.png"),
        _read_img("heatmap.png"),
        bar_chart,
        pdf_path,
        summary,
    )


# ─── Gradio UI ───────────────────────────────────────────────────────────────
CSS = """
#title    { text-align: center; }
#subtitle { text-align: center; color: #94a3b8; margin-bottom: 1rem; }
#patient-box { background: #1e293b; border-radius: 12px; padding: 12px; }
footer    { display: none !important; }
"""

HF_SPACE_URL = "https://swaggersamantaray55-neurovr.hf.space"

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

    # ── Patient info ────────────────────────────────────────────────────────
    with gr.Group(elem_id="patient-box"):
        gr.Markdown("#### 👤 Patient Information")
        with gr.Row():
            patient_name = gr.Textbox(label="Patient Name", placeholder="e.g. John Doe",
                                      scale=2)
            patient_age  = gr.Textbox(label="Age", placeholder="e.g. 45", scale=1)
            scan_date    = gr.Textbox(
                label="Scan Date",
                value=datetime.date.today().isoformat(),
                scale=1,
            )

    # ── Upload + results ─────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            inp     = gr.Image(type="pil", label="📤 Upload MRI Scan",
                               sources=["upload"], height=280)
            run_btn = gr.Button("🔬 Run Analysis", variant="primary", size="lg")
            pdf_out = gr.File(label="📄 Download PDF Report", visible=True)

        with gr.Column(scale=2):
            result_text = gr.Markdown(
                "*Upload an MRI scan and click **Run Analysis** to see results.*"
            )
            bar_chart_out = gr.Image(label="📊 Confidence by Class",
                                     show_label=True, height=200)

    # ── Segmentation gallery ──────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📊 Segmentation Output Gallery")
    with gr.Row():
        out_original = gr.Image(label="Original MRI",        show_label=True, height=200)
        out_mask     = gr.Image(label="Binary Mask",         show_label=True, height=200)
        out_overlay  = gr.Image(label="Green Overlay",       show_label=True, height=200)
        out_contour  = gr.Image(label="Contour",             show_label=True, height=200)
        out_heatmap  = gr.Image(label="Probability Heatmap", show_label=True, height=200)

    run_btn.click(
        fn=run_inference,
        inputs=[patient_name, patient_age, scan_date, inp],
        outputs=[
            out_original, out_mask, out_overlay, out_contour, out_heatmap,
            bar_chart_out, pdf_out, result_text,
        ],
        show_progress="full",
    )

    # ── Embed snippet ─────────────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Accordion("🌐 Embed this Space in your website", open=False):
        gr.Markdown(
            f"Copy the code below and paste it into any HTML page:\n\n"
            f"```html\n"
            f'<iframe\n'
            f'  src="{HF_SPACE_URL}"\n'
            f'  width="100%"\n'
            f'  height="900"\n'
            f'  frameborder="0"\n'
            f'  allow="camera;microphone"\n'
            f'  style="border-radius:12px;box-shadow:0 4px 24px rgba(0,0,0,0.3)"\n'
            f'></iframe>\n'
            f"```\n\n"
            f"Or link directly: [{HF_SPACE_URL}]({HF_SPACE_URL})"
        )

    gr.Markdown(
        "---\n"
        "**Models**: EfficientNet-B4 (classification) · U-Net + ResNet34 (segmentation)  \n"
        "⚠️ *For educational purposes only — not a clinical diagnostic tool.*"
    )


demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
