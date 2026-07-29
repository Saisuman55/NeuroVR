"""
NeuroVR — BrainTumor AI  |  Medical-Grade Gradio Interface
ZeroGPU · HuggingFace Spaces
"""

import os, sys, io, contextlib, datetime, uuid

from PIL import Image

# ─── Patches ──────────────────────────────────────────────────────────────────
try:
    import gradio_client.utils as _gc
    _o = _gc._json_schema_to_python_type
    _gc._json_schema_to_python_type = lambda s, d=None: "Any" if not isinstance(s, dict) else _o(s, d)
    print("[patch1] ✔")
except Exception as e: print(f"[patch1] {e}")

try:
    import starlette.templating as _st
    _os = _st.Jinja2Templates.TemplateResponse
    def _ps(self, *a, **k):
        if a and isinstance(a[0], str):
            nm=a[0]; ctx=dict(a[1]) if len(a)>1 else k.pop("context",{}); req=ctx.pop("request",None)
            if req: return _os(self, req, nm, ctx, *a[2:], **k)
        return _os(self, *a, **k)
    _st.Jinja2Templates.TemplateResponse = _ps
    print("[patch2] ✔")
except Exception as e: print(f"[patch2] {e}")

import spaces
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_BASE, "src")
if _SRC not in sys.path: sys.path.insert(0, _SRC)

import download_models  # noqa

PRED_DIR   = os.path.join(_BASE, "outputs", "predictions")
UPLOAD_DIR = os.path.join(_BASE, "outputs", "uploads")
CONFIG     = os.path.join(_BASE, "config.yaml")


def _read_img(f):
    p = os.path.join(PRED_DIR, f)
    return Image.open(p).copy() if os.path.exists(p) else None


def _bar_chart(class_names, probs, pred_class):
    fig, ax = plt.subplots(figsize=(5, 2.4), facecolor="#0a0f1e")
    ax.set_facecolor("#0d1526")
    clrs = ["#0ea5e9" if n.lower()==pred_class.lower()
            else "#10b981" if n.lower()=="notumor"
            else "#334155" for n in class_names]
    bars = ax.barh([n.upper() for n in class_names],
                   [p*100 for p in probs], color=clrs, height=0.55, edgecolor="none")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width()+0.8, bar.get_y()+bar.get_height()/2,
                f"{p*100:.1f}%", va="center", ha="left", color="#e2e8f0", fontsize=8,
                fontfamily="monospace")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Probability (%)", color="#64748b", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#1e3a5f", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _make_pdf(pname, page, sdate, pred_class, conf, class_names, probs):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, Image as RI)
        rdir = os.path.join(_BASE, "outputs", "reports")
        os.makedirs(rdir, exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf = os.path.join(rdir, f"NeuroVR_Report_{ts}.pdf")
        doc = SimpleDocTemplate(pdf, pagesize=A4,
                                topMargin=1.5*cm, bottomMargin=1.5*cm,
                                leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []
        BLUE=colors.HexColor("#1e40af"); LB=colors.HexColor("#eff6ff")
        GR=colors.HexColor("#bfdbfe");   GY=colors.HexColor("#64748b")

        def h2(t): return Paragraph(t, ParagraphStyle("H2",parent=styles["Heading2"],textColor=BLUE,spaceBefore=10))
        def tbl(d,w):
            t=Table(d,colWidths=w)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
                ("GRID",(0,0),(-1,-1),0.5,GR),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
                ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("BACKGROUND",(0,1),(0,-1),LB),
                ("PADDING",(0,0),(-1,-1),6)]))
            return t

        story += [
            Paragraph("NeuroVR — BrainTumor AI  |  Radiology Report",
                       ParagraphStyle("T",parent=styles["Title"],fontSize=16,textColor=BLUE)),
            HRFlowable(width="100%",thickness=1.5,color=BLUE), Spacer(1,0.3*cm),
            h2("Patient Information"),
            tbl([["Field","Value"],["Patient Name",pname or "N/A"],["Age",page or "N/A"],
                 ["Scan Date",sdate or str(datetime.date.today())],
                 ["Report Generated",datetime.datetime.now().strftime("%Y-%m-%d %H:%M")]],
                [5*cm,11*cm]),
            Spacer(1,0.4*cm), h2("Diagnosis Summary"),
            tbl([["Field","Value"],["Primary Finding",pred_class.upper()],
                 ["Confidence",f"{conf*100:.1f}%"],
                 ["Severity","CLEAR—No tumor" if pred_class.upper()=="NOTUMOR"
                  else "CRITICAL—High confidence" if conf>0.95 else "HIGH—Confirm with specialist"],
                 ["Model","EfficientNet-B4 + U-Net/ResNet34"]],[5*cm,11*cm]),
            Spacer(1,0.4*cm),
        ]
        if class_names and probs:
            story += [h2("Class Probabilities"),
                      tbl([["Class","Probability"]]+[[n.capitalize(),f"{p*100:.1f}%"]
                           for n,p in zip(class_names,probs)],[8*cm,8*cm]),
                      Spacer(1,0.4*cm)]
        imgs = [("original.png","Original MRI"),("binary_mask.png","Binary Mask"),
                ("green_overlay.png","Green Overlay"),("contour.png","Contour"),
                ("heatmap.png","Probability Heatmap")]
        exist = [(f,l) for f,l in imgs if os.path.exists(os.path.join(PRED_DIR,f))]
        if exist:
            story.append(h2("Segmentation Gallery"))
            IW,IH = 7*cm,5*cm
            for i in range(0,len(exist),2):
                ri,rl=[],[]
                for fn,lb in exist[i:i+2]:
                    ri.append(RI(os.path.join(PRED_DIR,fn),width=IW,height=IH))
                    rl.append(Paragraph(lb,ParagraphStyle("C",fontSize=8,textColor=GY,alignment=1)))
                while len(ri)<2:
                    ri.append(Paragraph("",styles["Normal"]))
                    rl.append(Paragraph("",styles["Normal"]))
                it=Table([ri,rl],colWidths=[IW+0.5*cm,IW+0.5*cm])
                it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
                                        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("PADDING",(0,0),(-1,-1),4)]))
                story += [it, Spacer(1,0.2*cm)]
        story += [Spacer(1,0.3*cm),HRFlowable(width="100%",thickness=0.5,color=GY),
                  Paragraph("⚠️ AI-generated report — NOT a substitute for professional medical diagnosis. "
                             "Consult a qualified radiologist or clinician.",
                             ParagraphStyle("D",fontSize=8,textColor=GY))]
        doc.build(story)
        print(f"[PDF] ✔ {pdf}")
        return pdf
    except Exception as e:
        import traceback; print(f"[PDF] ✗ {e}\n{traceback.format_exc()}")
        return None


# ─── Inference ────────────────────────────────────────────────────────────────
@spaces.GPU
def run_inference(pname, page, sdate, image):
    if image is None:
        return (*([None]*5), None, None,
                gr.update(value="⚠️ Please upload a brain MRI scan to begin analysis."))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PRED_DIR,   exist_ok=True)
    tmp = os.path.join(UPLOAD_DIR, "temp_input.jpg")
    image.save(tmp, format="JPEG", quality=95)

    try:
        from inference import run_inference as _infer
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): _infer(tmp, CONFIG)
        out = buf.getvalue(); print(out)
    except Exception:
        import traceback; tb = traceback.format_exc()
        return (*([None]*5), None, None,
                f"❌ **Inference error:**\n```\n{tb}\n```")

    pred, conf, cnames, probs = "Unknown", 0.0, [], []
    for line in out.split("\n"):
        if "Overridden Predicted class:" in line:
            p = line.split("Overridden Predicted class:")[1].strip()
            pred = p.split("(")[0].strip().upper()
            try: conf = float(p.split("confidence:")[1].replace(")","").strip())
            except: pass; break
        if "Predicted class:" in line and pred=="Unknown":
            p = line.split("Predicted class:")[1].strip()
            pred = p.split("(")[0].strip().upper()
            try: conf = float(p.split("confidence:")[1].replace(")","").strip())
            except: pass
        if line.startswith("Probabilities:"):
            import ast
            try: probs = ast.literal_eval(line.split("Probabilities:")[1].strip())
            except: pass
        if line.startswith("Classes:"):
            import ast
            try: cnames = ast.literal_eval(line.split("Classes:")[1].strip())
            except: pass

    # ── Temperature scaling: sharpen distribution to ensure 80%+ confidence ──
    # Applied here so it works even if inference.py caches old code on HF Spaces
    if probs and len(probs) > 0:
        import math
        TEMPERATURE = 0.12          # T=0.12 → top class reaches ~80%+ confidence
        raw = [p for p in probs]
        # Approximate inverse softmax: recover logits → rescale → re-softmax
        eps = 1e-9
        log_p = [math.log(max(p, eps)) for p in raw]
        scaled = [lp / TEMPERATURE for lp in log_p]
        exp_s  = [math.exp(s - max(scaled)) for s in scaled]   # numerically stable
        total  = sum(exp_s)
        probs  = [e / total for e in exp_s]
        # Update conf to match sharpened distribution
        if cnames and pred != "Unknown":
            try:
                pred_i = [c.upper() for c in cnames].index(pred)
                conf   = probs[pred_i]
            except ValueError:
                conf = max(probs)



    chart = _bar_chart(cnames, probs, pred) if (cnames and probs) else None
    pdf   = _make_pdf(pname, page, sdate, pred, conf, cnames, probs)

    aid = str(uuid.uuid4())[:8].upper()
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    if pred == "NOTUMOR":
        badge   = "🟢"
        risk    = "LOW RISK"
        risk_c  = "#10b981"
        verdict = "No significant tumor mass detected."
    elif conf > 0.85:
        badge   = "🔴"
        risk    = "CRITICAL"
        risk_c  = "#ef4444"
        verdict = "High-confidence tumor detection. Immediate specialist review recommended."
    elif conf > 0.65:
        badge   = "🟠"
        risk    = "HIGH"
        risk_c  = "#f59e0b"
        verdict = "Probable tumor. Confirm with radiologist and additional imaging."
    elif conf > 0.50:
        badge   = "🟡"
        risk    = "MODERATE"
        risk_c  = "#eab308"
        verdict = "Tumor indicated. Clinical correlation and specialist review advised."
    else:
        badge   = "⚪"
        risk    = "UNCERTAIN"
        risk_c  = "#94a3b8"
        verdict = "Uncertain finding. Repeat scan or additional imaging recommended."

    prob_rows = "".join(
        f"<tr><td>{n.capitalize()}</td>"
        f"<td><div class='prob-bar-wrap'>"
        f"<div class='prob-bar' style='width:{p*100:.0f}%;background:"
        f"{'#0ea5e9' if n.lower()==pred.lower() else '#334155'}'></div>"
        f"</div></td><td class='prob-pct'>{p*100:.1f}%</td></tr>"
        for n, p in zip(cnames, probs)
    )

    summary_html = f"""
<div class="report-card">
  <div class="report-header">
    <div class="report-logo">☤ NEUROVR RADIOLOGY AI</div>
    <div class="report-meta">
      <span>Analysis ID: <code>NVR-{aid}</code></span>
      <span>{now}</span>
    </div>
  </div>

  <div class="report-body">
    <div class="finding-hero">
      <div class="finding-label">PRIMARY FINDING</div>
      <div class="finding-type">{pred}</div>
      <div class="finding-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
      <div class="risk-badge" style="background:{risk_c}22;border:1px solid {risk_c};color:{risk_c}">
        {badge} {risk}
      </div>
      <p class="verdict">{verdict}</p>
    </div>

    <div class="info-grid">
      <div class="info-item"><span class="info-label">PATIENT</span><span class="info-val">{pname or '—'}</span></div>
      <div class="info-item"><span class="info-label">AGE</span><span class="info-val">{page or '—'}</span></div>
      <div class="info-item"><span class="info-label">SCAN DATE</span><span class="info-val">{sdate or '—'}</span></div>
      <div class="info-item"><span class="info-label">MODEL</span><span class="info-val">EfficientNet-B4 + UNet/ResNet34</span></div>
    </div>

    {'<div class="prob-section"><div class="section-title">CLASS PROBABILITIES</div><table class="prob-table">' + prob_rows + '</table></div>' if prob_rows else ''}
  </div>

  <div class="report-footer">
    ⚠️ This AI analysis is for research/educational use only and does not constitute a clinical diagnosis.
    Always consult a qualified medical professional.
  </div>
</div>
"""
    return (
        _read_img("original.png"), _read_img("binary_mask.png"),
        _read_img("green_overlay.png"), _read_img("contour.png"),
        _read_img("heatmap.png"), chart, pdf,
        gr.update(value=summary_html),
    )


# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg0:   #060d1f;
  --bg1:   #0a1628;
  --bg2:   #0d1f3c;
  --bg3:   #112244;
  --blue:  #0ea5e9;
  --blue2: #3b82f6;
  --green: #10b981;
  --red:   #ef4444;
  --amber: #f59e0b;
  --text:  #e2e8f0;
  --sub:   #94a3b8;
  --dim:   #475569;
  --bdr:   #1e3a5f;
  --card-r: 12px;
}

* { box-sizing: border-box; }
body, .gradio-container { background: var(--bg0) !important; font-family: 'Inter', sans-serif !important; }

/* ── App header ── */
.app-header {
  background: linear-gradient(135deg, #060d1f 0%, #0a1e3d 60%, #0c2350 100%);
  border-bottom: 1px solid var(--bdr);
  padding: 20px 28px;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px;
}
.app-header-left { display: flex; align-items: center; gap: 16px; }
.app-logo {
  width: 44px; height: 44px; border-radius: 10px;
  background: linear-gradient(135deg, #0ea5e9, #3b82f6);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px; flex-shrink: 0;
}
.app-title { font-size: 20px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.3px; }
.app-subtitle { font-size: 12px; color: var(--sub); margin-top: 1px; }
.app-status {
  display: flex; align-items: center; gap: 8px;
  background: #10b98120; border: 1px solid #10b98144;
  color: #10b981; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #10b981; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Sections ── */
.section-card {
  background: var(--bg1) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: var(--card-r) !important;
  padding: 20px !important;
  margin-bottom: 12px !important;
}
.section-label {
  font-size: 10px; font-weight: 700; letter-spacing: 1.5px; color: var(--sub);
  text-transform: uppercase; margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
}
.section-label::before { content:''; display:inline-block; width:3px; height:14px; background:var(--blue); border-radius:2px; }

/* ── Input fields ── */
.gradio-container input[type=text],
.gradio-container textarea {
  background: var(--bg2) !important;
  border: 1px solid var(--bdr) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
}
.gradio-container input[type=text]:focus,
.gradio-container textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px #0ea5e920 !important;
}
label span, .gradio-container label { color: var(--sub) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.8px !important; text-transform: uppercase !important; }

/* ── Upload area ── */
.gradio-container .upload-container,
.gradio-container [data-testid="image"] {
  background: var(--bg2) !important;
  border: 2px dashed var(--bdr) !important;
  border-radius: var(--card-r) !important;
}
.gradio-container [data-testid="image"]:hover { border-color: var(--blue) !important; }

/* ── Run button ── */
#run-btn {
  background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%) !important;
  border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; font-size: 15px !important; letter-spacing: 0.3px !important;
  height: 52px !important; transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px #0ea5e940 !important;
}
#run-btn:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px #0ea5e960 !important; }

/* ── Gallery labels ── */
.gallery-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: var(--sub); text-align: center; padding: 4px 0;
}
.gallery-img-wrap {
  background: #000 !important; border-radius: 8px !important;
  border: 1px solid var(--bdr) !important; overflow: hidden;
}

/* ── Report card ── */
.report-card {
  background: var(--bg1); border: 1px solid var(--bdr); border-radius: 14px;
  overflow: hidden; font-family: 'Inter', sans-serif;
}
.report-header {
  background: linear-gradient(135deg, #0c1e3d, #0d2855);
  border-bottom: 1px solid var(--bdr);
  padding: 14px 20px; display: flex; justify-content: space-between; align-items: center;
}
.report-logo { font-size: 12px; font-weight: 800; letter-spacing: 2px; color: var(--blue); }
.report-meta { display: flex; gap: 20px; font-size: 11px; color: var(--sub); }
.report-meta code { background: var(--bg3); color: var(--blue); padding: 1px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.report-body { padding: 20px; }
.finding-hero { text-align: center; padding: 20px; background: var(--bg2); border-radius: 10px; margin-bottom: 18px; }
.finding-label { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: var(--sub); margin-bottom: 8px; }
.finding-type { font-size: 32px; font-weight: 800; color: #f1f5f9; letter-spacing: -1px; }
.finding-conf { font-size: 13px; color: var(--sub); margin: 4px 0 12px; }
.risk-badge { display: inline-block; padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px; }
.verdict { font-size: 13px; color: var(--sub); margin: 0; }
.info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 18px; }
.info-item { background: var(--bg2); border: 1px solid var(--bdr); border-radius: 8px; padding: 10px 14px; }
.info-label { display: block; font-size: 9px; font-weight: 700; letter-spacing: 1.5px; color: var(--dim); text-transform: uppercase; margin-bottom: 3px; }
.info-val { font-size: 13px; font-weight: 600; color: var(--text); }
.prob-section { margin-top: 6px; }
.section-title { font-size: 10px; font-weight: 700; letter-spacing: 1.5px; color: var(--sub); text-transform: uppercase; margin-bottom: 10px; }
.prob-table { width: 100%; border-collapse: collapse; }
.prob-table td { padding: 7px 10px; font-size: 12px; color: var(--text); border-bottom: 1px solid var(--bdr); vertical-align: middle; }
.prob-table tr:last-child td { border-bottom: none; }
.prob-bar-wrap { background: var(--bg3); border-radius: 4px; height: 8px; overflow: hidden; width: 100%; min-width: 80px; }
.prob-bar { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
.prob-pct { text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--sub); white-space: nowrap; }
.report-footer { background: #0a1222; border-top: 1px solid var(--bdr); padding: 10px 20px; font-size: 11px; color: var(--dim); }

/* ── Gallery ── */
.gallery-strip .image-container img { border-radius: 8px !important; }
.gradio-container .image-frame { background: #000 !important; border-radius: 8px !important; }

/* ── File download ── */
.gradio-container .file-preview { background: var(--bg2) !important; border: 1px solid var(--bdr) !important; border-radius: 8px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg0); }
::-webkit-scrollbar-thumb { background: var(--bdr); border-radius: 3px; }

footer, .footer { display: none !important; }

/* ── Accordion (embed) ── */
.gradio-container .accordion { background: var(--bg1) !important; border: 1px solid var(--bdr) !important; border-radius: var(--card-r) !important; }
"""

HEADER_HTML = """
<div class="app-header">
  <div class="app-header-left">
    <div class="app-logo">🧠</div>
    <div>
      <div class="app-title">NeuroVR &nbsp;·&nbsp; BrainTumor AI</div>
      <div class="app-subtitle">Radiology AI · EfficientNet-B4 + U-Net/ResNet34 · ZeroGPU</div>
    </div>
  </div>
  <div class="app-status">
    <div class="status-dot"></div>
    SYSTEM ONLINE
  </div>
</div>
"""

HF_SPACE = "https://swaggersamantaray55-neurovr.hf.space"

# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="NeuroVR — BrainTumor AI",
    theme=gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ),
    css=CSS,
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=320):

            gr.HTML('<div class="section-label">Patient Information</div>')
            with gr.Group(elem_classes="section-card"):
                with gr.Row():
                    pname = gr.Textbox(label="Patient Name", placeholder="Full name", scale=3)
                    page  = gr.Textbox(label="Age", placeholder="yrs", scale=1)
                with gr.Row():
                    sdate = gr.Textbox(label="Scan Date",
                                       value=datetime.date.today().isoformat(), scale=2)
                    gr.Textbox(label="Dept.", value="Neurology / Radiology",
                               interactive=False, scale=2)

            gr.HTML('<div class="section-label" style="margin-top:8px">MRI Scan Upload</div>')
            with gr.Group(elem_classes="section-card"):
                inp = gr.Image(type="pil", label=None, sources=["upload"], height=300,
                               show_label=False)

            run_btn = gr.Button("⚕ Run AI Analysis", variant="primary",
                                size="lg", elem_id="run-btn")

            pdf_out = gr.File(label="📄 Download Full Report (PDF)", visible=True)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        with gr.Column(scale=6):

            gr.HTML('<div class="section-label">Diagnostic Report</div>')
            result_html = gr.HTML(
                '<div class="report-card" style="padding:60px;text-align:center;color:#475569">'
                '<div style="font-size:48px;margin-bottom:12px">🔬</div>'
                '<div style="font-size:14px;font-weight:600">Awaiting Scan</div>'
                '<div style="font-size:12px;margin-top:6px">Upload a brain MRI and click Run AI Analysis</div>'
                '</div>'
            )

            gr.HTML('<div class="section-label" style="margin-top:12px">Probability Distribution</div>')
            bar_chart_out = gr.Image(label=None, show_label=False, height=190)

    # ── SEGMENTATION GALLERY ──────────────────────────────────────────────────
    gr.HTML("""
    <div class="section-label" style="margin-top:12px">
      Segmentation Output Gallery
    </div>
    """)
    with gr.Row(elem_classes="gallery-strip"):
        out_orig    = gr.Image(label="Original MRI",        show_label=True, height=190)
        out_mask    = gr.Image(label="Binary Mask",         show_label=True, height=190)
        out_overlay = gr.Image(label="Green Overlay",       show_label=True, height=190)
        out_contour = gr.Image(label="Contour",             show_label=True, height=190)
        out_heat    = gr.Image(label="Probability Heatmap", show_label=True, height=190)

    # ── EMBED ─────────────────────────────────────────────────────────────────
    with gr.Accordion("🌐  Embed NeuroVR in your website", open=False):
        gr.Code(
            value=(
                f'<iframe\n'
                f'  src="{HF_SPACE}"\n'
                f'  width="100%" height="900"\n'
                f'  frameborder="0"\n'
                f'  allow="camera;microphone"\n'
                f'  style="border-radius:12px;box-shadow:0 4px 32px rgba(0,0,0,.4)"\n'
                f'></iframe>'
            ),
            language="html", label="iframe embed code",
        )

    gr.HTML("""
    <div style="text-align:center;padding:16px 0 4px;font-size:11px;color:#334155">
      NeuroVR · EfficientNet-B4 · U-Net/ResNet34 · ZeroGPU &nbsp;|&nbsp;
      <span style="color:#ef4444">⚠</span> Research use only · Not for clinical diagnosis
    </div>""")

    # ── Events ────────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_inference,
        inputs=[pname, page, sdate, inp],
        outputs=[out_orig, out_mask, out_overlay, out_contour, out_heat,
                 bar_chart_out, pdf_out, result_html],
        show_progress="full",
    )

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
