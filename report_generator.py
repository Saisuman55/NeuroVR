import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile
import os

def generate_medical_report(pred_class, confidence, original_img_b64, mask_img_b64):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1))
    styles.add(ParagraphStyle(name='Heading', fontSize=18, spaceAfter=14, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubHeading', fontSize=14, spaceAfter=10, fontName='Helvetica-Bold'))
    
    elements = []
    
    # Header
    elements.append(Paragraph("BrainTumor AI - Clinical Diagnostics Report", styles['Heading']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Patient Info Table
    data = [
        ["Patient ID:", "ANON-94821", "Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Age/Sex:", "45/F", "Referring Dr:", "Dr. S. Samantaray"],
        ["Study:", "MRI Brain w/o Contrast", "AI Model:", "EfficientNet-B4 + U-Net"]
    ]
    t = Table(data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*inch))
    
    # Diagnosis
    elements.append(Paragraph("1. Primary Findings", styles['SubHeading']))
    diag_text = f"The Deep Learning Classification model analyzed the provided MRI sequences. Primary detection indicates a high probability of <b>{pred_class.upper()}</b>."
    elements.append(Paragraph(diag_text, styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    conf_text = f"Model Confidence Score: <b>{confidence*100:.2f}%</b>"
    elements.append(Paragraph(conf_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Clinical Notes
    clinical_notes = {
        "GLIOMA": "Gliomas are tumors that occur in the brain and spinal cord, arising from glial cells. They can be benign or malignant. Further histopathological evaluation and staging are recommended.",
        "MENINGIOMA": "Meningiomas arise from the meninges. The majority are slow-growing and benign, though they can cause symptoms via mass effect. Neurosurgical consultation is advised.",
        "PITUITARY": "Pituitary adenomas are typically benign tumors of the pituitary gland. Endocrine workup and visual field testing are recommended alongside surgical evaluation.",
        "NO TUMOR": "No significant tumor pathology detected by the AI model. This does not rule out other neurological conditions. Clinical correlation is necessary."
    }
    
    note = clinical_notes.get(pred_class.upper(), "Clinical correlation required.")
    elements.append(Paragraph("Clinical Considerations:", styles['Normal']))
    elements.append(Spacer(1, 0.05*inch))
    elements.append(Paragraph(f"<i>{note}</i>", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Images
    elements.append(Paragraph("2. Imaging & AI Segmentation", styles['SubHeading']))
    
    # We need to decode the base64 images and save them temporarily for ReportLab
    img_elements = []
    temp_files = []
    
    def add_image(b64_data, title):
        if not b64_data:
            return None
        try:
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
                
            img_data = base64.b64decode(b64_data)
            fd, path = tempfile.mkstemp(suffix='.png')
            with os.fdopen(fd, 'wb') as f:
                f.write(img_data)
            temp_files.append(path)
            
            rl_img = RLImage(path, width=2.5*inch, height=2.5*inch)
            return [Paragraph(title, styles['Center']), Spacer(1, 0.1*inch), rl_img]
        except Exception as e:
            print(f"Error processing image {title}: {e}")
            return None
            
    col1 = add_image(original_img_b64, "Original MRI")
    col2 = add_image(mask_img_b64, "AI Segmentation Overlay")
    
    if col1 and col2:
        img_table = Table([[col1, col2]], colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
        elements.append(img_table)
    
    elements.append(Spacer(1, 0.4*inch))
    
    # Footer
    elements.append(Paragraph("Disclaimer:", styles['SubHeading']))
    disclaimer = "This report is generated automatically by the BrainTumor AI clinical suite. It is intended to assist medical professionals and does not constitute a final clinical diagnosis. All findings must be verified by a board-certified radiologist or physician."
    elements.append(Paragraph(disclaimer, styles['Normal']))
    
    doc.build(elements)
    
    for path in temp_files:
        try:
            os.remove(path)
        except:
            pass
            
    buffer.seek(0)
    return buffer.read()
