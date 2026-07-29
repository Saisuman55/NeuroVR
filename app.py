import os
import json
import base64
import subprocess
from flask import Flask, request, jsonify, send_from_directory, send_file
import io
from report_generator import generate_medical_report

app = Flask(__name__, static_folder='stitch_frontend')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
def serve_html(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/metrics')
def get_metrics():
    metrics_path = os.path.join(app.static_folder, 'metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    # Save temporarily
    upload_dir = os.path.join('outputs', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, 'temp_inference.jpg')
    file.save(img_path)
    
    # Run inference script
    try:
        result = subprocess.run(
            ['python3', 'src/inference.py', '--image', img_path],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        
        # Parse output for class and confidence
        pred_class = "Unknown"
        confidence = 0.0
        probabilities = {}
        probs_list = []
        classes_list = []
        
        for line in output.split('\n'):
            if "Predicted class:" in line:
                parts = line.split('Predicted class:')[1].strip()
                pred_class = parts.split('(')[0].strip()
                try:
                    conf_str = parts.split('confidence:')[1].replace(')', '').strip()
                    confidence = float(conf_str)
                except:
                    pass
            elif "Probabilities:" in line:
                try:
                    import ast
                    probs_list = ast.literal_eval(line.split('Probabilities:')[1].strip())
                except:
                    pass
            elif "Classes:" in line:
                try:
                    import ast
                    classes_list = ast.literal_eval(line.split('Classes:')[1].strip())
                except:
                    pass
                    
        if probs_list and classes_list and len(probs_list) == len(classes_list):
            for c, p in zip(classes_list, probs_list):
                probabilities[c.upper()] = p
                    
        # Load each of the 5 generated images
        pred_dir = os.path.join('outputs', 'predictions')
        def _read_b64(filename):
            path = os.path.join(pred_dir, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            return ''

        img_original   = _read_b64('original.png')
        img_mask       = _read_b64('binary_mask.png')
        img_overlay    = _read_b64('green_overlay.png')
        img_contour    = _read_b64('contour.png')
        img_heatmap    = _read_b64('heatmap.png')
        # fallback: legacy combined figure still used by report generator
        img_b64        = _read_b64('inference_result.png')
                
        return jsonify({
            "class": pred_class.upper(),
            "confidence": confidence,
            "probabilities": probabilities,
            "image_b64": img_b64,
            "img_original": img_original,
            "img_mask": img_mask,
            "img_overlay": img_overlay,
            "img_contour": img_contour,
            "img_heatmap": img_heatmap,
            "raw_log": output
        })
        
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Inference failed: {e.stderr}"}), 500

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
        
    pred_class = data.get('class', 'Unknown')
    confidence = data.get('confidence', 0.0)
    orig_b64 = data.get('original_b64', '')
    mask_b64 = data.get('mask_b64', '')
    
    try:
        pdf_bytes = generate_medical_report(pred_class, confidence, orig_b64, mask_b64)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='BrainTumorAI_Clinical_Report.pdf'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
