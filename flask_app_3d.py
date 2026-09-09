"""
NeuroVR 3D — Flask Backend
3D MRI Brain Tumor Segmentation + AR/VR Visualization System

Runs on port 7861 (separate from legacy Gradio app on 7860).

API Routes:
  GET  /                         → Three.js frontend SPA
  GET  /static/<path>            → Frontend assets
  POST /api/upload               → Upload NIfTI files
  POST /api/upload_dicom         → Upload DICOM series directory
  POST /api/analyze/<session_id> → Run 3D AI pipeline
  GET  /api/status/<session_id>  → Poll analysis progress
  GET  /api/results/<session_id> → Measurements JSON
  GET  /api/mesh/<sid>/<type>    → Serve GLB mesh files
  GET  /api/demo                 → Run demo mode pipeline
  GET  /api/health               → System status
  DELETE /api/session/<sid>      → Clean up session files
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

# ── Suppress MPS before ANY torch import ──────────────────────────────────────
# PyTorch probes Metal/MPS at import time even when device='cpu'.
# On macOS Python 3.9 this causes [mutex.cc] lock blocking in Flask threads.
# Setting these env vars prevents the Metal framework from initializing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
# Disable MPS entirely — inference will run on CPU (sufficient for demo)
os.environ["DISABLE_MPS"] = "1"

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUTS_DIR = BASE_DIR / "outputs"
DEMO_DATA_DIR = BASE_DIR / "demo_data"
CONFIG_PATH = BASE_DIR / "config.yaml"

sys.path.insert(0, str(BASE_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalars and booleans JSON-serializable."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.bool_):   return bool(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating):return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
try:
    # Flask 2.2 and earlier
    app.json_encoder = _NumpyEncoder
except AttributeError:
    # Flask 2.3+: use json_provider_class
    import flask.json.provider as _fp
    import dataclasses as _dc
    class _NumpyProvider(_fp.DefaultJSONProvider):
        @staticmethod
        def default(obj):
            try:
                import numpy as np
                if isinstance(obj, np.bool_):    return bool(obj)
                if isinstance(obj, np.integer):  return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray):  return obj.tolist()
            except ImportError:
                pass
            return _fp.DefaultJSONProvider.default(obj)
    app.json_provider_class = _NumpyProvider
    app.json = _NumpyProvider(app)
CORS(app)

# Session state: {session_id: {"status": ..., "progress": ..., "error": ...}}
_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()

MESH_TYPES = {"brain", "tumor_whole", "tumor_core", "tumor_enhancing"}

# In-memory cache for 2D slice serving (keyed by session_id)
_vol_cache: Dict[str, Dict] = {}



# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_config() -> Dict:
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    except Exception as exc:
        print(f"[Flask] Config load failed: {exc}. Using defaults.")
        return {}


CONFIG = _load_config()


def _cfg(key_path: str, default=None):
    """Dot-notation config accessor."""
    keys = key_path.split(".")
    val = CONFIG
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is None:
            return default
    return val


# ──────────────────────────────────────────────────────────────────────────────
# Session helpers
# ──────────────────────────────────────────────────────────────────────────────

def _session_dir(session_id: str) -> Path:
    return OUTPUTS_DIR / session_id


def _set_status(session_id: str, status: str, progress: int = 0,
                message: str = "", error: str = "") -> None:
    with _sessions_lock:
        _sessions[session_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
        }


def _get_status(session_id: str) -> Optional[Dict]:
    with _sessions_lock:
        return _sessions.get(session_id)


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Serving frontend
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Health
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    import torch
    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif os.environ.get("DISABLE_MPS") != "1":
        # Only probe MPS when not in Flask-thread-safe CPU mode
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "Apple MPS"

    return jsonify({
        "status": "online",
        "device": device_info,
        "model": "MONAI BraTS MRI Segmentation",
        "version": "3.0.0",
        "disclaimer": "Research prototype — not for clinical use",
    })


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Upload NIfTI files
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_nifti():
    """Accept 1–4 NIfTI files and save them to a new session directory.

    Form fields expected:
      - t1, t1ce, t2, flair: individual NIfTI file uploads
      OR
      - auto_detect: directory with files (handled client-side by sending all files)

    Returns session_id and validation status.
    """
    session_id = str(uuid.uuid4())
    sess_dir = _session_dir(session_id)
    uploads_dir = sess_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["t1", "t1ce", "t2", "flair"]
    saved: Dict[str, str] = {}
    errors: list = []

    for modality in modalities:
        if modality in request.files:
            f = request.files[modality]
            if f.filename:
                fname = f.filename
                if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                    errors.append(f"{modality}: invalid extension (expected .nii or .nii.gz)")
                    continue
                dest = uploads_dir / f"{modality}.nii.gz"
                f.save(str(dest))
                # Validate
                try:
                    from preprocessing.nifti_loader import validate_nifti_file
                    valid, msg = validate_nifti_file(str(dest))
                    if valid:
                        saved[modality] = str(dest)
                    else:
                        errors.append(f"{modality}: {msg}")
                        dest.unlink(missing_ok=True)
                except Exception as exc:
                    errors.append(f"{modality} validation error: {exc}")

    if errors:
        return jsonify({"error": "; ".join(errors), "session_id": session_id}), 400

    if not saved:
        return jsonify({"error": "No valid NIfTI files received."}), 400

    # Store modality paths in session metadata
    meta_path = sess_dir / "modalities.json"
    with open(meta_path, "w") as fout:
        json.dump(saved, fout)

    missing = [m for m in modalities if m not in saved]
    _set_status(session_id, "uploaded", 0, f"Uploaded {len(saved)}/4 modalities")

    return jsonify({
        "session_id": session_id,
        "uploaded_modalities": list(saved.keys()),
        "missing_modalities": missing,
        "ready": len(missing) == 0,
    })


@app.route("/api/upload_dicom", methods=["POST"])
def upload_dicom():
    """Accept DICOM files for multiple modalities and convert to NIfTI.

    Form fields: t1_dir, t1ce_dir, t2_dir, flair_dir — each a zip or folder.
    For simplicity: accepts individual DICOM files per modality field.
    """
    session_id = str(uuid.uuid4())
    sess_dir = _session_dir(session_id)
    uploads_dir = sess_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["t1", "t1ce", "t2", "flair"]
    dicom_dirs: Dict[str, str] = {}
    errors: list = []

    for modality in modalities:
        files = request.files.getlist(modality)
        if files:
            mod_dir = uploads_dir / f"dicom_{modality}"
            mod_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                if f.filename:
                    f.save(str(mod_dir / f.filename))
            dicom_dirs[modality] = str(mod_dir)

    if not dicom_dirs:
        return jsonify({"error": "No DICOM files received."}), 400

    # Convert DICOM → NIfTI
    try:
        from preprocessing.dicom_loader import convert_dicom_modalities
        nifti_paths = convert_dicom_modalities(dicom_dirs, str(uploads_dir))
    except Exception as exc:
        return jsonify({"error": f"DICOM conversion failed: {exc}"}), 500

    meta_path = sess_dir / "modalities.json"
    with open(meta_path, "w") as fout:
        json.dump(nifti_paths, fout)

    missing = [m for m in modalities if m not in nifti_paths]
    _set_status(session_id, "uploaded", 0, f"Converted {len(nifti_paths)}/4 DICOM series")

    return jsonify({
        "session_id": session_id,
        "converted_modalities": list(nifti_paths.keys()),
        "missing_modalities": missing,
        "ready": len(missing) == 0,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Run 3D analysis (async)
# ──────────────────────────────────────────────────────────────────────────────

def _run_pipeline_async(session_id: str) -> None:
    """Background thread running the full 3D pipeline."""
    sess_dir = _session_dir(session_id)

    try:
        # ── Load modality paths ────────────────────────────────────────────────
        _set_status(session_id, "running", 5, "Loading modality paths...")
        meta_path = sess_dir / "modalities.json"
        with open(meta_path) as f:
            modality_paths = json.load(f)

        # ── Load NIfTI volumes ─────────────────────────────────────────────────
        _set_status(session_id, "running", 10, "Loading MRI volumes...")
        from preprocessing.nifti_loader import load_brats_case
        modality_volumes = load_brats_case(modality_paths)

        # ── Run or Load 3D segmentation ────────────────────────────────────────
        gt_masks_path = uploads_dir / "ground_truth_masks.npz"
        if gt_masks_path.exists():
            _set_status(session_id, "running", 20, "Loading ground-truth masks...")
            print("[Flask] Found ground-truth masks (NPZ) — bypassing segmentation.")
            npz = np.load(gt_masks_path)
            seg_result = {
                "tc_mask": npz["tc_mask"],
                "wt_mask": npz["wt_mask"],
                "et_mask": npz["et_mask"],
                "voxel_spacing": tuple(npz["voxel_spacing"]),
                "affine": npz["affine"],
                "inference_time_s": 0.0,
                "method": "ground_truth_masks",
            }
        else:
            _set_status(session_id, "running", 20, "Running 3D segmentation...")
            from inference.segmentation_numpy import run_numpy_segmentation
            seg_result = run_numpy_segmentation(modality_volumes)

        _set_status(session_id, "running", 65, "Post-processing segmentation masks...")

        # ── Post-process masks ─────────────────────────────────────────────────
        from reconstruction.mask_processing import postprocess_masks
        tc_clean, wt_clean, et_clean = postprocess_masks(
            seg_result["tc_mask"],
            seg_result["wt_mask"],
            seg_result["et_mask"],
            min_component_size=_cfg("reconstruction.minimum_component_size", 100),
        )

        _set_status(session_id, "running", 70, "Computing physical measurements...")

        # ── Measurements ──────────────────────────────────────────────────────
        from reconstruction.measurements import compute_all_measurements
        measurements = compute_all_measurements(
            tc_clean, wt_clean, et_clean,
            voxel_spacing=seg_result["voxel_spacing"],
            affine=seg_result["affine"],
            inference_time_s=seg_result["inference_time_s"],
        )

        meas_path = sess_dir / "measurements.json"
        with open(meas_path, "w") as fout:
            json.dump(measurements, fout, indent=2)

        # ── Anatomical localization ────────────────────────────────────────────
        # Non-blocking: failure here never stops the rest of the pipeline.
        _set_status(session_id, "running", 74, "Anatomical localization...")
        try:
            from inference.localization import localize_tumor
            loc_result = localize_tumor(
                wt_clean,
                seg_result["affine"],
                seg_result["voxel_spacing"],
            )
            with open(sess_dir / "localization.json", "w") as fout:
                json.dump(loc_result, fout, indent=2, cls=_NumpyEncoder)
            print(f"[Flask] Localization: {loc_result.get('full_label','?')}")
        except Exception as loc_exc:
            print(f"[Flask] Localization failed (non-fatal): {loc_exc}")
            import traceback as _tb
            _tb.print_exc()

        # Cache modality volumes for 2D slice serving (lightweight: shape + dtype only)
        try:
            _vol_cache[session_id] = {
                "t1":    modality_volumes["t1"]["data"],
                "masks": {"wt": wt_clean, "tc": tc_clean, "et": et_clean},
                "voxel_spacing": seg_result["voxel_spacing"],
                "affine": seg_result["affine"],
            }
        except Exception:
            pass

        _set_status(session_id, "running", 78, "Generating 3D meshes...")


        # ── Generate meshes ───────────────────────────────────────────────────
        from reconstruction.tumor_mesh import generate_tumor_meshes, generate_brain_surface
        mesh_paths = generate_tumor_meshes(
            tc_clean, wt_clean, et_clean,
            voxel_spacing=seg_result["voxel_spacing"],
            output_dir=str(sess_dir),
            smoothing_iterations=_cfg("reconstruction.smoothing", 3),
        )

        _set_status(session_id, "running", 92, "Generating brain surface...")

        # Brain surface from T1
        import numpy as np
        brain_path = str(sess_dir / "brain.glb")
        generate_brain_surface(
            modality_volumes["t1"]["data"],
            voxel_spacing=seg_result["voxel_spacing"],
            output_path=brain_path,
            # step_size=1 by default (full resolution — preserves cortical gyri/sulci)
            # smoothing_iterations=2 by default (minimal — keeps surface folds)
        )

        # Store available mesh list in session
        available_meshes = {}
        for name, path in mesh_paths.items():
            available_meshes[name] = bool(path and os.path.exists(path))
        available_meshes["brain"] = os.path.exists(brain_path)

        session_meta = {
            "measurements": str(meas_path),
            "meshes": available_meshes,
        }
        with open(sess_dir / "session_meta.json", "w") as fout:
            json.dump(session_meta, fout, indent=2)

        _set_status(session_id, "complete", 100, "Analysis complete")
        print(f"[Flask] Session {session_id} pipeline complete.")

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Flask] Pipeline error for {session_id}:\n{tb}")
        _set_status(session_id, "error", 0, "", error=str(exc))


@app.route("/api/analyze/<session_id>", methods=["POST"])
def analyze(session_id: str):
    """Start 3D analysis pipeline for an uploaded session."""
    sess_dir = _session_dir(session_id)
    if not (sess_dir / "modalities.json").exists():
        return jsonify({"error": "Session not found or no files uploaded."}), 404

    current = _get_status(session_id)
    if current and current.get("status") == "running":
        return jsonify({"error": "Analysis already running."}), 409

    _set_status(session_id, "queued", 0, "Queued for analysis")
    thread = threading.Thread(target=_run_pipeline_async, args=(session_id,), daemon=True)
    thread.start()
    return jsonify({"session_id": session_id, "status": "queued"})


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Status polling
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/status/<session_id>")
def get_status(session_id: str):
    status = _get_status(session_id)
    if status is None:
        return jsonify({"error": "Session not found."}), 404
    return jsonify({"session_id": session_id, **status})


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Results
# ──────────────────────────────────────────────────────────────────────────────

def _classify_brats_tumor(data: dict) -> dict:
    """Rule-based BraTS pattern classification.

    Based on the BraTS challenge convention (Menze et al. 2015):
      - Whole Tumor (WT) = all tumor classes (labels 1+2+4)
      - Tumor Core (TC)  = necrosis + enhancing (labels 1+4)
      - Enhancing Tumor (ET) = active enhancing region (label 4)

    Heuristic classification rules (NOT clinical diagnosis):
      ET + TC + WT  → High-Grade Glioma pattern (WHO Grade III–IV / GBM-like)
      TC + WT, no ET → Low-Grade / Non-enhancing Glioma pattern (WHO Grade II–III)
      WT only       → Non-enhancing or Metastasis-like pattern
      None detected → No tumor segmented

    Reference: Bakas et al. 2017, Nature Scientific Data; BraTS 2020 challenge.
    This is a RESEARCH PROTOTYPE — NOT a clinical diagnostic tool.
    """
    summary = data.get("summary", {})
    wt = summary.get("whole_tumor_detected", False)
    tc = summary.get("tumor_core_detected", False)
    et = summary.get("enhancing_tumor_detected", False)

    wt_vol = summary.get("whole_tumor_volume_cm3", 0) or 0
    tc_vol = summary.get("tumor_core_volume_cm3",  0) or 0
    et_vol = summary.get("enhancing_tumor_volume_cm3", 0) or 0

    # Core ratio: TC/WT  — High-grade tumors tend to have higher core ratio
    core_ratio = (tc_vol / wt_vol) if wt_vol > 0 else 0
    # Enhancing ratio: ET/TC
    enh_ratio  = (et_vol / tc_vol) if tc_vol > 0 else 0

    if not wt:
        return {
            "pattern":     "No Tumor Detected",
            "who_grade":   None,
            "subtype":     None,
            "confidence":  "N/A",
            "basis":       "No segmentation labels found",
            "color":       "var(--text-muted)",
        }

    if et and tc and wt:
        # Classic HGG / GBM pattern: all three components present
        if enh_ratio > 0.4:
            label    = "High-Grade Glioma Pattern"
            subtype  = "Glioblastoma (GBM)-like (WHO Grade IV)"
            grade    = "IV"
            conf     = "Moderate"
            color    = "var(--accent-red)"
        else:
            label    = "High-Grade Glioma Pattern"
            subtype  = "Anaplastic Glioma-like (WHO Grade III–IV)"
            grade    = "III–IV"
            conf     = "Low–Moderate"
            color    = "var(--accent-amber)"
    elif tc and wt and not et:
        # Non-enhancing core — LGG pattern
        label    = "Low-Grade Glioma Pattern"
        subtype  = "Non-Enhancing Glioma (WHO Grade II–III)"
        grade    = "II–III"
        conf     = "Low–Moderate"
        color    = "var(--accent-cyan)"
    else:
        # WT only — infiltrative or cystic
        label    = "Infiltrative / Non-Specific Pattern"
        subtype  = "Non-Enhancing or Cystic Lesion"
        grade    = "Indeterminate"
        conf     = "Low"
        color    = "var(--text-secondary)"

    basis = (
        f"BraTS pattern: WT={'✓' if wt else '✗'} ({wt_vol:.1f} cm³)  "
        f"TC={'✓' if tc else '✗'} ({tc_vol:.1f} cm³)  "
        f"ET={'✓' if et else '✗'} ({et_vol:.1f} cm³). "
        f"Core ratio {core_ratio:.0%}, Enhancing ratio {enh_ratio:.0%}."
    )

    return {
        "pattern":     label,
        "who_grade":   grade,
        "subtype":     subtype,
        "confidence":  conf,
        "basis":       basis,
        "color":       color,
        "disclaimer":  (
            "RESEARCH ESTIMATE ONLY. This classification is derived from "
            "segmentation label patterns using BraTS challenge conventions — "
            "NOT histopathological diagnosis. WHO grade requires biopsy and "
            "molecular profiling. Consult a qualified neuro-oncologist."
        ),
    }


@app.route("/api/results/<session_id>")
def get_results(session_id: str):
    """Return measurements + BraTS-pattern tumor classification for a session."""
    meas_path = _session_dir(session_id) / "measurements.json"
    if not meas_path.exists():
        return jsonify({"error": "Results not ready."}), 404
    with open(meas_path) as f:
        data = json.load(f)

    # Inject tumor type classification derived from segmentation pattern
    data["tumor_type"] = _classify_brats_tumor(data)

    return jsonify(data)



@app.route("/api/mesh/<session_id>/<mesh_type>")
def get_mesh(session_id: str, mesh_type: str):
    """Serve a GLB mesh file for the given session and type."""
    if mesh_type not in MESH_TYPES:
        return jsonify({"error": f"Unknown mesh type: {mesh_type}"}), 400

    glb_path = _session_dir(session_id) / f"{mesh_type}.glb"
    if not glb_path.exists():
        return jsonify({"error": f"Mesh '{mesh_type}' not available (no tumor detected?)."}), 404

    return send_file(
        str(glb_path),
        mimetype="model/gltf-binary",
        as_attachment=False,
        download_name=f"{mesh_type}.glb",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Anatomical localization
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/localization/<session_id>")
def get_localization(session_id: str):
    """Return anatomical localization for a completed session."""
    loc_path = _session_dir(session_id) / "localization.json"
    if not loc_path.exists():
        # Try to compute on demand if measurements are already done
        meas_path = _session_dir(session_id) / "measurements.json"
        if not meas_path.exists():
            return jsonify({"status": "unavailable",
                            "reason": "Analysis not yet complete."}), 404
        return jsonify({"status": "unavailable",
                        "reason": "Localization file not generated yet."}), 404
    with open(loc_path) as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/slice_info/<session_id>")
def get_slice_info(session_id: str):
    """Return slice counts and spacing for a session's MRI volume."""
    cached = _vol_cache.get(session_id)
    if cached is None:
        # Try loading from disk
        try:
            meta_path = _session_dir(session_id) / "modalities.json"
            if not meta_path.exists():
                return jsonify({"error": "Session not found."}), 404
            with open(meta_path) as f:
                paths = json.load(f)
            import nibabel as nib
            t1_path = paths.get("t1") or list(paths.values())[0]
            img = nib.load(t1_path)
            shape = img.header.get_data_shape()[:3]
            spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        shape = cached["t1"].shape
        spacing = cached["voxel_spacing"]

    return jsonify({
        "axial_count":    int(shape[2]),
        "coronal_count":  int(shape[1]),
        "sagittal_count": int(shape[0]),
        "shape": list(shape),
        "voxel_spacing_mm": [round(float(v), 3) for v in spacing],
    })


@app.route("/api/slice/<session_id>/<plane>/<int:index>")
def get_slice(session_id: str, plane: str, index: int):
    """Serve a 2D MRI slice as PNG with optional segmentation overlay.

    Planes: axial (z), coronal (y), sagittal (x).
    Overlay: wt=cyan, tc=red, et=yellow.
    """
    if plane not in ("axial", "coronal", "sagittal"):
        return jsonify({"error": "plane must be axial/coronal/sagittal"}), 400

    cached = _vol_cache.get(session_id)
    if cached is None:
        # Load from disk
        try:
            meta_path = _session_dir(session_id) / "modalities.json"
            with open(meta_path) as f:
                paths = json.load(f)
            import nibabel as nib
            t1_path = paths.get("t1") or list(paths.values())[0]
            img = nib.load(t1_path)
            t1_vol = img.get_fdata(dtype="float32")
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        masks = {}
    else:
        t1_vol = cached["t1"]
        masks  = cached.get("masks", {})

    # Extract slice
    try:
        import numpy as np
        import io

        if plane == "axial":
            index = max(0, min(index, t1_vol.shape[2] - 1))
            sl  = t1_vol[:, :, index]
            wt  = masks.get("wt", np.zeros_like(sl, dtype=np.uint8))[:, :, index] if masks else None
            tc  = masks.get("tc", np.zeros_like(sl, dtype=np.uint8))[:, :, index] if masks else None
            et  = masks.get("et", np.zeros_like(sl, dtype=np.uint8))[:, :, index] if masks else None
            orient = {"h_label": "R <- -> L", "v_label": "A ^ v P"}
        elif plane == "coronal":
            index = max(0, min(index, t1_vol.shape[1] - 1))
            sl  = t1_vol[:, index, :]
            wt  = masks.get("wt", np.zeros_like(sl, dtype=np.uint8))[:, index, :] if masks else None
            tc  = masks.get("tc", np.zeros_like(sl, dtype=np.uint8))[:, index, :] if masks else None
            et  = masks.get("et", np.zeros_like(sl, dtype=np.uint8))[:, index, :] if masks else None
            orient = {"h_label": "L <- -> R", "v_label": "S ^ v I"}
        else:  # sagittal
            index = max(0, min(index, t1_vol.shape[0] - 1))
            sl  = t1_vol[index, :, :]
            wt  = masks.get("wt", np.zeros_like(sl, dtype=np.uint8))[index, :, :] if masks else None
            tc  = masks.get("tc", np.zeros_like(sl, dtype=np.uint8))[index, :, :] if masks else None
            et  = masks.get("et", np.zeros_like(sl, dtype=np.uint8))[index, :, :] if masks else None
            orient = {"h_label": "A <- -> P", "v_label": "S ^ v I"}

        # Normalise slice to 0-255
        sl = np.rot90(sl)  # standard radiological orientation
        mn, mx = sl.min(), sl.max()
        if mx > mn:
            sl_norm = ((sl - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            sl_norm = np.zeros_like(sl, dtype=np.uint8)

        # Build RGB image
        h, w = sl_norm.shape
        rgb = np.stack([sl_norm, sl_norm, sl_norm], axis=-1)

        # Apply segmentation overlays
        if wt is not None:
            wt_r = np.rot90(wt)
            tc_r = np.rot90(tc) if tc is not None else np.zeros((h, w), dtype=np.uint8)
            et_r = np.rot90(et) if et is not None else np.zeros((h, w), dtype=np.uint8)
            # Whole tumor: cyan (#0ea5e9)
            rgb[wt_r > 0] = [14, 165, 233]
            # Tumor core: red
            rgb[tc_r > 0] = [239, 68, 68]
            # Enhancing: amber
            rgb[et_r > 0] = [245, 158, 11]

        # Encode as PNG
        try:
            from PIL import Image as PILImage
            img_out = PILImage.fromarray(rgb, mode="RGB")
            buf = io.BytesIO()
            img_out.save(buf, format="PNG")
            buf.seek(0)
        except ImportError:
            # Fallback: use matplotlib
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
            ax.imshow(rgb, aspect="auto")
            ax.axis("off")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            buf.seek(0)

        from flask import Response
        return Response(
            buf.read(),
            mimetype="image/png",
            headers={
                "X-Plane": plane,
                "X-Index": str(index),
                "X-Orient-H": orient["h_label"],
                "X-Orient-V": orient["v_label"],
                "Cache-Control": "no-cache",
            },
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Routes: Demo mode
# ──────────────────────────────────────────────────────────────────────────────

def _generate_demo_data(sess_dir: Path) -> Dict[str, str]:
    """Generate synthetic BraTS-compatible NIfTI volumes + ground-truth masks.

    Produces a 128³ volume with:
      - Anatomically shaped brain (gyri/sulci, cerebellum, brainstem)
      - Realistic T1/T2/FLAIR/T1ce contrast ratios
      - Off-centre glioma with proper BraTS WT/TC/ET hierarchy
      - Ground-truth segmentation masks saved as NPZ (bypasses intensity segmentation)

    The masks strictly satisfy: ET ⊆ TC ⊆ WT (BraTS convention).
    """
    import numpy as np
    import nibabel as nib
    from scipy import ndimage

    uploads_dir = sess_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    shape = (128, 128, 128)
    voxel_spacing = (1.5, 1.5, 1.5)   # 1.5mm isotropic — typical BraTS
    affine = np.diag([1.5, 1.5, 1.5, 1.0])

    rng = np.random.default_rng(42)
    Z, Y, X = np.mgrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = 64, 64, 64

    # ── Cerebral hemisphere base ellipsoid ────────────────────────────────────
    r_base = (
        ((X - cx) / 48)**2 +
        ((Y - cy) / 58)**2 +
        ((Z - cz) / 46)**2
    )

    # ── Add multi-scale gyri/sulci via spherical-like sine modulation ─────────
    dx = (X - cx).astype(float)
    dy = (Y - cy).astype(float)
    dz = (Z - cz).astype(float)
    r_xyz = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = np.arccos(np.clip(dz / r_xyz, -1, 1))
    phi   = np.arctan2(dy, dx)

    gyri_lo  = 0.06 * np.sin(4 * theta) * np.cos(4 * phi)
    gyri_mid = 0.04 * np.sin(7 * theta) * np.cos(7 * phi)
    gyri_hi  = 0.025 * np.sin(12 * theta) * np.cos(11 * phi)
    sulci_noise = 0.02 * rng.standard_normal(shape)

    brain_surface = r_base + gyri_lo + gyri_mid + gyri_hi + sulci_noise

    # ── Interhemispheric fissure ───────────────────────────────────────────────
    fissure_depth = np.exp(-((X - cx)**2) / (3**2))
    fissure_zone  = (np.abs(X - cx) < 5) & (Z > cz - 5)
    brain_surface[fissure_zone] += fissure_depth[fissure_zone] * 0.25

    brain = (brain_surface < 1.0)

    # ── Cerebellum ────────────────────────────────────────────────────────────
    cbm_cx, cbm_cy, cbm_cz = 64, 75, 30
    cbm_gyri = 0.05 * np.sin(14 * theta) * np.cos(9 * phi)
    cbm_surface = (
        ((X - cbm_cx) / 26)**2 +
        ((Y - cbm_cy) / 22)**2 +
        ((Z - cbm_cz) / 18)**2 + cbm_gyri
    )
    cbm = cbm_surface < 1.0

    # ── Brain stem stub ───────────────────────────────────────────────────────
    stem = (
        ((X - cx) / 8)**2 +
        ((Y - 80) / 8)**2 +
        ((Z - cz) / 14)**2 < 1.0
    )

    full_brain = brain | cbm | stem

    # ── White matter (inner region) ───────────────────────────────────────────
    wm_surface = r_base - 0.12
    wm = (wm_surface < 1.0) & full_brain

    # ── Tumor: RIGHT frontal-parietal glioma (BraTS hierarchy) ───────────────
    # Place tumor clearly within one hemisphere to demonstrate spatial hierarchy
    # RIGHT hemisphere: X > cx (X is left-right in our numpy axis convention)
    tx, ty, tz = 84, 45, 80   # right frontal-parietal region in voxel space

    # Outer WT: large irregular ellipsoid (edema + solid tumor)
    wt_base = ((X - tx)**2/20**2 + (Y - ty)**2/17**2 + (Z - tz)**2/18**2)
    # Add surface irregularity so it looks organic, not a perfect sphere
    wt_irr  = 0.08 * np.sin(5*theta) * np.cos(3*phi) + 0.05 * rng.standard_normal(shape)
    wt_r    = (wt_base + wt_irr) < 1.0

    # TC: 55% of WT radius (solid core inside WT)
    tc_r = ((X - tx)**2/11**2 + (Y - ty)**2/9**2 + (Z - tz)**2/10**2) < 1.0

    # NC (necrotic core): innermost dead tissue — not visible separately but subtracted from ET
    nc_r = ((X - tx)**2/5**2 + (Y - ty)**2/4**2 + (Z - tz)**2/4**2) < 1.0

    # ET: enhancing ring = TC minus necrotic core
    et_r = tc_r & (~nc_r)

    # ── ENFORCE BraTS hierarchy strictly: ET ⊆ TC ⊆ WT ──────────────────────
    tc_r  = tc_r.astype(np.uint8)
    wt_r  = wt_r.astype(np.uint8)
    et_r  = et_r.astype(np.uint8)

    # Guarantee containment (clip any voxels outside parent)
    tc_r  = np.where(wt_r, tc_r, 0).astype(np.uint8)
    et_r  = np.where(tc_r, et_r, 0).astype(np.uint8)

    # Confine to brain tissue
    wt_r  = np.where(full_brain, wt_r, 0).astype(np.uint8)
    tc_r  = np.where(full_brain, tc_r, 0).astype(np.uint8)
    et_r  = np.where(full_brain, et_r, 0).astype(np.uint8)

    # Apply light smoothing so marching cubes produces organic surface
    wt_smooth = ndimage.gaussian_filter(wt_r.astype(np.float32), sigma=0.8)
    tc_smooth = ndimage.gaussian_filter(tc_r.astype(np.float32), sigma=0.6)
    et_smooth = ndimage.gaussian_filter(et_r.astype(np.float32), sigma=0.5)

    # Re-threshold after smoothing
    wt_final = (wt_smooth > 0.40).astype(np.uint8)
    tc_final = (tc_smooth > 0.40).astype(np.uint8)
    et_final = (et_smooth > 0.40).astype(np.uint8)

    # Re-enforce hierarchy after smoothing (smoothing can create minor leaks)
    tc_final = np.where(wt_final, tc_final, 0).astype(np.uint8)
    et_final = np.where(tc_final, et_final, 0).astype(np.uint8)

    # Validate hierarchy
    if wt_final.sum() > 0:
        tc_overlap = (tc_final & wt_final).sum() / max(tc_final.sum(), 1)
        et_overlap = (et_final & tc_final).sum() / max(et_final.sum(), 1)
        print(f"[Demo] Mask hierarchy validation:")
        print(f"  WT={wt_final.sum()} vox  TC={tc_final.sum()} vox  ET={et_final.sum()} vox")
        print(f"  TC⊆WT overlap: {tc_overlap:.1%}  ET⊆TC overlap: {et_overlap:.1%}")
        if tc_overlap < 0.95 or et_overlap < 0.95:
            print(f"[Demo] WARNING: hierarchy violation — forcing containment")
            tc_final = np.where(wt_final, tc_final, 0).astype(np.uint8)
            et_final = np.where(tc_final, et_final, 0).astype(np.uint8)

    # ── Save ground-truth masks (bypasses intensity segmentation) ─────────────
    masks_path = str(uploads_dir / "ground_truth_masks.npz")
    np.savez_compressed(
        masks_path,
        wt_mask=wt_final,
        tc_mask=tc_final,
        et_mask=et_final,
        voxel_spacing=np.array(voxel_spacing),
        affine=affine,
    )
    print(f"[Demo] Ground-truth masks saved → {masks_path}")

    # ── Tissue signal values per modality ─────────────────────────────────────
    def build_volume(wm_v, gm_v, csf_v, wt_v, tc_v, nc_v, et_v, noise_sd=0.04):
        vol = np.zeros(shape, dtype=np.float32)
        vol[full_brain] = gm_v
        vol[wm]         = wm_v
        vol[cbm]        = gm_v * 0.95
        vol[stem]       = wm_v * 0.85
        vol[wt_final > 0] = wt_v
        vol[tc_final > 0] = tc_v
        vol[nc_r]       = nc_v
        vol[et_final > 0] = et_v
        vol += rng.standard_normal(shape).astype(np.float32) * noise_sd
        vol *= full_brain.astype(np.float32)
        vol = ndimage.gaussian_filter(vol, sigma=0.8)
        return np.clip(vol, 0.0, 1.0)

    saved: Dict[str, str] = {}
    mods = {
        "t1":    build_volume(0.85, 0.55, 0.05, 0.40, 0.35, 0.12, 0.40),
        "t1ce":  build_volume(0.85, 0.55, 0.05, 0.40, 0.35, 0.12, 0.95),
        "t2":    build_volume(0.45, 0.65, 0.95, 0.88, 0.72, 0.82, 0.72),
        "flair": build_volume(0.45, 0.65, 0.08, 0.90, 0.76, 0.80, 0.74),
    }

    for modality, vol in mods.items():
        data = (vol * 1000).astype(np.int16)
        img = nib.Nifti1Image(data, affine)
        img.header.set_zooms(voxel_spacing)
        img.header.set_xyzt_units("mm")
        path = str(uploads_dir / f"{modality}.nii.gz")
        nib.save(img, path)
        saved[modality] = path
        print(f"[Demo] {modality}.nii.gz  shape={data.shape}  "
              f"range=[{data.min()},{data.max()}]")

    return saved


    import numpy as np
    import nibabel as nib
    from scipy import ndimage

    uploads_dir = sess_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    shape = (128, 128, 128)
    voxel_spacing = (1.5, 1.5, 1.5)   # 1.5mm isotropic — typical BraTS
    affine = np.diag([1.5, 1.5, 1.5, 1.0])

    rng = np.random.default_rng(42)
    Z, Y, X = np.mgrid[:shape[0], :shape[1], :shape[2]]
    cx, cy, cz = 64, 64, 64

    # ── Cerebral hemisphere base ellipsoid ────────────────────────────────────
    # Slightly flattened inferior-superior, elongated anterior-posterior
    r_base = (
        ((X - cx) / 48)**2 +
        ((Y - cy) / 58)**2 +
        ((Z - cz) / 46)**2
    )

    # ── Add multi-scale gyri/sulci via spherical-like sine modulation ─────────
    # Convert to spherical coords for surface modulation
    dx = (X - cx).astype(float)
    dy = (Y - cy).astype(float)
    dz = (Z - cz).astype(float)
    r_xyz = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    theta = np.arccos(np.clip(dz / r_xyz, -1, 1))   # polar
    phi   = np.arctan2(dy, dx)                        # azimuthal

    # Low-freq = large gyri; high-freq = fine sulci
    gyri_lo  = 0.06 * np.sin(4 * theta) * np.cos(4 * phi)
    gyri_mid = 0.04 * np.sin(7 * theta) * np.cos(7 * phi)
    gyri_hi  = 0.025 * np.sin(12 * theta) * np.cos(11 * phi)
    sulci_noise = 0.02 * rng.standard_normal(shape)   # stochastic sulci

    brain_surface = r_base + gyri_lo + gyri_mid + gyri_hi + sulci_noise

    # ── Interhemispheric fissure (central vertical plane) ─────────────────────
    fissure_depth = np.exp(-((X - cx)**2) / (3**2))           # narrow sagittal groove
    fissure_zone  = (np.abs(X - cx) < 5) & (Z > cz - 5)      # superior midline
    brain_surface[fissure_zone] += fissure_depth[fissure_zone] * 0.25

    # ── Full brain mask (hemisphere) ─────────────────────────────────────────
    brain = (brain_surface < 1.0)

    # ── Cerebellum: posterior-inferior ellipsoid ───────────────────────────────
    cbm_cx, cbm_cy, cbm_cz = 64, 75, 30
    cbm = (
        ((X - cbm_cx) / 26)**2 +
        ((Y - cbm_cy) / 22)**2 +
        ((Z - cbm_cz) / 18)**2 < 1.0
    )
    # Add fine folia texture to cerebellum
    cbm_gyri = 0.05 * np.sin(14 * theta) * np.cos(9 * phi)
    cbm_surface = (
        ((X - cbm_cx) / 26)**2 +
        ((Y - cbm_cy) / 22)**2 +
        ((Z - cbm_cz) / 18)**2 + cbm_gyri
    )
    cbm = cbm_surface < 1.0

    # ── Brain stem stub ───────────────────────────────────────────────────────
    stem = (
        ((X - cx) / 8)**2 +
        ((Y - 80) / 8)**2 +
        ((Z - cz) / 14)**2 < 1.0
    )

    full_brain = brain | cbm | stem

    # ── White matter (inner region) ───────────────────────────────────────────
    wm_surface = r_base - 0.12   # shrink inward
    wm = (wm_surface < 1.0) & full_brain





@app.route("/api/demo")
def demo():
    """Run the full 3D pipeline on demo data.

    Priority:
      1. data/real_brats/ — real MNI152 human brain MRI (downloaded from HCP)
      2. _generate_demo_data() — anatomical synthetic fallback
    """
    import shutil as _shutil

    session_id = "demo-" + str(uuid.uuid4())[:8]
    sess_dir = _session_dir(session_id)
    sess_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = sess_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # ── Try real downloaded MRI first ─────────────────────────────────────────
    REAL_DATA_DIR = BASE_DIR / "data" / "real_brats"
    MODALITIES = ["t1", "t1ce", "t2", "flair"]
    real_files = {m: REAL_DATA_DIR / f"{m}.nii.gz" for m in MODALITIES}
    has_real = all(f.exists() and f.stat().st_size > 50_000 for f in real_files.values())

    modality_paths: Dict[str, str] = {}
    notice = ""

    if has_real:
        for modality, src in real_files.items():
            dst = uploads_dir / f"{modality}.nii.gz"
            _shutil.copy2(str(src), str(dst))
            modality_paths[modality] = str(dst)
        notice = (
            "DEMO using REAL human brain MRI — MNI152 standard brain "
            "(averaged from 152 real scans, HCP/Washington University). "
            "Tumor overlay is synthetic for demonstration purposes."
        )
        print(f"[Demo] Using real MNI152 MRI from {REAL_DATA_DIR}")
    else:
        try:
            modality_paths = _generate_demo_data(sess_dir)
            notice = (
                "DEMO MODE: Anatomical synthetic brain (gyri/sulci modelled). "
                "Not a real patient scan."
            )
            print("[Demo] Real MRI not found — using synthetic anatomical brain")
        except Exception as exc:
            return jsonify({"error": f"Demo data generation failed: {exc}"}), 500

    meta_path = sess_dir / "modalities.json"
    with open(meta_path, "w") as fout:
        json.dump(modality_paths, fout)

    _set_status(session_id, "queued", 0,
                "Demo — real MRI" if has_real else "Demo — synthetic brain")
    thread = threading.Thread(target=_run_pipeline_async, args=(session_id,), daemon=True)
    thread.start()

    return jsonify({
        "session_id": session_id,
        "status": "queued",
        "demo": True,
        "data_source": "real_mni152" if has_real else "synthetic_anatomical",
        "notice": notice,
    })



# ──────────────────────────────────────────────────────────────────────────────
# Routes: Session cleanup
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Remove session files and state."""
    import shutil
    sess_dir = _session_dir(session_id)
    if sess_dir.exists():
        shutil.rmtree(str(sess_dir))
    with _sessions_lock:
        _sessions.pop(session_id, None)
    # Clear vol cache to free memory
    _vol_cache.pop(session_id, None)
    return jsonify({"deleted": session_id})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT_3D", 7861))
    print(f"\n{'='*60}")
    print("  NeuroVR 3D — AI Brain Tumor Segmentation + AR/VR")
    print(f"  http://localhost:{port}")
    print("  Research prototype — NOT for clinical use")
    print(f"{'='*60}\n")

    # ── Warm-start PyTorch in the MAIN THREAD ──────────────────────────────────
    # PyTorch on macOS initializes the Metal framework on first use.
    # If this happens inside a background thread (threading.Thread),
    # it causes a [mutex.cc] lock deadlock. Importing torch and running a
    # tiny CPU op here forces Metal to initialize in the main thread first,
    # so background threads inherit an already-initialized framework.
    print("[Flask] Warming up PyTorch (main thread)...")
    try:
        import torch
        _ = torch.zeros(1)  # Forces framework init in main thread
        print(f"[Flask] PyTorch {torch.__version__} ready on CPU")
    except Exception as e:
        print(f"[Flask] PyTorch warm-start skipped: {e}")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
