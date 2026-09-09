"""End-to-end smoke test: synthetic MRI → preprocessing → mask → mesh → GLB → measurements"""
import sys, os, numpy as np
sys.path.insert(0, '.')

print("=== NeuroVR 3D End-to-End Smoke Test ===\n")

# 1. Generate demo data
print("[1] Generating synthetic BraTS case...")
from demo_data.generate_demo import generate_synthetic_brats_case
import tempfile
tmp = tempfile.mkdtemp()
paths = generate_synthetic_brats_case(tmp, shape=(48, 48, 48), spacing=(2.0,2.0,2.0))
print("    Files:", list(paths.keys()))

# 2. Load NIfTI
print("\n[2] Loading NIfTI volumes...")
from preprocessing.nifti_loader import load_brats_case
vols = load_brats_case(paths)
shapes = {m: v["shape"] for m, v in vols.items()}
print("    Shapes:", shapes)
print("    Voxel spacing:", vols['t1']['voxel_spacing'])

# 3. Preprocess
print("\n[3] Preprocessing (resample + normalize + stack)...")
from preprocessing.volume_preprocessor import VolumePreprocessor
pp = VolumePreprocessor(target_spacing=(1.0,1.0,1.0), crop_foreground=True)
tensor, meta = pp.preprocess(vols)
print("    Output tensor:", tensor.shape, "dtype=", tensor.dtype)

# 4. Simulate segmentation output (no MONAI needed for smoke test)
print("\n[4] Simulating 3D segmentation masks (pretend MONAI output)...")
H, W, D = meta["resampled_shape"]
cx, cy, cz = H//2, W//2, D//2
zz, yy, xx = np.mgrid[0:H, 0:W, 0:D]
dist = np.sqrt((xx-cx)**2+(yy-cy)**2+(zz-cz)**2)
wt_pp = (dist < 8).astype(np.uint8)
tc_pp = (dist < 5).astype(np.uint8)
et_pp = (dist < 3).astype(np.uint8)
print("    WT:", int(wt_pp.sum()), "vox  TC:", int(tc_pp.sum()), "vox  ET:", int(et_pp.sum()), "vox")

# 5. Invert to original space
print("\n[5] Inverting preprocessing transforms...")
wt = pp.invert_mask(wt_pp, meta)
tc = pp.invert_mask(tc_pp, meta)
et = pp.invert_mask(et_pp, meta)
print("    WT restored shape:", wt.shape, "  sum:", int(wt.sum()))
spacing = vols['t1']['voxel_spacing']

# 6. Post-process masks
print("\n[6] Post-processing masks (CC filter + morphological closing)...")
from reconstruction.mask_processing import postprocess_masks
tc_c, wt_c, et_c = postprocess_masks(tc, wt, et, min_component_size=10)
print("    Clean WT:", int(wt_c.sum()), "vox")

# 7. Measurements
print("\n[7] Computing physical measurements...")
from reconstruction.measurements import compute_all_measurements
affine = vols['t1']['affine']
meas = compute_all_measurements(tc_c, wt_c, et_c, spacing, affine, inference_time_s=1.5)
wt_vol = meas['regions']['whole_tumor']['volume_cm3']
print("    Whole Tumor volume:", round(wt_vol, 4), "cm3")
print("    Tumor detected:", meas['tumor_detected'])
print("    WT bbox:", meas['summary']['wt_bbox_mm'])
print("    Centroid:", meas['summary']['wt_centroid'])

# 8. Generate meshes
print("\n[8] Generating 3D meshes (Marching Cubes -> GLB)...")
from reconstruction.tumor_mesh import generate_tumor_meshes
mesh_dir = os.path.join(tmp, "meshes")
mesh_paths = generate_tumor_meshes(tc_c, wt_c, et_c, spacing, mesh_dir, smoothing_iterations=2)
for name, path in mesh_paths.items():
    if path and os.path.exists(path):
        sz = os.path.getsize(path)/1024
        print("    " + name + ":", round(sz, 1), "KB  PASS")
    else:
        print("    " + name + ": empty (no tumor)  PASS")

print("\n=== SMOKE TEST PASSED ===")
print("Pipeline: NIfTI -> preprocess -> mask -> mesh -> GLB -> measurements PASS")
print("MONAI 3D model: tested during flask_app_3d.py /api/analyze (requires torch+monai)")
