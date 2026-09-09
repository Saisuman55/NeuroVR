"""
Tests for NeuroVR 3D preprocessing pipeline.

These tests use synthetic numpy-generated NIfTI volumes — no real patient data.
Run with: python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_synthetic_nifti(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0),
                           add_tumor=True, tmp_dir=None):
    """Create a synthetic NIfTI file in a temp directory. Returns path."""
    import nibabel as nib

    rng = np.random.default_rng(42)
    vol = rng.normal(0.5, 0.15, shape).astype(np.float32)

    if add_tumor:
        cx, cy, cz = [s // 2 for s in shape]
        zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
        vol[dist < 5] += 0.8

    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms(spacing)

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "test_volume.nii.gz")
    nib.save(img, path)
    return path


def _make_four_modalities(tmp_dir, spacing=(1.0, 1.0, 1.0), shape=(32, 32, 32)):
    """Create all 4 BraTS modalities as NIfTI files."""
    import nibabel as nib

    rng = np.random.default_rng(0)
    paths = {}
    for mod in ["t1", "t1ce", "t2", "flair"]:
        vol = rng.normal(0.5, 0.1, shape).astype(np.float32)
        affine = np.diag([*spacing, 1.0])
        img = nib.Nifti1Image(vol, affine)
        img.header.set_zooms(spacing)
        p = os.path.join(tmp_dir, f"{mod}.nii.gz")
        nib.save(img, p)
        paths[mod] = p
    return paths


# ─── NIfTI Loader Tests ────────────────────────────────────────────────────────

class TestNiftiLoader:
    def test_valid_nifti_loads(self, tmp_path):
        """A valid NIfTI file should load without error."""
        path = _make_synthetic_nifti(tmp_dir=str(tmp_path))
        from preprocessing.nifti_loader import load_nifti_volume
        vol = load_nifti_volume(path)

        assert vol["data"] is not None
        assert vol["data"].dtype == np.float32
        assert len(vol["data"].shape) == 3
        assert vol["voxel_spacing"] is not None
        assert len(vol["voxel_spacing"]) == 3

    def test_missing_file_raises(self):
        """Missing file should raise ValueError."""
        from preprocessing.nifti_loader import validate_nifti_file
        valid, msg = validate_nifti_file("/nonexistent/path/file.nii.gz")
        assert not valid
        assert "not found" in msg.lower()

    def test_invalid_extension_rejected(self, tmp_path):
        """Non-NIfTI extension should be flagged."""
        bad_path = str(tmp_path / "scan.jpg")
        Path(bad_path).write_bytes(b"not a nifti")
        from preprocessing.nifti_loader import validate_nifti_file
        valid, _ = validate_nifti_file(bad_path)
        assert not valid

    def test_corrupted_nifti_raises(self, tmp_path):
        """Corrupted file content should fail validation."""
        bad = str(tmp_path / "corrupt.nii.gz")
        Path(bad).write_bytes(b"\x00" * 100)
        from preprocessing.nifti_loader import validate_nifti_file
        valid, msg = validate_nifti_file(bad)
        assert not valid

    def test_load_brats_case_valid(self, tmp_path):
        """Four valid modalities should load successfully."""
        paths = _make_four_modalities(str(tmp_path))
        from preprocessing.nifti_loader import load_brats_case
        vols = load_brats_case(paths)

        assert set(vols.keys()) == {"t1", "t1ce", "t2", "flair"}
        for mod, v in vols.items():
            assert v["data"] is not None, f"{mod} data is None"

    def test_load_brats_case_missing_modality(self, tmp_path):
        """Missing required modality should raise ValueError."""
        paths = _make_four_modalities(str(tmp_path))
        del paths["flair"]
        from preprocessing.nifti_loader import load_brats_case
        with pytest.raises(ValueError, match="[Mm]issing"):
            load_brats_case(paths)

    def test_auto_detect_modalities(self, tmp_path):
        """Auto-detection should find all 4 modalities from filenames."""
        import nibabel as nib

        names = {
            "t1":    "case001_t1.nii.gz",
            "t1ce":  "case001_t1ce.nii.gz",
            "t2":    "case001_t2.nii.gz",
            "flair": "case001_flair.nii.gz",
        }
        rng = np.random.default_rng(1)
        for _, fname in names.items():
            vol = rng.normal(0, 1, (16, 16, 16)).astype(np.float32)
            img = nib.Nifti1Image(vol, np.eye(4))
            nib.save(img, str(tmp_path / fname))

        from preprocessing.nifti_loader import auto_detect_modalities
        detected = auto_detect_modalities(str(tmp_path))
        assert set(detected.keys()) == {"t1", "t1ce", "t2", "flair"}


# ─── Volume Preprocessor Tests ────────────────────────────────────────────────

class TestVolumePreprocessor:
    def test_output_tensor_shape(self, tmp_path):
        """Preprocessed output should be a [4, H, W, D] tensor."""
        import torch
        paths = _make_four_modalities(str(tmp_path), shape=(32, 32, 32))
        from preprocessing.nifti_loader import load_brats_case
        from preprocessing.volume_preprocessor import VolumePreprocessor

        vols = load_brats_case(paths)
        pp = VolumePreprocessor(target_spacing=(1.0, 1.0, 1.0), crop_foreground=False)
        tensor, meta = pp.preprocess(vols)

        assert tensor.ndim == 4
        assert tensor.shape[0] == 4       # 4 modalities
        assert tensor.dtype == torch.float32

    def test_normalization_is_applied(self, tmp_path):
        """After normalization, values should be centred near 0."""
        paths = _make_four_modalities(str(tmp_path), shape=(32, 32, 32))
        from preprocessing.nifti_loader import load_brats_case
        from preprocessing.volume_preprocessor import VolumePreprocessor

        vols = load_brats_case(paths)
        pp = VolumePreprocessor(normalize=True, crop_foreground=False)
        tensor, _ = pp.preprocess(vols)

        # Mean of first channel should be close to 0 after z-score
        first_channel = tensor[0].numpy()
        nonzero = first_channel[first_channel != 0]
        if nonzero.size > 10:
            assert abs(nonzero.mean()) < 1.0

    def test_invert_mask_shape(self, tmp_path):
        """Inverted mask should have the same shape as the original volume."""
        paths = _make_four_modalities(str(tmp_path), shape=(32, 32, 32))
        from preprocessing.nifti_loader import load_brats_case
        from preprocessing.volume_preprocessor import VolumePreprocessor

        vols = load_brats_case(paths)
        pp = VolumePreprocessor(crop_foreground=False)
        tensor, meta = pp.preprocess(vols)

        # Create a dummy mask in resampled space
        dummy_mask = np.ones(meta["resampled_shape"], dtype=np.uint8)
        inverted = pp.invert_mask(dummy_mask, meta)

        assert inverted.shape == (32, 32, 32)


# ─── Reconstruction Tests ─────────────────────────────────────────────────────

class TestReconstruction:
    def test_mask_to_mesh_returns_mesh(self):
        """A non-empty mask should produce a trimesh object."""
        pytest.importorskip("trimesh")
        pytest.importorskip("skimage")
        from reconstruction.tumor_mesh import mask_to_mesh

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # Solid cube
        spacing = np.array([1.0, 1.0, 1.0])

        mesh = mask_to_mesh(mask, spacing, smoothing_iterations=0)
        assert mesh is not None
        assert len(mesh.vertices) > 3
        assert len(mesh.faces) > 1

    def test_empty_mask_returns_none(self):
        """An empty mask should return None gracefully."""
        pytest.importorskip("trimesh")
        from reconstruction.tumor_mesh import mask_to_mesh

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        spacing = np.array([1.0, 1.0, 1.0])
        mesh = mask_to_mesh(mask, spacing)
        assert mesh is None

    def test_mesh_physical_scale(self):
        """Mesh vertices should be in meters (mm/1000)."""
        pytest.importorskip("trimesh")
        from reconstruction.tumor_mesh import mask_to_mesh

        # 10mm cube at 2mm spacing → expect vertices in range [0, 0.02] meters
        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[5:10, 5:10, 5:10] = 1
        spacing = np.array([2.0, 2.0, 2.0])

        mesh = mask_to_mesh(mask, spacing, smoothing_iterations=0)
        assert mesh is not None
        # All vertices should be within physical bounds (in meters)
        max_coord = np.abs(mesh.vertices).max()
        assert max_coord < 0.1, f"Vertices too large: {max_coord}m — scale not applied?"

    def test_glb_export(self, tmp_path):
        """GLB export should create a non-empty file."""
        pytest.importorskip("trimesh")
        from reconstruction.tumor_mesh import mask_to_mesh, export_mesh_glb

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1
        spacing = np.array([1.0, 1.0, 1.0])
        mesh = mask_to_mesh(mask, spacing, smoothing_iterations=0)

        out_path = str(tmp_path / "test.glb")
        success = export_mesh_glb(mesh, out_path)
        assert success
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 100

    def test_postprocess_removes_small_components(self):
        """Components below minimum size should be removed."""
        from reconstruction.mask_processing import filter_connected_components

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        # Large component
        mask[5:20, 5:20, 5:20] = 1
        # Small component (2 voxels)
        mask[28, 28, 28] = 1
        mask[28, 28, 29] = 1

        filtered, n = filter_connected_components(mask, min_size=50)
        assert n == 1  # Only large component kept
        assert filtered[28, 28, 28] == 0   # Small one removed
        assert filtered[10, 10, 10] == 1   # Large one kept


# ─── Measurements Tests ───────────────────────────────────────────────────────

class TestMeasurements:
    def test_volume_calculation_unit_cube(self):
        """1mm³ voxels × 1000 voxels = 1.0 cm³."""
        from reconstruction.measurements import calculate_volume_cm3

        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # 10×10×10 = 1000 voxels
        spacing = np.array([1.0, 1.0, 1.0])
        vol = calculate_volume_cm3(mask, spacing)
        assert abs(vol - 1.0) < 0.01, f"Expected ~1.0 cm³, got {vol}"

    def test_volume_scaling_with_spacing(self):
        """2mm spacing → volume 8× larger than 1mm spacing for same mask."""
        from reconstruction.measurements import calculate_volume_cm3

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1  # 10×10×10 = 1000 voxels

        v1 = calculate_volume_cm3(mask, np.array([1.0, 1.0, 1.0]))
        v2 = calculate_volume_cm3(mask, np.array([2.0, 2.0, 2.0]))
        assert abs(v2 / v1 - 8.0) < 0.01, f"Expected 8× ratio, got {v2/v1}"

    def test_empty_mask_volume_is_zero(self):
        """Empty mask should return 0.0 cm³."""
        from reconstruction.measurements import calculate_volume_cm3

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        assert calculate_volume_cm3(mask, np.array([1.0, 1.0, 1.0])) == 0.0

    def test_bounding_box(self):
        """Bounding box dimensions should match known cube size."""
        from reconstruction.measurements import calculate_bounding_box_mm

        mask = np.zeros((50, 50, 50), dtype=np.uint8)
        mask[10:20, 15:25, 5:15] = 1  # 10×10×10 voxel cube
        spacing = np.array([1.0, 1.0, 1.0])

        bbox = calculate_bounding_box_mm(mask, spacing)
        assert bbox is not None
        assert abs(bbox["width_mm"]  - 10.0) < 1e-3
        assert abs(bbox["height_mm"] - 10.0) < 1e-3
        assert abs(bbox["depth_mm"]  - 10.0) < 1e-3

    def test_centroid_center_of_mass(self):
        """Centroid should be at geometric center of a symmetric mask."""
        from reconstruction.measurements import calculate_centroid_mm

        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1  # Center at voxel [14.5, 14.5, 14.5]
        spacing = np.array([1.0, 1.0, 1.0])
        centroid = calculate_centroid_mm(mask, spacing)

        assert centroid is not None
        for v in centroid["centroid_voxel"]:
            assert abs(v - 14.5) < 0.5, f"Centroid off: {centroid}"

    def test_compute_all_measurements(self, tmp_path):
        """Full measurement computation should be JSON-serialisable."""
        from reconstruction.measurements import compute_all_measurements

        mask = np.zeros((32, 32, 32), dtype=np.uint8)
        mask[10:20, 10:20, 10:20] = 1
        spacing = np.array([1.0, 1.0, 1.0])

        result = compute_all_measurements(
            tc_mask=mask, wt_mask=mask, et_mask=mask,
            voxel_spacing=spacing,
        )

        # Should be JSON-serialisable
        out_path = str(tmp_path / "measurements.json")
        with open(out_path, "w") as f:
            json.dump(result, f)
        assert os.path.exists(out_path)
        assert result["tumor_detected"] is True
        assert result["regions"]["whole_tumor"]["volume_cm3"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
