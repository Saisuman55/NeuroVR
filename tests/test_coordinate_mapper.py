"""
Unit tests for coordinate mapping and mesh validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from reconstruction.coordinate_mapper import (
    RASToThreeJS,
    MM_TO_METERS,
    validate_coordinate_chain,
    validate_affine_consistency,
)


class TestRASToThreeJS:
    @pytest.fixture
    def identity_mapper(self):
        """Mapper with identity affine (1mm isotropic, no rotation)."""
        affine = np.eye(4, dtype=np.float64)
        return RASToThreeJS(affine)

    @pytest.fixture
    def real_affine_mapper(self):
        """Mapper with a realistic BraTS affine (1mm isotropic)."""
        affine = np.array([
            [-1,  0,  0, 120],
            [ 0,  1,  0, -120],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ], dtype=np.float64)
        return RASToThreeJS(affine)

    def test_voxel_to_ras_identity(self, identity_mapper):
        """With identity affine, voxel coords = RAS coords in mm."""
        voxel = np.array([10.0, 20.0, 30.0])
        ras = identity_mapper.voxel_to_ras(voxel)
        np.testing.assert_allclose(ras, [10.0, 20.0, 30.0], atol=1e-6)

    def test_voxel_to_threejs_scale(self, identity_mapper):
        """Three.js output should be in meters (1/1000 of mm)."""
        voxel = np.array([1000.0, 0.0, 0.0])
        threejs = identity_mapper.voxel_to_threejs(voxel)
        # X→X: 1000mm = 1.0m
        assert abs(threejs[0] - 1.0) < 1e-6

    def test_axis_remap_ras_to_threejs(self, identity_mapper):
        """Check axis mapping: RAS Z (Superior) → Three.js Y (Up)."""
        # Pure Z voxel (Superior direction)
        voxel = np.array([0.0, 0.0, 1000.0])
        threejs = identity_mapper.voxel_to_threejs(voxel)
        # Superior (Z) should map to Three.js Y
        assert abs(threejs[1] - 1.0) < 1e-6   # Y = 1m
        assert abs(threejs[0]) < 1e-6          # X = 0
        assert abs(threejs[2]) < 1e-6          # Z = 0

    def test_axis_remap_anterior_to_neg_z(self, identity_mapper):
        """RAS Y (Anterior) → Three.js -Z (Forward)."""
        voxel = np.array([0.0, 1000.0, 0.0])
        threejs = identity_mapper.voxel_to_threejs(voxel)
        assert abs(threejs[2] - (-1.0)) < 1e-6  # Z = -1m (negative forward)
        assert abs(threejs[0]) < 1e-6
        assert abs(threejs[1]) < 1e-6

    def test_round_trip_single_point(self, real_affine_mapper):
        """voxel → Three.js → voxel should recover original coordinates."""
        original = np.array([120.0, 80.0, 90.0])
        threejs = real_affine_mapper.voxel_to_threejs(original)
        recovered = real_affine_mapper.threejs_to_voxel(threejs)
        np.testing.assert_allclose(recovered, original, atol=1e-6)

    def test_round_trip_batch(self, real_affine_mapper):
        """Batch round-trip with N=50 random voxel coords."""
        rng = np.random.default_rng(42)
        voxels = rng.uniform(0, 240, (50, 3))
        threejs = real_affine_mapper.voxel_to_threejs(voxels)
        recovered = real_affine_mapper.threejs_to_voxel(threejs)
        np.testing.assert_allclose(recovered, voxels, atol=1e-5)

    def test_transform_matrix_shape(self, identity_mapper):
        matrix = identity_mapper.get_transform_matrix()
        assert matrix.shape == (4, 4)

    def test_metadata_json_serializable(self, real_affine_mapper):
        import json
        meta = real_affine_mapper.get_metadata()
        json.dumps(meta)  # Should not raise


class TestCoordinateChainValidation:
    def test_identity_affine_valid(self):
        affine = np.eye(4)
        result = validate_coordinate_chain(affine)
        assert result["valid"]
        assert result["max_error_mm"] < 0.1

    def test_realistic_brats_affine_valid(self):
        affine = np.array([
            [-1, 0, 0, 120],
            [0, 1, 0, -120],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        result = validate_coordinate_chain(affine)
        assert result["valid"]

    def test_returns_transform_matrix(self):
        affine = np.eye(4)
        result = validate_coordinate_chain(affine)
        matrix = np.array(result["transform_matrix"])
        assert matrix.shape == (4, 4)


class TestAffineConsistency:
    def test_identical_affines_valid(self):
        affine = np.eye(4) * 1.0
        result = validate_affine_consistency([affine, affine, affine])
        assert result["valid"]
        assert result["mismatched_modalities"] == []

    def test_mismatched_affine_detected(self):
        aff1 = np.eye(4)
        aff2 = np.eye(4)
        aff2[0, 3] = 100.0  # Translate X by 100mm
        result = validate_affine_consistency(
            [aff1, aff2], modality_names=["t1", "t1ce"]
        )
        assert not result["valid"]
        assert len(result["mismatched_modalities"]) == 1
        assert result["mismatched_modalities"][0]["modality"] == "t1ce"

    def test_empty_list(self):
        result = validate_affine_consistency([])
        assert result["valid"]

    def test_single_affine(self):
        result = validate_affine_consistency([np.eye(4)])
        assert result["valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
