"""
Unit tests for BraTS label utilities.

Tests containment enforcement, label conversion, multi-channel
round-trips, and edge cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from data.label_utils import (
    LABEL_BACKGROUND, LABEL_NCR_NET, LABEL_EDEMA, LABEL_ET,
    ContainmentViolationError,
    brats_seg_to_regions,
    create_whole_tumor_mask,
    create_tumor_core_mask,
    create_enhancing_tumor_mask,
    enforce_containment,
    validate_containment,
    masks_to_multichannel,
    multichannel_to_masks,
    regions_to_brats_seg,
)


class TestMaskCreation:
    def test_whole_tumor_includes_all_labels(self):
        seg = np.array([0, 1, 2, 4], dtype=np.int32)
        wt = create_whole_tumor_mask(seg)
        assert wt[0] == 0   # background excluded
        assert wt[1] == 1   # NCR included
        assert wt[2] == 1   # edema included
        assert wt[3] == 1   # ET included

    def test_tumor_core_excludes_edema(self):
        seg = np.array([0, 1, 2, 4], dtype=np.int32)
        tc = create_tumor_core_mask(seg)
        assert tc[0] == 0   # background excluded
        assert tc[1] == 1   # NCR included
        assert tc[2] == 0   # edema excluded
        assert tc[3] == 1   # ET included

    def test_enhancing_tumor_only_et(self):
        seg = np.array([0, 1, 2, 4], dtype=np.int32)
        et = create_enhancing_tumor_mask(seg)
        assert et[0] == 0
        assert et[1] == 0
        assert et[2] == 0
        assert et[3] == 1

    def test_brats_compat_label_3(self):
        """Some older BraTS datasets use label 3 instead of 4 for ET."""
        seg = np.array([0, 1, 2, 3], dtype=np.int32)
        et = create_enhancing_tumor_mask(seg)
        assert et[3] == 1

    def test_all_background(self):
        seg = np.zeros((10, 10, 10), dtype=np.int32)
        wt, tc, et = brats_seg_to_regions(seg)
        assert wt.sum() == 0
        assert tc.sum() == 0
        assert et.sum() == 0


class TestContainmentEnforcement:
    def test_valid_containment_unchanged(self):
        """Already-valid containment should not be modified."""
        # ET ⊂ TC ⊂ WT
        wt = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
        tc = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        et = np.array([0, 0, 1, 0, 0], dtype=np.uint8)
        wt_out, tc_out, et_out = enforce_containment(wt, tc, et)
        np.testing.assert_array_equal(wt_out, wt)
        np.testing.assert_array_equal(tc_out, tc)
        np.testing.assert_array_equal(et_out, et)

    def test_et_outside_tc_corrected(self):
        """ET voxel outside TC should be removed from ET."""
        wt = np.array([1, 1, 1, 0], dtype=np.uint8)
        tc = np.array([1, 0, 0, 0], dtype=np.uint8)  # TC only at index 0
        et = np.array([0, 0, 1, 0], dtype=np.uint8)  # ET at index 2 (outside TC!)
        _, tc_out, et_out = enforce_containment(wt, tc, et)
        # ET outside TC should now be within TC (ET is added to TC)
        assert et_out[2] == 1   # ET preserved
        assert tc_out[2] == 1   # TC expanded to include ET
        assert et_out[1] == 0   # No ET where there wasn't any

    def test_tc_outside_wt_corrected(self):
        """TC voxel outside WT should expand WT."""
        wt = np.array([1, 0, 0, 0], dtype=np.uint8)
        tc = np.array([1, 1, 0, 0], dtype=np.uint8)  # TC at index 1 (outside WT)
        et = np.array([0, 0, 0, 0], dtype=np.uint8)
        wt_out, _, _ = enforce_containment(wt, tc, et)
        assert wt_out[1] == 1  # WT expanded to include TC

    def test_containment_3d(self):
        """Test on 3D volumes."""
        shape = (30, 30, 30)
        wt = np.zeros(shape, dtype=np.uint8)
        tc = np.zeros(shape, dtype=np.uint8)
        et = np.zeros(shape, dtype=np.uint8)

        # WT sphere radius 12, TC radius 8, ET radius 4
        zz, yy, xx = np.mgrid[:30, :30, :30]
        d = np.sqrt((xx - 15) ** 2 + (yy - 15) ** 2 + (zz - 15) ** 2)
        wt[d < 12] = 1
        tc[d < 8] = 1
        et[d < 4] = 1

        wt_out, tc_out, et_out = enforce_containment(wt, tc, et)
        report = validate_containment(wt_out, tc_out, et_out)
        assert report.valid


class TestContainmentValidation:
    def test_valid_returns_true(self):
        wt = np.array([1, 1, 1, 0], dtype=np.uint8)
        tc = np.array([0, 1, 1, 0], dtype=np.uint8)
        et = np.array([0, 0, 1, 0], dtype=np.uint8)
        report = validate_containment(wt, tc, et)
        assert report.valid
        assert report.et_outside_tc_voxels == 0
        assert report.tc_outside_wt_voxels == 0

    def test_violation_detected(self):
        wt = np.array([1, 0, 0, 0], dtype=np.uint8)
        tc = np.array([1, 1, 0, 0], dtype=np.uint8)  # TC outside WT at index 1
        et = np.array([0, 0, 0, 0], dtype=np.uint8)
        report = validate_containment(wt, tc, et)
        assert not report.valid
        assert report.tc_outside_wt_voxels == 1

    def test_raise_on_violation(self):
        wt = np.zeros(5, dtype=np.uint8)
        tc = np.ones(5, dtype=np.uint8)  # TC everywhere but WT empty
        et = np.zeros(5, dtype=np.uint8)
        with pytest.raises(ContainmentViolationError):
            validate_containment(wt, tc, et, raise_on_violation=True)

    def test_both_empty(self):
        """All-empty masks should report valid containment."""
        z = np.zeros(10, dtype=np.uint8)
        report = validate_containment(z, z, z)
        assert report.valid
        assert report.wt_voxels == 0


class TestMultiChannelConversion:
    def test_round_trip(self):
        """masks_to_multichannel → multichannel_to_masks should recover masks."""
        wt = np.array([1, 1, 1, 0], dtype=np.uint8)
        tc = np.array([0, 1, 1, 0], dtype=np.uint8)
        et = np.array([0, 0, 1, 0], dtype=np.uint8)

        mc = masks_to_multichannel(wt, tc, et)
        assert mc.shape[0] == 3

        wt2, tc2, et2 = multichannel_to_masks(mc, threshold=0.5)
        np.testing.assert_array_equal(wt2, wt)
        np.testing.assert_array_equal(tc2, tc)
        np.testing.assert_array_equal(et2, et)

    def test_multichannel_shape(self):
        shape = (16, 16, 16)
        wt = np.ones(shape, dtype=np.uint8)
        tc = np.ones(shape, dtype=np.uint8)
        et = np.zeros(shape, dtype=np.uint8)
        mc = masks_to_multichannel(wt, tc, et)
        assert mc.shape == (3, 16, 16, 16)
        assert mc.dtype == np.float32


class TestRegionsToBratsSeg:
    def test_reconstruction(self):
        """Integer seg → regions → integer seg should produce valid labels."""
        seg = np.array([[0, 1, 2, 4]], dtype=np.int32)
        wt, tc, et = brats_seg_to_regions(seg, enforce=True)
        reconstructed = regions_to_brats_seg(wt, tc, et)
        valid_labels = {0, 1, 2, 4}
        assert set(np.unique(reconstructed).tolist()).issubset(valid_labels)

    def test_all_background_reconstruction(self):
        seg = np.zeros((5, 5, 5), dtype=np.int32)
        wt, tc, et = brats_seg_to_regions(seg)
        recon = regions_to_brats_seg(wt, tc, et)
        assert recon.sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
