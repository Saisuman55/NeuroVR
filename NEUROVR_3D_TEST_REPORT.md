# NEUROVR 3D APPLICATION TEST REPORT

Test date: 10 September 2026  
Application: NeuroVR 3D — Brain Tumor Analysis and Visualization System  
Scope: research and educational prototype; not a clinical diagnostic system  
Build tested: working tree after the fixes listed in section 12

## 1. APPLICATION STATUS

PASS

- Fresh Flask instance started on port 7862 without application exceptions.
- The configured port 7861 was already occupied by an older running NeuroVR process; this was an environment condition, not a startup-code failure.
- Python 3.9.6, PyTorch 2.8.0, YAML configuration, frontend assets, and both packaged model files loaded/detected.
- Existing automated suite: 24/24 tests passed after regression additions.
- End-to-end pipeline and three consecutive demo analyses completed successfully.
- `app.py`, `flask_app.py`, `monitor.py`, and `report_generator.py` are not present in this NeuroVR 3D repository. The actual entry point is `flask_app_3d.py`; this was treated as repository structure rather than inventing unused replacements.

## 2. BACKEND STATUS

PASS

- `/api/health`, `/api/demo`, status polling, results, localization, slice information, axial/coronal/sagittal PNG slices, and all four GLB mesh endpoints returned expected responses.
- Invalid upload extension, missing session, invalid mesh type, and invalid slice plane returned controlled 4xx JSON errors.
- Demo analysis completed in approximately 1–2 seconds on CPU; reported segmentation inference was 0.07–0.18 seconds for the supplied demo data.
- No Python traceback or critical backend error occurred in the final run.

## 3. FRONTEND STATUS

PASS

- Live Chrome UI loaded with title, subtitle, session, CPU/status indicator, panels, viewport, and research disclaimer.
- JavaScript syntax checks passed for `ui.js` and `viewer.js`.
- Chrome console showed viewer initialization, four successful mesh loads, and session completion. The sole favicon 404 found during testing was fixed with an embedded favicon.
- Static HTML, CSS, and JavaScript returned HTTP 200 with correct content types.

## 4. 3D BRAIN MODEL

PASS

- Brain GLB loaded successfully (3,441,028 bytes in the acceptance session) and contained only finite vertices.
- Tumor centroid lies within the brain bounding box.
- Camera presets, focus, reset, split view, orbit mode, and section modes were exercised through the live UI without console exceptions.
- The viewer uses OrbitControls with rotate, zoom, pan, damping, touch support inherited from Three.js, and bounded camera distance.

## 5. TUMOR ALIGNMENT

PASS

- Exact voxel validation after post-processing: `enhancing_outside_core_voxels = 0`; `core_outside_whole_voxels = 0`.
- The invariant passed on three consecutive demo analyses: Enhancing Tumor ⊆ Tumor Core ⊆ Whole Tumor.
- Brain, WT, TC, and ET now receive the same NIfTI affine and the same RAS-to-Three.js transform.
- No independent normalization or per-region tumor placement is used.
- Whole-tumor mesh centroid was 1.46 mm from the mask centroid transformed into scene coordinates (within surface/centroid discretization tolerance).

## 6. SEGMENTATION VISUALIZATION

PASS

- Brain, whole tumor, tumor core, and enhancing tumor GLBs all loaded.
- Colors remain cyan/blue (WT), red (TC), and amber/yellow (ET), with ordered transparent materials.
- Each visibility toggle was switched OFF and ON independently and its UI state updated correctly.
- Brain and tumor opacity controls were tested at their extremes (0% and 100%).
- Axial, coronal, sagittal, and 3D clipping controls were exercised.

## 7. ANATOMICAL LOCATION

PASS

- Demo result was derived from the mask centroid and NIfTI affine: LEFT PARIETAL LOBE at RAS approximately [-26.0, -57.8, 56.9] mm.
- The UI explicitly displays “Estimated from MNI coordinates,” and the API includes a coordinate-estimation/non-clinical disclaimer.
- Focus/callout placement now uses the same RAS-to-scene transform as the rendered meshes.

## 8. UI CONTROLS

PASS

- Demo, all region selectors, all view selectors, six camera presets, focus, reset, split, all visibility toggles, opacity sliders, and section controls were exercised in Chrome.
- Region selectors update selected state and educational reference content without changing tumor coordinates.
- AR and VR were correctly disabled on unsupported hardware with explanatory messages: Android/Chrome required for AR and a WebXR-capable headset required for VR.
- Repeated demo runs created new sessions but did not duplicate objects in the active Three.js scene; the viewer disposes old mesh geometry/materials before loading a session.

## 9. ANALYSIS RESULTS

PASS

- Demo mode is visibly marked `DEMO — SAMPLE DATA`.
- Tumor type is now `Not classified`; no unsupported WHO grade or diagnostic confidence is displayed.
- Volumes were recalculated from voxel counts and 2×2×2 mm spacing and matched exactly:
  - Whole Tumor: 3,171 voxels × 8 mm³ / 1000 = 25.368 cm³.
  - Tumor Core: 1,857 voxels × 8 mm³ / 1000 = 14.856 cm³.
  - Enhancing Tumor: 769 voxels × 8 mm³ / 1000 = 6.152 cm³.

## 10. PERFORMANCE

PASS (prototype/demo workload)

- Three consecutive complete analyses succeeded without hangs or progressive pipeline failure.
- Fresh backend resident memory after the repeated acceptance run was approximately 151 MB; output growth is per-session persisted data, not an in-scene duplicate-object leak.
- Chrome reported approximately 94 MB for the tested application tab after analysis and interactive control testing.
- No WebGL context loss, browser freeze, or memory warning appeared in the console.
- Responsive CSS provides reduced panel widths at 1024 px and a single-column, scrollable layout below 768 px. Desktop interaction was directly tested; physical tablet/WebXR hardware was not available.

## 11. ERRORS FOUND

### Error: Spatial affine ignored during mesh export

- File: `reconstruction/tumor_mesh.py`, calls in `flask_app_3d.py`
- Component: marching-cubes mesh coordinate conversion
- Cause: vertices used voxel spacing only and ignored NIfTI orientation/translation.
- Severity: CRITICAL
- Recommended fix: apply one NIfTI affine and one RAS-to-Three.js mapping to brain and every tumor mesh.

### Error: Nested-region invariant not guaranteed after cleanup

- File: `reconstruction/mask_processing.py`
- Component: `postprocess_masks`
- Cause: closing and connected-component filtering ran independently on ET, TC, and WT.
- Severity: CRITICAL
- Recommended fix: validate common shapes, then enforce ET within TC and TC within WT after cleanup.

### Error: Focus marker used an approximate coordinate formula

- File: `frontend/viewer.js`
- Component: `_placeCentroidMarker`
- Cause: RAS coordinates were scaled relative to a bounding-box diagonal rather than transformed into mesh coordinates.
- Severity: HIGH
- Recommended fix: use the exact shared RAS-to-scene transform.

### Error: Unsupported tumor classification and WHO-grade inference

- File: `flask_app_3d.py`, `frontend/ui.js`
- Component: `_classify_brats_tumor` / results panel
- Cause: segmentation ratios were presented as glioma subtype and WHO-grade patterns without a validated classification model.
- Severity: HIGH
- Recommended fix: display `Not classified` and explain that the application performs segmentation only.

### Error: Modalities checked for shape but not registration

- File: `preprocessing/nifti_loader.py`
- Component: `load_brats_case`
- Cause: equal dimensions with different affines could pass validation.
- Severity: HIGH
- Recommended fix: reject mismatched voxel-to-world affines before inference.

### Error: XR exit left meshes translated

- File: `frontend/viewer.js`
- Component: `enterAR` / `enterVR` session-end handlers
- Cause: session entry translated meshes, but session end reset only the camera.
- Severity: MEDIUM
- Recommended fix: restore every mesh group to the shared origin before resetting the camera.

### Error: Brain visibility status did not update

- File: `frontend/index.html`
- Component: Brain visibility row
- Cause: the status element lacked the ID expected by `ui.js`.
- Severity: LOW
- Recommended fix: add `id="vis-brain"`.

### Error: Inactive clipping slider initially visible

- File: `frontend/index.html`
- Component: Section Plane controls
- Cause: clip-position row was visible while section mode was 3D/none.
- Severity: LOW
- Recommended fix: hide it until a section plane is selected.

### Error: Browser favicon 404

- File: `frontend/index.html`
- Component: browser metadata
- Cause: no favicon was declared.
- Severity: LOW
- Recommended fix: embed a local/data favicon so startup produces no failed asset request.

### Environment warning: dependency conflicts outside NeuroVR requirements

- File: current global Python environment
- Component: `pip check`
- Cause: installed TensorFlow expects protobuf ≥5.28 while protobuf 4.25.9 is installed; grpcio reports platform support metadata. NeuroVR does not import TensorFlow or grpcio.
- Severity: LOW
- Recommended fix: use an isolated virtual environment for the project and install only `requirements.txt`.

### Environment warning: urllib3 LibreSSL compatibility

- File: current system Python environment
- Component: urllib3 startup warning
- Cause: system Python is linked against LibreSSL 2.8.3 while urllib3 v2 expects OpenSSL 1.1.1+.
- Severity: LOW
- Recommended fix: use a current Python virtual environment linked with supported OpenSSL.

## 12. FIXES APPLIED

1. Applied the shared NIfTI affine and RAS-to-Three.js transform to brain, WT, TC, and ET mesh vertices.
2. Added identical-shape validation and enforced ET ⊆ TC ⊆ WT after post-processing.
3. Added API-visible exact spatial validation with violating-voxel counts.
4. Added affine-registration validation across T1, T1ce, T2, and FLAIR inputs.
5. Replaced approximate focus-marker placement with the shared physical coordinate transform.
6. Removed rule-based glioma subtype/WHO-grade output; results now say `Not classified`.
7. Restored shared mesh positions when AR/VR sessions end.
8. Fixed the brain visibility ON/OFF indicator.
9. Hid the clip slider until a section plane is active.
10. Added an embedded favicon to eliminate the final browser asset error.
11. Added regression tests for nested-mask enforcement, spatial-validation metadata, and affine-driven mesh placement.
12. Made the Orbit toolbar control functional as an explicit enable/pause toggle.
13. Added a visible WebGL initialization failure state and disabled unusable viewport controls.
14. Added bounded polling retries and a user-facing backend-disconnection error instead of an indefinite loading state.
15. Added a regression test that rejects equal-sized but spatially misregistered MRI modalities.
16. Added four purpose-specific display presets—Anatomical, Transparent, Tumor Analysis, and MRI + 3D—without changing the validated analysis pipeline.
17. Added an accessible Auto Rotate on/off mode with smooth OrbitControls rotation, enabled only after a 3D model loads.

## 13. FINAL STATUS

**READY FOR PROJECT DEMONSTRATION**

This status applies to the tested research/demo workflow. It does not imply clinical validation, diagnostic accuracy, medical-device certification, or successful operation on untested AR/VR hardware.
