/**
 * NeuroVR 3D — UI Controller
 * Medical Imaging Workstation
 *
 * Fixes in this version:
 *  - Demo badge uses display:none toggle (no .hidden class conflict)
 *  - Modality badges turn green after demo/analysis completes
 *  - View resets to 3D after analysis completes
 *  - Auto split-view opens when 2D tab (Axial/Coronal/Sagittal) is clicked
 *  - Viewer empty state hidden correctly
 *  - window.viewer guard with retry for ES module race condition
 */

'use strict';

// ─── State ────────────────────────────────────────────────────────────────────
let currentSessionId = null;
let pollInterval     = null;
let localizationData = null;
let isDemoMode       = false;
let splitActive      = false;
let sliceInfo        = null;
let currentPlane     = 'axial';
let currentSliceIdx  = 0;
let pendingModalities= {};  // { t1: File, t1ce: File, ... }
let pollFailureCount = 0;

// Visibility state
const visibility = {
  brain:           true,
  tumor_whole:     true,
  tumor_core:      true,
  tumor_enhancing: true,
};

// ─── DOM helpers ─────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const qs = s  => document.querySelector(s);
const show = (el, d = 'block') => { if (el) el.style.display = d; };
const hide = el => { if (el) el.style.display = 'none'; };
const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };
const addClass    = (el, c) => el?.classList.add(c);
const removeClass = (el, c) => el?.classList.remove(c);

// ─── Viewer accessor (handles ES module loading race) ─────────────────────────
function getViewer() {
  return window.viewer ?? null;
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  _bindUpload();
  _bindButtons();
  _bindViewPresets();
  _bindSectionButtons();
  _bindToggleRows();
  _bindVisualizationModes();
  _bindRegionButtons();
  _checkSystemStatus();
  _initSliceViewer();
});

// ─── System status ────────────────────────────────────────────────────────────
async function _checkSystemStatus() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    setText('systemStatusText', 'ONLINE');
    setText('systemDevice',     d.device ?? '');
    setText('sb-device',        d.device ?? '—');
    const dot = $('systemDot');
    if (dot) dot.style.background = 'var(--accent-green)';
  } catch {
    setText('systemStatusText', 'OFFLINE');
    const dot = $('systemDot');
    if (dot) { dot.style.background = 'var(--accent-red)'; }
  }
}

function _setVisibility(name, visible) {
  visibility[name] = visible;
  const dot = $(`dot-${name}`);
  const lbl = $(`lbl-${name}`);
  const badge = $(`vis-${name}`);
  if (dot) dot.classList.toggle('off', !visible);
  if (lbl) lbl.classList.toggle('off', !visible);
  if (badge) badge.textContent = visible ? 'ON' : 'OFF';
  getViewer()?.setVisible(name, visible);
}

function _bindVisualizationModes() {
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-mode]').forEach(item => {
        item.classList.remove('active');
        item.setAttribute('aria-pressed', 'false');
      });
      button.classList.add('active');
      button.setAttribute('aria-pressed', 'true');
      _applyVisualizationMode(button.dataset.mode);
    });
  });
}

function _applyVisualizationMode(mode) {
  const presets = {
    anatomical:  { brain: 100, tumor: 0,   tumorsVisible: false },
    transparent: { brain: 35,  tumor: 75,  tumorsVisible: true },
    tumor:       { brain: 22,  tumor: 92,  tumorsVisible: true },
    mri3d:       { brain: 35,  tumor: 92,  tumorsVisible: true },
  };
  const preset = presets[mode];
  if (!preset) return;

  const brainSlider = $('brainOpacity');
  const tumorSlider = $('tumorOpacity');
  if (brainSlider) {
    brainSlider.value = preset.brain;
    brainSlider.setAttribute('value', String(preset.brain));
  }
  if (tumorSlider) {
    tumorSlider.value = preset.tumor;
    tumorSlider.setAttribute('value', String(preset.tumor));
  }
  window.ui.setBrainOpacity(preset.brain);
  window.ui.setTumorOpacity(preset.tumor);
  _setVisibility('brain', true);
  ['tumor_whole', 'tumor_core', 'tumor_enhancing'].forEach(name => {
    _setVisibility(name, preset.tumorsVisible);
  });

  if (mode === 'mri3d') {
    if (!splitActive) window.ui.toggleSplitView();
  } else if (splitActive) {
    window.ui.toggleSplitView();
  }
}

// ─── Upload ───────────────────────────────────────────────────────────────────
function _bindUpload() {
  const zone   = $('uploadZone');
  const fileIn = $('fileInput') || $('fileInputHidden');

  if (!zone) return;

  zone.addEventListener('click',    () => fileIn?.click());
  zone.addEventListener('keydown',  e => { if (e.key === 'Enter' || e.key === ' ') fileIn?.click(); });

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop',      e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    _handleFiles(Array.from(e.dataTransfer.files));
  });

  const fin2 = $('fileInputHidden');
  if (fin2) fin2.addEventListener('change', e => _handleFiles(Array.from(e.target.files)));
  if (fileIn) fileIn.addEventListener('change', e => _handleFiles(Array.from(e.target.files)));
}

function _handleFiles(files) {
  const nifti = files.filter(f => f.name.endsWith('.nii') || f.name.endsWith('.nii.gz'));
  if (!nifti.length) { _showUploadError('No NIfTI files found (.nii / .nii.gz).'); return; }

  pendingModalities = {};
  const patterns = {
    t1ce:  /t1c|t1ce|t1_ce|t1_gd/i,
    t1:    /^t1[^c]|_t1\.|t1\.nii/i,
    t2:    /t2/i,
    flair: /flair|fl/i,
  };

  for (const file of nifti) {
    for (const [mod, pat] of Object.entries(patterns)) {
      if (!pendingModalities[mod] && pat.test(file.name)) {
        pendingModalities[mod] = file;
        break;
      }
    }
  }

  if (nifti.length === 1 && !Object.keys(pendingModalities).length) {
    pendingModalities.t1 = nifti[0];
  }

  _updateModalityStatus(pendingModalities);

  const btn = $('btnAnalyze');
  if (btn) btn.disabled = false;
  hide($('uploadError'));
}

function _updateModalityStatus(mods) {
  const modList = $('modalityList');
  if (modList) show(modList, 'flex');
  modList && (modList.style.flexDirection = 'column');

  const badges = { t1: 'sb-t1', t1ce: 'sb-t1ce', t2: 'sb-t2', flair: 'sb-flair' };
  const modEls = { t1: 'mod-t1', t1ce: 'mod-t1ce', t2: 'mod-t2', flair: 'mod-flair' };

  for (const [mod, elId] of Object.entries(modEls)) {
    const el = $(elId);
    const sb = $(badges[mod]);
    if (mods[mod]) {
      if (el) { el.textContent = '✓'; el.className = 'modality-status ok'; }
      if (sb) { sb.className = 'sb-badge ok'; }
    } else {
      if (el) { el.textContent = '—'; el.className = 'modality-status miss'; }
      if (sb) { sb.className = 'sb-badge miss'; }
    }
  }
}

// Mark all 4 modality badges green (used after demo / 4-modality analysis)
function _markAllModalitiesOk() {
  ['sb-t1', 'sb-t1ce', 'sb-t2', 'sb-flair'].forEach(id => {
    const el = $(id);
    if (el) el.className = 'sb-badge ok';
  });
  const modList = $('modalityList');
  if (modList) show(modList, 'flex');
  ['mod-t1', 'mod-t1ce', 'mod-t2', 'mod-flair'].forEach(id => {
    const el = $(id);
    if (el) { el.textContent = '✓'; el.className = 'modality-status ok'; }
  });
}

function _showUploadError(msg) {
  const err = $('uploadError');
  if (!err) return;
  show(err);
  setText('uploadErrorMsg', msg);
}

// ─── Buttons ──────────────────────────────────────────────────────────────────
function _bindButtons() {
  const btnAn = $('btnAnalyze');
  if (btnAn) btnAn.addEventListener('click', _runAnalysis);

  const btnDemo = $('btnDemo');
  if (btnDemo) btnDemo.addEventListener('click', _runDemo);

  const btnFocus = $('btnFocusTumor');
  if (btnFocus) btnFocus.addEventListener('click', () => window.ui.focusTumor());

  const btnAR = $('btnAR');
  if (btnAR) {
    btnAR.addEventListener('click', () => getViewer()?.enterAR());
    btnAR.disabled = true;
  }
  const btnVR = $('btnVR');
  if (btnVR) {
    btnVR.addEventListener('click', () => getViewer()?.enterVR());
    btnVR.disabled = true;
  }

  const tbFocus = $('tbFocus');
  if (tbFocus) tbFocus.addEventListener('click', () => window.ui.focusTumor());
  const tbReset = $('tbReset');
  if (tbReset) tbReset.addEventListener('click', () => window.ui.resetView());
  const tbSplit = $('tbSplit');
  if (tbSplit) tbSplit.addEventListener('click', () => window.ui.toggleSplitView());
  const tbOrbit = $('tbOrbit');
  if (tbOrbit) {
    tbOrbit.setAttribute('aria-pressed', 'true');
    tbOrbit.addEventListener('click', () => {
      const enabled = getViewer()?.toggleOrbit();
      if (enabled == null) return;
      tbOrbit.classList.toggle('active', enabled);
      tbOrbit.setAttribute('aria-pressed', String(enabled));
      tbOrbit.setAttribute('data-tip', enabled ? 'Orbit / Rotate enabled' : 'Orbit / Rotate paused');
      if (!enabled) {
        getViewer()?.setAutoRotate(false);
        const autoRotate = $('btnAutoRotate');
        if (autoRotate) {
          autoRotate.classList.remove('active');
          autoRotate.setAttribute('aria-pressed', 'false');
          autoRotate.textContent = '⟳ Auto Rotate: Off';
        }
      }
    });
  }
  const btnAutoRotate = $('btnAutoRotate');
  if (btnAutoRotate) {
    btnAutoRotate.addEventListener('click', () => {
      const active = btnAutoRotate.getAttribute('aria-pressed') !== 'true';
      const enabled = getViewer()?.setAutoRotate(active);
      if (enabled == null) return;
      btnAutoRotate.classList.toggle('active', enabled);
      btnAutoRotate.setAttribute('aria-pressed', String(enabled));
      btnAutoRotate.textContent = enabled ? '⟳ Auto Rotate: On' : '⟳ Auto Rotate: Off';
      if (enabled) {
        const orbit = $('tbOrbit');
        if (orbit) {
          orbit.classList.add('active');
          orbit.setAttribute('aria-pressed', 'true');
          orbit.setAttribute('data-tip', 'Orbit / Rotate enabled');
        }
      }
    });
  }
}

// ─── View presets ─────────────────────────────────────────────────────────────
function _bindViewPresets() {
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      const v = btn.dataset.view;
      currentPlane = v === '3d' ? 'axial' : v;
      _onViewChange(v);
    });
  });

  document.querySelectorAll('[data-dir]').forEach(btn => {
    btn.addEventListener('click', () => {
      getViewer()?.setCameraPreset(btn.dataset.dir);
    });
  });

  const slider = $('sliceSlider');
  if (slider) {
    slider.addEventListener('input', () => {
      currentSliceIdx = parseInt(slider.value);
      _updateSliceDisplay();
      if (splitActive) _fetchSlice();
    });
  }
}

function _onViewChange(view) {
  const badge     = $('viewBadgeLabel');
  const viewLabel = $('currentViewLabel');
  const sliceCtrl = $('sliceControl');

  if (view === '3d') {
    if (badge)     badge.textContent = '3D PERSPECTIVE';
    if (viewLabel) viewLabel.textContent = '3D Perspective';
    hide(sliceCtrl);
    getViewer()?.setSectionMode('none');
    _resetSectionButtons();
    // Close split view when switching back to 3D
    if (splitActive) {
      splitActive = false;
      const tbSplit = $('tbSplit');
      if (tbSplit) tbSplit.classList.remove('active');
      _renderSplitView();
    }
  } else {
    const labels = { axial: 'AXIAL VIEW', coronal: 'CORONAL VIEW', sagittal: 'SAGITTAL VIEW' };
    if (badge)     badge.textContent = labels[view] ?? view.toUpperCase();
    if (viewLabel) viewLabel.textContent = labels[view] ?? view;
    show(sliceCtrl, 'flex');
    const sliceLabel = $('sliceLabel');
    if (sliceLabel) sliceLabel.textContent =
      view === 'axial' ? 'Axial Slice' : view === 'coronal' ? 'Coronal Slice' : 'Sagittal Slice';
    _updateSliceRange();

    // Auto-open split view when a 2D tab is selected (if session is active)
    if (currentSessionId && !splitActive) {
      splitActive = true;
      const tbSplit = $('tbSplit');
      if (tbSplit) tbSplit.classList.add('active');
      _renderSplitView();
    } else if (splitActive) {
      _fetchSlice();
    }
  }
}

function _updateSliceRange() {
  if (!sliceInfo) return;
  const slider = $('sliceSlider');
  if (!slider) return;
  const count = {
    axial:    sliceInfo.axial_count,
    coronal:  sliceInfo.coronal_count,
    sagittal: sliceInfo.sagittal_count,
  }[currentPlane] ?? 90;
  slider.max   = count - 1;
  slider.value = Math.floor(count / 2);
  currentSliceIdx = parseInt(slider.value);
  _updateSliceDisplay();
}

function _updateSliceDisplay() {
  const display = $('sliceIndexDisplay');
  const slider  = $('sliceSlider');
  if (display && slider) {
    display.textContent = `${currentSliceIdx} / ${slider.max}`;
  }
}

// ─── Section buttons (inside Visualization panel) ─────────────────────────────
function _bindSectionButtons() {
  document.querySelectorAll('[data-section]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-section]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.section;
      getViewer()?.setSectionMode(mode);
      const clipRow = $('clipRow');
      if (clipRow) clipRow.style.display = mode === 'none' ? 'none' : 'flex';
    });
  });
}

function _resetSectionButtons() {
  document.querySelectorAll('[data-section]').forEach(b => b.classList.remove('active'));
  const noneBtn = document.querySelector('[data-section="none"]');
  if (noneBtn) noneBtn.classList.add('active');
}

// ─── Visibility toggle rows ───────────────────────────────────────────────────
function _bindToggleRows() {
  document.querySelectorAll('[data-toggle]').forEach(el => {
    el.addEventListener('click', () => _handleToggle(el.dataset.toggle));
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') _handleToggle(el.dataset.toggle);
    });
  });
}

function _handleToggle(name) {
  _setVisibility(name, !visibility[name]);
}

// ─── Region buttons + Anatomy card ───────────────────────────────────────────

// Anatomy educational descriptions (not diagnosis — purely reference information)
const REGION_INFO = {
  frontal: {
    name: 'Frontal Lobe',
    color: '#60a5fa',
    location: 'Anterior region of the cerebral hemisphere, anterior to the central sulcus.',
    fn: 'Associated with executive function, voluntary motor control, language production (Broca\'s area), and working memory.',
  },
  parietal: {
    name: 'Parietal Lobe',
    color: '#34d399',
    location: 'Upper posterior region of the cerebral cortex, posterior to the central sulcus.',
    fn: 'Associated with sensory integration, spatial processing, and somatosensory cortex.',
  },
  temporal: {
    name: 'Temporal Lobe',
    color: '#f59e0b',
    location: 'Lateral region of the cerebral hemisphere, inferior to the Sylvian (lateral) fissure.',
    fn: 'Associated with auditory processing, memory formation, and language comprehension (Wernicke\'s area).',
  },
  occipital: {
    name: 'Occipital Lobe',
    color: '#a78bfa',
    location: 'Posterior region of the cerebral hemisphere, at the back of the skull.',
    fn: 'Primary visual cortex — processes visual information from the eyes via the optic radiation.',
  },
  cerebellum: {
    name: 'Cerebellum',
    color: '#f472b6',
    location: 'Inferior-posterior structure, below the cerebral hemispheres and posterior to the brainstem.',
    fn: 'Coordinates voluntary movement, balance, and fine motor control. Contains more neurons than the rest of the brain combined.',
  },
  brainstem: {
    name: 'Brainstem',
    color: '#fb923c',
    location: 'Connects the cerebrum to the spinal cord (midbrain → pons → medulla oblongata).',
    fn: 'Controls vital autonomic functions: breathing, heart rate, blood pressure, and consciousness arousal.',
  },
};

function _showAnatomyCard(regionName) {
  const card = document.getElementById('anatomyInfoCard');
  if (!card) return;
  const info = REGION_INFO[regionName];
  if (!info) { card.style.display = 'none'; return; }

  document.getElementById('anatomyRegionName').textContent = info.name;
  document.getElementById('anatomyRegionLoc').textContent  = info.location;
  document.getElementById('anatomyRegionFn').textContent   = info.fn;

  const dot = document.getElementById('anatomyCardDot');
  if (dot) dot.style.background = info.color;

  // Set left border color to match lobe color
  card.style.borderLeftColor = info.color;
  card.style.display = 'block';
}

function _hideAnatomyCard() {
  const card = document.getElementById('anatomyInfoCard');
  if (card) card.style.display = 'none';
}

function _bindRegionButtons() {
  const allBtn = $('regionAll');
  if (allBtn) {
    allBtn.addEventListener('click', () => {
      allBtn.classList.add('active');
      document.querySelectorAll('[data-region]').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      // Clear 3D highlight and hide anatomy card
      getViewer()?.clearRegionHighlight?.();
      _hideAnatomyCard();
    });
  }

  document.querySelectorAll('[data-region]').forEach(btn => {
    btn.addEventListener('click', () => {
      const region = btn.dataset.region;
      const allBtn2 = $('regionAll');
      if (allBtn2) allBtn2.classList.remove('active');
      const wasActive = btn.classList.contains('active');

      document.querySelectorAll('[data-region]').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });

      if (!wasActive) {
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
        // Highlight in 3D and show anatomy card
        getViewer()?.highlightRegion?.(region);
        _showAnatomyCard(region);
      } else {
        if (allBtn2) allBtn2.classList.add('active');
        getViewer()?.clearRegionHighlight?.();
        _hideAnatomyCard();
      }
    });
  });
}

// ─── Run Analysis ─────────────────────────────────────────────────────────────
async function _runAnalysis() {
  if (!Object.keys(pendingModalities).length) {
    _showUploadError('No files selected. Please upload NIfTI files first.');
    return;
  }
  isDemoMode = false;
  hide($('demoBadge'));

  const formData = new FormData();
  for (const [mod, file] of Object.entries(pendingModalities)) {
    formData.append(mod, file);
  }

  _setProcessingState(true);

  try {
    const upRes = await fetch('/api/upload', { method: 'POST', body: formData });
    const upData = await upRes.json();

    if (!upRes.ok || upData.error) {
      _showUploadError(upData.error ?? 'Upload failed.');
      _setProcessingState(false);
      return;
    }

    currentSessionId = upData.session_id;
    setText('headerSession', `Session: ${currentSessionId.slice(0, 12)}`);
    _updateModalityStatus(upData.uploaded_modalities.reduce((a, m) => { a[m] = true; return a; }, {}));

    await fetch(`/api/analyze/${currentSessionId}`, { method: 'POST' });
    _startPolling();
  } catch (err) {
    _showUploadError(`Network error: ${err.message}`);
    _setProcessingState(false);
  }
}

// ─── Demo Mode ────────────────────────────────────────────────────────────────
async function _runDemo() {
  isDemoMode = true;
  _setProcessingState(true);

  // Reset view to 3D before starting
  _switchToView('3d');

  try {
    const r  = await fetch('/api/demo');
    const d  = await r.json();
    if (!r.ok || d.error) { _showUploadError(d.error ?? 'Demo failed.'); _setProcessingState(false); return; }

    currentSessionId = d.session_id;
    setText('headerSession', `Demo: ${currentSessionId.slice(0, 12)}`);

    // Show demo badge (using inline style, no .hidden class conflict)
    const badge = $('demoBadge');
    if (badge) badge.style.display = 'block';

    if (d.status === 'complete') {
      _setProcessingState(false);
      await _onAnalysisComplete(d);
    } else {
      _startPolling();
    }
  } catch (err) {
    _showUploadError(`Demo error: ${err.message}`);
    _setProcessingState(false);
  }
}

// ─── Polling ──────────────────────────────────────────────────────────────────
function _startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollFailureCount = 0;
  pollInterval = setInterval(_poll, 1500);
}

async function _poll() {
  if (!currentSessionId) return;
  try {
    const r = await fetch(`/api/status/${currentSessionId}`);
    if (!r.ok) throw new Error(`Status request failed (${r.status})`);
    const d = await r.json();
    pollFailureCount = 0;

    _updatePipelineSteps(d.progress ?? 0, d.message ?? '');

    if (d.status === 'complete') {
      clearInterval(pollInterval);
      pollInterval = null;
      _setProcessingState(false);
      await _onAnalysisComplete(d);
    } else if (d.status === 'error') {
      clearInterval(pollInterval);
      pollInterval = null;
      _setProcessingState(false);
      _showAnalysisError(d.error ?? 'Unknown pipeline error.');
    }
  } catch (err) {
    pollFailureCount += 1;
    if (pollFailureCount >= 3) {
      clearInterval(pollInterval);
      pollInterval = null;
      _setProcessingState(false);
      _showAnalysisError('Backend connection lost. Check that the NeuroVR server is running, then try again.');
      const dot = $('systemDot');
      if (dot) dot.className = 'status-dot error';
      setText('systemStatusText', 'OFFLINE');
      console.error('[UI] Analysis status polling failed:', err);
    }
  }
}

function _updatePipelineSteps(progress, message) {
  const dot = $('systemDot');
  if (dot) { dot.className = 'status-dot processing'; }

  const fill = $('progressFill');
  if (fill) fill.style.width = `${progress}%`;

  const steps = [
    { id: 'step-load',       threshold: 15 },
    { id: 'step-preprocess', threshold: 25 },
    { id: 'step-segment',    threshold: 65 },
    { id: 'step-measure',    threshold: 75 },
    { id: 'step-localize',   threshold: 80 },
    { id: 'step-mesh',       threshold: 98 },
  ];

  steps.forEach(({ id, threshold }) => {
    const el  = $(id);
    if (!el) return;
    const ico = el.querySelector('.step-icon');
    if (progress >= threshold) {
      el.className = 'pipeline-step done';
      if (ico) ico.textContent = '✓';
    } else if (progress >= threshold - 15) {
      el.className = 'pipeline-step active';
      if (ico) ico.innerHTML = '<span class="step-spinner"></span>';
    } else {
      el.className = 'pipeline-step';
      if (ico) ico.textContent = '○';
    }
  });
}

// ─── Analysis complete ────────────────────────────────────────────────────────
async function _onAnalysisComplete(statusData) {
  const sid = currentSessionId;

  // Reset to 3D view first
  _switchToView('3d');

  // Mark all modality badges green
  _markAllModalitiesOk();

  // Load meshes into viewer — retry a few times in case viewer.js is still initializing
  let loaded = false;
  for (let attempt = 0; attempt < 5; attempt++) {
    const v = getViewer();
    if (v && typeof v.loadSession === 'function') {
      await v.loadSession(sid);
      loaded = true;
      break;
    }
    console.warn(`[UI] viewer not ready, retrying (attempt ${attempt + 1})...`);
    await new Promise(r => setTimeout(r, 300));
  }
  if (!loaded) {
    console.error('[UI] viewer.loadSession unavailable after retries');
  } else {
    document.querySelectorAll('[data-mode]').forEach(button => { button.disabled = false; });
    const autoRotate = $('btnAutoRotate');
    if (autoRotate) autoRotate.disabled = false;
    const activeMode = document.querySelector('[data-mode].active')?.dataset.mode ?? 'tumor';
    _applyVisualizationMode(activeMode);
  }

  // Fetch measurements
  try {
    const r = await fetch(`/api/results/${sid}`);
    const d = await r.json();
    _populateMeasurements(d);
    setText('sb-inference', d.inference_time_s != null ? `${d.inference_time_s}s` : '—');
  } catch {}

  // Fetch localization
  try {
    const r = await fetch(`/api/localization/${sid}`);
    if (r.ok) {
      const d = await r.json();
      localizationData = d;
      _populateLocalization(d);
    }
  } catch {}

  // Fetch slice info (for 2D viewer)
  try {
    const r = await fetch(`/api/slice_info/${sid}`);
    if (r.ok) {
      sliceInfo = await r.json();
      _updateSliceRange();
    }
  } catch {}

  // Enable focus tumor button
  const btnFocus = $('btnFocusTumor');
  if (btnFocus) btnFocus.disabled = false;

  // Update system status pill
  const dot = $('systemDot');
  if (dot) { dot.className = 'status-dot'; dot.style.background = 'var(--accent-green)'; }
  setText('systemStatusText', 'ANALYSIS DONE');
}

// ─── Switch active view tab programmatically ──────────────────────────────────
function _switchToView(view) {
  document.querySelectorAll('[data-view]').forEach(b => {
    b.classList.remove('active');
    b.setAttribute('aria-pressed', 'false');
  });
  const btn = document.querySelector(`[data-view="${view}"]`);
  if (btn) {
    btn.classList.add('active');
    btn.setAttribute('aria-pressed', 'true');
  }
  currentPlane = view === '3d' ? 'axial' : view;

  // Update badge labels and hide slice control
  const badge     = $('viewBadgeLabel');
  const viewLabel = $('currentViewLabel');
  const sliceCtrl = $('sliceControl');
  if (view === '3d') {
    if (badge)     badge.textContent = '3D PERSPECTIVE';
    if (viewLabel) viewLabel.textContent = '3D Perspective';
    hide(sliceCtrl);
  }
  _resetSectionButtons();
}

// ─── Measurements ─────────────────────────────────────────────────────────────
function _populateMeasurements(data) {
  const detected = data.tumor_detected ?? false;
  const icon     = $('detectionIcon');
  const text     = $('detectionText');
  const sub      = $('detectionSub');

  if (detected) {
    if (icon) icon.textContent = '🔴';
    if (text) { text.textContent = 'TUMOR DETECTED'; text.className = 'detection-text detected'; }
    if (sub)  sub.textContent = 'AI segmentation complete';
  } else {
    if (icon) icon.textContent = '🟢';
    if (text) { text.textContent = 'NO TUMOR DETECTED'; text.className = 'detection-text clear'; }
    if (sub)  sub.textContent = 'Segmentation complete — no significant regions found';
  }

  const meas = data.regions ?? {};
  const wt   = meas.whole_tumor?.volume_cm3  ?? 0;
  const tc   = meas.tumor_core?.volume_cm3   ?? 0;
  const et   = meas.enhancing_tumor?.volume_cm3 ?? 0;

  setText('measWT', wt.toFixed(2));
  setText('measTC', tc.toFixed(2));
  setText('measET', et.toFixed(2));

  // Populate tumor type classification
  _populateTumorType(data.tumor_type ?? null);

  const measSec = $('measSection');
  if (measSec) show(measSec, 'flex');
  measSec && (measSec.style.flexDirection = 'column');

  const cRAS = data.summary?.wt_centroid?.centroid_ras_mm;
  if (cRAS) {
    setText('coordX', cRAS[0].toFixed(1));
    setText('coordY', cRAS[1].toFixed(1));
    setText('coordZ', cRAS[2].toFixed(1));
  }

  const spacing = data.voxel_spacing_mm;
  if (spacing) {
    const sp = spacing.map(v => v.toFixed(2)).join(' × ');
    setText('voxelSpacing', `${sp} mm`);
  }

  const spatialSec = $('spatialSection');
  if (spatialSec) show(spatialSec, 'block');
}

// ─── Tumor Type Classification ────────────────────────────────────────────────
function _populateTumorType(tt) {
  const patternEl    = $('typePattern');
  const subtypeEl    = $('typeSubtype');
  const gradeRowEl   = $('typeGradeRow');
  const gradeEl      = $('typeGrade');
  const confEl       = $('typeConf');
  const basisEl      = $('typeBasis');
  const disclaimerEl = $('typeDisclaimer');

  if (!patternEl) return;

  if (!tt || !tt.pattern) {
    patternEl.textContent = 'Not classified';
    patternEl.style.color = 'var(--text-muted)';
    return;
  }

  // Pattern label (main)
  patternEl.textContent = tt.pattern;
  patternEl.style.color = tt.color ?? 'var(--text-primary)';

  // Subtype
  if (tt.subtype && subtypeEl) {
    subtypeEl.textContent = tt.subtype;
    show(subtypeEl, 'block');
  }

  // WHO grade row
  if (tt.who_grade && gradeRowEl && gradeEl) {
    gradeEl.textContent = tt.who_grade;
    show(gradeRowEl, 'flex');
    if (confEl && tt.confidence) {
      confEl.textContent = tt.confidence + ' confidence';
    }
  }

  // Basis (technical detail — collapsed by default via small text)
  if (tt.basis && basisEl) {
    basisEl.textContent = tt.basis;
    show(basisEl, 'block');
  }

  // Disclaimer
  if (tt.disclaimer && disclaimerEl) {
    disclaimerEl.textContent = '⚠ ' + tt.disclaimer;
    show(disclaimerEl, 'block');
  }
}

// ─── Localization ─────────────────────────────────────────────────────────────
function _populateLocalization(d) {
  const card = $('locationCard');
  if (card) show(card, 'block');

  const side = d.side ?? 'UNCERTAIN';
  const lobe = d.lobe_available ? (d.lobe ?? null) : null;
  const conf = d.side_confidence ?? 'Low';

  setText('locSide', side === 'UNCERTAIN' ? 'Location Uncertain' : side + ' HEMISPHERE');
  setText('locLobe', lobe ?? (side === 'MIDLINE' ? '(Midline / Deep)' : ''));

  const confEl  = $('locConf');
  const confBdg = $('locConfBadge');
  if (confEl && confBdg) {
    show(confEl, 'flex');
    confBdg.textContent = conf;
    confBdg.className   = `conf-badge ${conf.toLowerCase()}`;
  }

  const estEl = $('locEst');
  if (estEl && lobe) show(estEl, 'block');

  const calloutSide = $('calloutSide');
  const calloutVol  = $('calloutVol');
  if (calloutSide) {
    calloutSide.textContent = d.full_label ?? side;
  }

  if (d.centroid_ras_mm) {
    const [x, y, z] = d.centroid_ras_mm;
    if (calloutVol) calloutVol.textContent = '';
    getViewer()?.setCentroid(x, y, z);
  }

  setTimeout(() => {
    const wt = $('measWT')?.textContent;
    if (wt && calloutVol) calloutVol.textContent = `${wt} cm³`;
  }, 200);
}

// ─── Analysis error ───────────────────────────────────────────────────────────
function _showAnalysisError(msg) {
  const text = $('detectionText');
  const sub  = $('detectionSub');
  const icon = $('detectionIcon');

  if (icon) icon.textContent = '⚠';
  if (text) { text.textContent = 'ANALYSIS FAILED'; text.className = 'detection-text'; text.style.color = 'var(--accent-amber)'; }
  if (sub)  {
    sub.innerHTML = `
      <div class="error-card" style="margin-top:6px">
        <span class="error-title">Pipeline Error</span>
        <span class="error-msg">${_sanitize(msg)}</span>
      </div>
    `;
  }
}

function _sanitize(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ─── Processing state ─────────────────────────────────────────────────────────
function _setProcessingState(active) {
  const btn  = $('btnAnalyze');
  const prog = $('pipelineProgress');
  const dot  = $('systemDot');

  if (btn)  btn.disabled = active;
  if (prog) { prog.style.display = active ? 'flex' : 'none'; prog.style.flexDirection = 'column'; }

  if (active) {
    if (dot) { dot.className = 'status-dot processing'; }
    setText('systemStatusText', 'ANALYZING');
    setText('detectionText', 'ANALYZING…');
    $('detectionText')?.setAttribute('class', 'detection-text waiting');
    setText('detectionSub', 'Pipeline running');
    $('detectionIcon') && ($('detectionIcon').textContent = '⚙');
  }
}

// ─── Public API (window.ui) ───────────────────────────────────────────────────
window.ui = {
  setBrainOpacity(val) {
    const v = parseInt(val) / 100;
    setText('valBrainOpacity', `${val}%`);
    getViewer()?.setOpacity('brain', v);
  },

  setTumorOpacity(val) {
    const v = parseInt(val) / 100;
    setText('valTumorOpacity', `${val}%`);
    getViewer()?.setAllTumorOpacity(v);
  },

  setClipPosition(val) {
    setText('valClip', `${val}%`);
    getViewer()?.setClipPosition(val);
  },

  focusTumor() {
    getViewer()?.focusTumor();
  },

  resetView() {
    getViewer()?.resetCamera();
  },

  toggleSplitView() {
    splitActive = !splitActive;
    const tbSplit = $('tbSplit');
    if (tbSplit) tbSplit.classList.toggle('active', splitActive);
    _renderSplitView();
    if (splitActive && currentSessionId) _fetchSlice();
  },
};

// ─── 2D Slice Viewer ──────────────────────────────────────────────────────────
function _initSliceViewer() {
  // Slice canvas created lazily in _renderSplitView
}

function _renderSplitView() {
  const viewerCol = $('viewerCol');
  if (!viewerCol) return;

  const canvas3d  = viewerCol.querySelector('canvas');  // Three.js canvas
  let slicePanel  = viewerCol.querySelector('.slice-panel');

  if (splitActive) {
    // ── Build slice panel once ────────────────────────────────────
    if (!slicePanel) {
      slicePanel = document.createElement('div');
      slicePanel.className = 'slice-panel';
      slicePanel.innerHTML = `
        <div class="slice-header">
          <span class="slice-label" id="slicePlaneLabel">AXIAL</span>
          <div class="slice-header-controls">
            <button class="slice-plane-btn" id="slicePlaneToggle" title="Cycle plane">⟳ Plane</button>
            <button class="slice-close-btn" id="sliceClose" title="Close split view">✕</button>
          </div>
        </div>
        <canvas id="sliceCanvas" style="width:100%;height:auto;display:block;" aria-label="2D MRI slice view"></canvas>
        <div class="slice-toolbar">
          <input type="range" id="sliceSlider2" min="0" max="90" value="45"
                 style="flex:1" aria-label="2D slice position" />
          <span class="slice-index" id="sliceIdx2">45 / 90</span>
        </div>
        <div class="slice-orient-bar" id="sliceOrient">R ← → L · A ↑ ↓ P</div>
      `;
      viewerCol.appendChild(slicePanel);

      // Slider input
      const slider2 = slicePanel.querySelector('#sliceSlider2');
      if (slider2) {
        slider2.addEventListener('input', () => {
          currentSliceIdx = parseInt(slider2.value);
          _fetchSlice();
        });
      }

      // Close button
      const closeBtn = slicePanel.querySelector('#sliceClose');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          splitActive = false;
          const tbSplit = $('tbSplit');
          if (tbSplit) tbSplit.classList.remove('active');
          _renderSplitView();
        });
      }

      // Plane-cycle button
      const planes = ['axial', 'coronal', 'sagittal'];
      const planeBtn = slicePanel.querySelector('#slicePlaneToggle');
      if (planeBtn) {
        planeBtn.addEventListener('click', () => {
          const idx  = planes.indexOf(currentPlane === '3d' ? 'axial' : currentPlane);
          const next = planes[(idx + 1) % planes.length];
          currentPlane = next;
          const planeLbl = $('slicePlaneLabel');
          if (planeLbl) planeLbl.textContent = next.toUpperCase();
          _fetchSlice();
        });
      }
    }

    // ── Show panel + resize 3D canvas ────────────────────────────
    slicePanel.style.display = 'flex';
    viewerCol.classList.add('split-active');

    // Sync slider range from slice info
    const slider2 = slicePanel.querySelector('#sliceSlider2');
    if (slider2 && sliceInfo) {
      const count = {
        axial:    sliceInfo.axial_count,
        coronal:  sliceInfo.coronal_count,
        sagittal: sliceInfo.sagittal_count,
      }[currentPlane === '3d' ? 'axial' : currentPlane] ?? 90;
      slider2.max   = count - 1;
      slider2.value = Math.floor(count / 2);
      currentSliceIdx = parseInt(slider2.value);
      const idxLbl = slicePanel.querySelector('#sliceIdx2');
      if (idxLbl) idxLbl.textContent = `${currentSliceIdx} / ${slider2.max}`;
    }

    // Update plane label
    const planeLbl = slicePanel.querySelector('#slicePlaneLabel');
    if (planeLbl) planeLbl.textContent = (currentPlane === '3d' ? 'AXIAL' : currentPlane.toUpperCase());

    // Tell Three.js to resize into the new smaller canvas area
    setTimeout(() => {
      getViewer()?.resize?.();
      if (currentSessionId) _fetchSlice();
    }, 50);

  } else {
    // ── Hide panel + restore full-width 3D ───────────────────────
    if (slicePanel) slicePanel.style.display = 'none';
    viewerCol.classList.remove('split-active');
    setTimeout(() => getViewer()?.resize?.(), 50);
  }
}

async function _fetchSlice() {
  if (!currentSessionId || !splitActive) return;
  const plane = currentPlane === '3d' ? 'axial' : currentPlane;

  try {
    const r = await fetch(`/api/slice/${currentSessionId}/${plane}/${currentSliceIdx}`);
    if (!r.ok) return;

    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);

    const canvas = document.getElementById('sliceCanvas');
    if (!canvas) return;

    const img = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      _drawOrientLabels(ctx, canvas.width, canvas.height, plane);
    };
    img.src = url;

    // Update labels
    const planeLbl = document.getElementById('slicePlaneLabel');
    if (planeLbl) planeLbl.textContent = plane.toUpperCase();
    const idxLbl = document.getElementById('sliceIdx2');
    if (idxLbl) {
      const slider2 = document.getElementById('sliceSlider2');
      idxLbl.textContent = `${currentSliceIdx} / ${slider2?.max ?? '?'}`;
    }
    const orientLbl = document.getElementById('sliceOrient');
    if (orientLbl) {
      const labels = {
        axial:    'R ← → L  ·  A ↑ ↓ P',
        coronal:  'L ← → R  ·  S ↑ ↓ I',
        sagittal: 'A ← → P  ·  S ↑ ↓ I',
      };
      orientLbl.textContent = labels[plane] ?? '';
    }

    // Sync main slice slider
    const mainSlider = $('sliceSlider');
    if (mainSlider) mainSlider.value = currentSliceIdx;
    _updateSliceDisplay();

  } catch {}
}

function _drawOrientLabels(ctx, w, h, plane) {
  const labels = {
    axial:    { top: 'A', bottom: 'P', left: 'R', right: 'L' },
    coronal:  { top: 'S', bottom: 'I', left: 'L', right: 'R' },
    sagittal: { top: 'S', bottom: 'I', left: 'A', right: 'P' },
  }[plane] ?? {};

  ctx.font         = 'bold 14px Inter, sans-serif';
  ctx.fillStyle    = 'rgba(100,160,255,0.9)';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';

  const pad = 16;
  if (labels.top)    ctx.fillText(labels.top,    w / 2,     pad);
  if (labels.bottom) ctx.fillText(labels.bottom, w / 2,     h - pad);
  if (labels.left)   ctx.fillText(labels.left,   pad,       h / 2);
  if (labels.right)  ctx.fillText(labels.right,  w - pad,   h / 2);
}
