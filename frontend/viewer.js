/**
 * NeuroVR 3D — Three.js Viewer  (v4 — Anatomical Realism Upgrade)
 * Medical Imaging Workstation
 *
 * Improvements in this version:
 *  - Warm anatomical PBR material with sheen (subsurface-like tissue appearance)
 *  - 5-light medical visualization rig (key/fill/rim/inferior/ambient)
 *  - PMREMGenerator RoomEnvironment for image-based lighting
 *  - 10 camera view presets with smooth animated transitions
 *  - Anatomical region highlight system (Frontal/Parietal/Temporal/Occipital/Cerebellum/Brainstem)
 *  - All v3 functionality preserved (XR, gizmo, sections, focus tumor, opacity, slices)
 */

import * as THREE from 'three';
import { OrbitControls }            from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader }               from 'three/addons/loaders/GLTFLoader.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { RoomEnvironment }          from 'three/addons/environments/RoomEnvironment.js';

// Tissue colours
const COLORS = {
  brain:           0xc8b89a,
  tumor_whole:     0x0ea5e9,
  tumor_core:      0xef4444,
  tumor_enhancing: 0xf59e0b,
};

const LOBE_COLORS = {
  frontal:    0x60a5fa,
  parietal:   0x34d399,
  temporal:   0xf59e0b,
  occipital:  0xa78bfa,
  cerebellum: 0xf472b6,
  brainstem:  0xfb923c,
};

const INITIAL_OPACITY = {
  brain:           0.88,
  tumor_whole:     0.72,
  tumor_core:      0.90,
  tumor_enhancing: 0.96,
};

const SECTIONS = {
  none:     { normal: null,                       label: '3D PERSPECTIVE' },
  axial:    { normal: new THREE.Vector3(0, 1, 0), label: 'AXIAL VIEW' },
  coronal:  { normal: new THREE.Vector3(0, 0, 1), label: 'CORONAL VIEW' },
  sagittal: { normal: new THREE.Vector3(1, 0, 0), label: 'SAGITTAL VIEW' },
};

const CAM_PRESETS = {
  front:      { pos: [ 0,    0.05,  1   ], up: [0,  1,  0], label: 'ANTERIOR VIEW'   },
  back:       { pos: [ 0,    0.05, -1   ], up: [0,  1,  0], label: 'POSTERIOR VIEW'  },
  left:       { pos: [-1,    0.05,  0   ], up: [0,  1,  0], label: 'LEFT LATERAL'    },
  right:      { pos: [ 1,    0.05,  0   ], up: [0,  1,  0], label: 'RIGHT LATERAL'   },
  top:        { pos: [ 0,    1,     0   ], up: [0,  0, -1], label: 'SUPERIOR VIEW'   },
  bottom:     { pos: [ 0,   -1,     0   ], up: [0,  0,  1], label: 'INFERIOR VIEW'   },
  frontLeft:  { pos: [-0.6,  0.2,   0.8 ], up: [0,  1,  0], label: 'ANT-LEFT VIEW'   },
  frontRight: { pos: [ 0.6,  0.2,   0.8 ], up: [0,  1,  0], label: 'ANT-RIGHT VIEW'  },
  default3d:  { pos: [ 0.12, 0.22,  1.0 ], up: [0,  1,  0], label: '3D PERSPECTIVE'  },
};

// Lobe bounding fractions within brain bounding box (x/y/z fractions from center)
const LOBE_SCENE_FRACS = {
  frontal:    { xFrac: [-1, 1], yFrac: [0.0,  1.0], zFrac: [0.05, 1.0]  },
  parietal:   { xFrac: [-1, 1], yFrac: [0.4,  1.0], zFrac: [-0.6, 0.3]  },
  temporal:   { xFrac: [-1, 1], yFrac: [-0.5, 0.4], zFrac: [-0.5, 0.6]  },
  occipital:  { xFrac: [-1, 1], yFrac: [0.0,  0.8], zFrac: [-1.0,-0.4]  },
  cerebellum: { xFrac: [-1, 1], yFrac: [-1.0,-0.3], zFrac: [-1.0,-0.1]  },
  brainstem:  { xFrac: [-0.25, 0.25], yFrac: [-1.0, 0.1], zFrac: [-0.5, 0.1] },
};

class NeuroVRViewer {
  constructor(canvas) {
    this.canvas   = canvas;
    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.controls = null;
    this.clock    = new THREE.Clock();

    this._meshes     = {};
    this._sessionId  = null;
    this._xrActive   = false;
    this._arGroup    = new THREE.Group();

    this._sectionMode   = 'none';
    this._clipPlane     = new THREE.Plane();
    this._sectionOffset = 0;
    this._brainBounds   = new THREE.Box3();
    this._sectionCap    = null;

    this._centroidMarker     = null;
    this._centroidWorld      = null;
    this._pendingCentroidRAS = null;

    this._camTransition  = null;
    this._regionOverlay  = null;

    this._gizmoRenderer = null;
    this._gizmoScene    = null;
    this._gizmoCamera   = null;
    this._animId        = null;
  }

  init() {
    const col = this.canvas.parentElement;
    const W   = col.clientWidth  || window.innerWidth;
    const H   = col.clientHeight || window.innerHeight;

    this.renderer = new THREE.WebGLRenderer({
      canvas:          this.canvas,
      antialias:       true,
      alpha:           true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(W, H);
    this.renderer.shadowMap.enabled    = true;
    this.renderer.shadowMap.type       = THREE.PCFSoftShadowMap;
    this.renderer.xr.enabled           = true;
    this.renderer.localClippingEnabled = true;
    this.renderer.setClearColor(0x080d1a, 1.0);
    this.renderer.toneMapping          = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure  = 1.1;
    this.renderer.outputColorSpace     = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x080d1a, 4.0, 14.0);

    // PBR Environment (Image-Based Lighting)
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    pmrem.compileEquirectangularShader();
    const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = envTexture;
    pmrem.dispose();

    this.camera = new THREE.PerspectiveCamera(38, W / H, 0.001, 100);
    this.camera.position.set(0.04, 0.07, 0.42);

    // 1. Key light — warm, above-front, reveals gyri depth via soft shadows
    const key = new THREE.DirectionalLight(0xfff8f0, 1.8);
    key.position.set(1.2, 2.8, 1.8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near   = 0.01;
    key.shadow.camera.far    = 5.0;
    key.shadow.camera.top    =  0.5;
    key.shadow.camera.bottom = -0.5;
    key.shadow.camera.left   = -0.5;
    key.shadow.camera.right  =  0.5;
    key.shadow.bias = -0.001;
    this.scene.add(key);

    // 2. Fill light — cool, left-below, no harsh shadows on left hemisphere
    const fill = new THREE.DirectionalLight(0xe8f0ff, 0.55);
    fill.position.set(-2.0, -0.8, 1.0);
    this.scene.add(fill);

    // 3. Rim light — blue-white from behind, defines brain silhouette
    const rim = new THREE.DirectionalLight(0xb0c8ff, 0.65);
    rim.position.set(0.0, 0.6, -2.8);
    this.scene.add(rim);

    // 4. Inferior fill — prevents underside blackout (cerebellum visibility)
    const bottomL = new THREE.DirectionalLight(0xfff0e8, 0.35);
    bottomL.position.set(0.2, -2.5, 0.5);
    this.scene.add(bottomL);

    // 5. Ambient — warm overall scientific clarity
    const ambient = new THREE.AmbientLight(0xe8eeff, 0.45);
    this.scene.add(ambient);

    this.scene.add(this._arGroup);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.autoRotate     = false;
    this.controls.autoRotateSpeed = 1.25;
    this.controls.minDistance   = 0.04;
    this.controls.maxDistance   = 2.0;

    this._initGizmo();
    this.renderer.setAnimationLoop(this._animate.bind(this));
    window.addEventListener('resize', this._onResize.bind(this));
    setTimeout(() => this._onResize(), 100);
    this._setupXR();
    console.log('[NeuroVR] Viewer v4 ready — Anatomical Realism Mode');
  }

  _initGizmo() {
    const gCanvas = document.getElementById('gizmoCanvas');
    if (!gCanvas) return;

    this._gizmoRenderer = new THREE.WebGLRenderer({ canvas: gCanvas, antialias: true, alpha: true });
    this._gizmoRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._gizmoRenderer.setSize(80, 80);
    this._gizmoRenderer.setClearColor(0x000000, 0);

    this._gizmoScene  = new THREE.Scene();
    this._gizmoCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 50);
    this._gizmoCamera.position.set(0, 0, 3);

    const addAxis = (dir, color, label) => {
      const mat  = new THREE.MeshBasicMaterial({ color });
      const geo  = new THREE.CylinderGeometry(0.04, 0.04, 0.9, 8);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(dir.clone().multiplyScalar(0.45));
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      this._gizmoScene.add(mesh);
      const arrowGeo = new THREE.ConeGeometry(0.08, 0.22, 8);
      const arrow    = new THREE.Mesh(arrowGeo, mat);
      arrow.position.copy(dir.clone().multiplyScalar(0.9));
      arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      this._gizmoScene.add(arrow);
      this._addGizmoLabel(dir.clone().multiplyScalar(1.12), label, color);
    };

    addAxis(new THREE.Vector3( 1, 0, 0), 0xff4444, 'R');
    addAxis(new THREE.Vector3(-1, 0, 0), 0xff9999, 'L');
    addAxis(new THREE.Vector3( 0, 0,-1), 0x44cc44, 'A');
    addAxis(new THREE.Vector3( 0, 0, 1), 0x88cc88, 'P');
    addAxis(new THREE.Vector3( 0, 1, 0), 0x4488ff, 'S');
    addAxis(new THREE.Vector3( 0,-1, 0), 0x88aaff, 'I');

    const cGeo = new THREE.SphereGeometry(0.1, 8, 8);
    const cMat = new THREE.MeshBasicMaterial({ color: 0xffffff, opacity: 0.6, transparent: true });
    this._gizmoScene.add(new THREE.Mesh(cGeo, cMat));
  }

  _addGizmoLabel(pos, text, color) {
    const SIZE = 64;
    const cv = document.createElement('canvas');
    cv.width = cv.height = SIZE;
    const ctx = cv.getContext('2d');
    ctx.fillStyle    = '#' + color.toString(16).padStart(6, '0');
    ctx.font         = 'bold 36px Inter, sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, SIZE / 2, SIZE / 2);
    const tex    = new THREE.CanvasTexture(cv);
    const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.35, 0.35, 1);
    sprite.position.copy(pos);
    this._gizmoScene.add(sprite);
  }

  _renderGizmo() {
    if (!this._gizmoRenderer || !this._gizmoScene) return;
    this._gizmoCamera.position.copy(
      this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(3)
    );
    this._gizmoCamera.lookAt(0, 0, 0);
    this._gizmoCamera.up.copy(this.camera.up);
    this._gizmoRenderer.render(this._gizmoScene, this._gizmoCamera);
  }

  setCentroid(rasX, rasY, rasZ) {
    this._pendingCentroidRAS = [rasX, rasY, rasZ];
    this._placeCentroidMarker();
  }

  _placeCentroidMarker() {
    if (!this._pendingCentroidRAS || this._brainBounds.isEmpty()) return;
    const [rx, ry, rz] = this._pendingCentroidRAS;

    if (this._centroidMarker) {
      this.scene.remove(this._centroidMarker);
      this._centroidMarker.traverse(c => {
        if (c.geometry) c.geometry.dispose();
        if (c.material) c.material.dispose();
      });
      this._centroidMarker = null;
    }

    // Mesh vertices use the same NIfTI affine and RAS-to-Three mapping:
    // Three X=RAS X, Three Y=RAS Z, Three Z=-RAS Y (millimetres to metres).
    const x = rx / 1000;
    const y = rz / 1000;
    const z = -ry / 1000;
    this._centroidWorld = new THREE.Vector3(x, y, z);

    // Pulsing torus ring marker
    const geo = new THREE.TorusGeometry(0.006, 0.002, 8, 24);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
    const ring = new THREE.Mesh(geo, mat);
    ring.position.copy(this._centroidWorld);
    ring.renderOrder = 10;
    this._centroidMarker = ring;
    this.scene.add(ring);

    const sGeo = new THREE.SphereGeometry(0.003, 12, 12);
    const sMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const sph  = new THREE.Mesh(sGeo, sMat);
    sph.position.set(0, 0, 0);
    sph.renderOrder = 11;
    ring.add(sph);
  }

  focusTumor() {
    if (!this._centroidWorld) return;
    const target = this._centroidWorld.clone();
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this._animateCameraTo(
      target.clone().add(dir.multiplyScalar(0.18)),
      target,
      this.camera.up.clone(),
      600
    );
  }

  _updateCallout() {
    const callout = document.getElementById('tumorCallout');
    if (!callout || !this._centroidWorld) return;
    const vec = this._centroidWorld.clone().project(this.camera);
    const W = this.canvas.clientWidth;
    const H = this.canvas.clientHeight;
    const x = (vec.x  * 0.5 + 0.5) * W;
    const y = (-vec.y * 0.5 + 0.5) * H;
    if (vec.z > 1) { callout.style.display = 'none'; return; }
    callout.style.display = 'block';
    callout.style.left    = x + 'px';
    callout.style.top     = y + 'px';
  }

  _animateCameraTo(targetPos, targetLookAt, targetUp, durationMs) {
    durationMs = durationMs || 700;
    this._camTransition = {
      startPos:    this.camera.position.clone(),
      startTarget: this.controls.target.clone(),
      startUp:     this.camera.up.clone(),
      endPos:      targetPos,
      endTarget:   targetLookAt,
      endUp:       targetUp,
      startTime:   performance.now(),
      duration:    durationMs,
    };
  }

  _tickCameraTransition() {
    if (!this._camTransition) return;
    const t        = this._camTransition;
    const elapsed  = performance.now() - t.startTime;
    const progress = Math.min(elapsed / t.duration, 1.0);
    const ease     = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;

    this.camera.position.lerpVectors(t.startPos, t.endPos, ease);
    this.controls.target.lerpVectors(t.startTarget, t.endTarget, ease);
    this.camera.up.lerpVectors(t.startUp, t.endUp, ease).normalize();
    this.controls.update();

    if (progress >= 1.0) this._camTransition = null;
  }

  _animate() {
    if (!this._xrActive) {
      this._tickCameraTransition();
      if (!this._camTransition) this.controls.update();
      if (this._centroidMarker) {
        const s = 1 + 0.08 * Math.sin(performance.now() * 0.003);
        this._centroidMarker.scale.setScalar(s);
      }
    }
    this.renderer.render(this.scene, this.camera);
    this._renderGizmo();
    this._updateCallout();
  }

  _onResize() {
    // Use the canvas's own CSS-computed size (respects flex/split layout)
    const W = this.canvas.clientWidth  || this.canvas.parentElement?.clientWidth  || window.innerWidth;
    const H = this.canvas.clientHeight || this.canvas.parentElement?.clientHeight || window.innerHeight;
    if (W === 0 || H === 0) return;
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(W, H, false); // false = don't set CSS size (CSS controls that)
  }

  /** Public alias — called by ui.js after toggling split view */
  resize() { this._onResize(); }

  async loadSession(sessionId) {
    this._sessionId = sessionId;
    this._clearMeshes();
    this._sectionMode = 'none';
    this.renderer.clippingPlanes = [];

    const empty = document.getElementById('viewerEmpty');
    if (empty) empty.style.display = 'none';
    this._onResize();

    const loader = new GLTFLoader();
    const names  = ['brain', 'tumor_whole', 'tumor_core', 'tumor_enhancing'];
    let loadedCount = 0;

    for (const name of names) {
      try {
        const gltf = await new Promise((res, rej) =>
          loader.load('/api/mesh/' + sessionId + '/' + name, res, undefined, rej)
        );
        const mesh = this._setupMesh(gltf.scene, name);
        this._meshes[name] = mesh;
        this.scene.add(mesh);
        loadedCount++;
        console.log('[NeuroVR] Loaded mesh: ' + name);
      } catch (err) {
        console.warn('[NeuroVR] Mesh not available: ' + name, err && err.message ? err.message : '');
      }
    }

    if (loadedCount === 0) {
      console.error('[NeuroVR] No meshes loaded — showing empty state');
      if (empty) {
        empty.style.display = '';
        const sub = empty.querySelector('.viewer-empty-sub');
        if (sub) sub.textContent = 'No 3D meshes available for this session.';
      }
      return;
    }

    this._brainBounds = new THREE.Box3();
    Object.values(this._meshes).forEach(m => { if (m) this._brainBounds.expandByObject(m); });

    this._fitCamera();
    this._onResize();
    this._placeCentroidMarker();

    console.log('[NeuroVR] Session loaded: ' + loadedCount + ' meshes');
  }

  _setupMesh(group, name) {
    const color   = COLORS[name]          !== undefined ? COLORS[name]          : 0xffffff;
    const opacity = INITIAL_OPACITY[name] !== undefined ? INITIAL_OPACITY[name] : 0.8;
    const isBrain = name === 'brain';
    
    // Strict rendering order to solve transparency occlusion for nested meshes
    // Lower number renders first (outermost layer) -> higher renders last (innermost)
    const ORDER = {
      'brain': 1,
      'tumor_whole': 2,
      'tumor_core': 3,
      'tumor_enhancing': 4
    };
    const renderOrder = ORDER[name] || 5;

    group.traverse(child => {
      if (!child.isMesh) return;
      try { child.geometry.computeVertexNormals(); } catch (_) {}

      if (isBrain) {
        // Warm anatomical gray-beige with sheen (subsurface-like tissue appearance)
        child.material = new THREE.MeshPhysicalMaterial({
          color:              new THREE.Color(color),
          roughness:          0.82,
          metalness:          0.00,
          clearcoat:          0.05,
          clearcoatRoughness: 0.90,
          sheen:              0.22,
          sheenRoughness:     0.75,
          sheenColor:         new THREE.Color(0xd4a090),
          envMapIntensity:    0.35,
          transparent:        true,
          opacity,
          depthWrite:         false, // Crucial: allow inner objects to render through
          depthTest:          true,
          side:               THREE.DoubleSide,
          clippingPlanes:     [],
          clipIntersection:   false,
        });
      } else {
        child.material = new THREE.MeshPhysicalMaterial({
          color:              new THREE.Color(color),
          roughness:          0.35,
          metalness:          0.05,
          clearcoat:          0.60,
          clearcoatRoughness: 0.40,
          envMapIntensity:    0.55,
          transparent:        true,
          opacity,
          depthWrite:         false, // Crucial for nested transparent tumors
          depthTest:          true,
          side:               THREE.DoubleSide,
          clippingPlanes:     [],
          clipIntersection:   false,
        });
      }

      child.renderOrder = renderOrder;
      child.castShadow    = true;
      child.receiveShadow = true;
    });

    group.userData.name    = name;
    group.userData.opacity = opacity;
    return group;
  }

  setSectionMode(mode) {
    this._sectionMode = mode;
    this._applySection();
    const badge = document.getElementById('viewBadgeLabel');
    if (badge) badge.textContent = (SECTIONS[mode] && SECTIONS[mode].label) ? SECTIONS[mode].label : '3D PERSPECTIVE';
    const viewLabel = document.getElementById('currentViewLabel');
    if (viewLabel) viewLabel.textContent = (SECTIONS[mode] && SECTIONS[mode].label) ? SECTIONS[mode].label : '3D Perspective';
  }

  setSectionOffset(val) {
    this._sectionOffset = parseFloat(val);
    if (this._sectionMode !== 'none') this._applySection();
  }

  setClipZ(val)      { this.setSectionOffset(val); }
  setClipPosition(v) { this.setSectionOffset((v - 50) / 50); }

  _applySection() {
    const cfg = SECTIONS[this._sectionMode];

    if (this._sectionCap) {
      this.scene.remove(this._sectionCap);
      this._sectionCap.geometry.dispose();
      this._sectionCap.material.dispose();
      this._sectionCap = null;
    }

    if (!cfg.normal) {
      Object.values(this._meshes).forEach(mesh => {
        if (!mesh) return;
        mesh.traverse(c => { if (c.isMesh) c.material.clippingPlanes = []; });
      });
      return;
    }

    const center   = this._brainBounds.getCenter(new THREE.Vector3());
    const halfSize = this._brainBounds.getSize(new THREE.Vector3()).length() * 0.5;
    const offset   = this._sectionOffset * halfSize * 0.9;
    const n = cfg.normal.clone();
    const d = -(n.dot(center) + offset);
    this._clipPlane.set(n, d);

    Object.values(this._meshes).forEach(mesh => {
      if (!mesh) return;
      mesh.traverse(c => { if (c.isMesh) c.material.clippingPlanes = [this._clipPlane]; });
    });

    this._buildSectionCap(center, n, halfSize * 1.1, offset);
  }

  _buildSectionCap(center, normal, radius, offset) {
    const SIZE = 1024;
    const cv   = document.createElement('canvas');
    cv.width = cv.height = SIZE;
    const ctx = cv.getContext('2d');
    const cx = SIZE / 2, cy = SIZE / 2, r = SIZE * 0.46;
    ctx.clearRect(0, 0, SIZE, SIZE);

    const grad = ctx.createRadialGradient(cx, cy, r * 0.03, cx, cy, r);
    grad.addColorStop(0.00, '#0d0d0d');
    grad.addColorStop(0.10, '#1a1a1a');
    grad.addColorStop(0.18, '#c8b89a');
    grad.addColorStop(0.45, '#9e8e78');
    grad.addColorStop(0.68, '#6a5a48');
    grad.addColorStop(0.85, '#3a2a18');
    grad.addColorStop(0.96, '#181208');
    grad.addColorStop(1.00, '#000000');
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip();
    ctx.strokeStyle = 'rgba(0,0,0,0.18)';
    ctx.lineWidth   = 2.5;
    for (let i = 0; i < 24; i++) {
      const a0  = (i / 24) * Math.PI * 2;
      const a1  = a0 + 0.38;
      const rIn = r * (0.35 + 0.22 * ((i * 7919) % 17) / 17);
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a0) * rIn, cy + Math.sin(a0) * rIn);
      ctx.quadraticCurveTo(
        cx + Math.cos((a0 + a1) / 2) * (rIn + r) / 2,
        cy + Math.sin((a0 + a1) / 2) * (rIn + r) / 2,
        cx + Math.cos(a1) * r * 0.9,
        cy + Math.sin(a1) * r * 0.9
      );
      ctx.stroke();
    }
    ctx.restore();

    ctx.beginPath();
    ctx.arc(cx, cy, r - 2, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(80,180,255,0.22)';
    ctx.lineWidth   = 3;
    ctx.stroke();

    const tex = new THREE.CanvasTexture(cv);
    tex.minFilter = tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;

    const geo = new THREE.CircleGeometry(radius, 96);
    const mat = new THREE.MeshBasicMaterial({
      map: tex, side: THREE.DoubleSide,
      transparent: true, opacity: 0.9, depthWrite: false,
    });
    const cap = new THREE.Mesh(geo, mat);
    cap.renderOrder = 1;
    cap.material.polygonOffset       = true;
    cap.material.polygonOffsetFactor = -1;
    cap.material.polygonOffsetUnits  = -1;
    cap.position.copy(center.clone().add(normal.clone().multiplyScalar(offset - 0.0005)));
    cap.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    this._sectionCap = cap;
    this.scene.add(cap);
  }

  highlightRegion(regionName) {
    this._clearRegionOverlay();
    if (!regionName || this._brainBounds.isEmpty()) return;

    const fracs = LOBE_SCENE_FRACS[regionName];
    if (!fracs) return;

    const center = this._brainBounds.getCenter(new THREE.Vector3());
    const size   = this._brainBounds.getSize(new THREE.Vector3());

    const boxW = size.x * (fracs.xFrac[1] - fracs.xFrac[0]) * 0.5;
    const boxH = size.y * (fracs.yFrac[1] - fracs.yFrac[0]) * 0.5;
    const boxD = size.z * (fracs.zFrac[1] - fracs.zFrac[0]) * 0.5;
    const boxCx = center.x + size.x * (fracs.xFrac[0] + fracs.xFrac[1]) * 0.25;
    const boxCy = center.y + size.y * (fracs.yFrac[0] + fracs.yFrac[1]) * 0.25;
    const boxCz = center.z + size.z * (fracs.zFrac[0] + fracs.zFrac[1]) * 0.25;

    const geo = new THREE.BoxGeometry(Math.abs(boxW) * 2, Math.abs(boxH) * 2, Math.abs(boxD) * 2);
    const mat = new THREE.MeshBasicMaterial({
      color: LOBE_COLORS[regionName] !== undefined ? LOBE_COLORS[regionName] : 0xffffff,
      transparent: true, opacity: 0.12, depthWrite: false, side: THREE.FrontSide,
    });
    const box = new THREE.Mesh(geo, mat);
    box.position.set(boxCx, boxCy, boxCz);
    box.renderOrder = 2;
    this.scene.add(box);

    const edgesGeo = new THREE.EdgesGeometry(geo);
    const edgesMat = new THREE.LineBasicMaterial({
      color: LOBE_COLORS[regionName] !== undefined ? LOBE_COLORS[regionName] : 0xffffff,
      transparent: true, opacity: 0.45,
    });
    box.add(new THREE.LineSegments(edgesGeo, edgesMat));

    this._regionOverlay = box;
    if (this._meshes['brain']) this.setOpacity('brain', 0.35);
  }

  clearRegionHighlight() {
    this._clearRegionOverlay();
    if (this._meshes['brain']) this.setOpacity('brain', INITIAL_OPACITY.brain);
  }

  _clearRegionOverlay() {
    if (this._regionOverlay) {
      this.scene.remove(this._regionOverlay);
      this._regionOverlay.traverse(c => {
        if (c.geometry) c.geometry.dispose();
        if (c.material) c.material.dispose();
      });
      this._regionOverlay = null;
    }
  }

  setVisible(name, visible) {
    const m = this._meshes[name];
    if (m) m.visible = visible;
  }

  setOpacity(name, value) {
    const m = this._meshes[name];
    if (!m) return;
    m.traverse(child => {
      if (child.isMesh && child.material) {
        child.material.opacity     = value;
        child.material.transparent = value < 1.0;
        child.material.depthWrite  = value >= 0.99;
      }
    });
    m.userData.opacity = value;
  }

  setAllTumorOpacity(value) {
    ['tumor_whole', 'tumor_core', 'tumor_enhancing'].forEach(n => this.setOpacity(n, value));
  }

  setCameraPreset(preset) {
    const cfg = CAM_PRESETS[preset];
    if (!cfg) return;

    const box = new THREE.Box3();
    Object.values(this._meshes).forEach(m => { if (m) box.expandByObject(m); });
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const dist   = Math.max(size.x, size.y, size.z) * 1.85;
    const dir    = new THREE.Vector3(cfg.pos[0], cfg.pos[1], cfg.pos[2]).normalize();
    const endPos = center.clone().add(dir.multiplyScalar(dist));
    const endUp  = new THREE.Vector3(cfg.up[0], cfg.up[1], cfg.up[2]);

    this._animateCameraTo(endPos, center, endUp, 700);

    const badge = document.getElementById('viewBadgeLabel');
    if (badge) badge.textContent = cfg.label || '3D PERSPECTIVE';
  }

  _fitCamera() {
    const box = new THREE.Box3();
    Object.values(this._meshes).forEach(m => { if (m) box.expandByObject(m); });
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    this.controls.target.copy(center);
    this.camera.position.copy(
      center.clone().add(new THREE.Vector3(maxDim * 0.15, maxDim * 0.25, maxDim * 2.0))
    );
    this.camera.near = maxDim * 0.001;
    this.camera.far  = maxDim * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resetCamera() { this._fitCamera(); }

  toggleOrbit() {
    if (!this.controls) return null;
    this.controls.enabled = !this.controls.enabled;
    return this.controls.enabled;
  }

  setAutoRotate(enabled) {
    if (!this.controls) return false;
    this.controls.autoRotate = Boolean(enabled);
    // Auto rotation requires OrbitControls to remain enabled.
    if (enabled) this.controls.enabled = true;
    return this.controls.autoRotate;
  }

  hasMeshes() { return Object.keys(this._meshes).length > 0; }

  _clearMeshes() {
    this._clearRegionOverlay();
    Object.values(this._meshes).forEach(m => {
      if (!m) return;
      this.scene.remove(m);
      m.traverse(child => {
        if (child.isMesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material))
            child.material.forEach(mat => mat.dispose());
          else child.material.dispose();
        }
      });
    });
    this._meshes = {};

    if (this._sectionCap) {
      this.scene.remove(this._sectionCap);
      this._sectionCap.geometry.dispose();
      this._sectionCap.material.dispose();
      this._sectionCap = null;
    }
    if (this._centroidMarker) {
      this.scene.remove(this._centroidMarker);
      this._centroidMarker = null;
    }
    this._centroidWorld      = null;
    this._pendingCentroidRAS = null;
    this._camTransition      = null;
  }

  _setupXR() {
    const arBtn    = document.getElementById('btnAR');
    const vrBtn    = document.getElementById('btnVR');
    const arStatus = document.getElementById('arStatus');
    const vrStatus = document.getElementById('vrStatus');

    if (!navigator.xr) {
      if (arStatus) arStatus.textContent = 'WebXR not available in this browser.';
      if (vrStatus) vrStatus.textContent = 'WebXR not available in this browser.';
      return;
    }

    navigator.xr.isSessionSupported('immersive-ar').then(ok => {
      if (arStatus) arStatus.textContent = ok ? 'AR supported ✔' : 'AR: requires Android + Chrome.';
      if (arBtn && ok) arBtn.disabled = false;
    });

    navigator.xr.isSessionSupported('immersive-vr').then(ok => {
      if (vrStatus) vrStatus.textContent = ok ? 'VR supported ✔' : 'VR: requires a WebXR-capable headset.';
      if (vrBtn && ok) vrBtn.disabled = false;
    });
  }

  async enterAR() {
    if (!navigator.xr) { alert('WebXR not supported.'); return; }
    const ok = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
    if (!ok) { alert('AR not supported on this device.\nTry Chrome on Android.'); return; }
    try {
      this.renderer.xr.setReferenceSpaceType('local');
      const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local', 'hit-test'],
        optionalFeatures: ['dom-overlay'],
        domOverlay: { root: document.body },
      });
      this.renderer.xr.setSession(session);
      this._xrActive = true;
      Object.values(this._meshes).forEach(m => { if (m) m.position.set(0, 0, -0.3); });
      session.addEventListener('end', () => {
        this._xrActive = false;
        Object.values(this._meshes).forEach(m => { if (m) m.position.set(0, 0, 0); });
        this.resetCamera();
      });
    } catch (err) { alert('AR error: ' + err.message); }
  }

  async enterVR() {
    if (!navigator.xr) { alert('WebXR not supported.'); return; }
    const ok = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
    if (!ok) { alert('VR not supported on this device.'); return; }
    try {
      this.renderer.xr.setReferenceSpaceType('local-floor');
      const session = await navigator.xr.requestSession('immersive-vr', {
        optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking'],
      });
      this.renderer.xr.setSession(session);
      this._xrActive = true;
      Object.values(this._meshes).forEach(m => { if (m) m.position.set(0, 1.4, -0.5); });
      const cf = new XRControllerModelFactory();
      [0, 1].forEach(i => {
        const c = this.renderer.xr.getController(i);
        const g = this.renderer.xr.getControllerGrip(i);
        g.add(cf.createControllerModel(g));
        this.scene.add(c, g);
      });
      session.addEventListener('end', () => {
        this._xrActive = false;
        Object.values(this._meshes).forEach(m => { if (m) m.position.set(0, 0, 0); });
        this.resetCamera();
      });
    } catch (err) { alert('VR error: ' + err.message); }
  }
}

// Bootstrap
const canvas = document.getElementById('threeCanvas');
try {
  const viewer = new NeuroVRViewer(canvas);
  viewer.init();
  window.viewer = viewer;
} catch (err) {
  console.error('[NeuroVR] WebGL initialization failed:', err);
  window.viewer = null;
  const empty = document.getElementById('viewerEmpty');
  if (empty) {
    empty.style.display = 'flex';
    const title = empty.querySelector('.viewer-empty-title');
    const sub = empty.querySelector('.viewer-empty-sub');
    if (title) title.textContent = '3D Viewer Unavailable';
    if (sub) sub.textContent = 'WebGL could not be initialized. Enable hardware acceleration or use a supported browser.';
  }
  document.querySelectorAll('.viewer-toolbar button, #btnAR, #btnVR').forEach(button => {
    button.disabled = true;
  });
  const status = document.getElementById('systemStatusText');
  const dot = document.getElementById('systemDot');
  if (status) status.textContent = 'WEBGL ERROR';
  if (dot) dot.className = 'status-dot error';
}
