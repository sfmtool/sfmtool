// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

// The viewer of `sfm web-export`: mount(element, scene, options) draws the
// scene a `scene.json` describes into `element`. See
// specs/cli/visualization/web-export-command.md. three.js is imported by full
// URL, since a claude.ai artifact ignores an import map; jsDelivr rewrites
// OrbitControls' own `three` import to the same URL, so one copy loads.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.180.0/+esm";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js/+esm";

const STYLE = `
.wx{position:relative;overflow:hidden;background:var(--wx-bg,#17191e);border-radius:10px;touch-action:pan-y pinch-zoom;user-select:none;-webkit-user-select:none}
.wx canvas{display:block;width:100%;height:100%}
.wx.wx-active canvas{touch-action:none}
.wx-full{position:fixed!important;inset:0;z-index:1000;border-radius:0;width:auto!important;height:auto!important;aspect-ratio:auto!important}
.wx [hidden]{display:none!important}
.wx-bar{position:absolute;z-index:2;top:8px;right:8px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;max-width:calc(100% - 16px)}
.wx-full .wx-bar{top:calc(8px + env(safe-area-inset-top,0px));right:calc(8px + env(safe-area-inset-right,0px))}
.wx button{min-width:44px;min-height:44px;padding:0 12px;border:1px solid #ffffff2e;border-radius:8px;background:#23262dd9;color:#e9ecf2;font:500 13px/1 system-ui,sans-serif;cursor:pointer}
.wx button[aria-pressed="false"]{color:#8a92a3}
.wx button:focus-visible{outline:2px solid #7fb4ff;outline-offset:2px}
.wx-tap{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:transparent!important;border:0!important}
.wx-tap span{padding:12px 18px;border-radius:999px;background:#000000a6;color:#fff;font:600 14px/1 system-ui,sans-serif}
.wx-label,.wx-note{position:absolute;left:8px;bottom:8px;max-width:calc(100% - 16px);padding:8px 10px;border-radius:8px;background:#000000b3;color:#fff;font:13px/1.3 ui-monospace,Menlo,Consolas,monospace;pointer-events:none;overflow-wrap:anywhere}
.wx-note{top:8px;bottom:auto;right:auto;font-family:system-ui,sans-serif;max-width:60%}
.wx-full .wx-label{bottom:calc(8px + env(safe-area-inset-bottom,0px))}
.wx-stats{position:absolute;right:8px;bottom:8px;padding:6px 8px;border-radius:6px;background:#000000b3;color:#e9ecf2;font:12px/1.2 ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;pointer-events:none}
.wx-full .wx-stats{bottom:calc(8px + env(safe-area-inset-bottom,0px));right:calc(8px + env(safe-area-inset-right,0px))}
.wx-fallback{display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:center;height:100%;color:#e9ecf2;font:14px system-ui,sans-serif;padding:16px;box-sizing:border-box;text-align:center}
.wx-fallback div{width:128px;height:128px;border-radius:6px}
`;

function injectStyle() {
  if (document.getElementById("wx-style")) return;
  const s = document.createElement("style");
  s.id = "wx-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function loadTexture(url) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(url, resolve, undefined, () => reject(new Error(`could not load ${url.slice(0, 80)}`)));
  });
}

function decodeBase64(text) {
  const bin = atob(text);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

// One array of a byte block, as its typed array, or null when it is absent.
function view(buffer, arrays, name) {
  const a = arrays[name];
  if (!a) return null;
  const n = a.count * a.components;
  switch (a.type) {
    case "f32": return new Float32Array(buffer, a.offset, n);
    case "u32": return new Uint32Array(buffer, a.offset, n);
    case "u16": return new Uint16Array(buffer, a.offset, n);
    default: return new Uint8Array(buffer, a.offset, n);
  }
}

const PATCH_VS = `
attribute vec3 iCenter; attribute vec3 iU; attribute vec3 iV; attribute float iCell;
uniform vec2 atlasSize; uniform float cols; uniform float tile; uniform float border; uniform float size;
varying vec2 vUv;
void main(){
  vec3 n = cross(iU, iV);
  if (dot(n, cameraPosition - iCenter) <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 w = iCenter + position.x * iU + position.y * iV;
  gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
  float c = mod(iCell, cols); float r = floor(iCell / cols);
  vec2 st = vec2(position.x * 0.5 + 0.5, 1.0 - (position.y * 0.5 + 0.5));
  vUv = (vec2(c, r) * tile + border + st * size) / atlasSize;
}`;
const PATCH_FS = `
uniform sampler2D atlas; varying vec2 vUv;
void main(){ gl_FragColor = vec4(texture2D(atlas, vUv).rgb, 1.0); }`;

const SPLAT_VS = `
attribute vec3 aColor; attribute float aW; attribute float aHasPatch;
uniform float worldSize; uniform float projScale; uniform float infPx; uniform float showAll; uniform float pr;
varying vec3 vColor;
void main(){
  vColor = aColor;
  if (aW > 0.5 && aHasPatch > 0.5 && showAll < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec4 p = viewMatrix * vec4(position, aW > 0.5 ? 1.0 : 0.0);
  gl_Position = projectionMatrix * p;
  if (aW < 0.5) { gl_Position.z = gl_Position.w * 0.99999; gl_PointSize = infPx * pr; }
  else { gl_PointSize = max(2.0, worldSize * projScale / max(-p.z, 1e-9)); }
}`;
const SPLAT_FS = `
varying vec3 vColor;
void main(){ vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(vColor, 1.0); }`;

// A flat grey page or a notice when there is no WebGL2: the first camera's
// thumbnail, cut out of the atlas with CSS.
function fallback(el, scene, base, text) {
  el.classList.add("wx");
  const box = document.createElement("div");
  box.className = "wx-fallback";
  box.textContent = text;
  const th = scene.atlases && scene.atlases.thumbnails;
  const cam = scene.cameras[0];
  if (th && cam && cam.thumb) {
    const page = th.pages[cam.thumb[0]];
    const cell = cam.thumb[1];
    const img = document.createElement("div");
    const x = (cell % th.cols) * th.tile + th.border, y = Math.floor(cell / th.cols) * th.tile + th.border;
    img.style.background = `url("${new URL(page.file, base).href}") -${x}px -${y}px no-repeat`;
    box.appendChild(img);
  }
  el.appendChild(box);
}

/**
 * Let the web-export views in iframes on this page fill the window.
 *
 * A view in an iframe cannot grow past the iframe's box, so its Expand asks the
 * page that holds it. Call this once in that page: it answers each view's
 * greeting, and pins the asking iframe to the window on Expand and releases it
 * on Close. Without it, a view in an iframe falls back to the Fullscreen API
 * where the iframe allows it (`allow="fullscreen"`), and otherwise shows no
 * Expand button.
 */
export function hostFrames() {
  if (!document.getElementById("wx-host-style")) {
    const s = document.createElement("style");
    s.id = "wx-host-style";
    s.textContent = "iframe.wx-host-full{position:fixed!important;inset:0!important;width:100%!important;height:100%!important;max-width:none!important;max-height:none!important;aspect-ratio:auto!important;margin:0!important;border-radius:0!important;z-index:1000}";
    document.head.appendChild(s);
  }
  const frameOf = (source) => [...document.querySelectorAll("iframe")].find((f) => f.contentWindow === source);
  addEventListener("message", (e) => {
    const d = e.data;
    if (!d || !d.wxFrame) return;
    const frame = frameOf(e.source);
    if (!frame) return;
    if (d.wxFrame === "hello") e.source.postMessage({ wxHost: true }, "*");
    else frame.classList.toggle("wx-host-full", d.wxFrame === "expand");
  });
  // A view that greeted before this ran is answered now.
  for (const f of document.querySelectorAll("iframe")) {
    try { f.contentWindow.postMessage({ wxHost: true }, "*"); } catch (e) { /* not loaded yet */ }
  }
}

/**
 * Draw a web-export scene into `el`.
 *
 * `source` is the URL of a `scene.json` (relative to the page) or the parsed
 * scene itself, whose atlas page names then resolve against `opts.base`.
 * Options: `showPatches`, `showPoints`, `showCameras` (all default true),
 * `background` (a CSS colour), `startImage` (an image name to open through),
 * `startActive` (take gestures from the start rather than after a tap),
 * `showStats` (a frame-rate readout in the corner),
 * `fillsPage` (the view is the whole page, as in `index.html`, so in an iframe
 * its Expand asks the page holding the iframe), and `onEvent(key, value)` for
 * load and frame-rate reports.
 */
export async function mount(el, source, opts = {}) {
  injectStyle();
  const report = opts.onEvent || (() => {});
  const t0 = performance.now();
  let base, scene;
  if (typeof source === "string") {
    base = new URL(source, location.href);
    const response = await fetch(base);
    if (!response.ok) throw new Error(`could not load ${source}: ${response.status}`);
    scene = await response.json();
  } else {
    scene = source;
    base = new URL(opts.base || location.href, location.href);
  }
  report("scene.json", `${scene.points.count} points, ${scene.cameras.length} cameras`);
  if (opts.background) el.style.setProperty("--wx-bg", opts.background);

  const probe = document.createElement("canvas").getContext("webgl2");
  if (!probe) {
    fallback(el, scene, base, "This view needs WebGL2, which this browser does not offer.");
    return { scene, stats: {}, setActive() {}, setFull() {} };
  }
  const maxTexture = probe.getParameter(probe.MAX_TEXTURE_SIZE);
  const lose = probe.getExtension("WEBGL_lose_context");
  if (lose) lose.loseContext();

  const pointBuf = decodeBase64(scene.points_b64);
  const PA = scene.points.arrays;
  const P = scene.points.count;
  const position = view(pointBuf, PA, "position");
  const patchU = view(pointBuf, PA, "patch_u");
  const patchV = view(pointBuf, PA, "patch_v");
  const patchCell = view(pointBuf, PA, "patch_cell");
  const colorW = view(pointBuf, PA, "color_w");
  const observations = view(pointBuf, PA, "observations");
  const sourceIndex = view(pointBuf, PA, "source_index");
  const grid = view(decodeBase64(scene.frustums_b64), scene.frustum_arrays, "vertex");
  report("point arrays", `${pointBuf.byteLength} bytes`);

  el.classList.add("wx");
  const canvas = document.createElement("canvas");
  el.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  report("webgl", `${renderer.capabilities.isWebGL2 ? "WebGL2" : "WebGL1"}, max texture ${maxTexture}`);
  const pr = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pr);
  renderer.setClearColor(new THREE.Color(getComputedStyle(el).getPropertyValue("--wx-bg").trim() || "#17191e"));

  const s3 = new THREE.Scene();
  const R = scene.radius, F = scene.frustum_length;
  const cams = scene.cameras;
  let reach = R;
  for (const c of cams) reach = Math.max(reach, Math.hypot(...c.center));
  const cam = new THREE.PerspectiveCamera(50, 1, Math.min(R * 1e-3, F * 0.05), reach * 100);
  cam.up.set(0, 0, 1);

  // The initial view: the scene's own, or through the camera the options name.
  let v = scene.view;
  const startName = opts.startImage || v.start_image;
  const startIndex = startName ? cams.findIndex((c) => c.name === startName) : -1;
  if (opts.startImage && startIndex >= 0 && opts.startImage !== v.start_image) {
    const c = cams[startIndex];
    const [w, x, y, z] = c.rotation_wxyz;
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(new THREE.Quaternion(x, y, z, w));
    const eye = new THREE.Vector3(...c.center);
    const along = -eye.dot(fwd);
    const target = eye.clone().addScaledVector(fwd, along > 0.05 * R ? along : R);
    v = { eye: c.center, target: target.toArray(), fov: 60, start_image: c.name };
  }
  const home = { pos: new THREE.Vector3(...v.eye), target: new THREE.Vector3(...v.target), fov: v.fov };
  cam.fov = home.fov;
  cam.position.copy(home.pos);
  const controls = new OrbitControls(cam, canvas);
  controls.target.copy(home.target);
  controls.enableDamping = true;
  controls.screenSpacePanning = true;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  controls.update();

  const notices = [];

  // Patches: one instanced mesh per atlas page.
  const patchGroup = new THREE.Group();
  const hasPatch = new Float32Array(P);
  const pa = scene.atlases.patches;
  if (pa && patchCell && opts.showPatches !== false) {
    const tooLarge = pa.pages.some((p) => p.width > maxTexture || p.height > maxTexture);
    if (tooLarge) {
      notices.push(`Patch pages are larger than this device's ${maxTexture}-texel limit; drawing points.`);
    } else {
      const byPage = pa.pages.map(() => []);
      for (let i = 0; i < P; i++) {
        const cell = patchCell[i];
        if (cell !== 0xffffffff && colorW[i * 4 + 3] > 0) { byPage[cell >>> 24].push(i); hasPatch[i] = 1; }
      }
      const quad = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0];
      for (let page = 0; page < pa.pages.length; page++) {
        const idx = byPage[page];
        if (!idx.length) continue;
        const pg = pa.pages[page];
        const tex = await loadTexture(new URL(pg.file, base).href);
        tex.flipY = false;
        tex.colorSpace = THREE.NoColorSpace;
        tex.generateMipmaps = false;
        tex.minFilter = THREE.LinearFilter;
        tex.needsUpdate = true;
        const g = new THREE.InstancedBufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(quad, 3));
        g.setIndex([0, 1, 2, 0, 2, 3]);
        const c = new Float32Array(idx.length * 3), u = new Float32Array(idx.length * 3), w = new Float32Array(idx.length * 3), k = new Float32Array(idx.length);
        idx.forEach((p, j) => {
          for (let a = 0; a < 3; a++) { c[j * 3 + a] = position[p * 3 + a]; u[j * 3 + a] = patchU[p * 3 + a]; w[j * 3 + a] = patchV[p * 3 + a]; }
          k[j] = patchCell[p] & 0xffffff;
        });
        g.setAttribute("iCenter", new THREE.InstancedBufferAttribute(c, 3));
        g.setAttribute("iU", new THREE.InstancedBufferAttribute(u, 3));
        g.setAttribute("iV", new THREE.InstancedBufferAttribute(w, 3));
        g.setAttribute("iCell", new THREE.InstancedBufferAttribute(k, 1));
        g.instanceCount = idx.length;
        const mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
          vertexShader: PATCH_VS, fragmentShader: PATCH_FS, side: THREE.DoubleSide,
          uniforms: { atlas: { value: tex }, atlasSize: { value: new THREE.Vector2(pg.width, pg.height) },
            cols: { value: pa.cols }, tile: { value: pa.tile }, border: { value: pa.border }, size: { value: pa.size } },
        }));
        mesh.frustumCulled = false;
        patchGroup.add(mesh);
      }
      report("patch atlas", `${pa.pages.length} page(s)`);
    }
  }
  s3.add(patchGroup);
  const havePatches = patchGroup.children.length > 0;

  // Splats: points without a patch, points at infinity, and every point when
  // patches are off.
  const sg = new THREE.BufferGeometry();
  const sc = new Float32Array(P * 3), sw = new Float32Array(P);
  for (let i = 0; i < P; i++) {
    for (let a = 0; a < 3; a++) sc[i * 3 + a] = colorW[i * 4 + a] / 255;
    sw[i] = colorW[i * 4 + 3] > 0 ? 1 : 0;
  }
  sg.setAttribute("position", new THREE.BufferAttribute(position, 3));
  sg.setAttribute("aColor", new THREE.BufferAttribute(sc, 3));
  sg.setAttribute("aW", new THREE.BufferAttribute(sw, 1));
  sg.setAttribute("aHasPatch", new THREE.BufferAttribute(hasPatch, 1));
  const splatMat = new THREE.ShaderMaterial({
    vertexShader: SPLAT_VS, fragmentShader: SPLAT_FS,
    uniforms: { worldSize: { value: 2 * scene.point_size }, projScale: { value: 1 }, infPx: { value: 4 }, showAll: { value: havePatches ? 0 : 1 }, pr: { value: pr } },
  });
  const splats = new THREE.Points(sg, splatMat);
  splats.frustumCulled = false;
  splats.visible = opts.showPoints !== false;
  s3.add(splats);

  // Cameras: frustum outlines and each image's thumbnail on its far surface.
  // The camera the view opens through goes in a group of its own, hidden while
  // the eye is inside its frustum.
  const vertex = (c, gx, gy) => {
    const i = (c.grid[0] + gy * c.grid[1] + gx) * 3;
    return [grid[i], grid[i + 1], grid[i + 2]];
  };
  const th = scene.atlases.thumbnails;
  const thumbTextures = [];
  if (th && opts.showCameras !== false) {
    for (const pg of th.pages) {
      if (pg.width > maxTexture || pg.height > maxTexture) { thumbTextures.push(null); continue; }
      const tex = await loadTexture(new URL(pg.file, base).href);
      tex.flipY = false;
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      thumbTextures.push(tex);
    }
    report("thumbnail atlas", `${th.pages.length} page(s)`);
  }
  function frustumGroup(list) {
    const group = new THREE.Group();
    const lp = [];
    for (const c of list) {
      const G = c.grid[1];
      const corners = [vertex(c, 0, 0), vertex(c, G - 1, 0), vertex(c, G - 1, G - 1), vertex(c, 0, G - 1)];
      for (const q of corners) lp.push(...c.center, ...q);
      for (let e = 0; e < G - 1; e++) {
        lp.push(...vertex(c, e, 0), ...vertex(c, e + 1, 0), ...vertex(c, e, G - 1), ...vertex(c, e + 1, G - 1));
        lp.push(...vertex(c, 0, e), ...vertex(c, 0, e + 1), ...vertex(c, G - 1, e), ...vertex(c, G - 1, e + 1));
      }
    }
    group.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(lp, 3)),
      new THREE.LineBasicMaterial({ color: 0xf2c14e })));
    if (!th) return group;
    th.pages.forEach((pg, page) => {
      const tex = thumbTextures[page];
      if (!tex) return;
      const qp = [], qu = [], qi = [];
      for (const c of list) {
        if (!c.thumb || c.thumb[0] !== page) continue;
        const G = c.grid[1], o = qp.length / 3;
        const cx = c.thumb[1] % th.cols, cy = Math.floor(c.thumb[1] / th.cols);
        for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
          qp.push(...vertex(c, gx, gy));
          qu.push((cx * th.tile + th.border + (gx / (G - 1)) * th.size) / pg.width,
            (cy * th.tile + th.border + (gy / (G - 1)) * th.size) / pg.height);
        }
        for (let gy = 0; gy < G - 1; gy++) for (let gx = 0; gx < G - 1; gx++) {
          const a = o + gy * G + gx;
          qi.push(a, a + 1, a + G + 1, a, a + G + 1, a + G);
        }
      }
      if (!qi.length) return;
      const qg = new THREE.BufferGeometry();
      qg.setAttribute("position", new THREE.Float32BufferAttribute(qp, 3));
      qg.setAttribute("uv", new THREE.Float32BufferAttribute(qu, 2));
      qg.setIndex(qi);
      group.add(new THREE.Mesh(qg, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })));
    });
    return group;
  }
  const camGroup = new THREE.Group();
  camGroup.add(frustumGroup(cams.filter((_, i) => i !== startIndex)));
  const startGroup = startIndex >= 0 ? frustumGroup([cams[startIndex]]) : null;
  if (startGroup) camGroup.add(startGroup);
  camGroup.visible = opts.showCameras !== false;
  s3.add(camGroup);
  const startCentre = startIndex >= 0 ? new THREE.Vector3(...cams[startIndex].center) : null;

  // Picking targets: finite points and camera centres.
  const fin = [], finIdx = [];
  for (let i = 0; i < P; i++) if (sw[i]) { fin.push(position[i * 3], position[i * 3 + 1], position[i * 3 + 2]); finIdx.push(i); }
  const pickPts = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(fin, 3)));
  const pickCams = new THREE.Points(new THREE.BufferGeometry().setAttribute("position",
    new THREE.Float32BufferAttribute(cams.flatMap((c) => c.center), 3)));
  const ray = new THREE.Raycaster();

  // Overlay controls.
  const bar = document.createElement("div");
  bar.className = "wx-bar";
  el.appendChild(bar);
  const label = document.createElement("div");
  label.className = "wx-label";
  label.hidden = true;
  el.appendChild(label);
  // With showStats, a readout of the frame rate, measured while the view is
  // active, since an inactive view draws only when something changes.
  const statsBox = document.createElement("div");
  statsBox.className = "wx-stats";
  statsBox.textContent = "fps: move the view";
  statsBox.hidden = !opts.showStats;
  el.appendChild(statsBox);
  if (notices.length) {
    const note = document.createElement("div");
    note.className = "wx-note";
    note.textContent = notices.join(" ");
    el.appendChild(note);
  }
  const button = (text, title, onClick, pressed) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = text; b.title = title;
    if (pressed !== undefined) b.setAttribute("aria-pressed", String(pressed));
    b.addEventListener("click", (e) => { e.stopPropagation(); onClick(b); render(); });
    bar.appendChild(b);
    return b;
  };
  const pressed = (b, on) => b.setAttribute("aria-pressed", String(on));
  if (havePatches) button("Patches", "Show patches", (b) => { patchGroup.visible = !patchGroup.visible; splatMat.uniforms.showAll.value = patchGroup.visible ? 0 : 1; pressed(b, patchGroup.visible); }, true);
  button("Points", "Show points", (b) => { splats.visible = !splats.visible; pressed(b, splats.visible); }, splats.visible);
  button("Cameras", "Show cameras", (b) => { camGroup.visible = !camGroup.visible; pressed(b, camGroup.visible); }, camGroup.visible);
  button("Reset", "Return to the starting view", () => { cam.position.copy(home.pos); controls.target.copy(home.target); controls.update(); });
  const fullBtn = button("Expand", "Fill the window", () => setFull(!full));

  const tap = document.createElement("button");
  tap.type = "button"; tap.className = "wx-tap";
  tap.innerHTML = "<span>Tap to explore</span>";
  el.appendChild(tap);

  let active = false, running = false, frames = 0, frameStart = 0;
  const stats = { fps: null };
  function setActive(on) {
    active = on;
    el.classList.toggle("wx-active", on);
    controls.enabled = on;
    tap.hidden = on;
    if (on) loop();
  }
  // Expand. A view that is part of a page pins itself to the window with CSS,
  // whether or not that page is itself in a frame (an artifact page is). A view
  // that is the whole page of an iframe cannot grow past the iframe's box, so it
  // asks the page holding the iframe (which runs hostFrames()); with no answer
  // it uses the Fullscreen API where the iframe allows it, and otherwise has no
  // Expand.
  let inFrame = false;
  if (opts.fillsPage) {
    try { inFrame = window.parent !== window; } catch (e) { inFrame = true; }
  }
  let hosted = false, full = false;
  function showFull(on) {
    full = on;
    fullBtn.textContent = on ? "Close" : "Expand";
    fullBtn.title = on ? "Return to the page" : "Fill the window";
    if (on) setActive(true); else if (!opts.startActive) setActive(false);
    resize();
  }
  function setFull(on) {
    if (on === full) return;
    if (!inFrame) {
      el.classList.toggle("wx-full", on);
      showFull(on);
    } else if (hosted) {
      parent.postMessage({ wxFrame: on ? "expand" : "collapse" }, "*");
      showFull(on);
    } else if (document.fullscreenEnabled) {
      const request = on ? document.documentElement.requestFullscreen() : document.exitFullscreen();
      Promise.resolve(request).catch(() => {});
    }
  }
  if (inFrame) {
    fullBtn.hidden = !document.fullscreenEnabled;
    addEventListener("message", (e) => {
      if (e.source === parent && e.data && e.data.wxHost) { hosted = true; fullBtn.hidden = false; }
    });
    document.addEventListener("fullscreenchange", () => { if (!hosted) showFull(!!document.fullscreenElement); });
    parent.postMessage({ wxFrame: "hello" }, "*");
  }
  tap.addEventListener("click", () => setActive(true));
  document.addEventListener("pointerdown", (e) => {
    if (active && !opts.startActive && !el.contains(e.target) && !full) setActive(false);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && full) setFull(false); });

  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    splatMat.uniforms.projScale.value = (h * pr) / (2 * Math.tan((cam.fov * Math.PI) / 360));
    render();
  }
  new ResizeObserver(resize).observe(el);

  function render() {
    if (startGroup) startGroup.visible = cam.position.distanceTo(startCentre) > 2 * F;
    renderer.render(s3, cam);
  }
  function loop() {
    if (running) return;
    running = true; frames = 0; frameStart = performance.now();
    const step = () => {
      controls.update();
      render();
      frames++;
      const now = performance.now();
      if (now - frameStart > 2000) { stats.fps = Math.round((frames * 1000) / (now - frameStart)); report("frame rate", `${stats.fps} fps`); statsBox.textContent = `${stats.fps} fps`; frames = 0; frameStart = now; }
      if (active) requestAnimationFrame(step); else running = false;
    };
    requestAnimationFrame(step);
  }
  controls.addEventListener("change", render);

  // Tap or hover names what is under the pointer; a double-tap or double-click
  // orbits about it.
  let down = null, lastTap = 0;
  const pick = (e) => {
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), cam);
    // About ten CSS pixels at the orbit target's distance.
    const perPixel = (2 * Math.tan((cam.fov * Math.PI) / 360) * cam.position.distanceTo(controls.target)) / r.height;
    ray.params.Points.threshold = Math.max(10 * perPixel, scene.point_size);
    const hc = camGroup.visible ? ray.intersectObject(pickCams)[0] : null;
    const hp = splats.visible || havePatches ? ray.intersectObject(pickPts)[0] : null;
    if (hc && (!hp || hc.distance <= hp.distance)) return { kind: "camera", i: hc.index, at: new THREE.Vector3(...cams[hc.index].center) };
    if (hp) return { kind: "point", i: finIdx[hp.index], at: new THREE.Vector3(fin[hp.index * 3], fin[hp.index * 3 + 1], fin[hp.index * 3 + 2]) };
    return null;
  };
  const show = (h) => {
    if (!h) { label.hidden = true; return; }
    label.hidden = false;
    const index = sourceIndex ? sourceIndex[h.i] : h.i;
    label.textContent = h.kind === "camera" ? cams[h.i].name : `point ${index} · seen by ${observations[h.i]} images`;
  };
  canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener("pointerup", (e) => {
    if (!active || !down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) return;
    const h = pick(e);
    const now = performance.now();
    if (h && now - lastTap < 350) { controls.target.copy(h.at); controls.update(); }
    lastTap = now;
    show(h);
  });
  canvas.addEventListener("pointermove", (e) => { if (e.pointerType === "mouse" && active && e.buttons === 0) show(pick(e)); });

  setActive(!!opts.startActive);
  resize();
  report("first frame", `${Math.round(performance.now() - t0)} ms after mount`);
  return { scene, stats, setActive, setFull };
}
