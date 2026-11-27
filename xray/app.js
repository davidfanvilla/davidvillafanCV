// RAYOS X — escáner corporal en tiempo real.
// Arriba: cámara + HUD + rostro pixelado + puntos de pose + efectos.
// Abajo: tu cuerpo "transparente" (rayos X) con esqueleto dibujado sobre tu pose real.
// Todo corre en tu dispositivo con MediaPipe (PoseLandmarker + máscara de segmentación).
import { drawAnatomy } from "./anatomy.js";
import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = window.__WASM_URL || "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/";
const MODEL_LITE = MODEL_BASE + "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const MODEL_FULL = MODEL_BASE + "pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const $ = (id) => document.getElementById(id);
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const FONT = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const LIME = "#c8ff1f";

// ---------------------------------------------------------------- Paletas
function makeLut(stops) {
  const l = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 1;
    while (k < stops.length - 1 && stops[k][0] < t) k++;
    const [t0, c0] = stops[k - 1], [t1, c1] = stops[k];
    const f = clamp((t - t0) / (t1 - t0), 0, 1);
    for (let c = 0; c < 3; c++) l[i * 3 + c] = c0[c] + (c1[c] - c0[c]) * f;
  }
  return l;
}
const PALS = {
  xray:    { stops: [[0, [0, 0, 0]], [.22, [14, 19, 27]], [.5, [84, 99, 116]], [.78, [186, 199, 212]], [1, [250, 252, 255]]], glow: [150, 190, 230], bloom: 0.16 },
  neg:     { stops: [[0, [232, 236, 240]], [.4, [168, 176, 186]], [.75, [58, 64, 74]], [1, [8, 9, 12]]], glow: [30, 50, 80], bloom: 0, light: true },
  azul:    { stops: [[0, [0, 0, 0]], [.22, [3, 20, 60]], [.5, [26, 92, 196]], [.78, [118, 208, 255]], [1, [255, 255, 255]]], glow: [67, 184, 255], bloom: 0.38 },
  medico:  { stops: [[0, [0, 0, 0]], [.25, [0, 30, 16]], [.55, [10, 130, 66]], [.8, [120, 255, 170]], [1, [255, 255, 255]]], glow: [61, 255, 122], bloom: 0.3 },
  ambar:   { stops: [[0, [0, 0, 0]], [.25, [36, 16, 0]], [.55, [180, 90, 10]], [.8, [255, 190, 80]], [1, [255, 250, 220]]], glow: [255, 166, 61], bloom: 0.3 },
  termico: { stops: [[0, [0, 0, 0]], [.2, [80, 0, 10]], [.45, [230, 40, 0]], [.7, [255, 170, 0]], [1, [255, 255, 220]]], glow: [255, 90, 0], bloom: 0.3 },
};
for (const k in PALS) PALS[k].lut = makeLut(PALS[k].stops);
const CMAP = makeLut([[0, [8, 0, 30]], [.25, [90, 10, 120]], [.5, [230, 40, 40]], [.75, [255, 180, 20]], [1, [255, 255, 230]]]);

// ---------------------------------------------------------------- Estado
const cfg = { auto: true, blur: false, points: true, boxes: false, glitch: false, scan: true, cmap: false, edges: false, base: true, organs: true, bones: true };
const PRESETS = [
  { base: 1, points: 1, blur: 0, cmap: 0, glitch: 0, edges: 0, boxes: 0 },
  { base: 1, points: 0, blur: 1, cmap: 0, glitch: 0, edges: 0, boxes: 0 },
  { base: 0, points: 0, blur: 1, cmap: 1, glitch: 0, edges: 0, boxes: 0 },
  { base: 1, points: 1, blur: 0, cmap: 0, glitch: 1, edges: 0, boxes: 0 },
  { base: 1, points: 1, blur: 0, cmap: 0, glitch: 0, edges: 1, boxes: 1 },
  { base: 0, points: 1, blur: 0, cmap: 1, glitch: 1, edges: 0, boxes: 0 },
];
let presetIdx = 0;
const opt = { pal: "xray", tissue: 0.55, bone: 0.85, pix: 11 };
const FX_KEYS = ["auto", "blur", "points", "boxes", "glitch", "scan", "cmap", "edges", "base", "organs", "bones"];

// ---------------------------------------------------------------- DOM
const video = $("video"), topC = $("top"), botC = $("bot");
const tg = topC.getContext("2d"), bg = botC.getContext("2d");
const mk = (w = 2, h = 2, read = false) => {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return [c, c.getContext("2d", read ? { willReadFrequently: true } : undefined)];
};

// ---------------------------------------------------------------- Geometría / buffers
let PW = 1, PH = 1, DPR = 1, WW = 2, WH = 2, N = 4;
let crop = { x: 0, y: 0, w: 1, h: 1 };
let facing = "user", mirror = true;
let denC, denG, tex, den = null, workC, wg, rgbC, rgbG, chC, chG, pixC, pixG, cmC, cmG, edC, edG, xrC, xrG, blC, blG;
let mask, lum, ms, gw8 = 1, gh8 = 1, imgData, cmImg, edImg, xrImg;
const scanPat = (() => {
  const [c, g] = mk(4, 4);
  g.fillStyle = "rgba(0,0,0,.14)"; g.fillRect(0, 0, 4, 1);
  return c;
})();

function layout() {
  const r1 = topC.getBoundingClientRect();
  PW = Math.max(120, Math.round(r1.width)); PH = Math.max(120, Math.round(r1.height));
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  for (const c of [topC, botC]) { c.width = Math.round(PW * DPR); c.height = Math.round(PH * DPR); }
  WW = PW > 800 ? 560 : 480;
  WH = clamp(Math.round(WW * PH / PW), 200, 760);
  N = WW * WH;
  [workC, wg] = mk(WW, WH, true);
  [rgbC, rgbG] = mk(WW, WH);
  [chC, chG] = mk(WW, WH);
  [cmC, cmG] = mk(WW, WH);
  [edC, edG] = mk(WW, WH);
  [xrC, xrG] = mk(WW, WH);
  [blC, blG] = mk(Math.ceil(WW / 7), Math.ceil(WH / 7));
  [denC, denG] = mk(WW, WH, true);
  tex = makeTexture(WW, WH);
  [pixC, pixG] = mk(12, 12);
  mask = new Float32Array(N);
  lum = new Uint8Array(N);
  gw8 = Math.ceil(WW / 8); gh8 = Math.ceil(WH / 8);
  ms = new Float32Array(gw8 * gh8);
  imgData = null;
  cmImg = cmG.createImageData(WW, WH);
  edImg = edG.createImageData(WW, WH);
  xrImg = xrG.createImageData(WW, WH);
  xrG.fillStyle = "#000"; xrG.fillRect(0, 0, WW, WH);
  computeCrop();
}

// Textura trabecular del hueso (ruido suave fijo, varias escalas)
function makeTexture(w, h) {
  const t = new Float32Array(w * h);
  for (const [cell, amp] of [[3, 0.5], [7, 0.35], [17, 0.25]]) {
    const gw = Math.ceil(w / cell) + 2, gh = Math.ceil(h / cell) + 2, r = new Float32Array(gw * gh);
    for (let i = 0; i < r.length; i++) r[i] = Math.random() * 2 - 1;
    for (let y = 0; y < h; y++) {
      const gy = y / cell, y0 = gy | 0, fy = gy - y0;
      for (let x = 0; x < w; x++) {
        const gx = x / cell, x0 = gx | 0, fx = gx - x0, a = y0 * gw + x0;
        t[y * w + x] += amp * ((r[a] * (1 - fx) + r[a + 1] * fx) * (1 - fy) + (r[a + gw] * (1 - fx) + r[a + gw + 1] * fx) * fy);
      }
    }
  }
  return t;
}

function computeCrop() {
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const target = PW / PH;
  if (vw / vh > target) { const cw = vh * target; crop = { x: (vw - cw) / 2, y: 0, w: cw, h: vh }; }
  else { const ch = vw / target; crop = { x: 0, y: (vh - ch) * 0.3, w: vw, h: ch }; }
}

// ---------------------------------------------------------------- Suavizado (One Euro)
class OneEuro {
  constructor(minC = 1.6, beta = 9, dc = 1) { this.minC = minC; this.beta = beta; this.dc = dc; this.t = null; this.x = 0; this.dx = 0; }
  static a(dt, fc) { const tau = 1 / (2 * Math.PI * fc); return 1 / (1 + tau / dt); }
  reset() { this.t = null; }
  f(x, t) {
    if (this.t == null) { this.t = t; this.x = x; this.dx = 0; return x; }
    const dt = Math.max(1e-3, t - this.t); this.t = t;
    const d = (x - this.x) / dt, ad = OneEuro.a(dt, this.dc);
    this.dx = ad * d + (1 - ad) * this.dx;
    const a = OneEuro.a(dt, this.minC + this.beta * Math.abs(this.dx));
    this.x = a * x + (1 - a) * this.x;
    return this.x;
  }
}
const NL = 33;
const filt = Array.from({ length: NL }, () => ({ x: new OneEuro(), y: new OneEuro() }));
const lm = Array.from({ length: NL }, () => ({ x: .5, y: .5, v: 0 }));
const Lp = Array.from({ length: NL }, () => ({ x: 0, y: 0, v: 0 })); // en px del panel
let present = false, lastSeen = 0;

function updateLandmarks(res, now) {
  const l = res.landmarks && res.landmarks[0];
  if (l) {
    const t = now / 1000;
    for (let i = 0; i < NL; i++) {
      const q = l[i];
      lm[i].x = filt[i].x.f(q.x, t);
      lm[i].y = filt[i].y.f(q.y, t);
      lm[i].v = lerp(lm[i].v, q.visibility ?? 1, 0.6);
    }
    present = true; lastSeen = now;
  } else if (now - lastSeen > 350) {
    if (present) for (const f of filt) { f.x.reset(); f.y.reset(); }
    present = false;
    for (const q of lm) q.v *= 0.8;
  }
}
function toPx() { for (let i = 0; i < NL; i++) { Lp[i].x = lm[i].x * PW; Lp[i].y = lm[i].y * PH; Lp[i].v = present ? lm[i].v : 0; } }

// ---------------------------------------------------------------- Cámara y modelo
let stream = null, pose = null;
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("La cámara requiere abrir la página con https://");
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
  });
  video.srcObject = stream;
  await video.play();
  if (!video.videoWidth) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
  mirror = facing === "user";
  computeCrop();
}

async function loadModel(forceCpu = false) {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const coarse = matchMedia("(pointer: coarse)").matches;
  const make = (delegate, model) => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: model, delegate },
    runningMode: "VIDEO", numPoses: 1, outputSegmentationMasks: true,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const model = coarse ? MODEL_LITE : MODEL_FULL;
  try { if (forceCpu || location.search.includes("cpu")) throw new Error("CPU forzado"); pose = await make("GPU", model); }
  catch (e) { console.warn("Usando CPU:", e.message); const old = pose; pose = await make("CPU", MODEL_LITE); if (old && old !== pose) try { old.close(); } catch (_) {} }
}

// ---------------------------------------------------------------- Procesamiento por cuadro
let lastVT = -1, lastTs = 0, frameNo = 0, meanLum = 128, inkDark = true;
let bbox = null, poseConf = 0, noMaskFrames = 0, cpuTried = false;
let fpsCount = 0, fpsT = performance.now(), fps = 0;

function grab(now) {
  wg.save();
  if (mirror) { wg.translate(WW, 0); wg.scale(-1, 1); }
  wg.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, WW, WH);
  wg.restore();
  imgData = wg.getImageData(0, 0, WW, WH);

  const ts = Math.max(lastTs + 1, now); lastTs = ts;
  let gotMask = false;
  pose.detectForVideo(workC, ts, (res) => {
    updateLandmarks(res, now);
    const m = res.segmentationMasks && res.segmentationMasks[0];
    if (m) {
      gotMask = true;
      const d = m.getAsFloat32Array(), mw = m.width, mh = m.height;
      if (mw === WW && mh === WH) {
        for (let i = 0; i < N; i++) mask[i] = mask[i] * 0.4 + d[i] * 0.6;
      } else {
        for (let y = 0; y < WH; y++) {
          const sy = Math.min(mh - 1, (y * mh / WH) | 0) * mw;
          for (let x = 0; x < WW; x++) { const i = y * WW + x; mask[i] = mask[i] * 0.4 + d[sy + Math.min(mw - 1, (x * mw / WW) | 0)] * 0.6; }
        }
      }
    }
  });
  if (!gotMask) for (let i = 0; i < N; i++) mask[i] *= 0.6;

  // Mapa de densidad anatómica (mismo cuadro que la máscara → todo queda alineado)
  toPx();
  denG.setTransform(1, 0, 0, 1, 0, 0);
  denG.globalCompositeOperation = "source-over";
  denG.fillStyle = "#000"; denG.fillRect(0, 0, WW, WH);
  if (present) {
    denG.setTransform(WW / PW, 0, 0, WH / PH, 0, 0);
    drawAnatomy(denG, Lp, { bones: cfg.bones, organs: cfg.organs, mirror, t: now / 1000 });
    den = denG.getImageData(0, 0, WW, WH).data;
  } else den = null;

  buildLayers();
  frameNo++; fpsCount++;
  if (present && !bbox && !cpuTried && ++noMaskFrames > 60) { cpuTried = true; loadModel(true).catch(() => {}); }
  if (bbox) noMaskFrames = 0;
  if (now - fpsT > 1000) { fps = fpsCount * 1000 / (now - fpsT); fpsCount = 0; fpsT = now; $("fps").textContent = fps.toFixed(0) + " fps"; }
}

function buildLayers() {
  const px = imgData.data, w = WW, h = WH;
  // 1) luminancia, promedio y mapa de color
  let sum = 0;
  const wantCm = cfg.cmap, cd = cmImg.data;
  for (let i = 0, j = 0; i < N; i++, j += 4) {
    const l = (px[j] * 77 + px[j + 1] * 150 + px[j + 2] * 29) >> 8;
    lum[i] = l; sum += l;
    if (wantCm) {
      const k = clamp((l - 128) * 1.3 + 128, 0, 255) | 0;
      cd[j] = CMAP[k * 3]; cd[j + 1] = CMAP[k * 3 + 1]; cd[j + 2] = CMAP[k * 3 + 2]; cd[j + 3] = 255;
    }
  }
  if (wantCm) cmG.putImageData(cmImg, 0, 0);
  meanLum = sum / N;
  if (inkDark && meanLum < 95) inkDark = false; else if (!inkDark && meanLum > 150) inkDark = true;

  // 2) contorno (Sobel)
  if (cfg.edges) {
    const e = edImg.data;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = -lum[i - w - 1] + lum[i - w + 1] - 2 * lum[i - 1] + 2 * lum[i + 1] - lum[i + w - 1] + lum[i + w + 1];
      const gy = -lum[i - w - 1] - 2 * lum[i - w] - lum[i - w + 1] + lum[i + w - 1] + 2 * lum[i + w] + lum[i + w + 1];
      const g = clamp((Math.abs(gx) + Math.abs(gy)) * 1.1 - 40, 0, 255), j = i * 4;
      e[j] = 190; e[j + 1] = 255; e[j + 2] = 40; e[j + 3] = g;
    }
    edG.putImageData(edImg, 0, 0);
  }

  // 3) rayos X: densidad del cuerpo a partir de la máscara
  const B = 8;
  ms.fill(0);
  for (let y = 0; y < h; y++) {
    const row = ((y / B) | 0) * gw8, yi = y * w;
    for (let x = 0; x < w; x++) ms[row + ((x / B) | 0)] += mask[yi + x];
  }
  for (let i = 0; i < ms.length; i++) ms[i] /= B * B;

  const lut = PALS[opt.pal].lut, o = xrImg.data, tis = opt.tissue;
  let x0 = w, x1 = 0, y0 = h, y1 = 0, cnt = 0;
  const fr = Math.imul(frameNo, 7919) & 0xfffff;
  for (let y = 0; y < h; y++) {
    const gy = clamp(y / B - 0.5, 0, gh8 - 1.001), gy0 = gy | 0, fy = gy - gy0;
    for (let x = 0; x < w; x++) {
      const i = y * w + x, m = mask[i];
      let rim = 0;
      if (x > 3 && x < w - 4 && y > 3 && y < h - 4) {
        rim = Math.min(1, (Math.abs(mask[i + 3] - mask[i - 3]) + Math.abs(mask[i + 3 * w] - mask[i - 3 * w])) * 1.7);
      }
      let v = 0;
      if (m > 0.02 || rim > 0.02 || (den && den[i * 4] > 0)) {
        const gx = clamp(x / B - 0.5, 0, gw8 - 1.001), gx0 = gx | 0, fx = gx - gx0;
        const a = gy0 * gw8 + gx0;
        const T = (ms[a] * (1 - fx) + ms[a + 1] * fx) * (1 - fy) + (ms[a + gw8] * (1 - fx) + ms[a + gw8 + 1] * fx) * fy;
        // Tejido blando: proporcional al grosor del cuerpo (más denso al centro)
        let Dn = m * (0.12 + 0.62 * T) * tis;
        if (den) {
          const j4 = i * 4, air = den[j4 + 1] / 255, sft = den[j4 + 2] / 255, bon = den[j4] / 255;
          Dn = Dn * (1 - 0.8 * air) + sft * 0.32 * m * tis;
          Dn += bon * (0.82 + 0.28 * tex[i]) * 1.25 * (0.35 + 0.65 * m);
        }
        const n = (Math.imul((i + fr) | 0, 2654435761) >>> 24) / 255 - 0.5;
        v = 1 - Math.exp(-2.1 * Dn) + rim * 0.1 + n * 0.045 * (0.3 + m);
        if (m > 0.5) { cnt++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      }
      const k = clamp(v, 0, 1) * 255 | 0, j = i * 4;
      o[j] = lut[k * 3]; o[j + 1] = lut[k * 3 + 1]; o[j + 2] = lut[k * 3 + 2]; o[j + 3] = 255;
    }
  }
  xrG.putImageData(xrImg, 0, 0);
  bbox = cnt > N * 0.01 ? { x0: x0 / w, x1: x1 / w, y0: y0 / h, y1: y1 / h } : null;
}

// ---------------------------------------------------------------- Utilidades de dibujo
function ink() {
  return inkDark ? { fg: "#101010", halo: "rgba(255,255,255,.35)" } : { fg: "#f2f2f2", halo: "rgba(0,0,0,.55)" };
}
function txt(g, s, x, y, align, c, fs, keepIn) {
  g.font = `600 ${fs}px ${FONT}`;
  if (keepIn) { const hw = g.measureText(s).width / 2 + 6; x = clamp(x, hw, PW - hw); }
  g.textAlign = align; g.textBaseline = "top";
  g.lineJoin = "round"; g.lineWidth = 3; g.strokeStyle = c.halo; g.strokeText(s, x, y);
  g.fillStyle = c.fg; g.fillText(s, x, y);
}

function faceBox() {
  const eL = Lp[5], eR = Lp[2], aL = Lp[8], aR = Lp[7];
  if (eL.v < 0.4 || eR.v < 0.4) return null;
  const ex = (eL.x + eR.x) / 2, ey = (eL.y + eR.y) / 2;
  const ed = Math.hypot(eR.x - eL.x, eR.y - eL.y);
  let e = aL.v > 0.3 && aR.v > 0.3 ? Math.hypot(aR.x - aL.x, aR.y - aL.y) : ed * 2.3;
  e = Math.max(e, ed * 2.0);
  const w = e * 1.38, top = ey - e * 1.02, bottom = ey + e * 0.98;
  return { x: ex - w / 2, y: top, w, h: bottom - top, cx: ex, cy: (top + bottom) / 2 };
}
let smFace = null;

// ---------------------------------------------------------------- Panel superior
const LABELS = ["INDEX 001   →   FACE", "FACE   →   ISOLATION"];
const TOP_PTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16];
const TOP_LINES = [[8, 5], [5, 0], [0, 2], [2, 7], [12, 11], [12, 14], [14, 16], [11, 13], [13, 15]];
let hudT = 0, gpu = 1800;

function keypoints(g, sizeK) {
  g.lineWidth = 1; g.strokeStyle = "rgba(200,255,31,.6)";
  for (const [a, b] of TOP_LINES) {
    if (Lp[a].v < 0.45 || Lp[b].v < 0.45) continue;
    g.beginPath(); g.moveTo(Lp[a].x, Lp[a].y); g.lineTo(Lp[b].x, Lp[b].y); g.stroke();
  }
  g.fillStyle = LIME; g.shadowColor = LIME; g.shadowBlur = 10;
  for (const i of TOP_PTS) {
    if (Lp[i].v < 0.45) continue;
    g.beginPath(); g.arc(Lp[i].x, Lp[i].y, 3.2 * sizeK, 0, 6.2832); g.fill();
  }
  g.shadowBlur = 0;
}

function drawTop(now) {
  const g = tg;
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.globalAlpha = 1; g.globalCompositeOperation = "source-over"; g.imageSmoothingEnabled = true;
  g.fillStyle = "#000"; g.fillRect(0, 0, PW, PH);

  // Glitch RGB: separa canales (se prepara en un canvas pequeño)
  let src = workC;
  const t = now / 1000;
  if (cfg.glitch) {
    const burst = Math.max(0, Math.sin(t * 2.3) * Math.sin(t * 5.1));
    const off = 2 + burst * 14 + Math.random() * 2;
    rgbG.globalCompositeOperation = "source-over"; rgbG.fillStyle = "#000"; rgbG.fillRect(0, 0, WW, WH);
    for (const [col, dx] of [["#f00", -off], ["#0f0", 0], ["#00f", off]]) {
      chG.globalCompositeOperation = "source-over"; chG.drawImage(workC, 0, 0);
      chG.globalCompositeOperation = "multiply"; chG.fillStyle = col; chG.fillRect(0, 0, WW, WH);
      rgbG.globalCompositeOperation = "lighter"; rgbG.drawImage(chC, dx, 0);
    }
    rgbG.globalCompositeOperation = "source-over";
    src = rgbC;
  }

  if (cfg.base) g.drawImage(src, 0, 0, PW, PH);
  else { g.globalAlpha = cfg.cmap ? 0 : 0.28; g.drawImage(src, 0, 0, PW, PH); g.globalAlpha = 1; }
  if (cfg.cmap) g.drawImage(cmC, 0, 0, PW, PH);
  if (cfg.edges) g.drawImage(edC, 0, 0, PW, PH);

  if (cfg.glitch && Math.random() < 0.55) {
    g.setTransform(1, 0, 0, 1, 0, 0);
    const bands = 2 + (Math.random() * 4 | 0);
    for (let k = 0; k < bands; k++) {
      const sy = Math.random() * topC.height, sh = 4 + Math.random() * topC.height * 0.06;
      const dx = (Math.random() - 0.5) * topC.width * 0.14;
      g.drawImage(topC, 0, sy, topC.width, sh, dx, sy, topC.width, sh);
    }
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  // Rostro
  const fb = present ? faceBox() : null;
  if (fb) {
    if (!smFace) smFace = { ...fb };
    for (const k of ["x", "y", "w", "h", "cx", "cy"]) smFace[k] = lerp(smFace[k], fb[k], 0.55);
  } else smFace = null;
  const fs = clamp(PW * 0.0105, 9, 13);

  if (cfg.blur && smFace) {
    const f = smFace, sx = WW / PW, sy = WH / PH;
    const cells = opt.pix, ph = Math.max(2, Math.round(cells * f.h / f.w));
    if (pixC.width !== cells || pixC.height !== ph) { pixC.width = cells; pixC.height = ph; }
    pixG.imageSmoothingQuality = "high";
    pixG.drawImage(src, f.x * sx, f.y * sy, f.w * sx, f.h * sy, 0, 0, cells, ph);
    g.imageSmoothingEnabled = false;
    g.drawImage(pixC, f.x, f.y, f.w, f.h);
    g.imageSmoothingEnabled = true;
    g.strokeStyle = "#ff2e2e"; g.lineWidth = 1.5; g.strokeRect(f.x, f.y, f.w, f.h);
    const bh = Math.round(fs * 2.1);
    g.fillStyle = LIME; g.fillRect(f.x, f.y + f.h, f.w, bh);
    g.fillStyle = "#141414"; g.font = `700 ${fs}px ${FONT}`; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(LABELS[((now / 2200) | 0) % 2], f.x + f.w / 2, f.y + f.h + bh / 2 + 1, f.w - 6);
  }

  if (cfg.boxes && bbox) {
    const bx = bbox.x0 * PW, by = bbox.y0 * PH, bw = (bbox.x1 - bbox.x0) * PW, bh = (bbox.y1 - bbox.y0) * PH, c = Math.min(26, bw * 0.12);
    g.strokeStyle = LIME; g.lineWidth = 2; g.beginPath();
    for (const [x, y, dx, dy] of [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]]) {
      g.moveTo(x + dx * c, y); g.lineTo(x, y); g.lineTo(x, y + dy * c);
    }
    g.stroke();
    g.fillStyle = LIME; g.font = `700 ${fs}px ${FONT}`; g.textAlign = "left"; g.textBaseline = "bottom";
    g.fillStyle = LIME; const lw = g.measureText("INDEX 001  →  BODY").width + 12;
    g.fillRect(bx, by - fs * 1.9, lw, fs * 1.9);
    g.fillStyle = "#141414"; g.fillText("INDEX 001  →  BODY", bx + 6, by - 4);
    if (Lp[0].v > 0.4) {
      g.strokeStyle = "rgba(200,255,31,.85)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(Lp[0].x - 18, Lp[0].y); g.lineTo(Lp[0].x + 18, Lp[0].y); g.moveTo(Lp[0].x, Lp[0].y - 18); g.lineTo(Lp[0].x, Lp[0].y + 18); g.stroke();
      g.beginPath(); g.arc(Lp[0].x, Lp[0].y, 9, 0, 6.2832); g.stroke();
    }
  }
  if (cfg.points && present) keypoints(g, 1);

  if (cfg.scan) {
    const sy = ((now / 2600) % 1) * (PH + 80) - 40;
    const gr = g.createLinearGradient(0, sy - 70, 0, sy);
    gr.addColorStop(0, "rgba(200,255,31,0)"); gr.addColorStop(1, "rgba(200,255,31,.28)");
    g.fillStyle = gr; g.fillRect(0, sy - 70, PW, 70);
    g.fillStyle = "rgba(200,255,31,.9)"; g.fillRect(0, sy, PW, 1.5);
    g.fillStyle = g.createPattern(scanPat, "repeat"); g.fillRect(0, 0, PW, PH);
  }

  // HUD
  if (now - hudT > 140) { hudT = now; gpu = 1780 + ((Math.random() * 30) | 0); }
  const c = ink(), pad = fs * 1.0, lh = fs * 1.5;
  const conf = present ? (0.86 + Math.min(0.04, (poseConf - 0.9) * 0.4 + 0.02)).toFixed(3) : "0.000";
  const fx = smFace ? (smFace.cx / PW).toFixed(3) : "-.---", fy = smFace ? (smFace.cy / PH).toFixed(3) : "-.---";
  txt(g, "GPU USAGE", pad, pad, "left", c, fs); txt(g, `[ ${gpu}  /  5402 ]`, pad, pad + lh, "left", c, fs);
  txt(g, "ENTITY STATUS", PW - pad, pad, "right", c, fs); txt(g, present ? "[ ARTIFICIAL ]" : "[ NO SIGNAL ]", PW - pad, pad + lh, "right", c, fs);
  const my = PH * 0.5 - lh * 1.5;
  txt(g, "FACE ISOLATION", pad, my, "left", c, fs);
  txt(g, `[ X: ${fx}  Y: ${fy} ]`, pad, my + lh, "left", c, fs);
  txt(g, `[ CONF: ${conf} ]`, pad, my + lh * 2, "left", c, fs);
  txt(g, "FOOTAGE", PW - pad, my, "right", c, fs);
  txt(g, `[ BASE STREAM: ${cfg.base ? "ON" : "OFF"} ]`, PW - pad, my + lh, "right", c, fs);
  txt(g, `[ COLOR MAPS: ${cfg.cmap ? "ON" : "OFF"} ]`, PW - pad, my + lh * 2, "right", c, fs);
  txt(g, "FRAMES PER SECOND", pad, PH - pad - lh * 2, "left", c, fs); txt(g, fps.toFixed(1), pad, PH - pad - lh, "left", c, fs);
  let pts = 0, vs = 0;
  for (let i = 0; i < NL; i++) { if (Lp[i].v > 0.5) pts++; vs += Lp[i].v; }
  poseConf = vs / NL;
  txt(g, "POSE ESTIMATION", PW - pad, PH - pad - lh * 2, "right", c, fs);
  txt(g, `[ POINTS: ${pts.toFixed(1)} ]  [ ${(poseConf * 100).toFixed(1)} % ]`, PW - pad, PH - pad - lh, "right", c, fs);
}

let pal = PALS.xray;

// ---------------------------------------------------------------- Panel inferior
const BOT_LABELS = [
  [0.08, 1, 12, "LEFT SHOULDER"], [0.22, 0, 8, "LEFT EAR"], [0.36, 1, 5, "LEFT EYE"], [0.5, 0, 0, "NOSE"],
  [0.64, 1, 7, "RIGHT EAR"], [0.78, 0, 2, "RIGHT EYE"], [0.92, 1, 11, "RIGHT SHOULDER"],
];

function drawBottom(now) {
  const g = bg; pal = PALS[opt.pal];
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.globalAlpha = 1; g.globalCompositeOperation = "source-over"; g.imageSmoothingEnabled = true; g.lineCap = "round"; g.lineJoin = "round";
  g.drawImage(xrC, 0, 0, PW, PH);

  // Resplandor (versión pequeña de la imagen, ampliada)
  blG.drawImage(xrC, 0, 0, blC.width, blC.height);
  if (pal.bloom) { g.globalCompositeOperation = "lighter"; g.globalAlpha = pal.bloom; g.drawImage(blC, 0, 0, PW, PH); g.globalAlpha = 1; }
  g.globalCompositeOperation = "source-over";

  // Cuadrícula
  const cell = PW / 9;
  g.strokeStyle = pal.light ? "rgba(0,0,0,.14)" : "rgba(255,255,255,.14)"; g.lineWidth = 1; g.beginPath();
  for (let x = cell; x < PW; x += cell) { g.moveTo(Math.round(x) + .5, 0); g.lineTo(Math.round(x) + .5, PH); }
  for (let y = cell * 0.9; y < PH; y += cell) { g.moveTo(0, Math.round(y) + .5); g.lineTo(PW, Math.round(y) + .5); }
  g.stroke();

  g.globalCompositeOperation = pal.light ? "source-over" : "lighter";
  if (cfg.scan) {
    const sy = (((now / 3400) + 0.4) % 1) * (PH + 120) - 60;
    const gr = g.createLinearGradient(0, sy - 60, 0, sy + 60);
    gr.addColorStop(0, rgba(pal.glow, 0)); gr.addColorStop(0.5, rgba(pal.glow, 0.16)); gr.addColorStop(1, rgba(pal.glow, 0));
    g.fillStyle = gr; g.fillRect(0, sy - 60, PW, 120);
    g.fillStyle = rgba(pal.glow, 0.45); g.fillRect(0, sy, PW, 1);
  }
  g.globalCompositeOperation = "source-over";
  if (cfg.points && present) keypoints(g, 1);

  // Etiquetas del fondo
  const fs = clamp(PW * 0.0105, 9, 13), lh = fs * 1.45, c = pal.light ? { fg: "#111", halo: "rgba(255,255,255,.7)" } : { fg: "#f2f2f2", halo: "rgba(0,0,0,.7)" };
  for (const [xf, row, idx, name] of BOT_LABELS) {
    const y = PH - fs * (row ? 3.4 : 6.3), x = PW * xf, v = present ? clamp(Lp[idx].v, 0, 1) : 0;
    const tw = Math.max(name.length, 8) * fs * 0.36 + 4;
    const cx = clamp(x, tw, PW - tw);
    txt(g, `[ ${v >= 0.995 ? "1.0" : v.toFixed(2)} ]`, cx, y, "center", c, fs);
    txt(g, name, cx, y + lh, "center", c, fs);
  }
}

// ---------------------------------------------------------------- Bucle
function tick(now) {
  requestAnimationFrame(tick);
  if (!pose || video.readyState < 2) return;
  if (video.currentTime !== lastVT) { lastVT = video.currentTime; grab(now); }
  if (!imgData) return;
  toPx();
  drawTop(now);
  drawBottom(now);
}

// ---------------------------------------------------------------- Interfaz
const chipEls = [...document.querySelectorAll(".chip")];
function refreshChips() { for (const c of chipEls) c.classList.toggle("on", !!cfg[c.dataset.fx]); }
function applyPreset(i) { Object.assign(cfg, Object.fromEntries(Object.entries(PRESETS[i]).map(([k, v]) => [k, !!v]))); refreshChips(); }
setInterval(() => { if (cfg.auto) { presetIdx = (presetIdx + 1) % PRESETS.length; applyPreset(presetIdx); } }, 3800);

for (const c of chipEls) c.addEventListener("click", () => {
  const k = c.dataset.fx;
  cfg[k] = !cfg[k];
  if (k === "auto") { if (cfg.auto) applyPreset(presetIdx); }
  else if (["blur", "points", "boxes", "glitch", "cmap", "edges", "base"].includes(k)) cfg.auto = false;
  refreshChips();
});
$("palette").addEventListener("change", (e) => { opt.pal = e.target.value; });
$("tissue").addEventListener("input", (e) => { opt.tissue = +e.target.value / 100; });
$("bone").addEventListener("input", (e) => { opt.bone = +e.target.value / 100; });
$("pix").addEventListener("input", (e) => { opt.pix = +e.target.value; });
$("gear").addEventListener("click", () => { $("panel").hidden = !$("panel").hidden; });
$("flip").addEventListener("click", async () => {
  facing = facing === "user" ? "environment" : "user";
  try { await startCamera(); for (const f of filt) { f.x.reset(); f.y.reset(); } } catch (e) { facing = facing === "user" ? "environment" : "user"; await startCamera().catch(() => {}); }
});
if (!document.documentElement.requestFullscreen) $("full").hidden = true;
$("full").addEventListener("click", () => { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => {}); });
$("shot").addEventListener("click", () => {
  const horiz = matchMedia("(min-aspect-ratio: 23/20)").matches;
  const [oc, og] = mk(horiz ? topC.width * 2 : topC.width, horiz ? topC.height : topC.height * 2);
  og.drawImage(topC, 0, 0); og.drawImage(botC, horiz ? topC.width : 0, horiz ? 0 : topC.height);
  oc.toBlob((b) => {
    if (!b) return;
    const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "rayos-x.png";
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, "image/png");
});

let resizeT = 0;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => { layout(); }, 120); });

const startBtn = $("start"), statusEl = $("status");
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  try {
    statusEl.textContent = "Pidiendo acceso a la cámara…";
    await startCamera();
    statusEl.textContent = "Cargando modelo de pose (unos segundos)…";
    layout();
    await loadModel();
    $("intro").hidden = true; $("gear").hidden = false; $("panel").hidden = true;
    refreshChips(); layout();
    requestAnimationFrame(tick);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "No se pudo iniciar: " + (e && e.message ? e.message : e);
    startBtn.disabled = false;
  }
});
refreshChips();
