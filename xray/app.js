// RAYOS X — escáner corporal en tiempo real.
// Arriba: cámara + HUD + rostro pixelado + puntos de pose + efectos.
// Abajo: tu cuerpo "transparente" (rayos X) con esqueleto dibujado sobre tu pose real.
// Todo corre en tu dispositivo con MediaPipe (PoseLandmarker + máscara de segmentación).
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
  xray:    { stops: [[0, [0, 0, 0]], [.22, [3, 20, 60]], [.5, [26, 92, 196]], [.78, [118, 208, 255]], [1, [255, 255, 255]]], bone: [233, 248, 255], glow: [67, 184, 255] },
  medico:  { stops: [[0, [0, 0, 0]], [.25, [0, 34, 18]], [.55, [10, 140, 70]], [.8, [120, 255, 170]], [1, [255, 255, 255]]], bone: [232, 255, 240], glow: [61, 255, 122] },
  ambar:   { stops: [[0, [0, 0, 0]], [.25, [40, 18, 0]], [.55, [190, 95, 10]], [.8, [255, 190, 80]], [1, [255, 250, 220]]], bone: [255, 243, 214], glow: [255, 166, 61] },
  termico: { stops: [[0, [0, 0, 0]], [.2, [80, 0, 10]], [.45, [230, 40, 0]], [.7, [255, 170, 0]], [1, [255, 255, 220]]], bone: [255, 255, 255], glow: [255, 90, 0] },
  mono:    { stops: [[0, [0, 0, 0]], [.5, [110, 110, 115]], [1, [255, 255, 255]]], bone: [255, 255, 255], glow: [180, 190, 200] },
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
const opt = { pal: "xray", tissue: 0.42, bone: 0.85, pix: 11 };
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
let workC, wg, rgbC, rgbG, chC, chG, pixC, pixG, cmC, cmG, edC, edG, xrC, xrG, blC, blG;
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
  const fr = frameNo * 40503;
  for (let y = 0; y < h; y++) {
    const gy = clamp(y / B - 0.5, 0, gh8 - 1.001), gy0 = gy | 0, fy = gy - gy0;
    for (let x = 0; x < w; x++) {
      const i = y * w + x, m = mask[i];
      let rim = 0;
      if (x > 3 && x < w - 4 && y > 3 && y < h - 4) {
        rim = Math.min(1, (Math.abs(mask[i + 3] - mask[i - 3]) + Math.abs(mask[i + 3 * w] - mask[i - 3 * w])) * 1.7);
      }
      let v = 0;
      if (m > 0.02 || rim > 0.02) {
        const gx = clamp(x / B - 0.5, 0, gw8 - 1.001), gx0 = gx | 0, fx = gx - gx0;
        const a = gy0 * gw8 + gx0;
        const T = (ms[a] * (1 - fx) + ms[a + 1] * fx) * (1 - fy) + (ms[a + gw8] * (1 - fx) + ms[a + gw8 + 1] * fx) * fy;
        const d = 1 - lum[i] / 255;
        const core = m * m * (0.10 + 0.50 * T * T);
        const detail = m * (d * 0.40 - 0.06);
        const n = ((Math.imul(i, 2654435761) ^ fr) & 255) / 255 - 0.5;
        v = (core + detail) * tis * 1.5 + rim * 0.9 + n * 0.05 * m;
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

// ---------------------------------------------------------------- Esqueleto de rayos X
let G = null; // contexto de dibujo actual
let pal = PALS.xray;

const gs = (path, w, al) => {
  G.strokeStyle = rgba(pal.glow, al * 0.22); G.lineWidth = w * 3.2; G.stroke(path);
  G.strokeStyle = rgba(pal.bone, al); G.lineWidth = w; G.stroke(path);
};
function circle(x, y, r, al, fill = 0.25) {
  const p = new Path2D(); p.arc(x, y, Math.max(0.5, r), 0, 6.2832);
  G.fillStyle = rgba(pal.bone, al * fill); G.fill(p); gs(p, Math.max(1, r * 0.16), al * 0.9);
}
function bone(a, b, r, al) {
  const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1, nx = -dy / L * r * 0.42, ny = dx / L * r * 0.42;
  const sh = new Path2D();
  sh.moveTo(a.x + nx, a.y + ny); sh.lineTo(b.x + nx, b.y + ny); sh.lineTo(b.x - nx, b.y - ny); sh.lineTo(a.x - nx, a.y - ny); sh.closePath();
  G.fillStyle = rgba(pal.bone, al * 0.22); G.fill(sh); gs(sh, Math.max(1, r * 0.18), al * 0.8);
  circle(a.x, a.y, r * 0.85, al); circle(b.x, b.y, r * 0.7, al);
}
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const add = (a, b, k = 1) => ({ x: a.x + b.x * k, y: a.y + b.y * k });

function drawSkull(al) {
  const eL = Lp[5], eR = Lp[2];
  if (eL.v < 0.5 || eR.v < 0.5) return;
  const ed = Math.hypot(eR.x - eL.x, eR.y - eL.y) || 1;
  const e = Math.max(ed * 2.0, Lp[8].v > 0.3 && Lp[7].v > 0.3 ? Math.hypot(Lp[7].x - Lp[8].x, Lp[7].y - Lp[8].y) : 0);
  const d = { x: (eR.x - eL.x) / ed, y: (eR.y - eL.y) / ed }, dn = { x: -d.y, y: d.x };
  const O = mid(eL, eR);
  const lp = (x, y) => ({ x: O.x + d.x * x * e + dn.x * y * e, y: O.y + d.y * x * e + dn.y * y * e });
  // Cráneo
  const sk = new Path2D();
  for (let i = 0; i <= 48; i++) {
    const a = Math.PI + (i / 48) * Math.PI, p = lp(Math.cos(a) * 0.53, -0.24 + Math.sin(a) * 0.56);
    i ? sk.lineTo(p.x, p.y) : sk.moveTo(p.x, p.y);
  }
  // Mandíbula
  for (const [x, y] of [[0.53, -0.1], [0.5, 0.12], [0.42, 0.45], [0.22, 0.72], [0, 0.8], [-0.22, 0.72], [-0.42, 0.45], [-0.5, 0.12], [-0.53, -0.1]]) { const p = lp(x, y); sk.lineTo(p.x, p.y); }
  G.fillStyle = rgba(pal.bone, al * 0.12); G.fill(sk); gs(sk, Math.max(1.2, e * 0.022), al);
  // Cuencas de los ojos
  for (const [x, s] of [[-0.31, 1], [0.31, 1]]) {
    const c = lp(x, 0), p = new Path2D();
    p.ellipse(c.x, c.y, e * 0.15, e * 0.12, Math.atan2(d.y, d.x), 0, 6.2832);
    G.fillStyle = "rgba(0,0,0,.65)"; G.globalCompositeOperation = "source-over"; G.fill(p); G.globalCompositeOperation = "lighter";
    gs(p, Math.max(1, e * 0.02), al * 0.95);
  }
  // Nariz, pómulos y dientes
  const nz = new Path2D(); let p0 = lp(0, 0.1); nz.moveTo(p0.x, p0.y);
  for (const [x, y] of [[-0.07, 0.36], [0.07, 0.36]]) { const p = lp(x, y); nz.lineTo(p.x, p.y); } nz.closePath();
  gs(nz, Math.max(1, e * 0.016), al * 0.8);
  const ck = new Path2D();
  for (const s of [-1, 1]) { let p = lp(0.2 * s, 0.16); ck.moveTo(p.x, p.y); p = lp(0.48 * s, 0.14); ck.quadraticCurveTo(lp(0.36 * s, 0.26).x, lp(0.36 * s, 0.26).y, p.x, p.y); }
  gs(ck, Math.max(1, e * 0.014), al * 0.6);
  const th = new Path2D(); let pa = lp(-0.17, 0.52), pb = lp(0.17, 0.52); th.moveTo(pa.x, pa.y); th.lineTo(pb.x, pb.y);
  for (let i = 0; i <= 8; i++) { const p = lp(-0.17 + i * 0.0425, 0.52), q = lp(-0.17 + i * 0.0425, 0.58); th.moveTo(p.x, p.y); th.lineTo(q.x, q.y); }
  gs(th, Math.max(1, e * 0.012), al * 0.7);
  return lp;
}

function drawBody(t) {
  const A = opt.bone;
  const hasSh = Lp[11].v > 0.5 && Lp[12].v > 0.5;
  const lpHead = cfg.bones ? drawSkull(A) : null;
  if (!hasSh) return;
  const a = Lp[12], b = Lp[11];                 // izquierda y derecha de la pantalla
  const sw = Math.hypot(b.x - a.x, b.y - a.y) || 1;
  const dsh = { x: (b.x - a.x) / sw, y: (b.y - a.y) / sw };
  const S0 = mid(a, b);
  let ax = { x: -dsh.y, y: dsh.x }, len = sw * 1.4, hipsOK = false;
  if (Lp[23].v > 0.55 && Lp[24].v > 0.55) {
    const H = mid(Lp[23], Lp[24]), v = { x: H.x - S0.x, y: H.y - S0.y }, L = Math.hypot(v.x, v.y);
    if (L > sw * 0.9 && L < sw * 2.6) { ax = { x: v.x / L, y: v.y / L }, len = L; hipsOK = true; }
  }
  const P = (k, s) => ({ x: S0.x + ax.x * len * k + dsh.x * sw * s, y: S0.y + ax.y * len * k + dsh.y * sw * s });
  const tA = hipsOK ? 1 : 0.8;

  // Órganos (oscurecen / brillan suavemente)
  if (cfg.organs) {
    G.globalCompositeOperation = "source-over";
    const rot = Math.atan2(ax.y, ax.x) - Math.PI / 2;
    for (const s of [-1, 1]) {
      const c = P(0.26, 0.2 * s), p = new Path2D();
      p.ellipse(c.x, c.y, sw * 0.16, len * 0.2, rot, 0, 6.2832);
      G.fillStyle = "rgba(0,0,0,.32)"; G.fill(p);
      G.strokeStyle = rgba(pal.glow, 0.18 * A); G.lineWidth = 1; G.stroke(p);
    }
    G.globalCompositeOperation = "lighter";
    const beat = Math.pow(Math.max(0, Math.sin(t * 6.9)), 6) * 0.7 + Math.pow(Math.max(0, Math.sin(t * 6.9 - 0.9)), 6) * 0.4;
    const hc = P(0.3, 0.07), hr = sw * 0.1 * (1 + beat * 0.14);
    const gr = G.createRadialGradient(hc.x, hc.y, 0, hc.x, hc.y, hr * 1.8);
    gr.addColorStop(0, rgba(pal.glow, (0.4 + beat * 0.4) * A)); gr.addColorStop(1, rgba(pal.glow, 0));
    G.fillStyle = gr; G.beginPath(); G.arc(hc.x, hc.y, hr * 1.8, 0, 6.2832); G.fill();
  }
  G.globalCompositeOperation = "lighter";
  if (!cfg.bones) return;

  // Columna: cervical + torácica
  const base = lpHead ? lpHead(0, 0.5) : add(S0, ax, -sw * 0.7);
  const nb = add(S0, ax, -sw * 0.1);
  const neck = new Path2D(); neck.moveTo(base.x, base.y); neck.lineTo(nb.x, nb.y);
  gs(neck, 1, A * 0.35);
  for (let i = 0; i < 7; i++) {
    const k = i / 6, c = { x: lerp(base.x, nb.x, k), y: lerp(base.y, nb.y, k) }, vw = sw * 0.045;
    const v = new Path2D(); v.moveTo(c.x - dsh.x * vw, c.y - dsh.y * vw); v.lineTo(c.x + dsh.x * vw, c.y + dsh.y * vw);
    G.lineCap = "round"; gs(v, sw * 0.032, A * 0.9);
  }
  for (let i = 0; i < 13; i++) {
    const k = 0.0 + (i / 12) * 0.72 * tA, c = P(k - 0.04, 0), vw = sw * (0.05 + k * 0.03);
    const al = A * (1 - (i / 12) * (hipsOK ? 0 : 0.55));
    const v = new Path2D(); v.moveTo(c.x - dsh.x * vw, c.y - dsh.y * vw); v.lineTo(c.x + dsh.x * vw, c.y + dsh.y * vw);
    gs(v, sw * (0.04 + k * 0.025), al);
  }

  // Clavículas
  for (const s of [a, b]) {
    const sgn = s === a ? -1 : 1, n0 = add(S0, ax, sw * 0.02);
    const cl = new Path2D(); cl.moveTo(n0.x, n0.y);
    const cp = { x: (n0.x + s.x) / 2 - ax.x * sw * 0.07, y: (n0.y + s.y) / 2 - ax.y * sw * 0.07 };
    cl.quadraticCurveTo(cp.x, cp.y, s.x - dsh.x * sgn * 0, s.y);
    gs(cl, sw * 0.03, A * 0.95);
  }

  // Costillas
  const nR = 7;
  for (let k = 0; k < nR; k++) {
    const tt = 0.06 + k * 0.085, al = A * 0.85 * (1 - (k / nR) * (hipsOK ? 0.15 : 0.4));
    const wmax = sw * 0.5 * (0.62 + 0.38 * Math.sin(Math.PI * clamp((tt + 0.12) / 0.85, 0, 1)));
    for (const s of [-1, 1]) {
      const st = P(tt, 0), peak = P(tt + 0.01, s * wmax * 0.78 / sw), end = P(tt + 0.07 + k * 0.004, s * wmax / sw), front = P(tt + 0.14, s * wmax * 0.12 / sw);
      const r = new Path2D(); r.moveTo(st.x, st.y);
      r.quadraticCurveTo(peak.x, peak.y - 0 * len, end.x, end.y);
      gs(r, sw * 0.022, al);
      const cc = new Path2D(); cc.moveTo(end.x, end.y);
      cc.quadraticCurveTo(P(tt + 0.13, s * wmax * 0.62 / sw).x, P(tt + 0.13, s * wmax * 0.62 / sw).y, front.x, front.y);
      gs(cc, sw * 0.014, al * 0.45);
    }
  }
  const stn = new Path2D(); const s1 = P(0.04, 0.01), s2 = P(0.5 * tA, 0.01); stn.moveTo(s1.x, s1.y); stn.lineTo(s2.x, s2.y);
  gs(stn, sw * 0.014, A * 0.35);

  // Hombros y brazos
  const arm = (sh, el, wr, hand) => {
    if (sh.v < 0.5) return;
    circle(sh.x, sh.y, sw * 0.075, A, 0.3);
    if (el.v < 0.45) return;
    bone(sh, el, sw * 0.05, A);
    if (wr.v < 0.45) return;
    const dx = wr.x - el.x, dy = wr.y - el.y, L = Math.hypot(dx, dy) || 1, ox = -dy / L * sw * 0.017, oy = dx / L * sw * 0.017;
    bone({ x: el.x + ox, y: el.y + oy }, { x: wr.x + ox * 0.4, y: wr.y + oy * 0.4 }, sw * 0.03, A * 0.95);
    bone({ x: el.x - ox, y: el.y - oy }, { x: wr.x - ox * 0.4, y: wr.y - oy * 0.4 }, sw * 0.028, A * 0.95);
    for (const h of hand) {
      if (h.v < 0.3) continue;
      const hb = new Path2D(); hb.moveTo(wr.x, wr.y); hb.lineTo(h.x, h.y);
      gs(hb, sw * 0.014, A * 0.85);
      const tip = { x: h.x + (h.x - wr.x) * 0.55, y: h.y + (h.y - wr.y) * 0.55 };
      const fb = new Path2D(); fb.moveTo(h.x, h.y); fb.lineTo(tip.x, tip.y);
      gs(fb, sw * 0.012, A * 0.7); circle(h.x, h.y, sw * 0.014, A * 0.9); circle(tip.x, tip.y, sw * 0.01, A * 0.8);
    }
  };
  arm(Lp[12], Lp[14], Lp[16], [Lp[18], Lp[20], Lp[22]]);
  arm(Lp[11], Lp[13], Lp[15], [Lp[17], Lp[19], Lp[21]]);

  // Pelvis y piernas (solo si se ven)
  if (hipsOK) {
    const hl = Lp[24], hr = Lp[23], hw = Math.hypot(hr.x - hl.x, hr.y - hl.y);
    const pel = new Path2D();
    for (const s of [hl, hr]) {
      const sg = s === hl ? -1 : 1, top = { x: s.x - ax.x * hw * 0.45 + dsh.x * sg * hw * 0.12, y: s.y - ax.y * hw * 0.45 + dsh.y * sg * hw * 0.12 };
      pel.moveTo(S0.x + ax.x * len * 0.98, S0.y + ax.y * len * 0.98);
      pel.quadraticCurveTo(top.x, top.y, s.x + dsh.x * sg * hw * 0.2, s.y);
    }
    gs(pel, hw * 0.035, A * 0.8);
    circle(hl.x, hl.y, hw * 0.1, A, 0.3); circle(hr.x, hr.y, hw * 0.1, A, 0.3);
    const leg = (h, k, an, foot) => {
      if (k.v < 0.45) return; bone(h, k, hw * 0.1, A);
      if (an.v < 0.45) return; bone(k, an, hw * 0.08, A);
      if (foot.v > 0.4) bone(an, foot, hw * 0.05, A * 0.8);
    };
    leg(hl, Lp[26], Lp[28], Lp[32]); leg(hr, Lp[25], Lp[27], Lp[31]);
  }
}

// ---------------------------------------------------------------- Panel inferior
const BOT_LABELS = [
  [0.08, 1, 12, "LEFT SHOULDER"], [0.22, 0, 8, "LEFT EAR"], [0.36, 1, 5, "LEFT EYE"], [0.5, 0, 0, "NOSE"],
  [0.64, 1, 7, "RIGHT EAR"], [0.78, 0, 2, "RIGHT EYE"], [0.92, 1, 11, "RIGHT SHOULDER"],
];

function drawBottom(now) {
  const g = bg; G = g; pal = PALS[opt.pal];
  g.setTransform(DPR, 0, 0, DPR, 0, 0);
  g.globalAlpha = 1; g.globalCompositeOperation = "source-over"; g.imageSmoothingEnabled = true; g.lineCap = "round"; g.lineJoin = "round";
  g.drawImage(xrC, 0, 0, PW, PH);

  // Resplandor (versión pequeña de la imagen, ampliada)
  blG.drawImage(xrC, 0, 0, blC.width, blC.height);
  g.globalCompositeOperation = "lighter"; g.globalAlpha = 0.45; g.drawImage(blC, 0, 0, PW, PH); g.globalAlpha = 1;

  // Cuadrícula
  const cell = PW / 9;
  g.strokeStyle = "rgba(255,255,255,.17)"; g.lineWidth = 1; g.beginPath();
  for (let x = cell; x < PW; x += cell) { g.moveTo(Math.round(x) + .5, 0); g.lineTo(Math.round(x) + .5, PH); }
  for (let y = cell * 0.9; y < PH; y += cell) { g.moveTo(0, Math.round(y) + .5); g.lineTo(PW, Math.round(y) + .5); }
  g.stroke();

  if (present) drawBody(now / 1000);

  g.globalCompositeOperation = "lighter";
  if (cfg.scan) {
    const sy = (((now / 3400) + 0.4) % 1) * (PH + 120) - 60;
    const gr = g.createLinearGradient(0, sy - 60, 0, sy + 60);
    gr.addColorStop(0, rgba(pal.glow, 0)); gr.addColorStop(0.5, rgba(pal.glow, 0.16)); gr.addColorStop(1, rgba(pal.glow, 0));
    g.fillStyle = gr; g.fillRect(0, sy - 60, PW, 120);
    g.fillStyle = rgba(pal.bone, 0.6); g.fillRect(0, sy, PW, 1);
  }
  g.globalCompositeOperation = "source-over";
  if (cfg.points && present) keypoints(g, 1);

  // Etiquetas del fondo
  const fs = clamp(PW * 0.0105, 9, 13), lh = fs * 1.45, c = { fg: "#f2f2f2", halo: "rgba(0,0,0,.7)" };
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
