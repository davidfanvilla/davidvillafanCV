// Cuerpo ASCII 3D — segmentación de persona con MediaPipe (GPU) + render Canvas 2D optimizado.
// El fondo es el video real de la cámara; solo el cuerpo se reemplaza por caracteres en 3D.
import { FilesetResolver, ImageSegmenter } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";

// Mismo conjunto de caracteres que el sketch original
const CHAR_POOL = [" ", ".", "-", "=", "+", "π", "ℏ", "ψ", "λ", "Ω", "Σ", "∞", "∇", "42", "137", "e", "c", "G", "X", "Z", "Q", "M", "CIA", "MK", "UFO"];
const SLOTS = 20;            // caracteres activos (como asciiChars en el original)
const SHUFFLE_MS = 250;      // ≈ cada 15 frames a 60 fps
const SEG_SIZE = 256;        // resolución de entrada para la segmentación (lado mayor)
const GLYPH = 48;            // tamaño de cada glifo en el atlas (px)

const PALETTES = {
  cyan:   { front: "#eaffff", glow: "#19f6ff", mid: "#11a9b8", back: "#04363b" },
  matrix: { front: "#d8ffd8", glow: "#00ff66", mid: "#00a845", back: "#003314" },
  fire:   { front: "#fff3d6", glow: "#ff7a00", mid: "#d13b00", back: "#3d0b00" },
  mono:   { front: "#ffffff", glow: "#9fb3ff", mid: "#8a8a8a", back: "#262626" },
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const video = $("video"), stage = $("stage"), ctx = stage.getContext("2d", { alpha: false });
const ui = { dur: $("dur"), cell: $("cell"), depth: $("depth"), spin: $("spin"), palette: $("palette"), fps: $("fps") };

// Canvas pequeño: entrada para la segmentación y para leer el brillo
const seg = document.createElement("canvas");
const segCtx = seg.getContext("2d", { willReadFrequently: true });
// Canvas de la silueta (oscurece el cuerpo para que el ASCII resalte)
const sil = document.createElement("canvas");
const silCtx = sil.getContext("2d");
let silImg;
// --- Borrado del cuerpo real ---
// Se aprende una "placa" del fondo (solo con los píxeles donde no hay persona) y se
// pinta encima del cuerpo al mismo ritmo en que se deshace el código.
const mk = (w = 1, h = 1) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
const PLATE_SIZE = 480;                  // resolución de la placa del fondo (lado mayor)
const silBase = mk(), eraseC = mk(), learnMaskC = mk(), cellKC = mk();
const frameC = mk(), learnC = mk(), plateC = mk(), fillC = mk(), outC = mk();
const lv = [mk(), mk()];                 // versiones pequeñas para rellenar huecos del fondo
const c2 = (c) => c.getContext("2d");
let eraseImg, learnImg, cellKImg, dmask = new Float32Array(0), tmpMask = new Float32Array(0);
let kCell = new Float32Array(0);

// ---------- Estado ----------
let segmenter, stream, facing = "user";
let W = 0, H = 0, DPR = 1, cell = 12, cols = 0, rows = 0;
let crop = { x: 0, y: 0, w: 1, h: 1 };
let mask = new Float32Array(0);          // máscara suavizada (temporal)
let cellIdx = new Int32Array(0);         // celda -> índice en la máscara
let atlas, atlasCols = 0;
let slots = new Uint8Array(SLOTS);       // índices a CHAR_POOL
let lastShuffle = 0, lastVideoTime = -1;
let pivotX = 0, pivotY = 0;
let px = new Uint8ClampedArray(0);       // píxeles del frame pequeño (brillo)
// Desintegración: el código aparece, se sostiene y luego se deshace hasta desaparecer
const HOLD_MS = 3000;                    // tiempo con el cuerpo completo antes de deshacerse
const FADE_WIN = 0.18;                   // cuánto dura la caída de cada carácter (fracción)
let dissolveStart = 0;                   // 0 = esperando detectar un cuerpo
let rndA = new Float32Array(0), rndB = new Float32Array(0);
// Buffers reutilizables (sin asignaciones por frame)
let bufX, bufY, bufS, bufBX, bufBY, bufBS, bufA, bufG;

// ---------- Atlas de glifos (pre-render con glow: cero costo de sombras por frame) ----------
function buildAtlas(name) {
  const p = PALETTES[name];
  atlasCols = CHAR_POOL.length;
  atlas = document.createElement("canvas");
  atlas.width = atlasCols * GLYPH;
  atlas.height = GLYPH * 3; // fila 0: frente, 1: medio, 2: fondo (extrusión)
  const a = atlas.getContext("2d");
  a.textAlign = "center";
  a.textBaseline = "middle";
  const rowsDef = [[p.front, p.glow, 10], [p.mid, null, 0], [p.back, null, 0]];
  rowsDef.forEach(([color, glow, blur], r) => {
    CHAR_POOL.forEach((ch, i) => {
      if (ch === " ") return;
      const size = GLYPH * 0.72;
      a.font = `bold ${size}px ui-monospace, Menlo, Consolas, monospace`;
      const w = a.measureText(ch).width;
      const k = Math.min(1, (GLYPH * 0.86) / w);
      a.save();
      a.translate(i * GLYPH + GLYPH / 2, r * GLYPH + GLYPH / 2);
      a.scale(k, k);
      a.fillStyle = color;
      if (glow) { a.shadowColor = glow; a.shadowBlur = blur; }
      a.fillText(ch, 0, 0);
      a.restore();
    });
  });
  document.documentElement.style.setProperty("--accent", p.glow);
}

function shuffle() {
  for (let i = 0; i < SLOTS; i++) slots[i] = (Math.random() * CHAR_POOL.length) | 0;
}

// ---------- Layout ----------
function layout() {
  DPR = Math.min(window.devicePixelRatio || 1, 1.5);
  W = window.innerWidth; H = window.innerHeight;
  stage.width = Math.round(W * DPR); stage.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.imageSmoothingEnabled = true;

  // Recorte "cover" del video para llenar la pantalla sin deformar
  const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
  const s = Math.max(W / vw, H / vh);
  crop.w = W / s; crop.h = H / s;
  crop.x = (vw - crop.w) / 2; crop.y = (vh - crop.h) / 2;

  // Canvas de segmentación con el mismo aspecto que la pantalla
  if (W >= H) { seg.width = SEG_SIZE; seg.height = Math.max(32, Math.round(SEG_SIZE * H / W)); }
  else { seg.height = SEG_SIZE; seg.width = Math.max(32, Math.round(SEG_SIZE * W / H)); }
  for (const c of [sil, silBase, eraseC, learnMaskC]) { c.width = seg.width; c.height = seg.height; }
  silImg = silCtx.createImageData(sil.width, sil.height);
  eraseImg = silCtx.createImageData(sil.width, sil.height);
  learnImg = silCtx.createImageData(sil.width, sil.height);
  mask = new Float32Array(seg.width * seg.height);
  dmask = new Float32Array(mask.length); tmpMask = new Float32Array(mask.length);
  const pk = PLATE_SIZE / Math.max(W, H);
  const pw = Math.max(16, Math.round(W * pk)), ph = Math.max(16, Math.round(H * pk));
  for (const c of [frameC, learnC, plateC, fillC, outC]) { c.width = pw; c.height = ph; }
  lv.forEach((c, i) => { const f = 4 * 2 ** i; c.width = Math.max(2, Math.round(pw / f)); c.height = Math.max(2, Math.round(ph / f)); });
  pivotX = W / 2; pivotY = H / 2;
  layoutGrid();
}

function layoutGrid() {
  cell = +ui.cell.value;
  cols = Math.ceil(W / cell); rows = Math.ceil(H / cell);
  const n = cols * rows;
  cellIdx = new Int32Array(n);
  for (let gy = 0; gy < rows; gy++) {
    const sy = Math.min(seg.height - 1, (((gy + 0.5) * cell) / H * seg.height) | 0);
    for (let gx = 0; gx < cols; gx++) {
      const sx = Math.min(seg.width - 1, (((gx + 0.5) * cell) / W * seg.width) | 0);
      cellIdx[gy * cols + gx] = sy * seg.width + sx;
    }
  }
  bufX = new Float32Array(n); bufY = new Float32Array(n); bufS = new Float32Array(n);
  bufBX = new Float32Array(n); bufBY = new Float32Array(n); bufBS = new Float32Array(n);
  bufA = new Float32Array(n); bufG = new Uint8Array(n);
  kCell = new Float32Array(n);
  cellKC.width = cols; cellKC.height = rows;
  cellKImg = c2(cellKC).createImageData(cols, rows);
  // Umbral de desintegración por celda: ruido + de arriba hacia abajo (como ceniza)
  rndA = new Float32Array(n); rndB = new Float32Array(n);
  for (let gy = 0; gy < rows; gy++) for (let gx = 0; gx < cols; gx++) {
    const i = gy * cols + gx;
    rndA[i] = Math.random() * 0.6 + (gy / rows) * 0.4;
    rndB[i] = Math.random() - 0.5;
  }
}

// Progreso de la desintegración: 0 = completo, >1 = no queda nada
function dissolveProgress(now) {
  if (!dissolveStart) return 0;
  const dur = +ui.dur.value * 1000;
  return Math.max(0, (now - dissolveStart - HOLD_MS) / dur) * (1 + FADE_WIN);
}

// ---------- Cámara ----------
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
  layout();
}

async function loadModel() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });
  try { segmenter = await ImageSegmenter.createFromOptions(fileset, opts("GPU")); }
  catch (e) { console.warn("GPU no disponible, usando CPU", e); segmenter = await ImageSegmenter.createFromOptions(fileset, opts("CPU")); }
}

// ---------- Bucle principal ----------
let fpsFrames = 0, fpsT = performance.now();

function frame(now) {
  requestAnimationFrame(frame);
  if (video.readyState < 2) return;
  const mirror = facing === "user";
  const sw = seg.width, sh = seg.height;
  const prog = dissolveProgress(now);
  const finished = prog >= 1 + FADE_WIN;

  // 1) Frame pequeño de la cámara (espejado si es la cámara frontal)
  const newFrame = video.currentTime !== lastVideoTime;
  if (newFrame) {
    lastVideoTime = video.currentTime;
    segCtx.save();
    if (mirror) { segCtx.translate(sw, 0); segCtx.scale(-1, 1); }
    segCtx.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, sw, sh);
    segCtx.restore();

    // 2) Segmentación (solo personas) + suavizado temporal para evitar parpadeo
    segmenter.segmentForVideo(seg, now, (res) => {
      const m = res.confidenceMasks && res.confidenceMasks[0];
      if (!m) return;
      const data = m.getAsFloat32Array();
      for (let i = 0; i < data.length; i++) mask[i] = mask[i] * 0.35 + data[i] * 0.65;
    });

    // Máscara dilatada: cubre bien los bordes del cuerpo (cabello, manos)
    dilate(mask, tmpMask, sw, sh); dilate(tmpMask, dmask, sw, sh);
    const d = silImg.data, e = eraseImg.data, l = learnImg.data;
    for (let i = 0, j = 3; i < mask.length; i++, j += 4) {
      d[j] = mask[i] * 215;                                  // silueta oscura
      e[j] = Math.min(1, dmask[i] * 1.6) * 255;              // zona a borrar
      l[j] = Math.max(0, 1 - dmask[i] * 3) * 255;            // zona de fondo limpio
    }
    c2(silBase).putImageData(silImg, 0, 0);
    c2(eraseC).putImageData(eraseImg, 0, 0);
    c2(learnMaskC).putImageData(learnImg, 0, 0);
    px = segCtx.getImageData(0, 0, sw, sh).data;
    updatePlate(mirror);
  }
  if (px.length !== sw * sh * 4) return;

  // Progreso de desintegración por celda (0 intacto → 1 desaparecido)
  const ka = cellKImg.data;
  for (let i = 0, j = 3; i < kCell.length; i++, j += 4) {
    const k = prog <= rndA[i] ? 0 : Math.min(1, (prog - rndA[i]) / FADE_WIN);
    kCell[i] = k; ka[j] = k * 255;
  }
  c2(cellKC).putImageData(cellKImg, 0, 0);

  // 3) Mundo real de fondo (video a resolución completa)
  ctx.save();
  if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, W, H);
  ctx.restore();
  // Escala para dibujar la rejilla de celdas sobre otro canvas (la rejilla puede exceder la pantalla)
  const gwk = (cols * cell) / W, ghk = (rows * cell) / H;
  if (prog > 0) {
    // Tu cuerpo real se deshace: el fondo aprendido lo reemplaza celda por celda
    const o = c2(outC), ow = outC.width, oh = outC.height;
    o.globalCompositeOperation = "copy";
    o.drawImage(eraseC, 0, 0, ow, oh);
    o.globalCompositeOperation = "destination-in";
    o.drawImage(cellKC, 0, 0, ow * gwk, oh * ghk);
    o.globalCompositeOperation = "source-in";
    o.drawImage(fillC, 0, 0);
    o.globalCompositeOperation = "source-over";
    ctx.drawImage(outC, 0, 0, W, H);
  }
  // La silueta oscura desaparece junto con cada carácter
  const sc = silCtx;
  sc.globalCompositeOperation = "copy";
  sc.drawImage(silBase, 0, 0);
  if (prog > 0) {
    sc.globalCompositeOperation = "destination-out";
    sc.drawImage(cellKC, 0, 0, sil.width * gwk, sil.height * ghk);
  }
  sc.globalCompositeOperation = "source-over";
  if (!finished) ctx.drawImage(sil, 0, 0, W, H);

  // 4) Cambiar caracteres periódicamente (como el sketch original)
  if (now - lastShuffle > SHUFFLE_MS) { shuffle(); lastShuffle = now; }

  // 5) Proyección 3D de cada carácter del cuerpo
  const t = now / 1000;
  const spin = +ui.spin.value / 100;
  const depth = (+ui.depth.value / 100) * cell * 14;
  const extr = (+ui.depth.value / 100) * cell * 1.6 + 2;
  const ay = Math.sin(t * 0.7) * 0.6 * spin, ax = Math.sin(t * 0.43) * 0.3 * spin;
  const cyA = Math.cos(ay), syA = Math.sin(ay), cxA = Math.cos(ax), sxA = Math.sin(ax);
  const F = Math.max(W, H) * 1.2; // distancia focal
  const half = cell / 2;
  let count = 0, sumX = 0, sumY = 0;

  for (let gy = 0; gy < rows; gy++) {
    const y0 = gy * cell + half;
    const ripple = Math.sin(t * 2.2 + gy * 0.22) * cell * 0.6 * (depth > 0 ? 1 : 0);
    for (let gx = 0; gx < cols; gx++) {
      const ci = gy * cols + gx;
      const mi = cellIdx[ci];
      const conf = mask[mi];
      if (conf < 0.5) continue;
      // Desintegración: k = 0 intacto, 1 desaparecido
      const k = kCell[ci];
      if (k >= 1) continue;
      const p = mi * 4;
      const lum = (px[p] * 0.299 + px[p + 1] * 0.587 + px[p + 2] * 0.114) / 255;
      const g = slots[Math.min(SLOTS - 1, (lum * SLOTS) | 0)];
      if (g === 0) continue; // espacio
      // Los caracteres que se deshacen flotan hacia arriba y se dispersan como ceniza
      const k2 = k * k;
      const x0 = gx * cell + half + rndB[ci] * k2 * cell * 10 + Math.sin(t * 3 + ci) * k * cell;
      const yk = y0 - k2 * cell * 9;
      sumX += gx * cell + half; sumY += y0;

      // Lo más brillante "sale" hacia la cámara
      const z = -(lum - 0.35) * depth + ripple - k2 * depth * 0.8;
      const dx = x0 - pivotX, dy = yk - pivotY;
      // Rotación Y y luego X
      let rx = dx * cyA + z * syA;
      let rz = -dx * syA + z * cyA;
      let ry = dy * cxA - rz * sxA;
      rz = dy * sxA + rz * cxA;
      let s = F / (F + rz);
      bufX[count] = pivotX + rx * s; bufY[count] = pivotY + ry * s; bufS[count] = s;
      // Capa de extrusión (misma posición desplazada hacia atrás en Z)
      const zb = z + extr;
      rx = dx * cyA + zb * syA;
      rz = -dx * syA + zb * cyA;
      ry = dy * cxA - rz * sxA;
      rz = dy * sxA + rz * cxA;
      s = F / (F + rz);
      bufBX[count] = pivotX + rx * s; bufBY[count] = pivotY + ry * s; bufBS[count] = s;
      bufA[count] = Math.min(1, (conf - 0.5) * 3 + 0.2) * (0.55 + lum * 0.45) * (1 - k);
      bufG[count] = g;
      count++;
    }
  }
  // El pivote de giro sigue al cuerpo, para que el ASCII no se despegue de la persona
  // Empieza la cuenta en cuanto aparece un cuerpo
  if (!dissolveStart && count > 80) dissolveStart = now;
  if (count > 0) { pivotX += (sumX / count - pivotX) * 0.15; pivotY += (sumY / count - pivotY) * 0.15; }

  // 6) Dibujo: primero toda la extrusión (fondo + medio), luego el frente
  const size = cell * 1.35;
  for (let pass = 2; pass >= 0; pass--) {
    const sy = pass * GLYPH;
    for (let i = 0; i < count; i++) {
      let x, y, s;
      if (pass === 0) { x = bufX[i]; y = bufY[i]; s = bufS[i]; }
      else if (pass === 2) { x = bufBX[i]; y = bufBY[i]; s = bufBS[i]; }
      else { x = (bufX[i] + bufBX[i]) * 0.5; y = (bufY[i] + bufBY[i]) * 0.5; s = (bufS[i] + bufBS[i]) * 0.5; }
      const d = size * s;
      ctx.globalAlpha = pass === 0 ? bufA[i] : bufA[i] * 0.9;
      ctx.drawImage(atlas, bufG[i] * GLYPH, sy, GLYPH, GLYPH, x - d / 2, y - d / 2, d, d);
    }
  }
  ctx.globalAlpha = 1;

  if (finished) $("again").hidden = false;
  fpsFrames++;
  if (now - fpsT > 500) {
    const fps = Math.round(fpsFrames * 1000 / (now - fpsT));
    ui.fps.textContent = finished ? `${fps} fps · el código y tu cuerpo se deshicieron` : `${fps} fps · ${count} caracteres`;
    fpsFrames = 0; fpsT = now;
  }
}

// Dilatación 3x3 (máximo) de la máscara
function dilate(src, dst, w, h) {
  for (let y = 0; y < h; y++) {
    const y0 = y > 0 ? y - 1 : 0, y1 = y < h - 1 ? y + 1 : y;
    for (let x = 0; x < w; x++) {
      const x0 = x > 0 ? x - 1 : 0, x1 = x < w - 1 ? x + 1 : x;
      let m = 0;
      for (let yy = y0; yy <= y1; yy++) {
        const r = yy * w;
        for (let xx = x0; xx <= x1; xx++) { const v = src[r + xx]; if (v > m) m = v; }
      }
      dst[y * w + x] = m;
    }
  }
}

// Aprende el fondo (todo en GPU con composición de canvas) y rellena lo que nunca se ha visto
function updatePlate(mirror) {
  const pw = frameC.width, ph = frameC.height;
  const f = c2(frameC);
  f.save();
  if (mirror) { f.translate(pw, 0); f.scale(-1, 1); }
  f.drawImage(video, crop.x, crop.y, crop.w, crop.h, 0, 0, pw, ph);
  f.restore();
  // Solo los píxeles donde NO hay persona
  const l = c2(learnC);
  l.globalCompositeOperation = "copy";
  l.drawImage(learnMaskC, 0, 0, pw, ph);
  l.globalCompositeOperation = "source-in";
  l.drawImage(frameC, 0, 0);
  l.globalCompositeOperation = "source-over";
  // Mezcla lenta: el fondo se actualiza (luz, objetos) sin copiar al cuerpo
  const p = c2(plateC);
  p.globalAlpha = 0.2;
  p.drawImage(learnC, 0, 0);
  p.globalAlpha = 1;
  // Relleno de huecos (zonas que siempre han estado tapadas por el cuerpo):
  // en una versión pequeña se "esparcen" los colores vecinos hacia el hueco (jump flood)
  const sm = lv[0], g = c2(sm), sw2 = sm.width, sh2 = sm.height;
  g.globalCompositeOperation = "copy";
  g.drawImage(plateC, 0, 0, sw2, sh2);
  g.globalCompositeOperation = "destination-over";
  for (let d = 1; d < Math.max(sw2, sh2); d *= 2) {
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2;
      g.drawImage(sm, Math.round(Math.cos(ang) * d), Math.round(Math.sin(ang) * d));
    }
  }
  g.globalCompositeOperation = "source-over";
  // Suavizado (abajo y arriba) para que el relleno no se vea rayado
  const b1 = lv[1], gb = c2(b1);
  gb.globalCompositeOperation = "copy";
  gb.drawImage(sm, 0, 0, b1.width, b1.height);
  g.globalCompositeOperation = "copy";
  g.drawImage(b1, 0, 0, sw2, sh2);           // re-escalado en dos pasos = más suave
  g.globalCompositeOperation = "source-over";
  const fl = c2(fillC);
  fl.globalCompositeOperation = "copy";
  fl.drawImage(sm, 0, 0, pw, ph);
  fl.globalCompositeOperation = "source-over";
  fl.drawImage(plateC, 0, 0);
}

// ---------- UI ----------
$("start").addEventListener("click", async () => {
  const btn = $("start"), status = $("status");
  btn.disabled = true;
  try {
    status.textContent = "Cargando modelo de segmentación…";
    buildAtlas(ui.palette.value); shuffle();
    await Promise.all([loadModel(), startCamera()]);
    $("intro").hidden = true; $("panel").hidden = false;
    // En teléfonos el panel arranca cerrado para no tapar la imagen
    if (window.matchMedia("(max-width: 760px), (pointer: coarse)").matches) $("panel").classList.add("collapsed");
    requestAnimationFrame(frame);
  } catch (e) {
    console.error(e);
    btn.disabled = false;
    status.textContent = e && e.name === "NotAllowedError"
      ? "Necesito permiso para usar la cámara. Revisa los permisos del navegador."
      : "No se pudo iniciar: " + (e.message || e);
  }
});

ui.cell.addEventListener("input", layoutGrid);
$("again").addEventListener("click", () => { dissolveStart = 0; mask.fill(0); $("again").hidden = true; });
ui.palette.addEventListener("change", () => buildAtlas(ui.palette.value));
$("togglePanel").addEventListener("click", () => $("panel").classList.toggle("collapsed"));
$("flip").addEventListener("click", async () => { facing = facing === "user" ? "environment" : "user"; try { await startCamera(); } catch (e) { console.error(e); } });
$("full").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
});
// --- Grabación de video (del mismo canvas que ves, sin volver a procesar nada) ---
let recorder, recChunks = [], recStart = 0, recTimer;
function pickMime() {
  const opts = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return opts.find((m) => window.MediaRecorder?.isTypeSupported(m)) || "";
}
function stopRec() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}
$("rec").addEventListener("click", () => {
  const btn = $("rec");
  if (recorder && recorder.state === "recording") { stopRec(); return; }
  if (!window.MediaRecorder || !stage.captureStream) {
    ui.fps.textContent = "Tu navegador no puede grabar video";
    return;
  }
  const mime = pickMime();
  try {
    recorder = new MediaRecorder(stage.captureStream(30), mime ? { mimeType: mime, videoBitsPerSecond: 8e6 } : undefined);
  } catch (e) {
    console.error(e);
    ui.fps.textContent = "No se pudo iniciar la grabación";
    return;
  }
  recChunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
  recorder.onstop = () => {
    clearInterval(recTimer);
    btn.textContent = "● Grabar";
    btn.classList.remove("recording");
    $("recDot").hidden = true;
    const type = recorder.mimeType || mime || "video/webm";
    const blob = new Blob(recChunks, { type });
    recChunks = [];
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.download = "cuerpo-ascii-3d." + (type.includes("mp4") ? "mp4" : "webm");
    a.href = url;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  };
  recorder.start(1000);
  recStart = performance.now();
  btn.textContent = "■ Detener";
  btn.classList.add("recording");
  $("recDot").hidden = false;
  recTimer = setInterval(() => {
    const sec = Math.floor((performance.now() - recStart) / 1000);
    $("recTime").textContent = `${(sec / 60) | 0}:${String(sec % 60).padStart(2, "0")}`;
  }, 250);
});
window.addEventListener("pagehide", stopRec);

$("shot").addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "cuerpo-ascii-3d.png";
  a.href = stage.toDataURL("image/png");
  a.click();
});
let resizeT;
window.addEventListener("resize", () => {
  if (recorder && recorder.state === "recording") return; // redimensionar cortaría la grabación
  clearTimeout(resizeT);
  resizeT = setTimeout(() => video.videoWidth && layout(), 120);
});
