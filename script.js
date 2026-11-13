"use strict";

/* ============================================================
   1) Aparición de secciones al hacer scroll
   (independiente de p5: si p5 no carga, el CV se sigue viendo)
   ============================================================ */
(function revealOnScroll() {
  const items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) return;
  document.documentElement.classList.add("js");

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    }
  }, { threshold: 0.12 });

  items.forEach((el) => observer.observe(el));
})();

/* ============================================================
   2) Fondo generativo con p5.js
   Capas: cielo degradado · portal HUD giratorio · horizonte
   brillante · piso con rejilla en perspectiva · pulsos de luz ·
   red de partículas 3D (plexus) que reacciona al cursor
   ============================================================ */

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Colores (RGB)
const CYAN = [34, 230, 255];
const MAGENTA = [255, 43, 214];
const VIOLET = [138, 92, 255];

let particles = [];
let pulses = [];
let horizonY;
let focal;
let gridOffset = 0;
let angle = 0;

// Interacción suavizada
let pointer = { x: -9999, y: -9999, active: false };
let camShift = 0;
let lastScrollY = 0;
let scrollBoost = 0;

const FIELD = { depthNear: 250, depthFar: 1600, center: 900 };

function setup() {
  const canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent("canvas-container");
  // Limitar la densidad de píxeles: en celulares 3x es carísimo y no se nota
  pixelDensity(Math.min(window.devicePixelRatio || 1, 1.5));

  initScene();

  // Ratón / toque: solo seguimos la posición, el scroll lo maneja el navegador
  window.addEventListener("pointermove", (e) => {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.active = true;
  }, { passive: true });
  document.addEventListener("pointerleave", () => { pointer.active = false; });
  window.addEventListener("pointerup", (e) => {
    if (e.pointerType !== "mouse") pointer.active = false;
  }, { passive: true });

  if (REDUCED_MOTION) {
    noLoop(); // Dibuja un solo cuadro estático
  }
}

function initScene() {
  horizonY = height * 0.68;
  focal = Math.min(width, height) * 0.9;

  // Cantidad de partículas según el tamaño de pantalla (menos en móvil)
  const count = constrain(Math.floor((width * height) / 9000), 60, 170);
  particles = [];
  for (let i = 0; i < count; i++) {
    particles.push({
      x: random(-1, 1) * width * 1.3,
      y: random(-1.1, 0.45) * height,
      z: random(FIELD.depthNear, FIELD.depthFar),
      vx: random(-0.25, 0.25),
      vy: random(-0.2, 0.2),
      vz: random(-0.6, 0.6),
      phase: random(TWO_PI),
      hue: random(), // 0 = cian, 1 = violeta
      sx: 0, sy: 0, s: 0, // posición proyectada (se calcula en cada cuadro)
    });
  }
  pulses = [];
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  initScene();
  if (REDUCED_MOTION) redraw();
}

function draw() {
  const t = millis() / 1000;

  // Velocidad de scroll -> acelera la escena un momento
  const sy = window.scrollY || 0;
  scrollBoost = lerp(scrollBoost, Math.abs(sy - lastScrollY), 0.1);
  lastScrollY = sy;

  // Parallax suave con el cursor
  const targetShift = pointer.active ? (pointer.x / width - 0.5) : 0;
  camShift = lerp(camShift, targetShift, 0.04);

  angle = t * 0.04 + sy * 0.0004 + camShift * 0.35;
  gridOffset += 1.2 + scrollBoost * 0.25;

  drawSky();
  drawPortal(t);
  drawHorizonGlow();
  drawGrid(t);
  drawParticles(t);
}

/* ---------- Cielo ---------- */
function drawSky() {
  const ctx = drawingContext;
  const g = ctx.createLinearGradient(0, 0, 0, height);
  g.addColorStop(0, "#03040b");
  g.addColorStop(0.45, "#0a0b24");
  g.addColorStop(0.68, "#2a0c3f");
  g.addColorStop(0.69, "#07061a");
  g.addColorStop(1, "#020208");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, height);
}

/* ---------- Portal / anillos HUD sobre el horizonte ---------- */
function drawPortal(t) {
  const cx = width / 2 + camShift * -40;
  const cy = horizonY;
  const R = Math.min(width, height) * 0.32;

  push();
  // Recortar todo lo que esté debajo del horizonte
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.rect(0, 0, width, horizonY);
  drawingContext.clip();

  // Relleno del "sol" con degradado
  const ctx = drawingContext;
  const g = ctx.createLinearGradient(0, cy - R, 0, cy);
  g.addColorStop(0, "rgba(34,230,255,0.18)");
  g.addColorStop(1, "rgba(255,43,214,0.35)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(cx, cy, R * 0.8, 0, TWO_PI);
  ctx.fill();

  // Franjas horizontales que "cortan" el sol (estilo retro-futurista)
  ctx.fillStyle = "#1a0a30";
  for (let i = 0; i < 7; i++) {
    const h = 2 + i * 1.6;
    const y = cy - R * 0.05 - i * (R * 0.09) - ((t * 8) % (R * 0.09));
    ctx.fillRect(cx - R, y, R * 2, h);
  }

  noFill();
  blendMode(ADD);

  // Anillos giratorios con segmentos
  const rings = [
    { r: 0.92, w: 2, speed: 0.25, segs: 3, gap: 0.6, col: CYAN, a: 160 },
    { r: 1.05, w: 1, speed: -0.15, segs: 24, gap: 0.12, col: MAGENTA, a: 120 },
    { r: 1.18, w: 3, speed: 0.08, segs: 2, gap: 1.4, col: VIOLET, a: 110 },
    { r: 1.32, w: 1, speed: -0.05, segs: 60, gap: 0.06, col: CYAN, a: 70 },
  ];
  for (const ring of rings) {
    stroke(ring.col[0], ring.col[1], ring.col[2], ring.a);
    strokeWeight(ring.w);
    const step = TWO_PI / ring.segs;
    const rot = t * ring.speed;
    for (let i = 0; i < ring.segs; i++) {
      const a0 = rot + i * step;
      arc(cx, cy, R * ring.r * 2, R * ring.r * 2, a0, a0 + step - ring.gap);
    }
  }

  // Marcas tipo medidor
  stroke(CYAN[0], CYAN[1], CYAN[2], 90);
  strokeWeight(1);
  const ticks = 72;
  for (let i = 0; i < ticks; i++) {
    const a = (i / ticks) * TWO_PI - t * 0.03;
    const r1 = R * 1.42;
    const r2 = r1 + (i % 6 === 0 ? 12 : 5);
    line(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1, cx + Math.cos(a) * r2, cy + Math.sin(a) * r2);
  }

  drawingContext.restore();
  pop();
}

/* ---------- Brillo del horizonte ---------- */
function drawHorizonGlow() {
  const ctx = drawingContext;
  const g = ctx.createLinearGradient(0, horizonY - 60, 0, horizonY + 40);
  g.addColorStop(0, "rgba(255,43,214,0)");
  g.addColorStop(0.6, "rgba(255,43,214,0.35)");
  g.addColorStop(0.62, "rgba(255,255,255,0.9)");
  g.addColorStop(0.66, "rgba(34,230,255,0.4)");
  g.addColorStop(1, "rgba(34,230,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, horizonY - 60, width, 100);
}

/* ---------- Piso con rejilla en perspectiva ---------- */
function drawGrid(t) {
  const ctx = drawingContext;
  const cx = width / 2 + camShift * -120;
  const camH = 120;          // altura de la cámara sobre el piso
  const spacing = 80;        // separación de la rejilla en "metros"
  const zNear = 40;
  const zFar = 3000;

  push();
  blendMode(ADD);
  strokeWeight(1);
  stroke(255); // activa el trazo; el color real lo pone el degradado
  const g = ctx.createLinearGradient(0, horizonY, 0, height);
  g.addColorStop(0, "rgba(255,43,214,0)");
  g.addColorStop(0.15, "rgba(255,43,214,0.35)");
  g.addColorStop(1, "rgba(34,230,255,0.75)");
  ctx.strokeStyle = g;

  // Líneas horizontales que avanzan hacia el espectador
  const off = gridOffset % spacing;
  for (let z = zFar - off; z > zNear; z -= spacing) {
    const y = horizonY + (camH * focal) / z;
    if (y > height + 2) continue;
    line(0, y, width, y);
  }

  // Líneas que convergen al punto de fuga
  const lines = 26;
  for (let i = -lines; i <= lines; i++) {
    const X = i * spacing;
    const xNear = cx + (X * focal) / zNear;
    const yNear = horizonY + (camH * focal) / zNear;
    const xFar = cx + (X * focal) / zFar;
    line(xNear, yNear, xFar, horizonY);
  }

  // Pulsos de luz que viajan por la rejilla
  if (!REDUCED_MOTION && random() < 0.05 && pulses.length < 10) {
    pulses.push({ i: Math.floor(random(-10, 11)), z: zFar, speed: random(18, 40) });
  }
  for (let k = pulses.length - 1; k >= 0; k--) {
    const p = pulses[k];
    p.z -= p.speed + scrollBoost;
    if (p.z < zNear + 20) { pulses.splice(k, 1); continue; }
    const X = p.i * spacing;
    const z2 = p.z + 260;
    const x1 = cx + (X * focal) / p.z, y1 = horizonY + (camH * focal) / p.z;
    const x2 = cx + (X * focal) / z2, y2 = horizonY + (camH * focal) / z2;
    stroke(CYAN[0], CYAN[1], CYAN[2], 220);
    strokeWeight(2.5);
    line(x1, y1, x2, y2);
    noStroke();
    fill(255, 255, 255, 200);
    circle(x1, y1, 4);
  }
  pop();
}

/* ---------- Red de partículas 3D ---------- */
function drawParticles(t) {
  const cx = width / 2;
  const cy = height * 0.4;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const f = focal * 0.9;

  // Mover y proyectar
  for (const p of particles) {
    if (!REDUCED_MOTION) {
      p.x += p.vx;
      p.y += p.vy;
      p.z += p.vz;
    }
    // Rebote suave dentro del volumen (sin aceleración infinita)
    if (p.x < -width * 1.3 || p.x > width * 1.3) p.vx *= -1;
    if (p.y < -height * 1.1 || p.y > height * 0.45) p.vy *= -1;
    if (p.z < FIELD.depthNear || p.z > FIELD.depthFar) p.vz *= -1;

    // Rotación alrededor del eje Y
    const dz = p.z - FIELD.center;
    const rx = p.x * cosA - dz * sinA;
    const rz = p.x * sinA + dz * cosA + FIELD.center;
    const scale = f / Math.max(rz, 60);
    p.sx = cx + rx * scale;
    p.sy = cy + p.y * scale;
    p.s = scale;
    p.depth = rz;
  }

  push();
  blendMode(ADD);

  // Conexiones (comparamos distancias al cuadrado: más rápido que dist())
  const maxD = Math.min(width, height) * 0.17;
  const maxD2 = maxD * maxD;
  strokeWeight(1);
  for (let i = 0; i < particles.length; i++) {
    const a = particles[i];
    if (a.depth < 60) continue;
    for (let j = i + 1; j < particles.length; j++) {
      const b = particles[j];
      const dx = a.sx - b.sx;
      const dy = a.sy - b.sy;
      const d2 = dx * dx + dy * dy;
      if (d2 < maxD2) {
        const k = 1 - d2 / maxD2;
        const mix = (a.hue + b.hue) / 2;
        stroke(
          lerp(CYAN[0], VIOLET[0], mix),
          lerp(CYAN[1], VIOLET[1], mix),
          lerp(CYAN[2], VIOLET[2], mix),
          k * 90 * Math.min(1, (a.s + b.s))
        );
        line(a.sx, a.sy, b.sx, b.sy);
      }
    }
  }

  // Líneas hacia el cursor / dedo
  if (pointer.active) {
    const r = maxD * 1.4;
    const r2 = r * r;
    for (const p of particles) {
      const dx = p.sx - pointer.x;
      const dy = p.sy - pointer.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < r2) {
        stroke(MAGENTA[0], MAGENTA[1], MAGENTA[2], (1 - d2 / r2) * 200);
        line(p.sx, p.sy, pointer.x, pointer.y);
      }
    }
    noFill();
    stroke(MAGENTA[0], MAGENTA[1], MAGENTA[2], 120);
    circle(pointer.x, pointer.y, 18 + Math.sin(t * 4) * 4);
  }

  // Nodos con halo (brillo) y núcleo
  noStroke();
  for (const p of particles) {
    if (p.sx < -40 || p.sx > width + 40 || p.sy < -40 || p.sy > height + 40) continue;
    const size = constrain(p.s * 6, 1, 9);
    const twinkle = 0.6 + 0.4 * Math.sin(t * 2 + p.phase);
    const c = [
      lerp(CYAN[0], VIOLET[0], p.hue),
      lerp(CYAN[1], VIOLET[1], p.hue),
      lerp(CYAN[2], VIOLET[2], p.hue),
    ];
    fill(c[0], c[1], c[2], 28 * twinkle);
    circle(p.sx, p.sy, size * 5);
    fill(c[0], c[1], c[2], 200 * twinkle);
    circle(p.sx, p.sy, size);
  }
  pop();
}
