// Anatomía para rayos X.
// En lugar de dibujar líneas, se pinta un "mapa de densidad" en un canvas:
//   rojo  = hueso   (suma densidad → se ve blanco)
//   verde = aire    (pulmones, senos, órbitas → resta densidad, se ve oscuro)
//   azul  = tejido blando denso (corazón, mediastino, vasos)
// Luego app.js combina ese mapa con el grosor del cuerpo y lo revela como una película de rayos X.
// Cada hueso tiene borde cortical (más brillante) y centro medular (más tenue), como en una radiografía.

let D = null;
const CH = { bone: "255,0,0", air: "0,255,0", soft: "0,0,255" };
const col = (ch, a) => `rgba(${CH[ch]},${a < 0 ? 0 : a > 1 ? 1 : a})`;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const add = (a, b, k = 1) => ({ x: a.x + b.x * k, y: a.y + b.y * k });
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const mixP = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) });
const dist = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
const dot = (a, b) => a.x * b.x + a.y * b.y;
const unit = (v) => { const l = Math.hypot(v.x, v.y) || 1; return { x: v.x / l, y: v.y / l }; };
const perp = (v) => ({ x: -v.y, y: v.x });

// ------------------------------------------------------------------ primitivas
function spline(cp, closed, seg = 6) {
  const n = cp.length, out = [];
  const get = (i) => (closed ? cp[(i + n) % n] : cp[clamp(i, 0, n - 1)]);
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    const p0 = get(i - 1), p1 = get(i), p2 = get(i + 1), p3 = get(i + 2);
    for (let s = 0; s < seg; s++) {
      const t = s / seg, t2 = t * t, t3 = t2 * t;
      out.push({
        x: 0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
        y: 0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
      });
    }
  }
  if (!closed) out.push(cp[n - 1]);
  return out;
}
const subdiv = (a, b, n = 10) => Array.from({ length: n + 1 }, (_, i) => mixP(a, b, i / n));

// Hueso alargado: banda con ancho variable, extremos redondeados y borde cortical.
function ribbon(cp, w, aF, aE, ch = "bone") {
  const pts = cp.length > 2 ? spline(cp, false, 6) : subdiv(cp[0], cp[1], 12);
  const n = pts.length;
  if (n < 2) return;
  const Lf = [], Rt = [], hw = [];
  let avg = 0, a0 = 0, a1 = 0;
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
    const d = unit(sub(b, a)), h = Math.max(0.3, (typeof w === "function" ? w(i / (n - 1)) : w) / 2);
    hw.push(h); avg += h;
    Lf.push({ x: pts[i].x - d.y * h, y: pts[i].y + d.x * h });
    Rt.push({ x: pts[i].x + d.y * h, y: pts[i].y - d.x * h });
    if (i === 0) a0 = Math.atan2(d.y, d.x);
    if (i === n - 1) a1 = Math.atan2(d.y, d.x);
  }
  avg /= n;
  if (avg * 2 < 0.4) return;
  const p = new Path2D();
  p.moveTo(Lf[0].x, Lf[0].y);
  for (let i = 1; i < n; i++) p.lineTo(Lf[i].x, Lf[i].y);
  p.arc(pts[n - 1].x, pts[n - 1].y, hw[n - 1], a1 + Math.PI / 2, a1 - Math.PI / 2, true);
  for (let i = n - 1; i >= 0; i--) p.lineTo(Rt[i].x, Rt[i].y);
  p.arc(pts[0].x, pts[0].y, hw[0], a0 - Math.PI / 2, a0 - Math.PI * 1.5, true);
  p.closePath();
  if (aF > 0) { D.fillStyle = col(ch, aF); D.fill(p); }
  if (aE > 0) {
    D.lineWidth = Math.max(0.8, avg * 0.75); D.strokeStyle = col(ch, aE * 0.3); D.stroke(p);
    D.lineWidth = Math.max(0.5, avg * 0.2); D.strokeStyle = col(ch, aE * 0.8); D.stroke(p);
  }
}

// Hueso plano / forma cerrada
function blob(cp, aF, aE, ew, ch = "bone") {
  const pts = spline(cp, true, 5), p = new Path2D();
  p.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) p.lineTo(pts[i].x, pts[i].y);
  p.closePath();
  if (aF > 0) { D.fillStyle = col(ch, aF); D.fill(p); }
  if (aE > 0 && ew > 0.3) { D.lineWidth = ew; D.strokeStyle = col(ch, aE); D.stroke(p); }
}
const ellPts = (c, rx, ry, u = { x: 1, y: 0 }, n = 10) => {
  const v = perp(u);
  return Array.from({ length: n }, (_, i) => {
    const a = (i / n) * Math.PI * 2;
    return { x: c.x + u.x * Math.cos(a) * rx + v.x * Math.sin(a) * ry, y: c.y + u.y * Math.cos(a) * rx + v.y * Math.sin(a) * ry };
  });
};
// Bordes suaves (órganos): varias pasadas concéntricas
function soft(cp, a, ch, passes = 5) {
  let cx = 0, cy = 0;
  for (const p of cp) { cx += p.x; cy += p.y; }
  cx /= cp.length; cy /= cp.length;
  for (let k = 0; k < passes; k++) {
    const s = 1.08 - (k / (passes - 1)) * 0.2;
    blob(cp.map((p) => ({ x: cx + (p.x - cx) * s, y: cy + (p.y - cy) * s })), a / passes, 0, 0, ch);
  }
}
function ring(c, rx, ry, u, w, aF, aE) {
  const path = (r1, r2) => { const p = new Path2D(), v = perp(u); for (let i = 0; i <= 28; i++) { const a = (i / 28) * Math.PI * 2, x = c.x + u.x * Math.cos(a) * r1 + v.x * Math.sin(a) * r2, y = c.y + u.y * Math.cos(a) * r1 + v.y * Math.sin(a) * r2; i ? p.lineTo(x, y) : p.moveTo(x, y); } return p; };
  D.strokeStyle = col("bone", aF); D.lineWidth = w; D.stroke(path(rx, ry));
  D.strokeStyle = col("bone", aE); D.lineWidth = Math.max(0.6, w * 0.22);
  D.stroke(path(rx + w / 2, ry + w / 2)); D.stroke(path(rx - w / 2, ry - w / 2));
}

// Semillas fijas para vasos pulmonares (no parpadean)
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const VESSELS = Array.from({ length: 9 }, (_, i) => ({ ang: -1.0 + (i / 8) * 2.1 + (rnd() - 0.5) * 0.2, len: 0.22 + rnd() * 0.12, bend: (rnd() - 0.5) * 0.5, br: 0.4 + rnd() * 0.3, bang: (rnd() - 0.5) * 1.2 }));

// ------------------------------------------------------------------ marcos de referencia
// Cabeza en 3D: el giro (yaw) se estima con ojos + nariz y el cráneo se proyecta.
// Unidades: e ≈ ancho de cabeza. Origen en el centro del cráneo, y = 0 a la altura de los ojos, z hacia el frente.
function headFrame(L) {
  const eL = L[5], eR = L[2], n = L[0], mL = L[10], mR = L[9];
  if (eL.v < 0.5 || eR.v < 0.5) return null;
  const sep = dist(eL, eR) || 1, d = unit(sub(eR, eL)), dn = perp(d), EM = mid(eL, eR);
  const nOff = n.v > 0.4 ? dot(sub(n, EM), d) / sep : 0;
  // Giro: combina nariz y orejas (si se ven) para no exagerarlo
  let t1 = Math.atan(1.6 * nOff);
  const aL = L[8], aR = L[7];
  if (aL.v > 0.5 && aR.v > 0.5) {
    const ea = dot(sub(mid(aL, aR), EM), d) / sep;           // orejas desplazadas hacia atrás al girar
    t1 = 0.6 * t1 + 0.4 * Math.atan(-ea * 1.1);
  }
  const th = clamp(t1, -0.8, 0.8), c = Math.cos(th), s = Math.sin(th);
  const e = sep / (0.42 * Math.max(0.72, c));
  const Hc = add(EM, d, -0.38 * s * e);
  const F2 = (x, y) => ({ x: Hc.x + d.x * x * e + dn.x * y * e, y: Hc.y + d.y * x * e + dn.y * y * e });
  const P = (x, y, z) => F2(x * c + z * s, y);
  const M = mL.v > 0.4 && mR.v > 0.4 ? mid(mL, mR) : null;
  const my = M ? clamp(dot(sub(M, EM), dn) / e, 0.3, 0.6) : 0.43;
  const ny = n.v > 0.4 ? clamp(dot(sub(n, EM), dn) / e, 0.14, 0.42) : 0.3;
  return { P, F2, e, c, s, th, d, dn, eL, eR, my, ny };
}

function chestFrame(L) {
  const a = L[12], b = L[11];
  if (a.v < 0.5 || b.v < 0.5) return null;
  const sw = dist(a, b) || 1, dsh = unit(sub(b, a)), S0 = mid(a, b);
  let ax = perp(dsh), fy = 1, hipsOK = false;
  if (L[23].v > 0.55 && L[24].v > 0.55) {
    const v = sub(mid(L[23], L[24]), S0), l = Math.hypot(v.x, v.y);
    if (l > sw * 0.9 && l < sw * 2.6) { ax = { x: v.x / l, y: v.y / l }; fy = l / (1.35 * sw); hipsOK = true; }
  }
  const C = (x, y) => ({ x: S0.x + dsh.x * x * sw + ax.x * y * sw * fy, y: S0.y + dsh.y * x * sw + ax.y * y * sw * fy });
  return { C, sw, dsh, ax, fy, hipsOK };
}

// ------------------------------------------------------------------ cráneo
const VAULT = [ // y, medio ancho, media profundidad (corte horizontal del cráneo)
  [-0.8, 0.02, 0.03], [-0.76, 0.22, 0.32], [-0.66, 0.37, 0.47], [-0.5, 0.47, 0.58], [-0.3, 0.51, 0.63], [-0.1, 0.51, 0.62], [0.1, 0.5, 0.57],
];
function skull(h) {
  const { P, F2, e, c, s, d, dn, my, ny } = h;
  // Silueta de la bóveda proyectada según el giro
  const side = (sg) => VAULT.map(([y, a, b]) => F2(-0.05 * s + sg * Math.sqrt(a * a * c * c + b * b * s * s), y));
  const left = side(-1).reverse(), right = side(1);
  const vault = [...left, ...right.slice(1)];
  const mand3 = [[0.43, 0.1, -0.05], [0.43, 0.3, -0.04], [0.41, 0.52, -0.02], [0.33, 0.64, 0.12], [0.18, my + 0.3, 0.33], [0, my + 0.36, 0.42]];
  const mandR = mand3.map(([x, y, z]) => P(x, y, z)), mandL = mand3.map(([x, y, z]) => P(-x, y, z));
  // Volumen general (huesos superpuestos)
  blob([...vault, ...mandR.slice(1), ...mandL.slice(1, -1).reverse()], 0.12, 0, 0);
  // Tablas externa e interna de la bóveda
  ribbon(vault, (t) => e * (0.028 + 0.036 * Math.sin(Math.PI * t)), 0.22, 0.42);
  ribbon([P(-0.2, -0.7, 0.2), P(0, -0.74, 0.1), P(0.2, -0.7, 0.2)], e * 0.012, 0.04, 0.07); // sutura coronal
  for (const sx of [-1, 1]) {
    ribbon([P(sx * 0.12, 0.2, -0.1), P(sx * 0.3, 0.17, -0.15), P(sx * 0.44, 0.12, -0.2)], e * 0.03, 0.06, 0.06); // crestas petrosas
    blob(ellPts(P(sx * 0.44, 0.32, -0.12), e * 0.045, e * 0.06, d, 8), 0.16, 0.18, e * 0.01);                // mastoides
  }
  // Seno frontal
  const fs = P(0, -0.17, 0.42);
  soft(ellPts(fs, e * 0.09 * c + e * 0.02, e * 0.055, d, 9), 0.35, "air", 3);
  // Órbitas: aire dentro, reborde óseo grueso (escorzo según el giro)
  for (const [eye, sx] of [[h.eL, -1], [h.eR, 1]]) {
    const cc = add(eye, dn, -0.012 * e), k = Math.max(0.3, Math.cos(h.th + sx * 0.35));
    const w = 0.135 * e * k, hh = 0.12 * e;
    const Q = (x, y) => ({ x: cc.x + d.x * x + dn.x * y, y: cc.y + d.y * x + dn.y * y });
    const pts = Array.from({ length: 12 }, (_, i) => {
      const a = (i / 12) * Math.PI * 2, r = 1 + 0.06 * Math.cos(4 * a);
      return Q(Math.cos(a) * w * r * (Math.cos(a) * sx > 0 ? 1.05 : 0.95), Math.sin(a) * hh * r * (Math.sin(a) > 0 ? 1.02 : 0.9));
    });
    soft(pts, 0.5, "air", 4);
    blob(pts, 0, 0.4, e * 0.03);
    blob(pts.map((p) => mixP(cc, p, 0.8)), 0, 0.1, e * 0.012); // línea innominada
  }
  // Abertura nasal (pera) + tabique
  const nas = [[0, 0.1], [0.045, 0.19], [0.075, 0.3], [0.06, ny + 0.03], [0, ny + 0.05], [-0.06, ny + 0.03], [-0.075, 0.3], [-0.045, 0.19]].map(([x, y]) => P(x, y, 0.47));
  soft(nas, 0.55, "air", 4); blob(nas, 0, 0.26, e * 0.018);
  ribbon([P(0, 0.13, 0.47), P(0, ny + 0.04, 0.48)], e * 0.016, 0.18, 0.08);
  for (const sx of [-1, 1]) {
    const ms = P(sx * 0.17, 0.25, 0.33);
    soft(ellPts(ms, e * (0.03 + 0.07 * c), e * 0.085, d, 10), 0.3, "air", 3); // senos maxilares
    ribbon([P(sx * 0.28, 0.15, 0.33), P(sx * 0.42, 0.15, 0.18), P(sx * 0.48, 0.12, 0.0)], (t) => e * (0.045 - 0.015 * t), 0.14, 0.16); // arco cigomático
  }
  // Maxilar y cuerpo de la mandíbula (relleno tenue)
  blob([P(-0.22, 0.33, 0.3), P(0.22, 0.33, 0.3), P(0.2, my - 0.02, 0.35), P(-0.2, my - 0.02, 0.35)], 0.1, 0, 0);
  blob([P(-0.22, my + 0.08, 0.33), P(0.22, my + 0.08, 0.33), P(0.16, my + 0.3, 0.36), P(0, my + 0.34, 0.42), P(-0.16, my + 0.3, 0.36)], 0.1, 0, 0);
  // Dientes sobre el arco dental (en 3D): corona de esmalte densa + raíz tenue
  for (const jaw of [-1, 1]) {
    for (let i = 0; i < 10; i++) {
      const u = ((i + 0.5) / 10) * 2 - 1, phi = u * 1.35;
      const X = 0.2 * Math.sin(phi), Z = 0.42 - 0.3 * (1 - Math.cos(phi));
      const tx = 0.2 * Math.cos(phi), tz = -0.3 * Math.sin(phi), tl = Math.hypot(tx, tz);
      const k = Math.abs((tx * c + tz * s) / tl);
      const wT = 0.05 * (Math.abs(u) < 0.25 ? 1.1 : Math.abs(u) > 0.6 ? 1.25 : 0.9) * k;
      if (wT < 0.006) continue;
      const root = jaw < 0 ? 0.15 : 0.14;
      ribbon([P(X, my + jaw * 0.05, Z), P(X, my + jaw * root, Z - 0.02)], (t) => e * wT * 0.65 * (1 - t * 0.6), 0.07, 0.07);
      ribbon([P(X, my + jaw * 0.01, Z), P(X, my + jaw * 0.058, Z)], (t) => e * wT * (0.85 + 0.15 * Math.sin(Math.PI * t)), 0.24, 0.22);
    }
  }
  // Mandíbula completa (rama + ángulo + cuerpo + mentón)
  ribbon([...mandL, ...mandR.slice(0, -1).reverse()], (t) => e * (0.06 + 0.035 * Math.sin(Math.PI * t)), 0.24, 0.4);
  ribbon([P(-0.4, 0.1, -0.03), P(-0.37, 0.22, 0.02)], e * 0.04, 0.12, 0.14); // apófisis coronoides
  ribbon([P(0.4, 0.1, -0.03), P(0.37, 0.22, 0.02)], e * 0.04, 0.12, 0.14);
}

// ------------------------------------------------------------------ columna
function vertebra(c, axd, w, h, type) {
  const px = { x: axd.y, y: -axd.x };
  const V = (x, y) => ({ x: c.x - px.x * x * w + axd.x * y * h, y: c.y - px.y * x * w + axd.y * y * h });
  blob([V(-0.5, -0.5), V(0, -0.44), V(0.5, -0.5), V(0.54, 0), V(0.5, 0.5), V(0, 0.44), V(-0.5, 0.5), V(-0.54, 0)], 0.2, 0.34, w * 0.07);
  for (const s of [-1, 1]) {
    blob(ellPts(V(s * 0.3, -0.12), w * 0.1, h * 0.17, px, 8), 0.04, 0.34, w * 0.05); // pedículos
    if (type === "t") ribbon([V(s * 0.45, -0.15), V(s * 0.85, -0.35)], h * 0.32, 0.14, 0.2);
    else if (type === "l") ribbon([V(s * 0.5, -0.05), V(s * 1.05, -0.12)], h * 0.28, 0.14, 0.22);
    else ribbon([V(s * 0.48, 0), V(s * 0.72, 0.06)], h * 0.32, 0.14, 0.18);
  }
  ribbon([V(0, -0.05), V(0, 0.6)], (t) => w * 0.13 * (1 - t * 0.6), 0.12, 0.14); // apófisis espinosa
}

function spine(ch, head) {
  const { C, sw, fy } = ch;
  const top = head ? head.P(0, head.my + 0.2, -0.1) : C(0, -0.55);
  const t1 = C(0, -0.04);
  const cd = unit(sub(t1, top)), seg = dist(top, t1) / 6;
  for (let i = 0; i < 6; i++) vertebra(mixP(top, t1, (i + 0.55) / 6.1), cd, sw * 0.1, seg * 0.66, "c");
  const ad = unit(sub(C(0, 1), C(0, 0)));
  for (let k = 0; k < 12; k++) vertebra(C(0, 0.083 * k + 0.02), ad, sw * (0.085 + 0.004 * k), sw * fy * (0.055 + 0.0015 * k), "t");
  for (let k = 0; k < 5; k++) vertebra(C(0, 1.0 + 0.078 * k), ad, sw * (0.13 + 0.006 * k), sw * fy * 0.06, "l");
}

// ------------------------------------------------------------------ tórax
const RIB_W = [0.2, 0.3, 0.36, 0.4, 0.43, 0.45, 0.46, 0.46, 0.45, 0.43, 0.38, 0.3];

function thorax(ch) {
  const { C, sw } = ch;
  for (const s of [-1, 1]) {
    const Q = (x, y) => C(s * x, y);
    // Escápula (detrás de las costillas)
    blob([[0.25, -0.05], [0.33, -0.06], [0.41, -0.03], [0.44, 0.04], [0.42, 0.14], [0.38, 0.28], [0.33, 0.43], [0.29, 0.4], [0.25, 0.25], [0.23, 0.08]].map(([x, y]) => Q(x, y)), 0.1, 0.24, sw * 0.012);
    ribbon([Q(0.25, 0.04), Q(0.36, 0.0), Q(0.46, -0.04), Q(0.53, -0.03)], sw * 0.034, 0.16, 0.26); // espina y acromion
    ribbon([Q(0.36, 0.0), Q(0.41, 0.05), Q(0.44, 0.06)], sw * 0.028, 0.14, 0.18);           // coracoides
    ribbon([Q(0.44, -0.01), Q(0.455, 0.04), Q(0.44, 0.09)], sw * 0.022, 0.2, 0.2);          // glenoide
    // Costillas: arco posterior (más denso) y anterior (más tenue)
    for (let k = 0; k < 12; k++) {
      const y = 0.083 * k, W = RIB_W[k];
      if (k === 0) { ribbon([[0.05, 0.0], [0.1, -0.045], [0.18, -0.045], [0.215, 0.01], [0.18, 0.065], [0.11, 0.09]].map(([x, yy]) => Q(x, yy)), sw * 0.03, 0.18, 0.32); continue; }
      if (k >= 10) { ribbon([Q(0.05, y), Q(0.15, y + 0.005), Q(0.75 * W, y + 0.06), Q(W, y + 0.13)], (t) => sw * 0.032 * (1 - 0.45 * t), 0.17, 0.3); continue; }
      ribbon([Q(0.05, y), Q(0.15, y - 0.03), Q(0.6 * W, y - 0.015), Q(0.9 * W, y + 0.05), Q(W, y + 0.13), Q(0.97 * W, y + 0.21)], (t) => sw * (0.031 - 0.007 * t), 0.16, 0.34);
      const dA = k <= 5 ? 0.4 : [0.32, 0.24, 0.16, 0.1][k - 6], A = k <= 5 ? 0.13 + 0.012 * k : [0.22, 0.28, 0.33, 0.37][k - 6];
      ribbon([Q(0.97 * W, y + 0.21), Q(0.8 * W, y + 0.21 + (dA - 0.21) * 0.55), Q(Math.max(A, 0.55 * W), y + dA * 0.95), Q(A, y + dA)], (t) => sw * (0.024 - 0.006 * t), 0.06, 0.15);
    }
    // Clavícula en S
    ribbon([[0.04, 0.05], [0.12, 0.025], [0.22, 0.015], [0.32, -0.015], [0.42, -0.035], [0.49, -0.03]].map(([x, y]) => Q(x, y)), (t) => sw * (0.05 - 0.017 * Math.sin(Math.PI * t)), 0.2, 0.38);
  }
  // Esternón
  blob([C(-0.06, 0.06), C(0.06, 0.06), C(0.045, 0.17), C(-0.045, 0.17)], 0.08, 0.1, sw * 0.008);
  ribbon([C(0, 0.18), C(0, 0.56)], sw * 0.06, 0.06, 0.08);
}

function organs(ch, o) {
  const { C, sw } = ch, t = o.t, hs = o.mirror ? -1 : 1;
  const breath = Math.sin((t * 2 * Math.PI) / 4.6) * 0.02;
  // Pulmones (aire)
  for (const s of [-1, 1]) {
    const dome = 0.66 + breath + (s === hs ? 0.03 : 0);
    const pts = [[0.15, -0.12], [0.25, -0.09], [0.36, 0.02], [0.42, 0.18], [0.45, 0.38], [0.46, dome - 0.06], [0.44, dome + 0.07], [0.35, dome], [0.22, dome - 0.05], [0.11, dome - 0.03], [0.09, 0.45], [0.08, 0.25], [0.09, 0.08], [0.11, -0.06]].map(([x, y]) => C(s * x, y));
    soft(pts, 0.85, "air", 5);
    // Vasos pulmonares desde el hilio
    const hil = C(s * 0.13, 0.3);
    for (const v of VESSELS) {
      const dir = unit({ x: Math.cos(v.ang) * s, y: Math.sin(v.ang) });
      const L = v.len * sw, side = perp(dir);
      const p1 = add(add(hil, dir, L * 0.5), side, L * v.bend * 0.3), p2 = add(add(hil, dir, L), side, L * v.bend);
      ribbon([hil, p1, p2], (tt) => sw * 0.018 * (1 - tt * 0.8), 0.16, 0, "soft");
      const bd = unit(add(dir, side, v.bang));
      const b0 = mixP(hil, p2, v.br);
      ribbon([b0, add(b0, bd, L * 0.45)], (tt) => sw * 0.009 * (1 - tt * 0.7), 0.12, 0, "soft");
    }
  }
  // Mediastino, botón aórtico y corazón (late)
  soft([C(-0.075, -0.16), C(0.075, -0.16), C(0.09, 0.3), C(-0.09, 0.3)], 0.45, "soft", 4);
  soft(ellPts(C(hs * 0.11, 0.16), sw * 0.06, sw * 0.055), 0.4, "soft", 3);
  const beat = Math.pow(Math.max(0, Math.sin(t * 7.2)), 8) + 0.6 * Math.pow(Math.max(0, Math.sin(t * 7.2 - 1)), 8);
  const k = 1 + beat * 0.035, hc = C(hs * 0.1, 0.5);
  const heart = [[-0.11, 0.3], [-0.14, 0.42], [-0.13, 0.56], [-0.08, 0.66], [0.05, 0.7], [0.2, 0.69], [0.31, 0.62], [0.35, 0.5], [0.29, 0.37], [0.17, 0.3], [0.05, 0.28]]
    .map(([x, y]) => C(hs * x, y)).map((p) => ({ x: hc.x + (p.x - hc.x) * k, y: hc.y + (p.y - hc.y) * k }));
  soft(heart, 0.6, "soft", 5);
  // Burbuja gástrica bajo el diafragma izquierdo
  soft(ellPts(C(hs * 0.27, 0.8 + breath), sw * 0.08, sw * 0.045), 0.6, "air", 4);
}

// ------------------------------------------------------------------ brazos y manos
function hand(W, P, I, T, sw) {
  if (P.v < 0.3 || I.v < 0.3) return;
  const K = mid(P, I), hl = dist(W, K);
  if (hl < 4) return;
  const h = unit(sub(K, W)), pr = perp(h);
  // Carpo: dos filas de huesos pequeños
  for (let row = 0; row < 2; row++) for (let j = 0; j < 4; j++) {
    const c = add(add(W, h, hl * (0.1 + row * 0.14)), pr, (j - 1.5) * hl * 0.17 * (row ? 1.1 : 1));
    blob(ellPts(c, hl * 0.075, hl * 0.065, h, 8), 0.16, 0.26, hl * 0.02);
  }
  const fac = [0.8, 0.92, 1, 0.95];
  for (let j = 0; j < 4; j++) {
    const u = j / 3, kp = mixP(P, I, u);
    const Kj = add(kp, h, hl * 0.12 * Math.sin(Math.PI * u));
    const Bj = add(add(W, h, hl * 0.32), pr, dot(sub(kp, K), pr) * 0.55);
    ribbon([Bj, Kj], (t) => hl * (t < 0.12 || t > 0.88 ? 0.11 : 0.075), 0.16, 0.36); // metacarpo
    const fd = unit(sub(Kj, Bj));
    let p0 = add(Kj, fd, hl * 0.07);
    for (const [len, wd] of [[0.48, 0.085], [0.3, 0.07], [0.22, 0.06]]) {
      const p1 = add(p0, fd, len * hl * fac[j]);
      ribbon([p0, p1], (t) => hl * wd * (t < 0.15 || t > 0.85 ? 1.15 : 0.8), 0.16, 0.36); // falanges
      p0 = add(p1, fd, hl * 0.05);
    }
  }
  if (T.v > 0.3) {
    const sT = Math.sign(dot(sub(T, W), pr)) || 1;
    const tb = add(add(W, h, hl * 0.18), pr, sT * hl * 0.28), td = unit(sub(T, tb)), m = mixP(tb, T, 0.48);
    ribbon([tb, m], hl * 0.1, 0.16, 0.36);
    ribbon([add(m, td, hl * 0.06), T], hl * 0.085, 0.16, 0.36);
    ribbon([add(T, td, hl * 0.06), add(T, td, hl * 0.32)], hl * 0.075, 0.16, 0.36);
  }
}

function arms(ch, L) {
  const { sw, ax } = ch;
  for (const [sh, el, wr, pk, ix, th] of [[L[12], L[14], L[16], L[18], L[20], L[22]], [L[11], L[13], L[15], L[17], L[19], L[21]]]) {
    if (sh.v < 0.5) continue;
    const elP = el.v > 0.3 ? el : add(sh, ax, sw * 0.75);
    const hd = unit(sub(elP, sh));
    blob(ellPts(add(sh, hd, sw * 0.02), sw * 0.088, sw * 0.088, hd, 12), 0.22, 0.36, sw * 0.014); // cabeza humeral
    ribbon([add(sh, hd, sw * 0.05), elP], (t) => sw * (t < 0.15 ? lerp(0.13, 0.072, t / 0.15) : t > 0.85 ? lerp(0.072, 0.135, (t - 0.85) / 0.15) : 0.072), 0.16, 0.42);
    if (el.v < 0.3 || wr.v < 0.3) continue;
    const dir = unit(sub(wr, el)), pr = perp(dir), o = sw * 0.024;
    const sU = pk.v > 0.3 ? Math.sign(dot(sub(pk, wr), pr)) || 1 : 1;
    ribbon([add(el, pr, o * sU), add(wr, pr, o * sU * 0.8)], (t) => sw * lerp(0.08, 0.034, t), 0.15, 0.4);   // cúbito
    ribbon([add(el, pr, -o * sU * 0.6), add(wr, pr, -o * sU)], (t) => sw * lerp(0.042, 0.078, t), 0.15, 0.4); // radio
    hand(wr, pk, ix, th, sw);
  }
}

// ------------------------------------------------------------------ pelvis y piernas
function pelvisLegs(ch, L) {
  const hl = L[24], hr = L[23], hw = dist(hl, hr);
  if (hw < 4) return;
  const dh = unit(sub(hr, hl)), dn = ch.ax, H = mid(hl, hr);
  const B = (x, y) => ({ x: H.x + dh.x * x * hw + dn.x * y * hw, y: H.y + dh.y * x * hw + dn.y * y * hw });
  blob([B(-0.18, -0.62), B(0.18, -0.62), B(0.12, -0.25), B(0.04, 0), B(-0.04, 0), B(-0.12, -0.25)], 0.16, 0.24, hw * 0.015); // sacro
  for (let j = 0; j < 4; j++) for (const s of [-1, 1]) soft(ellPts(B(s * 0.08, -0.52 + j * 0.11), hw * 0.025, hw * 0.022), 0.3, "air", 2);
  for (const s of [-1, 1]) {
    const Q = (x, y) => B(s * x, y);
    blob([[0.1, -0.6], [0.3, -0.8], [0.55, -0.85], [0.75, -0.72], [0.8, -0.5], [0.68, -0.3], [0.5, -0.12], [0.33, -0.2], [0.15, -0.35]].map(([x, y]) => Q(x, y)), 0.12, 0.32, hw * 0.022);
    ribbon([Q(0.12, -0.3), Q(0.3, -0.12), Q(0.4, 0.05), Q(0.3, 0.22), Q(0.06, 0.26)], hw * 0.035, 0.16, 0.24); // estrecho pélvico
    ring(Q(0.24, 0.22), hw * 0.13, hw * 0.15, dh, hw * 0.06, 0.18, 0.24);                                   // agujero obturador
    ribbon([Q(0.32, -0.12), Q(0.45, -0.19), Q(0.6, -0.1)], hw * 0.04, 0.2, 0.26);                          // techo acetabular
    blob(ellPts(Q(0.45, -0.02), hw * 0.14, hw * 0.14, dh, 12), 0.2, 0.36, hw * 0.02);                      // cabeza femoral
    ribbon([Q(0.45, -0.02), Q(0.62, 0.14)], hw * 0.15, 0.16, 0.3);                                          // cuello femoral
    blob(ellPts(Q(0.68, 0.1), hw * 0.09, hw * 0.1, dh, 10), 0.14, 0.26, hw * 0.015);                        // trocánter
  }
  for (const [knee, ank, heel, toe, s] of [[L[26], L[28], L[30], L[32], -1], [L[25], L[27], L[29], L[31], 1]]) {
    if (knee.v < 0.4) continue;
    const start = B(s * 0.6, 0.14);
    ribbon([start, knee], (t) => hw * (t > 0.85 ? lerp(0.13, 0.26, (t - 0.85) / 0.15) : 0.13), 0.16, 0.42);
    blob(ellPts(knee, hw * 0.09, hw * 0.1, dh, 10), 0.12, 0.22, hw * 0.015); // rótula
    if (ank.v < 0.4) continue;
    const ld = unit(sub(ank, knee)), lat = perp(ld), sl = Math.sign(dot(sub(knee, H), dh)) * (dot(lat, dh) > 0 ? 1 : -1) || 1;
    ribbon([knee, ank], (t) => hw * (t < 0.12 ? lerp(0.25, 0.13, t / 0.12) : t > 0.9 ? lerp(0.11, 0.15, (t - 0.9) / 0.1) : 0.11), 0.16, 0.42); // tibia
    ribbon([add(knee, lat, sl * hw * 0.15), add(ank, lat, sl * hw * 0.11)], (t) => hw * (t < 0.1 || t > 0.9 ? 0.07 : 0.045), 0.12, 0.32); // peroné
    if (toe.v > 0.4) {
      const hp = heel.v > 0.4 ? heel : add(ank, ld, hw * 0.15);
      blob(ellPts(add(ank, ld, hw * 0.04), hw * 0.1, hw * 0.08, dh, 10), 0.18, 0.28, hw * 0.015);
      ribbon([add(ank, ld, hw * 0.08), hp], hw * 0.14, 0.16, 0.3);
      const fd = unit(sub(toe, mid(ank, hp))), fp = perp(fd);
      for (let j = -2; j <= 2; j++) {
        const b0 = add(add(mid(ank, hp), fd, hw * 0.12), fp, j * hw * 0.04);
        ribbon([b0, add(toe, fp, j * hw * 0.05)], hw * 0.04, 0.14, 0.3);
      }
    }
  }
}

// ------------------------------------------------------------------ entrada
export function drawAnatomy(ctx, L, o) {
  D = ctx;
  D.globalCompositeOperation = "lighter";
  D.lineCap = "round"; D.lineJoin = "round";
  const head = headFrame(L), chest = chestFrame(L);
  if (chest && o.organs) organs(chest, o);
  if (!o.bones) return;
  if (chest) { thorax(chest); spine(chest, head); }
  if (head) skull(head);
  if (chest) { arms(chest, L); if (chest.hipsOK) pelvisLegs(chest, L); }
}
