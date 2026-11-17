/* ============================================================
   Fondo 3D futurista (Three.js + bloom)
   - Terreno de malla de neón que avanza (montañas a los lados)
   - Núcleo abstracto: icosaedro de cristal, aristas de luz,
     anillos orbitales y un enjambre de octaedros
   - Formas abstractas flotantes (cristal con brillo de borde)
   - Sol retro-futurista al fondo y campo de estrellas
   - Reacciona al cursor, al scroll y a la música 8 bits
   ============================================================ */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const container = document.getElementById("canvas-container");
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const MOBILE = window.matchMedia("(max-width: 700px), (pointer: coarse)").matches;

// Paleta (la misma del CV)
const CYAN = new THREE.Color("#22e6ff");
const MAGENTA = new THREE.Color("#ff2bd6");
const VIOLET = new THREE.Color("#8a5cff");
const BG = new THREE.Color("#05060f");

/* ---------- Render ---------- */
// Si el navegador no tiene WebGL, esto falla y se queda el fondo oscuro del CSS
const renderer = new THREE.WebGLRenderer({ antialias: !MOBILE, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MOBILE ? 1 : 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(BG);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070616, 0.016);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
const CAM_BASE = new THREE.Vector3(0, 6, 24);
camera.position.copy(CAM_BASE);
const LOOK = new THREE.Vector3(0, 6, -10);

/* ---------- Materiales reutilizables ---------- */

// Cristal con brillo en los bordes (efecto fresnel)
function fresnelMaterial(colorA, colorB, intensity = 1.2) {
  return new THREE.ShaderMaterial({
    uniforms: THREE.UniformsUtils.merge([
      THREE.UniformsLib.fog,
      {
        uColorA: { value: colorA.clone() },
        uColorB: { value: colorB.clone() },
        uIntensity: { value: intensity },
      },
    ]),
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vView;
      #include <fog_pars_vertex>
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-mvPosition.xyz);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColorA;
      uniform vec3 uColorB;
      uniform float uIntensity;
      varying vec3 vNormal;
      varying vec3 vView;
      #include <fog_pars_fragment>
      void main() {
        float f = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
        vec3 col = mix(uColorA, uColorB, f) * (0.03 + f * uIntensity * 0.6);
        gl_FragColor = vec4(col, 0.1 + f * 0.6);
        #include <fog_fragment>
      }`,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    fog: true,
  });
}

function edgeMaterial(color, opacity = 0.9) {
  return new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    fog: true,
  });
}

/* ---------- 1. Terreno de neón ---------- */
const NOISE_GLSL = /* glsl */ `
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
  }
  float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
    return v;
  }`;

const segs = MOBILE ? 70 : 120;
const terrainGeo = new THREE.PlaneGeometry(260, 260, segs, segs);
const terrainMat = new THREE.ShaderMaterial({
  uniforms: {
    uOffset: { value: 0 },
    uAmp: { value: 14 },
    uCyan: { value: CYAN },
    uMagenta: { value: MAGENTA },
    uFog: { value: 0.016 },
    uPulse: { value: 0 },
  },
  vertexShader: /* glsl */ `
    uniform float uOffset;
    uniform float uAmp;
    varying float vH;
    varying float vDepth;
    ${NOISE_GLSL}
    void main() {
      vec3 p = position;
      // Valle plano al centro y montañas a los lados
      float valley = smoothstep(7.0, 38.0, abs(p.x));
      float h = fbm(vec2(p.x * 0.045, (p.y + uOffset) * 0.045));
      p.z = pow(h, 1.6) * uAmp * 2.2 * valley;
      vH = clamp(p.z / (uAmp * 1.4), 0.0, 1.0);
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      vDepth = -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uCyan;
    uniform vec3 uMagenta;
    uniform float uFog;
    uniform float uPulse;
    varying float vH;
    varying float vDepth;
    void main() {
      vec3 col = mix(uMagenta, uCyan, smoothstep(0.0, 0.8, vH));
      float fog = exp(-pow(uFog * vDepth, 2.0));
      float a = fog * (0.16 + vH * 0.45 + uPulse * 0.25);
      gl_FragColor = vec4(col * a, a);
    }`,
  wireframe: true,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
const terrain = new THREE.Mesh(terrainGeo, terrainMat);
terrain.rotation.x = -Math.PI / 2;
terrain.position.set(0, -2, -90);
scene.add(terrain);

/* ---------- 2. Sol retro-futurista ---------- */
const sunMat = new THREE.ShaderMaterial({
  uniforms: { uTime: { value: 0 }, uPulse: { value: 0 }, uCyan: { value: CYAN }, uMagenta: { value: MAGENTA } },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform float uTime;
    uniform float uPulse;
    uniform vec3 uCyan;
    uniform vec3 uMagenta;
    varying vec2 vUv;
    void main() {
      vec2 p = vUv * 2.0 - 1.0;
      float r = length(p);
      if (r > 1.0) discard;
      vec3 col = mix(uMagenta, uCyan, smoothstep(-0.6, 0.9, p.y));
      // Franjas que se desplazan en la mitad inferior
      float bands = step(0.5, fract((p.y - uTime * 0.05) * 9.0));
      float cut = p.y < 0.15 ? mix(1.0, bands, smoothstep(0.15, -0.2, p.y) * 0.9 + 0.1) : 1.0;
      float glow = smoothstep(1.0, 0.85, r);
      gl_FragColor = vec4(col * cut * glow * (0.32 + uPulse * 0.25), 1.0);
    }`,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  fog: false,
});
const sun = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), sunMat);
sun.position.set(0, 10, -170);
scene.add(sun);

/* ---------- 3. Núcleo abstracto ---------- */
const core = new THREE.Group();
core.position.set(0, 7, -10);
scene.add(core);

const crystal = new THREE.Mesh(new THREE.IcosahedronGeometry(3.4, 1), fresnelMaterial(VIOLET, CYAN, 1.3));
core.add(crystal);

const crystalEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(3.45, 1)),
  edgeMaterial(CYAN, 0.85)
);
core.add(crystalEdges);

const shellEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.DodecahedronGeometry(5.2, 0)),
  edgeMaterial(MAGENTA, 0.55)
);
core.add(shellEdges);

const heart = new THREE.Mesh(
  new THREE.OctahedronGeometry(0.9, 0),
  new THREE.MeshBasicMaterial({ color: new THREE.Color("#9ff3ff") })
);
core.add(heart);

// Anillos orbitales
const rings = [];
[
  { r: 7.0, color: CYAN, tilt: [1.2, 0.2, 0], speed: 0.35 },
  { r: 8.3, color: MAGENTA, tilt: [0.4, 1.0, 0.3], speed: -0.25 },
  { r: 9.6, color: VIOLET, tilt: [1.6, -0.5, 0.8], speed: 0.18 },
].forEach((cfg) => {
  const pivot = new THREE.Group();
  pivot.rotation.set(...cfg.tilt);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(cfg.r, 0.045, 8, 220),
    new THREE.MeshBasicMaterial({ color: cfg.color, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  pivot.add(ring);
  // Pequeño "satélite" sobre cada anillo
  const sat = new THREE.Mesh(new THREE.OctahedronGeometry(0.28, 0), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  sat.position.x = cfg.r;
  ring.add(sat);
  core.add(pivot);
  rings.push({ ring, speed: cfg.speed });
});

// Enjambre de octaedros en órbita (una sola malla instanciada)
const SWARM = MOBILE ? 60 : 120;
const swarm = new THREE.InstancedMesh(
  new THREE.OctahedronGeometry(0.16, 0),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }),
  SWARM
);
const swarmData = [];
for (let i = 0; i < SWARM; i++) {
  swarmData.push({
    r: 5.8 + Math.random() * 5,
    a: Math.random() * Math.PI * 2,
    y: (Math.random() - 0.5) * 3,
    s: 0.15 + Math.random() * 0.35,
  });
  swarm.setColorAt(i, [CYAN, MAGENTA, VIOLET][i % 3]);
}
core.add(swarm);
const dummy = new THREE.Object3D();

/* ---------- 4. Formas abstractas flotantes ---------- */
const shapeFactories = [
  () => new THREE.OctahedronGeometry(1.6, 0),
  () => new THREE.TetrahedronGeometry(1.8, 0),
  () => new THREE.TorusGeometry(1.4, 0.45, 10, 32),
  () => new THREE.IcosahedronGeometry(1.5, 0),
  () => new THREE.TorusKnotGeometry(1.1, 0.32, 80, 10, 2, 3),
  () => new THREE.DodecahedronGeometry(1.4, 0),
];
const floaters = [];
const FLOATERS = MOBILE ? 9 : 16;
for (let i = 0; i < FLOATERS; i++) {
  const geo = shapeFactories[i % shapeFactories.length]();
  const palette = [[CYAN, MAGENTA], [MAGENTA, VIOLET], [VIOLET, CYAN]][i % 3];
  const g = new THREE.Group();
  g.add(new THREE.Mesh(geo, fresnelMaterial(palette[0], palette[1], 1.3)));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 20), edgeMaterial(palette[1], 0.7)));

  // Repartir a los lados del núcleo, a distintas profundidades
  const side = i % 2 === 0 ? -1 : 1;
  const x = side * (9 + Math.random() * 24);
  const y = 1 + Math.random() * 20;
  const z = -50 + Math.random() * 55;
  g.position.set(x, y, z);
  const s = 0.6 + Math.random() * 1.1;
  g.scale.setScalar(s);
  scene.add(g);
  floaters.push({
    g,
    base: g.position.clone(),
    rot: new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(0.6),
    phase: Math.random() * Math.PI * 2,
  });
}

/* ---------- 5. Estrellas ---------- */
const STARS = MOBILE ? 700 : 1600;
const starPos = new Float32Array(STARS * 3);
const starCol = new Float32Array(STARS * 3);
for (let i = 0; i < STARS; i++) {
  const r = 80 + Math.random() * 160;
  const th = Math.random() * Math.PI * 2;
  const ph = Math.acos(Math.random() * 0.9); // solo hemisferio superior
  starPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
  starPos[i * 3 + 1] = r * Math.cos(ph) * 0.6 + 4;
  starPos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th) - 60;
  const c = [CYAN, MAGENTA, VIOLET, new THREE.Color(1, 1, 1)][i % 4];
  starCol.set([c.r, c.g, c.b], i * 3);
}
const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
starGeo.setAttribute("color", new THREE.BufferAttribute(starCol, 3));
const stars = new THREE.Points(
  starGeo,
  new THREE.PointsMaterial({ size: 0.45, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, fog: false })
);
scene.add(stars);

/* ---------- Post-procesado: resplandor de neón ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2),
  0.6,  // intensidad
  0.5,  // radio
  0.18  // umbral
);
composer.addPass(bloom);
composer.addPass(new OutputPass());

/* ---------- Interacción ---------- */
const pointer = { x: 0, y: 0 };
window.addEventListener("pointermove", (e) => {
  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = (e.clientY / window.innerHeight) * 2 - 1;
}, { passive: true });

let lastScroll = window.scrollY;
let scrollSpeed = 0;
let musicBeats = 0;
let beatFlash = 0;

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  // En pantallas verticales, alejar un poco la cámara para que quepa el núcleo
  camera.fov = w < h ? 72 : 60;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloom.resolution.set(w / 2, h / 2);
  if (REDUCED_MOTION) render(0, 0);
}
window.addEventListener("resize", resize);

/* ---------- Animación ---------- */
const clock = new THREE.Clock();
let offset = 0;

function render(t, dt) {
  // Música (0 si está apagada)
  const m = window.cvMusic ? window.cvMusic.read() : { level: 0, pulse: 0, beats: 0 };
  if (m.beats > musicBeats) beatFlash = 1;
  musicBeats = m.beats;
  beatFlash *= 0.9;

  // Scroll
  const sy = window.scrollY;
  scrollSpeed += (Math.abs(sy - lastScroll) - scrollSpeed) * 0.1;
  lastScroll = sy;
  const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
  const progress = sy / maxScroll;

  // Cámara: parallax con el cursor y avance con el scroll
  const target = new THREE.Vector3(
    CAM_BASE.x + pointer.x * 3.5,
    CAM_BASE.y - pointer.y * 2 + progress * 3,
    CAM_BASE.z - progress * 9
  );
  camera.position.lerp(target, REDUCED_MOTION ? 1 : 0.04);
  camera.lookAt(LOOK);

  // Terreno
  offset += dt * (9 + scrollSpeed * 0.6 + m.level * 14);
  terrainMat.uniforms.uOffset.value = offset;
  terrainMat.uniforms.uAmp.value = 14 + m.level * 6;
  terrainMat.uniforms.uPulse.value = m.pulse;

  // Sol
  sunMat.uniforms.uTime.value = t;
  sunMat.uniforms.uPulse.value = m.pulse;

  // Núcleo
  const beatScale = 1 + m.pulse * 0.07;
  core.rotation.y = t * 0.15 + progress * Math.PI;
  crystal.rotation.set(t * 0.2, t * 0.27, 0);
  crystalEdges.rotation.copy(crystal.rotation);
  crystal.scale.setScalar(beatScale);
  crystalEdges.scale.setScalar(beatScale);
  shellEdges.rotation.set(-t * 0.1, t * 0.12, t * 0.05);
  shellEdges.material.opacity = 0.45 + beatFlash * 0.5;
  heart.rotation.set(t, t * 1.3, 0);
  heart.scale.setScalar(1 + m.pulse * 0.4);
  core.position.y = 7 + Math.sin(t * 0.8) * 0.4;
  for (const r of rings) r.ring.rotation.z = t * r.speed * (1 + m.level * 2);

  for (let i = 0; i < SWARM; i++) {
    const d = swarmData[i];
    const a = d.a + t * d.s;
    dummy.position.set(Math.cos(a) * d.r, d.y + Math.sin(a * 2 + i) * 0.6, Math.sin(a) * d.r);
    dummy.rotation.set(a, a * 1.5, 0);
    dummy.scale.setScalar(1 + m.pulse * 0.8);
    dummy.updateMatrix();
    swarm.setMatrixAt(i, dummy.matrix);
  }
  swarm.instanceMatrix.needsUpdate = true;

  // Formas flotantes
  for (const f of floaters) {
    f.g.rotation.x += f.rot.x * dt;
    f.g.rotation.y += f.rot.y * dt;
    f.g.rotation.z += f.rot.z * dt;
    f.g.position.y = f.base.y + Math.sin(t * 0.6 + f.phase) * 1.2;
  }

  stars.rotation.y = t * 0.005;
  bloom.strength = 0.6 + m.pulse * 0.45 + m.level * 0.3;

  composer.render();
}

resize();

if (REDUCED_MOTION) {
  render(0, 0); // un solo cuadro, sin animación
} else {
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05);
    render(clock.elapsedTime, dt);
  });
}
