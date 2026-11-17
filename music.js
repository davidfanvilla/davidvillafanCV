"use strict";

/* ============================================================
   Música 8 bits original, generada en tiempo real (Web Audio)
   Ambiente espacial: órgano lento, arpegio repetitivo, bajo,
   tic-tac de reloj y melodía con eco.
   No usa archivos de audio: todo se sintetiza en el navegador.

   Expone window.cvMusic.read() -> { level, pulse, beats }
   para que el fondo (script.js) reaccione al ritmo.
   ============================================================ */
(function () {
  const BPM = 66;
  const STEP = 60 / BPM / 4;      // duración de una semicorchea (s)
  const STEPS_PER_BAR = 16;
  const LOOKAHEAD = 0.15;         // cuánto adelantamos la programación (s)

  // Progresión en La menor: 4 acordes, 2 compases cada uno (8 compases)
  // root = bajo (MIDI), tones = notas del acorde
  const CHORDS = [
    { root: 45, tones: [57, 60, 64] },     // Am
    { root: 41, tones: [53, 57, 60, 64] }, // Fmaj7
    { root: 48, tones: [55, 60, 64, 67] }, // C
    { root: 40, tones: [55, 59, 64] },     // Em
  ];

  // Patrón del arpegio (índices sobre las notas del acorde extendidas 2 octavas)
  const ARP = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 5, 4, 3, 2];

  // Melodía original: por compás, [paso, nota MIDI, duración en pasos]
  const MELODY = [
    [[0, 76, 8], [8, 74, 4], [12, 72, 4]],
    [[0, 71, 12]],
    [[0, 72, 8], [8, 69, 8]],
    [[0, 76, 16]],
    [[0, 79, 8], [8, 76, 4], [12, 74, 4]],
    [[0, 72, 16]],
    [[0, 71, 8], [8, 74, 8]],
    [[0, 76, 16]],
  ];

  const midiToHz = (m) => 440 * Math.pow(2, (m - 69) / 12);

  let ctx = null;
  let master, analyser, echoIn, noiseBuffer, pulse25, pulse125;
  let timer = null;
  let playing = false;
  let step = 0;          // paso global desde que empezó
  let nextTime = 0;
  const beatTimes = [];  // momentos (tiempo de audio) de cada tiempo fuerte
  let beatsPlayed = 0;
  const waveData = new Uint8Array(256);

  /* ---------- Construcción del sintetizador ---------- */
  function pulseWave(duty) {
    // Onda de pulso (el sonido clásico de las consolas de 8 bits)
    const n = 32;
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    for (let k = 1; k < n; k++) {
      real[k] = (2 / (k * Math.PI)) * Math.sin(2 * Math.PI * k * duty);
      imag[k] = (2 / (k * Math.PI)) * (1 - Math.cos(2 * Math.PI * k * duty));
    }
    return ctx.createPeriodicWave(real, imag);
  }

  function build() {
    ctx = new (window.AudioContext || window.webkitAudioContext)();

    master = ctx.createGain();
    master.gain.value = 0;

    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18;
    comp.ratio.value = 4;

    analyser = ctx.createAnalyser();
    analyser.fftSize = 256;

    master.connect(comp);
    comp.connect(analyser);
    analyser.connect(ctx.destination);

    // Eco espacial (corchea con puntillo)
    echoIn = ctx.createGain();
    const delay = ctx.createDelay(2);
    delay.delayTime.value = STEP * 3;
    const feedback = ctx.createGain();
    feedback.gain.value = 0.38;
    const tone = ctx.createBiquadFilter();
    tone.type = "lowpass";
    tone.frequency.value = 2200;
    const wet = ctx.createGain();
    wet.gain.value = 0.35;
    echoIn.connect(delay);
    delay.connect(tone);
    tone.connect(feedback);
    feedback.connect(delay);
    tone.connect(wet);
    wet.connect(master);

    // Ruido blanco para el tic-tac
    noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 0.1, ctx.sampleRate);
    const data = noiseBuffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;

    pulse25 = pulseWave(0.25);
    pulse125 = pulseWave(0.125);
  }

  /* ---------- Voces ---------- */
  function envGain(time, peak, attack, hold, release) {
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, time);
    g.gain.exponentialRampToValueAtTime(peak, time + attack);
    g.gain.setValueAtTime(peak, time + attack + hold);
    g.gain.exponentialRampToValueAtTime(0.0001, time + attack + hold + release);
    return g;
  }

  function voice({ time, midi, wave, peak, attack, hold, release, detune = 0, cutoff = 0, echo = false }) {
    const osc = ctx.createOscillator();
    if (typeof wave === "string") osc.type = wave;
    else osc.setPeriodicWave(wave);
    osc.frequency.value = midiToHz(midi);
    osc.detune.value = detune;

    const g = envGain(time, peak, attack, hold, release);
    let out = osc;
    if (cutoff) {
      const f = ctx.createBiquadFilter();
      f.type = "lowpass";
      f.frequency.value = cutoff;
      osc.connect(f);
      out = f;
    }
    out.connect(g);
    g.connect(master);
    if (echo) g.connect(echoIn);

    const end = time + attack + hold + release + 0.05;
    osc.start(time);
    osc.stop(end);
  }

  function tick(time, accent) {
    const src = ctx.createBufferSource();
    src.buffer = noiseBuffer;
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass";
    hp.frequency.value = accent ? 5000 : 7500;
    const g = envGain(time, accent ? 0.09 : 0.045, 0.002, 0.005, 0.04);
    src.connect(hp);
    hp.connect(g);
    g.connect(master);
    src.start(time);
    src.stop(time + 0.1);
  }

  /* ---------- Secuenciador ---------- */
  function schedule(globalStep, time) {
    const bar = Math.floor(globalStep / STEPS_PER_BAR);
    const s = globalStep % STEPS_PER_BAR;
    const loop = Math.floor(bar / 8);           // vuelta completa de 8 compases
    const barInLoop = bar % 8;
    const chord = CHORDS[Math.floor(barInLoop / 2)];
    const firstBarOfChord = barInLoop % 2 === 0;

    // Órgano: acorde sostenido al empezar cada acorde
    if (s === 0 && firstBarOfChord) {
      for (const n of chord.tones) {
        voice({ time, midi: n, wave: "square", peak: 0.028, attack: 1.2, hold: STEP * 26, release: 1.8, detune: -6, cutoff: 1400 });
        voice({ time, midi: n + 12, wave: pulse25, peak: 0.012, attack: 1.6, hold: STEP * 24, release: 1.8, detune: 7, cutoff: 1800 });
      }
    }

    // Arpegio (entra suave en la primera vuelta)
    const tones = chord.tones.slice(0, 3);
    const ext = [...tones, ...tones.map((t) => t + 12), tones[0] + 24];
    const arpPeak = loop === 0 ? 0.02 + 0.03 * (bar / 8) : 0.05;
    voice({ time, midi: ext[ARP[s]], wave: pulse125, peak: arpPeak, attack: 0.005, hold: STEP * 0.4, release: STEP * 1.2, echo: true });

    // Bajo (desde la segunda vuelta)
    if (loop >= 1 && (s === 0 || s === 8 || s === 14)) {
      voice({ time, midi: chord.root, wave: "triangle", peak: 0.22, attack: 0.01, hold: STEP * (s === 14 ? 1 : 5), release: 0.25 });
    }

    // Melodía (desde la segunda vuelta; descansa cada cuarta vuelta)
    if (loop >= 1 && loop % 4 !== 0) {
      const octave = loop % 4 === 3 ? 12 : 0;
      for (const [at, midi, len] of MELODY[barInLoop]) {
        if (at === s) {
          voice({ time, midi: midi + octave, wave: pulse25, peak: 0.06, attack: 0.04, hold: STEP * len * 0.8, release: 0.6, detune: 4, echo: true });
        }
      }
    }

    // Tic-tac de reloj en cada tiempo
    if (s % 4 === 0) {
      tick(time, s === 0);
      beatTimes.push(time);
    }
  }

  function scheduler() {
    while (nextTime < ctx.currentTime + LOOKAHEAD) {
      schedule(step, nextTime);
      step++;
      nextTime += STEP;
    }
  }

  /* ---------- Control ---------- */
  async function start() {
    if (!ctx) build();
    await ctx.resume();
    playing = true;
    nextTime = ctx.currentTime + 0.1;
    clearInterval(timer);
    timer = setInterval(scheduler, 25);
    master.gain.cancelScheduledValues(ctx.currentTime);
    master.gain.setValueAtTime(master.gain.value, ctx.currentTime);
    master.gain.linearRampToValueAtTime(0.9, ctx.currentTime + 1.5);
  }

  function stop() {
    if (!ctx) return;
    playing = false;
    const now = ctx.currentTime;
    master.gain.cancelScheduledValues(now);
    master.gain.setValueAtTime(master.gain.value, now);
    master.gain.linearRampToValueAtTime(0, now + 0.8);
    setTimeout(() => {
      if (!playing) {
        clearInterval(timer);
        ctx.suspend();
      }
    }, 900);
  }

  // Lectura para las visuales del fondo
  function read() {
    if (!ctx || ctx.state !== "running") return { level: 0, pulse: 0, beats: beatsPlayed };
    analyser.getByteTimeDomainData(waveData);
    let sum = 0;
    for (let i = 0; i < waveData.length; i++) {
      const v = (waveData[i] - 128) / 128;
      sum += v * v;
    }
    const level = Math.min(1, Math.sqrt(sum / waveData.length) * 4);

    const now = ctx.currentTime;
    let last = -Infinity;
    while (beatTimes.length && beatTimes[0] <= now) {
      last = beatTimes.shift();
      beatsPlayed++;
    }
    if (last > -Infinity) read.lastBeat = last;
    const pulse = read.lastBeat ? Math.exp(-(now - read.lastBeat) * 5) : 0;
    return { level, pulse, beats: beatsPlayed };
  }

  window.cvMusic = { read, isPlaying: () => playing };

  /* ---------- Botón ---------- */
  const button = document.getElementById("sound-toggle");
  if (!button || !(window.AudioContext || window.webkitAudioContext)) {
    if (button) button.hidden = true;
    return;
  }

  function render() {
    button.classList.toggle("is-on", playing);
    button.setAttribute("aria-pressed", String(playing));
    button.setAttribute("aria-label", playing ? "Pausar música" : "Reproducir música");
    button.querySelector(".sound-toggle__label").textContent = playing ? "Música ON" : "Música OFF";
  }

  button.addEventListener("click", async () => {
    if (playing) stop();
    else await start();
    render();
  });

  // Pausa cuando la pestaña no está visible y reanuda al volver
  let resumeOnReturn = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && playing) {
      resumeOnReturn = true;
      stop();
    } else if (!document.hidden && resumeOnReturn) {
      resumeOnReturn = false;
      start().then(render);
    }
    render();
  });

  render();
})();
