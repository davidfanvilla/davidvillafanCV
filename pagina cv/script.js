"use strict";

let points = [];
let numPoints = 800;
let maxSize = 20;
let cvGraphics;
let contentHeight = 4000; // Altura total del contenido
let scrollY = 0;
let velocity = 0;
let lastTime = 0;

function setup() {
  // Ajusta el canvas para llenar el ancho de la ventana
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent('canvas-container');
  
  for (let i = 0; i < numPoints; i++) {
    points.push({
      x: random(width),
      y: random(contentHeight), // Los puntos se distribuyen en toda la altura del contenido
      size: random(3, maxSize),
      grow: random(0.02, 0.05),
      direction: random([1, -1]),
      speedX: random(-1, 1),
      speedY: random(-1, 1),
      accelX: random(-0.01, 0.01),
      accelY: random(-0.01, 0.01),
    });
  }

  cvGraphics = createGraphics(windowWidth, contentHeight);
  cvGraphics.textFont('Arial');
  cvGraphics.textAlign(CENTER, TOP);

  dibujarCV();
}

function draw() {
  background(10);

  let currentTime = millis();
  let deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;

  scrollY += velocity * deltaTime;
  velocity *= 0.95; // Reduce la velocidad gradualmente para un efecto de inercia

  scrollY = constrain(scrollY, 0, contentHeight - height); // Limita el scroll al contenido

  push();
  translate(0, -scrollY); // Aplica el desplazamiento del scroll

  // Dibuja los puntos y las conexiones entre ellos
  for (let i = 0; i < points.length; i++) {
    let p = points[i];
    
    p.size += p.grow * p.direction;
    if (p.size > maxSize || p.size < 3) {
      p.direction *= -1;
    }
    
    p.speedX += p.accelX;
    p.speedY += p.accelY;
    p.x += p.speedX;
    p.y += p.speedY;
    
    if (p.x < 0 || p.x > width) p.speedX *= -1;
    if (p.y < 0 || p.y > contentHeight) p.speedY *= -1;
    
    fill(58, 128, 189);
    noStroke();
    ellipse(p.x, p.y, p.size, p.size);
    
    for (let j = i + 1; j < points.length; j++) {
      let p2 = points[j];
      let d = dist(p.x, p.y, p2.x, p2.y);
      if (d < 150) {
        stroke(58, 128, 189, 50);
        line(p.x, p.y, p2.x, p2.y);
      }
    }
  }

  image(cvGraphics, 0, 0);

  pop();
}

function dibujarCV() {
  cvGraphics.clear();
  let titleSize = min(windowWidth * 0.05, 28);
  let subtitleSize = min(windowWidth * 0.035, 20);
  let sectionTitleSize = min(windowWidth * 0.045, 24);
  let textGap = min(windowWidth * 0.04, 25);

  let margin = 20;
  let effectiveWidth = windowWidth - 2 * margin;

  function drawTextWithShadow(text, x, y, maxWidth, lineHeight) {
    let words = text.split(' ');
    let line = '';
    for (let n = 0; n < words.length; n++) {
      let testLine = line + words[n] + ' ';
      let testWidth = cvGraphics.textWidth(testLine);
      if (testWidth > maxWidth && n > 0) {
        cvGraphics.fill(0);
        cvGraphics.text(line, x + 1, y + 1);
        cvGraphics.fill(255);
        cvGraphics.text(line, x, y);
        y += lineHeight;
        line = words[n] + ' ';
      } else {
        line = testLine;
      }
    }
    cvGraphics.fill(0);
    cvGraphics.text(line, x + 1, y + 1);
    cvGraphics.fill(255);
    cvGraphics.text(line, x, y);
    return y + lineHeight;
  }

  cvGraphics.textSize(titleSize);
  let startY = 50;
  startY = drawTextWithShadow("David Villafán Sánchez", windowWidth / 2, startY, effectiveWidth, titleSize * 1.5);

  cvGraphics.textSize(subtitleSize);
  startY = drawTextWithShadow("Fecha de nacimiento: 18/02/1990", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("Correo: davidvillafan180290@hotmail.com", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("Teléfono: 56 61 28 89 93", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);

  cvGraphics.textSize(sectionTitleSize);
  startY = drawTextWithShadow("Habilidades:", windowWidth / 2, startY + textGap * 2, effectiveWidth, sectionTitleSize * 1.5);

  cvGraphics.textSize(subtitleSize);
  startY = drawTextWithShadow("Buena comunicación / Resolución de problemas", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("Liderazgo / Adaptación / Persuasivo", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);

  cvGraphics.textSize(sectionTitleSize);
  startY = drawTextWithShadow("Experiencia Laboral:", windowWidth / 2, startY + textGap * 2, effectiveWidth, sectionTitleSize * 1.5);

  cvGraphics.textSize(subtitleSize);
  startY = drawTextWithShadow("Escuela de Aviación México", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("· Jefe de Operaciones Aéreas", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("· Area de Compras de partes aeronáuticas", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("· Administrativo en el Área de Control Escolar", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);

  cvGraphics.textSize(sectionTitleSize);
  startY = drawTextWithShadow("Herramientas de trabajo:", windowWidth / 2, startY + textGap * 2, effectiveWidth, sectionTitleSize * 1.5);

  cvGraphics.textSize(subtitleSize);
  startY = drawTextWithShadow("Uso de prompts para IA: ChatGPT, Gemini (google), Groq, Claude, Dall-e (generacion de imagenes con texto) hago Vibe Coding y puedo hacer una pagina web desde cero", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("Software que uso: Visual Studio Code, Cursor, Blender, Processing, P5.js. Con estos dos ultimos tengo proyectos de Arte Generativo", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
  startY = drawTextWithShadow("Lenguajes de Programación: HTML, Javascript, CSS", windowWidth / 2, startY + textGap, effectiveWidth, subtitleSize * 1.5);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  cvGraphics.resizeCanvas(windowWidth, contentHeight);
  dibujarCV();
}

function mouseWheel(event) {
  velocity += event.delta * 1.0;
  return false;
}