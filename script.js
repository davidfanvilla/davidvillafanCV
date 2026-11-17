"use strict";

/* ============================================================
   Aparición de secciones al hacer scroll
   (el fondo 3D está en bg3d.js y la música en music.js)
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
