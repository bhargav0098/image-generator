import { useRef, useEffect } from "react";
import bgImage from "../assets/background.jpg";

/**
 * HeroBackground
 * Renders the dark cinematic background image with an animated
 * canvas overlay of soft, slowly-moving light reflections.
 */
export default function HeroBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const prefersReduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let raf;
    let w, h, dpr;

    function resize() {
      const rect = canvas.parentElement.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = rect.width;
      h = rect.height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener("resize", resize);

    // Light reflection parameters
    const lights = [
      { x: 0.25, y: 0.72, rx: 140, ry: 30, speed: 0.08, phase: 0, opacity: 0.12 },
      { x: 0.45, y: 0.75, rx: 180, ry: 35, speed: 0.05, phase: 1.2, opacity: 0.10 },
      { x: 0.65, y: 0.70, rx: 120, ry: 25, speed: 0.07, phase: 2.5, opacity: 0.14 },
      { x: 0.80, y: 0.74, rx: 100, ry: 20, speed: 0.09, phase: 3.8, opacity: 0.08 },
      { x: 0.35, y: 0.78, rx: 200, ry: 40, speed: 0.03, phase: 5.0, opacity: 0.06 },
      { x: 0.55, y: 0.68, rx: 90, ry: 18, speed: 0.06, phase: 0.7, opacity: 0.10 },
    ];

    let t = 0;

    function draw() {
      ctx.clearRect(0, 0, w, h);

      for (const light of lights) {
        const cx = (light.x + Math.sin(t * light.speed + light.phase) * 0.08) * w;
        const cy = light.y * h + Math.cos(t * light.speed * 0.7 + light.phase) * 8;
        const scaleX = (w / 1200);
        const rx = light.rx * scaleX;
        const ry = light.ry * scaleX;
        const currentOpacity =
          light.opacity * (0.7 + 0.3 * Math.sin(t * light.speed * 1.5 + light.phase));

        ctx.save();
        ctx.globalAlpha = currentOpacity;
        ctx.beginPath();
        ctx.ellipse(cx, cy, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2);

        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, 1));
        grad.addColorStop(0, "rgba(210, 220, 235, 0.9)");
        grad.addColorStop(0.4, "rgba(190, 205, 225, 0.4)");
        grad.addColorStop(1, "rgba(160, 180, 210, 0)");

        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
      }

      t += 0.016; // ~60fps time step
      raf = requestAnimationFrame(draw);
    }

    if (prefersReduced) {
      // Draw one static frame
      t = 0;
      ctx.clearRect(0, 0, w, h);
      for (const light of lights) {
        const cx = light.x * w;
        const cy = light.y * h;
        const scaleX = w / 1200;
        const rx = light.rx * scaleX;
        const ry = light.ry * scaleX;

        ctx.save();
        ctx.globalAlpha = light.opacity * 0.7;
        ctx.beginPath();
        ctx.ellipse(cx, cy, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, 1));
        grad.addColorStop(0, "rgba(210, 220, 235, 0.9)");
        grad.addColorStop(0.4, "rgba(190, 205, 225, 0.4)");
        grad.addColorStop(1, "rgba(160, 180, 210, 0)");
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
      }
    } else {
      draw();
    }

    // Pause when not visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting && raf) {
            cancelAnimationFrame(raf);
            raf = null;
          } else if (entry.isIntersecting && !raf && !prefersReduced) {
            draw();
          }
        });
      },
      { threshold: 0 }
    );
    observer.observe(canvas.parentElement);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      observer.disconnect();
    };
  }, []);

  return (
    <div className="hero-bg" aria-hidden="true">
      <div
        className="hero-bg-image"
        style={{ backgroundImage: `url(${bgImage})` }}
      />
      <canvas ref={canvasRef} />
      <div className="hero-vignette" />
    </div>
  );
}
