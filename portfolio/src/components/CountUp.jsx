import { useEffect, useState } from "react";

/** Counts from 0 to the numeric value once the intro finishes. Keeps leading zeros ("06"). */
export default function CountUp({ value, duration = 1200 }) {
  const target = parseInt(value, 10) || 0;
  const width = String(value).length;
  const [n, setN] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const run = () => {
      if (reduce) { setN(target); return; }
      const t0 = performance.now();
      const step = (now) => {
        const p = Math.min(1, (now - t0) / duration);
        setN(Math.round((1 - Math.pow(1 - p, 3)) * target));
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    if (document.documentElement.classList.contains("intro-done")) {
      run();
    } else {
      window.addEventListener("intro:done", run, { once: true });
    }
    return () => { cancelAnimationFrame(raf); window.removeEventListener("intro:done", run); };
  }, [target, duration]);

  return <>{String(n).padStart(width, "0")}</>;
}
