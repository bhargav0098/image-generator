import { useEffect, useState } from "react";
import { PERSONAL } from "../data/portfolio";

const DURATION = 1700; // ms for the counter to reach 100

/**
 * Opening curtain: counter + name, then the panel lifts away.
 * Adds `intro-done` to <html> when finished so the hero can play its entrance.
 */
export default function IntroLoader() {
  const [count, setCount] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const finish = () => {
      root.classList.add("intro-done");
      window.dispatchEvent(new Event("intro:done"));
    };
    if (reduce) {
      finish();
      setGone(true);
      return;
    }

    document.body.style.overflow = "hidden";
    let raf = 0;
    let t0 = 0;
    let timer = 0;
    const step = (now) => {
      if (!t0) t0 = now;
      const p = Math.min(1, (now - t0) / DURATION);
      const eased = 1 - Math.pow(1 - p, 2.2);
      setCount(Math.round(eased * 100));
      if (p < 1) {
        raf = requestAnimationFrame(step);
      } else {
        setLeaving(true);
        timer = window.setTimeout(() => {
          document.body.style.overflow = "";
          finish();
          window.setTimeout(() => setGone(true), 900);
        }, 350);
      }
    };
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(timer);
      document.body.style.overflow = "";
    };
  }, []);

  if (gone) return null;

  return (
    <div className={`intro-loader${leaving ? " is-leaving" : ""}`} role="status" aria-label="Loading portfolio">
      <div className="intro-center">
        <span className="intro-brand">{PERSONAL.shortBrand}</span>
        <span className="intro-name">{PERSONAL.name}</span>
        <div className="intro-bar"><i style={{ transform: `scaleX(${count / 100})` }} /></div>
      </div>
      <span className="intro-count">{String(count).padStart(3, "0")}</span>
      <span className="intro-hint">AI ENGINEER · FULL-STACK DEVELOPER</span>
    </div>
  );
}
