import { useEffect, useRef, useState } from "react";
import { SIX_PROJECT_PANELS } from "../data/portfolio";
import DetailModal from "../components/DetailModal";

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = (t) => 1 - Math.pow(1 - t, 3);
const easeBack = (t) => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2);

// Six different entry vectors (x vw, y vh, z px) — one per card
const STARTS = [
  [-95, -10, 0],
  [95, 12, 0],
  [10, -90, 0],
  [-12, 90, 0],
  [0, 0, 900],
  [0, 0, -1400],
];

export default function ProjectsRibbon() {
  const wrap = useRef(null);
  const cards = useRef([]);
  const textL = useRef(null);
  const textR = useRef(null);
  const bar = useRef(null);
  const [modal, setModal] = useState(null);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const update = () => {
      raf = 0;
      const el = wrap.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const total = r.height - window.innerHeight;
      const p = reduce ? 1 : clamp(-r.top / total);

      cards.current.forEach((c, i) => {
        if (!c) return;
        const w = clamp((p - 0.04 - i * 0.07) / 0.5);
        const e = ease(w);
        const [sx, sy, sz] = STARTS[i];
        const arc = Math.sin(w * Math.PI) * (i % 2 ? 16 : -16); // ribbon curve (vh)
        const twist = (1 - e) * (i % 2 ? 55 : -55);
        const s = 0.55 + 0.45 * easeBack(w);
        c.style.opacity = clamp(w * 4);
        c.style.transform = `perspective(1400px) translate3d(${(1 - e) * sx}vw, ${
          (1 - e) * sy + arc
        }vh, ${(1 - e) * sz}px) rotateY(${twist}deg) rotateX(${(1 - e) * 85}deg) rotateZ(${
          (1 - e) * (i % 2 ? 8 : -8)
        }deg) scale(${s})`;
      });

      const t = ease(clamp(p / 0.55));
      const fade = 1 - clamp((p - 0.6) / 0.25);
      if (textL.current) {
        textL.current.style.transform = `translateX(${(1 - t) * -70}vw)`;
        textL.current.style.opacity = fade;
      }
      if (textR.current) {
        textR.current.style.transform = `translateX(${(1 - t) * 70}vw)`;
        textR.current.style.opacity = fade;
      }
      if (bar.current) bar.current.style.width = `${p * 100}%`;
    };
    const onScroll = () => !raf && (raf = requestAnimationFrame(update));
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <section id="projects" ref={wrap} className="pr-wrap" aria-label="Projects">
      <div className="pr-stage">
        <div className="pr-top">
          <span className="pr-tag">04 / SELECTED WORK</span>
          <span className="pr-tag">6 CORE SYSTEMS · 6 SPATIAL VECTORS</span>
        </div>

        <div className="pr-grid">
          {SIX_PROJECT_PANELS.map((p, i) => (
            <button
              key={p.id}
              ref={(n) => (cards.current[i] = n)}
              className="pr-card"
              onClick={() => setModal(p)}
              aria-label={`Open ${p.title}`}
            >
              <img referrerPolicy="no-referrer" src={p.image} alt={p.title} loading="lazy" />
              <div className="pr-info">
                <span className="pr-view">{p.panelView}</span>
                <strong>{p.name || p.title.split(" — ")[0]}</strong>
                <em>{p.tagline}</em>
              </div>
              <span className="pr-cta">View case study ↗</span>
            </button>
          ))}
        </div>

        <div className="pr-foot">
          <p>Scroll — six systems converge from six distinct spatial vectors.</p>
          <div className="pr-track"><div ref={bar} /></div>
        </div>
      </div>
      {modal && <DetailModal item={modal} onClose={() => setModal(null)} />}
    </section>
  );
}
