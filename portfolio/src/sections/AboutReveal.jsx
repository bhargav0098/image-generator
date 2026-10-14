import { useEffect, useRef } from "react";
import { PERSONAL } from "../data/portfolio";

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const out3 = (t) => 1 - Math.pow(1 - t, 3);
const io = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

// Scroll-scrubbed: giant words slide in from the edges, shrink to the top,
// the lead text "inks in" word by word, then two cards rise in with a 3D tilt.
export default function AboutReveal() {
  const wrap = useRef(null);
  const w1 = useRef(null);
  const w2 = useRef(null);
  const words = useRef([]);
  const cards = useRef([]);
  const bar = useRef(null);
  const lead = PERSONAL.about.lead.split(" ");
  const { education: edu } = PERSONAL.about;
  const job = PERSONAL.internship;

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const update = () => {
      raf = 0;
      const el = wrap.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const p = reduce ? 1 : clamp(-r.top / (r.height - window.innerHeight));

      const a = out3(clamp(p / 0.2)); // slide in
      const s = io(clamp((p - 0.22) / 0.2)); // move up + shrink
      const sc = 1 - s * 0.55;
      const ty = -s * 30;
      w1.current.style.transform = `translate(${(1 - a) * -80}vw, ${ty}vh) scale(${sc})`;
      w2.current.style.transform = `translate(${(1 - a) * 80}vw, ${ty}vh) scale(${sc})`;

      words.current.forEach((w, i) => {
        if (!w) return;
        const k = clamp((p - 0.3 - (i / lead.length) * 0.3) / 0.05);
        w.style.opacity = (0.16 + 0.84 * k) * s;
      });

      cards.current.forEach((c, i) => {
        if (!c) return;
        const k = out3(clamp((p - 0.68 - i * 0.08) / 0.2));
        c.style.opacity = clamp(k * 3);
        c.style.transform = `perspective(1200px) translateY(${(1 - k) * 60}vh) rotateX(${(1 - k) * 55}deg) rotateZ(${(1 - k) * (i ? 5 : -5)}deg)`;
      });
      bar.current.style.width = `${p * 100}%`;
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
  }, [lead.length]);

  return (
    <section id="about" ref={wrap} className="ab-wrap" aria-label="About">
      <div className="ab-stage">
        <div className="ab-top"><span>01 / ABOUT</span><span>KURNOOL, INDIA</span></div>

        <div className="ab-title" aria-hidden="true">
          <span ref={w1}>WHO I</span>
          <span ref={w2}>AM.</span>
        </div>

        <p className="ab-lead">
          {lead.map((w, i) => (
            <span key={i} ref={(n) => (words.current[i] = n)}>{w} </span>
          ))}
        </p>

        <div className="ab-cards">
          <article ref={(n) => (cards.current[0] = n)} className="ab-card">
            <span className="ab-chip">EDUCATION</span>
            <h3>{edu.degree}</h3>
            <p>{edu.institution}</p>
            <small>{edu.location} · {edu.period}</small>
          </article>
          <article ref={(n) => (cards.current[1] = n)} className="ab-card">
            <span className="ab-chip">INTERNSHIP · {job.period}</span>
            <h3>{job.role}</h3>
            <p>{job.company}</p>
            <small>{job.duration}</small>
            <ul>{job.highlights.slice(0, 2).map((h) => <li key={h}>{h}</li>)}</ul>
          </article>
        </div>

        <div className="ab-foot"><div className="ab-track"><div ref={bar} /></div></div>
      </div>
    </section>
  );
}
