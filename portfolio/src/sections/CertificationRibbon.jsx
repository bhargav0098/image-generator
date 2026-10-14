import { useEffect, useRef, useState } from "react";
import { CERTIFICATES } from "../data/portfolio";
import CertificateModal from "../components/CertificateModal";

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const out3 = (t) => 1 - Math.pow(1 - t, 3);
const N = CERTIFICATES.length;

// Piece-wise linear keyframes: [[t, value], ...]
const keys = (k, t) => {
  for (let i = 1; i < k.length; i++) {
    if (t <= k[i][0]) {
      const [t0, v0] = k[i - 1], [t1, v1] = k[i];
      const u = (t - t0) / (t1 - t0);
      const e = u * u * (3 - 2 * u);
      return v0 + (v1 - v0) * e;
    }
  }
  return k[k.length - 1][1];
};
// word travel (vw): both drift to the middle, then swap sides behind the strip, as in the reference
const W1 = [[0, 0], [0.3, 24], [0.55, 50], [1, 74]];
const W2 = [[0, 0], [0.3, -24], [0.55, -50], [1, -74]];

// Grid entry vectors (vw, vh, rotZ) for 12 credentials
const DROPS = [
  [-45, -70, -8], [-10, -80, 6], [55, -75, 9], [75, -40, -7],
  [-70, 60, 8], [-5, 85, -6], [30, 90, 7], [80, 55, -9],
  [-50, -60, 5], [60, 70, -6], [-65, 40, -5], [70, -50, 7],
];

export default function CertificationRibbon() {
  const wrap = useRef(null);
  const strip = useRef([]);
  const t1 = useRef(null);
  const t2 = useRef(null);
  const intro = useRef(null);
  const grid = useRef([]);
  const bar = useRef(null);
  const cap = useRef(null);
  const [modal, setModal] = useState(null);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const update = () => {
      raf = 0;
      const el = wrap.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const p = reduce ? 1 : clamp(-r.top / (r.height - window.innerHeight));

      // geometry of the track (px, scaled to the viewport)
      const U = Math.max(0.4, Math.min(window.innerWidth / 1440, window.innerHeight / 900));
      const W = 520 * U, H = W / 1.42, R = 900 * U, Z0 = 450 * U, K = 147 * U, DX = 80 * U;
      const dphi = (W + 8 * U) / R;
      const PH0 = -2.6, PH1 = 2.9;
      const F = PH1 - PH0 + N * dphi + 0.3;

      // 1) words slide in from the edges
      const a = out3(clamp(p / 0.08));
      // 2) strip flows along the track (.08 → .78)
      const fp = clamp((p - 0.08) / 0.7);
      const f = fp * F;

      let nearest = 0, best = 9;
      strip.current.forEach((c, i) => {
        if (!c) return;
        const phi = PH0 - 0.15 + f - i * dphi;
        const vis = clamp((phi - (PH0 - 0.45)) / 0.4) * (1 - clamp((phi - PH1) / 0.35));
        c.style.width = `${W}px`;
        c.style.height = `${H}px`;
        c.style.marginLeft = `${-W / 2}px`;
        c.style.marginTop = `${-H / 2}px`;
        c.style.opacity = vis;
        c.style.visibility = vis > 0 ? "visible" : "hidden";
        const x = R * Math.sin(phi) + DX * phi;
        const y = -K * phi;
        const z = R * Math.cos(phi) - Z0;
        c.style.transform = `translate3d(${x}px, ${y}px, ${z}px) rotateY(${(phi * 180) / Math.PI + (Math.cos(phi) < 0 ? 180 : 0)}deg) rotateZ(-4deg)`;
        const d = Math.abs(phi - 0.2);
        if (vis > 0.9 && d < best) { best = d; nearest = i; }
      });
      const live = fp > 0.02 && fp < 0.98;
      cap.current.textContent = `${String(nearest + 1).padStart(2, "0")} / ${String(N).padStart(2, "0")} — ${CERTIFICATES[nearest].title}`;
      cap.current.style.opacity = live ? 1 : 0;

      // words: slide in, converge, swap sides, leave
      const fadeW = 1 - clamp((p - 0.72) / 0.08);
      t1.current.style.transform = `translateX(${(1 - a) * -70 + keys(W1, fp)}vw)`;
      t2.current.style.transform = `translateX(${(1 - a) * 70 + keys(W2, fp)}vw)`;
      t1.current.style.opacity = fadeW;
      t2.current.style.opacity = fadeW;
      intro.current.style.opacity = a * (1 - clamp((p - 0.55) / 0.1));

      // 3) certificates drop into the grid one by one (.78 → 1)
      grid.current.forEach((c, i) => {
        if (!c) return;
        const w = clamp((p - 0.77 - i * 0.011) / 0.09); // every card finishes by p≈0.98
        const e = out3(w);
        const [dx, dy, rz] = DROPS[i % DROPS.length];
        c.style.opacity = clamp(w * 5);
        if (w >= 1) { c.style.transform = "none"; return; }
        c.style.transform = `translate3d(${(1 - e) * dx}vw, ${(1 - e) * dy}vh, 0) rotate(${(1 - e) * rz}deg) scale(${0.8 + 0.2 * e})`;
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
  }, []);

  return (
    <section id="certifications" ref={wrap} className="cr-wrap" aria-label="Certifications">
      <div className="cr-stage">
        <div className="cr-top">
          <span>03 / CERTIFICATIONS &amp; ACHIEVEMENTS</span>
          <span>{N} VERIFIED CREDENTIALS</span>
        </div>

        <div className="cr-words" aria-hidden="true">
          <span ref={t1} className="cr-w1">CERTIFICATIONS &amp;</span>
          <span ref={t2} className="cr-w2">ACHIEVEMENTS</span>
        </div>

        <div ref={intro} className="cr-intro">
          <p className="cr-center">PROVEN MASTERY THROUGH<br />RIGOROUS CERTIFICATION</p>
          <p className="cr-left">Oracle, Google Cloud, AWS, Microsoft, SAP &amp; Kaggle credentials — verified and earned through practical application.</p>
          <a className="cr-right" href="https://www.linkedin.com/in/telugu-bhargav-ram-a2a33b2bb/" target="_blank" rel="noreferrer">SEE ON LINKEDIN ↗</a>
        </div>

        <div className="cr-persp" aria-hidden="true">
          <div className="cr-ring">
            {CERTIFICATES.map((c, i) => (
              <div key={c.id} ref={(n) => (strip.current[i] = n)} className="cr-rcard">
                <img referrerPolicy="no-referrer" src={c.image} alt="" />
              </div>
            ))}
          </div>
        </div>

        <p ref={cap} className="cr-active" />

        <div className="cr-grid">
          {CERTIFICATES.map((c, i) => (
            <button key={c.id} ref={(n) => (grid.current[i] = n)} className="cr-gcard" onClick={() => setModal(c)} aria-label={c.title}>
              <img referrerPolicy="no-referrer" src={c.image} alt={c.title} loading="lazy" />
              <span className="cr-cap"><b>{c.issuer}</b>{c.title}</span>
            </button>
          ))}
        </div>

        <div className="cr-foot"><div className="cr-track"><div ref={bar} /></div></div>
      </div>
      {modal && <CertificateModal cert={modal} onClose={() => setModal(null)} />}
    </section>
  );
}
