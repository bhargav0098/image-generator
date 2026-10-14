import { useState, useEffect } from "react";
import { PERSONAL } from "../data/portfolio";
import bgImage from "../assets/background.jpg";
import RoleTyper from "../components/RoleTyper";
import CountUp from "../components/CountUp";

export default function HeroCinematic() {
  const [mouseOffset, setMouseOffset] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 20;
      const y = (e.clientY / window.innerHeight - 0.5) * 20;
      setMouseOffset({ x, y });
    };
    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  const scrollToProjects = (e) => {
    e.preventDefault();
    const elem = document.querySelector("#projects");
    if (elem) {
      elem.scrollIntoView({ behavior: "smooth" });
      if (window.history && window.history.pushState) {
        window.history.pushState(null, "", "#projects");
      }
    }
  };

  return (
    <section id="hero" className="scene-hero-cinematic" aria-label="Landing Scene">
      {/* Dark Cinematic Reflective Environment Layer */}
      <div className="hero-dark-environment">
        <div
          className="environment-floor-layer"
          style={{
            backgroundImage: `url(${bgImage})`,
            transform: `perspective(1000px) rotateX(12deg) translateY(${mouseOffset.y * 0.4}px) scale(1.05)`,
          }}
        />
        <div className="environment-silver-vignette" />
        <div className="environment-subtle-glow" />
      </div>

      <div className="hero-content-wrapper">
        {/* Subtle Status Label */}
        <div className="hero-status-tag hx" style={{ "--d": "0s" }}>
          <span className="status-live-beacon"></span>
          <span className="status-copy">{PERSONAL.statusLabel}</span>
        </div>

        {/* Hero Headline */}
        <div className="hero-headline-block">
          <p className="hero-pre-title hx" style={{ "--d": ".1s" }}>Hi, I'm</p>
          <h1 className="hero-candidate-name" aria-label={PERSONAL.name}>
            {PERSONAL.name.split(" ").map((word, wi, arr) => (
              <span key={wi} className="hero-word" aria-hidden="true">
                {word.split("").map((ch, ci) => (
                  <span key={ci} className="hero-char" style={{ "--i": wi * 4 + ci }}>{ch}</span>
                ))}
                {wi < arr.length - 1 ? "\u00A0" : ""}
              </span>
            ))}
          </h1>
          <div className="hero-animated-role hx" style={{ "--d": ".75s" }}>
            <span className="role-prefix">// </span>
            <RoleTyper roles={PERSONAL.roles} />
          </div>
        </div>

        {/* Professional Introduction */}
        <p className="hero-professional-intro hx" style={{ "--d": ".9s" }}>
          {PERSONAL.bioIntroduction}
        </p>

        {/* Technology Highlights */}
        <div className="hero-tech-highlights-row hx" style={{ "--d": "1.05s" }} aria-label="Key Technologies">
          {PERSONAL.techHighlights.map((tech) => (
            <span key={tech} className="understated-pill">
              {tech}
            </span>
          ))}
        </div>

        {/* Hero Actions */}
        <div className="hero-action-buttons hx" style={{ "--d": "1.2s" }}>
          <a
            href="#projects"
            className="action-btn action-primary"
            onClick={scrollToProjects}
          >
            <span>View Projects</span>
            <span className="action-arrow">↓</span>
          </a>

          <a
            href={PERSONAL.resumePath}
            target="_blank"
            rel="noreferrer"
            className="action-btn action-secondary"
          >
            <span>View Resume (PDF)</span>
            <span className="action-arrow">↗</span>
          </a>

          <div className="hero-verified-socials">
            <a
              href={PERSONAL.github}
              target="_blank"
              rel="noreferrer"
              className="social-icon-btn"
              aria-label="GitHub Profile"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.5 0 12.3c0 5.43 3.44 10.03 8.2 11.65.6.12.82-.27.82-.6 0-.3-.01-1.08-.02-2.12-3.34.74-4.04-1.65-4.04-1.65-.55-1.43-1.34-1.82-1.34-1.82-1.1-.77.08-.76.08-.76 1.21.09 1.85 1.27 1.85 1.27 1.08 1.88 2.83 1.34 3.52 1.02.1-.79.42-1.34.76-1.65-2.66-.31-5.47-1.37-5.47-6.1 0-1.35.46-2.45 1.22-3.32-.12-.31-.53-1.56.12-3.25 0 0 1-.33 3.3 1.26a11.2 11.2 0 0 1 6 0c2.28-1.59 3.29-1.26 3.29-1.26.65 1.69.24 2.94.12 3.25.76.87 1.22 1.97 1.22 3.32 0 4.74-2.81 5.78-5.49 6.09.43.38.81 1.13.81 2.28 0 1.65-.02 2.98-.02 3.39 0 .33.22.72.83.6C20.56 22.32 24 17.72 24 12.3 24 5.5 18.63 0 12 0Z"/>
              </svg>
            </a>

            <a
              href={PERSONAL.linkedin}
              target="_blank"
              rel="noreferrer"
              className="social-icon-btn"
              aria-label="LinkedIn Profile"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.61 0 4.28 2.38 4.28 5.47v6.27ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45Z"/>
              </svg>
            </a>
          </div>
        </div>

        {/* Verified Metrics (No fake 3 internships or fabricated percentages) */}
        <div className="hero-verified-metrics hx" style={{ "--d": "1.35s" }}>
          {PERSONAL.metrics.map((m) => (
            <div key={m.label} className="metric-box">
              <span className="metric-digit"><CountUp value={m.value} /></span>
              <div className="metric-text-group">
                <span className="metric-main-label">{m.label}</span>
                <span className="metric-sub-detail">{m.detail}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Scroll indicator prompt */}
      <div className="hero-scroll-cue" aria-hidden="true">
        <span className="scroll-hint-label">SCROLL TO TURN PAGE</span>
        <div className="scroll-chevron-line">
          <div className="scroll-chevron-dot" />
        </div>
      </div>
    </section>
  );
}
