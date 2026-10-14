import { useState, useEffect, useRef } from "react";
import { PERSONAL } from "../data/portfolio";

export default function AboutPageFlip() {
  const containerRef = useRef(null);
  const [flipProgress, setFlipProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const totalScrollDistance = rect.height - windowHeight;

      if (totalScrollDistance <= 0) return;

      // When the top of container enters view, progress starts
      const currentScroll = -rect.top;
      const rawProgress = currentScroll / totalScrollDistance;
      const clamped = Math.max(0, Math.min(1, rawProgress));
      setFlipProgress(clamped);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Turning sheet angle (0deg at start -> -115deg fully open)
  const turnAngle = flipProgress * -115;
  const shadowOpacity = Math.min(1, flipProgress * 1.2);

  return (
    <section
      id="about"
      ref={containerRef}
      className="scene-page-flip-container"
      aria-label="About Scene with Page-Flip Transition"
    >
      <div className="sticky-book-stage">
        <div className="portfolio-book-viewport">
          {/* Base Inner Sheet (Revealed as the top page turns) */}
          <div className="book-page book-page-revealed">
            <div className="page-content-wrapper">
              <div className="page-header-row">
                <span className="section-roman-tag">01 / WHO I AM</span>
                <span className="page-number-indicator">FOLIO // 01</span>
              </div>

              <h2 className="revealed-section-title">About & Background</h2>
              <p className="revealed-lead-copy">{PERSONAL.about.lead}</p>

              <div className="revealed-two-column-layout">
                {/* Left Column: Education & Narrative */}
                <div className="revealed-col-primary">
                  <div className="editorial-narrative">
                    {PERSONAL.about.narrative.map((p, idx) => (
                      <p key={idx} className="narrative-paragraph">
                        {p}
                      </p>
                    ))}
                  </div>

                  {/* Verified Education Card */}
                  <div className="verified-education-card">
                    <div className="edu-card-icon">🎓</div>
                    <div className="edu-card-info">
                      <h3 className="edu-degree-title">{PERSONAL.about.education.degree}</h3>
                      <p className="edu-college-name">{PERSONAL.about.education.institution}</p>
                      <div className="edu-meta-row">
                        <span>{PERSONAL.about.education.location}</span>
                        <span className="meta-sep">•</span>
                        <span className="edu-tenure">{PERSONAL.about.education.period}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Column: Industry Internship */}
                <div id="experience" className="revealed-col-secondary">
                  <div className="verified-internship-card">
                    <div className="internship-badge-row">
                      <span className="internship-status-badge">VERIFIED INDUSTRY INTERNSHIP</span>
                      <span className="internship-period-badge">{PERSONAL.internship.period}</span>
                    </div>

                    <h3 className="internship-role-title">{PERSONAL.internship.role}</h3>
                    <p className="internship-org-name">{PERSONAL.internship.company}</p>
                    <p className="internship-duration-note">{PERSONAL.internship.duration}</p>

                    <ul className="internship-bullet-list">
                      {PERSONAL.internship.highlights.map((pt, i) => (
                        <li key={i}>{pt}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Turning Sheet (Flipping Page responding directly to scroll) */}
          <div
            className="book-page book-page-turning"
            style={{
              transform: `perspective(2000px) rotateY(${turnAngle}deg)`,
              transformOrigin: "left center",
              pointerEvents: flipProgress > 0.85 ? "none" : "auto",
            }}
          >
            {/* Front of Turning Sheet */}
            <div className="turning-page-front">
              <div className="cover-sheen-overlay" style={{ opacity: shadowOpacity * 0.4 }} />
              <div className="cover-crease-shadow" style={{ opacity: shadowOpacity }} />

              <div className="cover-inner-content">
                <span className="cover-folio-tag">DIGITAL EXHIBITION // 2026</span>
                <div className="cover-title-group">
                  <span className="cover-sub-label">CHAPTER 01</span>
                  <h3 className="cover-headline">Who I Am & Academic Foundations</h3>
                </div>
                <div className="cover-prompt-cue">
                  <span className="cue-glyph">↶</span>
                  <span>SCROLL DOWN TO FLIP PAGE</span>
                </div>
              </div>
            </div>

            {/* Back of Turning Sheet with shading */}
            <div
              className="turning-page-back"
              style={{
                opacity: flipProgress > 0.4 ? 1 : 0,
              }}
            >
              <div className="back-shadow-gradient" style={{ opacity: 1 - flipProgress * 0.5 }} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
