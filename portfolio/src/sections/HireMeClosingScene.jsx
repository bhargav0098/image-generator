import { PERSONAL } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function HireMeClosingScene() {
  return (
    <section id="contact" className="scene-closing-hire-me" aria-label="Scene 6: Contact & Hire Me">
      <div className="closing-ambient-vignette" />

      <div className="closing-content-stage">
        <ScrollReveal>
          <div className="closing-header-block text-center">
            <span className="closing-eyebrow-tag">06 / THE CONCLUSION</span>
            <h2 className="closing-headline">LET'S BUILD SOMETHING MEANINGFUL.</h2>
            <p className="closing-lead-invitation">
              Open to final-year internships and full-time software engineering, AI/ML, and Generative AI roles. Available to discuss immediate contributions to production teams.
            </p>
          </div>
        </ScrollReveal>

        {/* Converged Focal Hub */}
        <ScrollReveal delay={120}>
          <div className="converged-contact-hub">
            {/* Direct Email Action Button */}
            <div className="hub-primary-action">
              <a
                href={`mailto:${PERSONAL.email}`}
                className="btn btn-primary btn-large-glow"
              >
                <span>Email Directly: {PERSONAL.email}</span>
                <span className="btn-arrow">✉</span>
              </a>
            </div>

            {/* Verified Actions Grid */}
            <div className="hub-verified-links-grid">
              <a
                href={PERSONAL.github}
                target="_blank"
                rel="noreferrer"
                className="hub-channel-card"
              >
                <div className="channel-icon-wrap">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.5 0 12.3c0 5.43 3.44 10.03 8.2 11.65.6.12.82-.27.82-.6 0-.3-.01-1.08-.02-2.12-3.34.74-4.04-1.65-4.04-1.65-.55-1.43-1.34-1.82-1.34-1.82-1.1-.77.08-.76.08-.76 1.21.09 1.85 1.27 1.85 1.27 1.08 1.88 2.83 1.34 3.52 1.02.1-.79.42-1.34.76-1.65-2.66-.31-5.47-1.37-5.47-6.1 0-1.35.46-2.45 1.22-3.32-.12-.31-.53-1.56.12-3.25 0 0 1-.33 3.3 1.26a11.2 11.2 0 0 1 6 0c2.28-1.59 3.29-1.26 3.29-1.26.65 1.69.24 2.94.12 3.25.76.87 1.22 1.97 1.22 3.32 0 4.74-2.81 5.78-5.49 6.09.43.38.81 1.13.81 2.28 0 1.65-.02 2.98-.02 3.39 0 .33.22.72.83.6C20.56 22.32 24 17.72 24 12.3 24 5.5 18.63 0 12 0Z"/>
                  </svg>
                </div>
                <div className="channel-text">
                  <span className="channel-label">GITHUB</span>
                  <span className="channel-value">github.com/bhargav0098</span>
                </div>
                <span className="channel-arrow">↗</span>
              </a>

              <a
                href={PERSONAL.linkedin}
                target="_blank"
                rel="noreferrer"
                className="hub-channel-card"
              >
                <div className="channel-icon-wrap">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.61 0 4.28 2.38 4.28 5.47v6.27ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45Z"/>
                  </svg>
                </div>
                <div className="channel-text">
                  <span className="channel-label">LINKEDIN</span>
                  <span className="channel-value">Telugu Bhargav Ram</span>
                </div>
                <span className="channel-arrow">↗</span>
              </a>

              <a
                href={PERSONAL.resumePath}
                target="_blank"
                rel="noreferrer"
                className="hub-channel-card"
              >
                <div className="channel-icon-wrap">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                </div>
                <div className="channel-text">
                  <span className="channel-label">RESUME PDF</span>
                  <span className="channel-value">View Verified PDF</span>
                </div>
                <span className="channel-arrow">↗</span>
              </a>
            </div>
          </div>
        </ScrollReveal>
      </div>
    </section>
  );
}
