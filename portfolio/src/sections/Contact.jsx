import { PERSONAL } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function Contact() {
  return (
    <section id="contact" className="section contact-section" aria-label="Contact">
      <div className="section-inner">
        <div className="contact-content text-center">
          <ScrollReveal>
            <div className="section-head text-center">
              <span className="section-label">06 / GET IN TOUCH</span>
              <h2 className="section-title">Let's Build Something High-Impact</h2>
              <p className="section-sub mx-auto">
                Open to internships and full-time software engineering, AI/ML, and Generative AI roles. Let's talk.
              </p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={100}>
            <div className="contact-cta-panel">
              <a href={`mailto:${PERSONAL.email}`} className="btn btn-primary contact-main-btn">
                <span>Hire Me & Email Directly</span>
                <span className="btn-arrow">✉</span>
              </a>

              <a
                href={PERSONAL.resumePath}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-ghost"
              >
                <span>Download Resume (PDF)</span>
                <span className="btn-arrow">↓</span>
              </a>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={150}>
            <div className="contact-links-grid">
              <a
                href={`mailto:${PERSONAL.email}`}
                className="contact-card-link"
              >
                <span className="contact-icon">✉</span>
                <div className="contact-card-text">
                  <span className="contact-label">Email Directly</span>
                  <span className="contact-val">{PERSONAL.email}</span>
                </div>
              </a>

              <a
                href={PERSONAL.github}
                target="_blank"
                rel="noopener noreferrer"
                className="contact-card-link"
              >
                <span className="contact-icon">⌥</span>
                <div className="contact-card-text">
                  <span className="contact-label">GitHub Profile</span>
                  <span className="contact-val">github.com/bhargav0098</span>
                </div>
              </a>

              <a
                href={PERSONAL.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="contact-card-link"
              >
                <span className="contact-icon">in</span>
                <div className="contact-card-text">
                  <span className="contact-label">LinkedIn Connect</span>
                  <span className="contact-val">Telugu Bhargav Ram</span>
                </div>
              </a>
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
