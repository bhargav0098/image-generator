import { EDUCATION, CERTIFICATIONS, ACHIEVEMENTS } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function Education() {
  return (
    <section id="education" className="section" aria-label="Education, Certifications, and Achievements">
      <div className="section-inner">
        {/* Education */}
        <ScrollReveal>
          <div className="section-head">
            <span className="section-label">EDUCATION</span>
            <h2 className="section-title">Education</h2>
          </div>
        </ScrollReveal>

        <ScrollReveal>
          <div className="education-card">
            <div className="edu-icon" aria-hidden="true">🎓</div>
            <div>
              <div className="edu-title">{EDUCATION.degree}</div>
              <div className="edu-sub">
                {EDUCATION.institution}, {EDUCATION.location}
              </div>
              <div className="edu-period">{EDUCATION.period}</div>
            </div>
          </div>
        </ScrollReveal>

        {/* Certifications */}
        <div style={{ marginTop: "72px" }}>
          <ScrollReveal>
            <div className="section-head">
              <span className="section-label">CERTIFICATIONS</span>
              <h3 className="section-title" style={{ fontSize: "clamp(24px, 3vw, 34px)" }}>
                Certifications
              </h3>
            </div>
          </ScrollReveal>

          <div className="certs-grid">
            {CERTIFICATIONS.map((cert, idx) => (
              <ScrollReveal key={cert} delay={idx * 60}>
                <div className="cert-item">
                  <span className="cert-icon" aria-hidden="true">✦</span>
                  <span>{cert}</span>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>

        {/* Achievements */}
        <div style={{ marginTop: "72px" }}>
          <ScrollReveal>
            <div className="section-head">
              <span className="section-label">ACHIEVEMENTS</span>
              <h3 className="section-title" style={{ fontSize: "clamp(24px, 3vw, 34px)" }}>
                Achievements
              </h3>
            </div>
          </ScrollReveal>

          <div className="achievements-list">
            {ACHIEVEMENTS.map((ach, idx) => (
              <ScrollReveal key={idx} delay={idx * 80}>
                <div className="achievement-item">
                  <span className="achievement-icon" aria-hidden="true">◆</span>
                  <div>
                    <div className="achievement-title">{ach.title}</div>
                    <div className="achievement-desc">{ach.description}</div>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
