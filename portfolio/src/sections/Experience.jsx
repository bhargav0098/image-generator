import { EXPERIENCE } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function Experience() {
  return (
    <section id="experience" className="section" aria-label="Experience">
      <div className="section-inner">
        <ScrollReveal>
          <div className="section-head">
            <span className="section-label">02 / CAREER PATH</span>
            <h2 className="section-title">Experience & Internships</h2>
            <p className="section-sub">
              Hands-on engineering across machine learning models, production web architectures, and cloud deployments.
            </p>
          </div>
        </ScrollReveal>

        <div className="timeline">
          {EXPERIENCE.map((exp, idx) => (
            <ScrollReveal key={idx} delay={idx * 120}>
              <div className="timeline-item">
                <div className="timeline-dot" />
                <div className="timeline-card">
                  <div className="timeline-header">
                    <div>
                      <h3 className="timeline-role">{exp.role}</h3>
                      <p className="timeline-company">{exp.company}</p>
                    </div>
                    <div className="timeline-meta-badges">
                      <span className="timeline-badge">{exp.duration}</span>
                      <span className="timeline-period-badge">{exp.period}</span>
                    </div>
                  </div>

                  <ul className="timeline-points">
                    {exp.points.map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
