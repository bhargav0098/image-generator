import { PERSONAL } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function About() {
  const { aboutMe } = PERSONAL;

  return (
    <section id="about" className="section" aria-label="About Me">
      <div className="section-inner">
        <ScrollReveal>
          <div className="section-head">
            <span className="section-label">01 / WHO I AM</span>
            <h2 className="section-title">About Me</h2>
            <p className="section-sub">{aboutMe.headline}</p>
          </div>
        </ScrollReveal>

        <div className="about-grid">
          {/* Narrative Column */}
          <ScrollReveal delay={100}>
            <div className="about-narrative">
              {aboutMe.paragraphs.map((p, idx) => (
                <p key={idx} className="about-p">
                  {p}
                </p>
              ))}

              {/* Education Card */}
              <div className="about-edu-card">
                <div className="edu-icon-wrap">🎓</div>
                <div className="edu-content">
                  <h4 className="edu-title">{aboutMe.education.degree}</h4>
                  <p className="edu-school">{aboutMe.education.institution}</p>
                  <span className="edu-period">{aboutMe.education.period}</span>
                </div>
              </div>
            </div>
          </ScrollReveal>

          {/* Stats & Highlights Column */}
          <ScrollReveal delay={200}>
            <div className="about-stats-matrix">
              {aboutMe.stats.map((stat, i) => (
                <div key={i} className="about-stat-box">
                  <span className="stat-big-num">{stat.num}</span>
                  <span className="stat-sub-label">{stat.label}</span>
                </div>
              ))}
            </div>
          </ScrollReveal>
        </div>
      </div>
    </section>
  );
}
