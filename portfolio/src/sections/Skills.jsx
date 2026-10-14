import { SKILL_METRICS, SKILLS_CATEGORIES } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

const CATEGORY_ICONS = {
  "Programming Languages": "{ }",
  "AI, ML & Generative AI": "◎",
  "Web Development": "▣",
  Databases: "▤",
  "Cloud & DevOps": "☁",
};

export default function Skills() {
  return (
    <section id="skills" className="section" aria-label="Technical Skills">
      <div className="section-inner">
        <ScrollReveal>
          <div className="section-head">
            <span className="section-label">05 / TECHNICAL EXPERTISE</span>
            <h2 className="section-title">Core Skills & Proficiency</h2>
            <p className="section-sub">
              Engineering capabilities across deep learning pipelines, generative intelligence, full-stack microservices, and cloud DevOps.
            </p>
          </div>
        </ScrollReveal>

        {/* Proficiency Metric Bars */}
        <div className="skills-metrics-row">
          {SKILL_METRICS.map((metric, i) => (
            <ScrollReveal key={metric.name} delay={i * 80}>
              <div className="skill-meter-box">
                <div className="meter-head">
                  <span className="meter-name">{metric.name}</span>
                  <span className="meter-percent">{metric.percent}%</span>
                </div>
                <div className="meter-track">
                  <div
                    className="meter-bar-fill"
                    style={{ width: `${metric.percent}%` }}
                  />
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>

        {/* Categorized Skill Pill Cards */}
        <div className="skills-grid mt-8">
          {Object.entries(SKILLS_CATEGORIES).map(([category, items], idx) => (
            <ScrollReveal key={category} delay={idx * 100}>
              <div className="skill-card">
                <div className="skill-card-head">
                  <span className="skill-icon" aria-hidden="true">
                    {CATEGORY_ICONS[category] || "•"}
                  </span>
                  <h3 className="skill-cat-title">{category}</h3>
                </div>

                <div className="skill-pills">
                  {items.map((item) => (
                    <span key={item} className="tag-pill">
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
