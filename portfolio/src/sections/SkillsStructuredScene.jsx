import { useState } from "react";
import { SKILLS_DATA } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";

export default function SkillsStructuredScene() {
  const [activeLayer, setActiveLayer] = useState(null);

  return (
    <section id="skills" className="scene-skills-container" aria-label="Scene 5: Technical Skills">
      <div className="skills-inner-content">
        <ScrollReveal>
          <div className="skills-scene-header">
            <span className="section-roman-tag">05 / TECHNICAL TAXONOMY</span>
            <h2 className="skills-scene-title">Engineering Stack & Disciplines</h2>
            <p className="skills-scene-sub">
              Verified technical disciplines directly from active development and project deployments, structured across core architectural layers.
            </p>
          </div>
        </ScrollReveal>

        {/* Structured Visual Layers with Subtle Depth */}
        <div className="skills-layered-matrix">
          {SKILLS_DATA.map((layer, idx) => {
            const isHovered = activeLayer === idx;

            return (
              <ScrollReveal key={layer.category} delay={idx * 100}>
                <div
                  className={`skill-layer-card ${isHovered ? "layer-focused" : ""}`}
                  onMouseEnter={() => setActiveLayer(idx)}
                  onMouseLeave={() => setActiveLayer(null)}
                >
                  <div className="layer-header-bar">
                    <span className="layer-icon" aria-hidden="true">{layer.icon}</span>
                    <div className="layer-text-wrap">
                      <span className="layer-stratum-label">LAYER 0{idx + 1}</span>
                      <h3 className="layer-category-name">{layer.category}</h3>
                    </div>
                    <span className="layer-skill-count">{layer.skills.length} Technologies</span>
                  </div>

                  <div className="layer-pills-cluster">
                    {layer.skills.map((skill) => (
                      <span key={skill} className="structured-skill-pill">
                        <span className="pill-dot">•</span>
                        <span>{skill}</span>
                      </span>
                    ))}
                  </div>
                </div>
              </ScrollReveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
