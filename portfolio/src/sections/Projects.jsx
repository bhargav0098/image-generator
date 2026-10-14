import { useState } from "react";
import { PROJECTS } from "../data/portfolio";
import ScrollReveal from "../components/ScrollReveal";
import ProjectModal from "../components/ProjectModal";

export default function Projects() {
  const [filter, setFilter] = useState("all");
  const [activeModalProject, setActiveModalProject] = useState(null);

  const filteredProjects = PROJECTS.filter((p) => {
    if (filter === "all") return true;
    if (filter === "ai") return p.category.toLowerCase().includes("ai") || p.category.toLowerCase().includes("ml");
    if (filter === "genai") return p.techStack.some((t) => t.toLowerCase().includes("gemini") || t.toLowerCase().includes("llm") || t.toLowerCase().includes("prompt"));
    if (filter === "cloud") return p.techStack.some((t) => t.toLowerCase().includes("azure") || t.toLowerCase().includes("aws") || t.toLowerCase().includes("prometheus") || t.toLowerCase().includes("redis"));
    return true;
  });

  return (
    <section id="projects" className="section projects-showcase-section" aria-label="Projects">
      <div className="section-inner">
        <ScrollReveal>
          <div className="section-head">
            <div className="section-head-flex">
              <div>
                <span className="section-label">03 / SELECTED WORK</span>
                <h2 className="section-title">Projects Showcase</h2>
                <p className="section-sub">
                  End-to-end production AI applications, multi-agent frameworks, and cloud-native systems.
                </p>
              </div>

              {/* Category Filter Tabs */}
              <div className="project-filter-tabs">
                <button
                  type="button"
                  className={`filter-btn ${filter === "all" ? "active" : ""}`}
                  onClick={() => setFilter("all")}
                >
                  All ({PROJECTS.length})
                </button>
                <button
                  type="button"
                  className={`filter-btn ${filter === "ai" ? "active" : ""}`}
                  onClick={() => setFilter("ai")}
                >
                  AI/ML
                </button>
                <button
                  type="button"
                  className={`filter-btn ${filter === "genai" ? "active" : ""}`}
                  onClick={() => setFilter("genai")}
                >
                  Generative AI
                </button>
                <button
                  type="button"
                  className={`filter-btn ${filter === "cloud" ? "active" : ""}`}
                  onClick={() => setFilter("cloud")}
                >
                  Cloud & Systems
                </button>
              </div>
            </div>
          </div>
        </ScrollReveal>

        {/* 7 to 10 Sec Video Scroll Showcase Grid */}
        <div className="projects-scroll-grid">
          {filteredProjects.map((proj, idx) => (
            <ScrollReveal key={proj.id} delay={idx * 100}>
              <article
                className="project-scroll-card"
                onClick={() => setActiveModalProject(proj)}
              >
                <div className="scroll-card-media">
                  <img
                    src={proj.image}
                    alt={proj.name}
                    loading="lazy"
                    className="scroll-card-img"
                  />
                  <div className="scroll-card-badges">
                    <span className="card-cat-badge">{proj.category}</span>
                    <span className="card-yr-badge">{proj.year}</span>
                  </div>
                </div>

                <div className="scroll-card-body">
                  <div className="scroll-card-header">
                    <h3 className="scroll-card-title">{proj.name}</h3>
                  </div>

                  <p className="scroll-card-tagline">{proj.tagline}</p>
                  <p className="scroll-card-desc">{proj.description}</p>

                  <div className="scroll-card-tags">
                    {proj.techStack.map((tech) => (
                      <span key={tech} className="tag-pill">
                        {tech}
                      </span>
                    ))}
                  </div>

                  <div className="scroll-card-actions">
                    {proj.live && (
                      <a
                        href={proj.live}
                        target="_blank"
                        rel="noreferrer"
                        className="proj-btn proj-btn-primary"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Live Demo ↗
                      </a>
                    )}
                    {proj.github && (
                      <a
                        href={proj.github}
                        target="_blank"
                        rel="noreferrer"
                        className="proj-btn"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Source Code ↗
                      </a>
                    )}
                  </div>
                </div>
              </article>
            </ScrollReveal>
          ))}
        </div>
      </div>

      {activeModalProject && (
        <ProjectModal
          card={{
            title: activeModalProject.name,
            subtitle: activeModalProject.tagline,
            category: activeModalProject.category,
            year: activeModalProject.year,
            image: activeModalProject.image,
            description: activeModalProject.description,
            tags: activeModalProject.techStack,
            live: activeModalProject.live,
            github: activeModalProject.github,
          }}
          onClose={() => setActiveModalProject(null)}
        />
      )}
    </section>
  );
}
