import { useState, useEffect, useRef } from "react";
import { SIX_PROJECT_PANELS } from "../data/portfolio";
import DetailModal from "../components/DetailModal";

export default function ProjectsSixDirections() {
  const containerRef = useRef(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeModalItem, setActiveModalItem] = useState(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const totalScrollDistance = rect.height - windowHeight;

      if (totalScrollDistance <= 0) return;

      const currentScroll = -rect.top;
      const progress = Math.max(0, Math.min(1, currentScroll / totalScrollDistance));
      setScrollProgress(progress);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Compute transform for each direction based on scrollProgress
  // We want the panels to converge into their final grid position as progress goes from 0 to 0.75,
  // and remain settled from 0.75 to 1.0.
  const getPanelTransform = (direction, idx) => {
    // Stagger start points slightly
    const startOffset = idx * 0.04;
    const effectiveProgress = Math.max(0, Math.min(1, (scrollProgress - startOffset) / 0.65));

    // Inverted progress (1 at start, 0 at settled)
    const inv = 1 - effectiveProgress;

    switch (direction) {
      case "from-left":
        return {
          transform: `translateX(${inv * -120}vw)`,
          opacity: Math.min(1, effectiveProgress * 2.5),
        };
      case "from-right":
        return {
          transform: `translateX(${inv * 120}vw)`,
          opacity: Math.min(1, effectiveProgress * 2.5),
        };
      case "from-top":
        return {
          transform: `translateY(${inv * -100}vh)`,
          opacity: Math.min(1, effectiveProgress * 2.5),
        };
      case "from-bottom":
        return {
          transform: `translateY(${inv * 100}vh)`,
          opacity: Math.min(1, effectiveProgress * 2.5),
        };
      case "from-foreground":
        return {
          transform: `perspective(1200px) translateZ(${inv * 600}px) scale(${1 + inv * 1.5})`,
          opacity: Math.min(1, effectiveProgress * 2),
        };
      case "from-background":
        return {
          transform: `perspective(1200px) translateZ(${inv * -600}px) scale(${1 - inv * 0.7})`,
          opacity: Math.min(1, effectiveProgress * 2),
        };
      default:
        return { transform: "none", opacity: 1 };
    }
  };

  return (
    <section
      id="projects"
      ref={containerRef}
      className="scene-six-directions-container"
      aria-label="Scene 4: Six-Direction Converging Projects Showcase"
    >
      <div className="sticky-six-stage">
        {/* Scene Head */}
        <div className="six-stage-header">
          <div className="header-meta-tag">
            <span className="section-roman-tag">04 / SELECTED WORK</span>
            <span className="convergence-indicator">
              {scrollProgress < 0.7 ? "CONVERGING FROM 6 VECTORS" : "SETTLED ARRANGEMENT"}
            </span>
          </div>
          <h2 className="six-stage-title">Projects Showcase</h2>
          <p className="six-stage-sub">
            Three core resume systems presented across six focused architectural and functional views, converging from distinct spatial vectors.
          </p>
        </div>

        {/* 6-Direction Converging Grid Viewport */}
        <div className="converging-grid-viewport">
          <div className="six-panels-grid">
            {SIX_PROJECT_PANELS.map((panel, idx) => {
              const motionStyle = getPanelTransform(panel.direction, idx);

              return (
                <div
                  key={panel.id}
                  className={`six-direction-panel ${panel.direction}`}
                  style={motionStyle}
                  onClick={() => setActiveModalItem(panel)}
                >
                  <div className="panel-surface-shell">
                    {/* Visual Media Header */}
                    <div className="panel-visual-area">
                      <img referrerPolicy="no-referrer" src={panel.image} alt={panel.title} loading="lazy" />
                      <div className="panel-top-badges">
                        <span className="vector-badge">{panel.directionLabel}</span>
                        <span className="view-badge">{panel.panelView}</span>
                      </div>
                    </div>

                    {/* Content Body */}
                    <div className="panel-content-body">
                      <h3 className="panel-project-name">{panel.title}</h3>
                      <p className="panel-tagline-text">{panel.tagline}</p>

                      <div className="panel-problem-built">
                        <p className="panel-problem">
                          <strong>Problem:</strong> {panel.problem}
                        </p>
                        <p className="panel-built">
                          <strong>Built:</strong> {panel.built}
                        </p>
                      </div>

                      <div className="panel-stack-pills">
                        {panel.techStack.map((tech) => (
                          <span key={tech} className="stack-pill">
                            {tech}
                          </span>
                        ))}
                      </div>

                      <div className="panel-action-bar">
                        {panel.live && (
                          <a
                            href={panel.live}
                            target="_blank"
                            rel="noreferrer"
                            className="panel-btn primary"
                            onClick={(e) => e.stopPropagation()}
                          >
                            Live Demo ↗
                          </a>
                        )}
                        {panel.github && (
                          <a
                            href={panel.github}
                            target="_blank"
                            rel="noreferrer"
                            className="panel-btn secondary"
                            onClick={(e) => e.stopPropagation()}
                          >
                            Code ↗
                          </a>
                        )}
                        <button
                          className="panel-btn tertiary"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActiveModalItem(panel);
                          }}
                        >
                          Details
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Scroll status progress bar */}
        <div className="six-stage-progress-track">
          <div
            className="six-stage-progress-fill"
            style={{ width: `${Math.round(scrollProgress * 100)}%` }}
          />
        </div>
      </div>

      {activeModalItem && (
        <DetailModal item={activeModalItem} onClose={() => setActiveModalItem(null)} />
      )}
    </section>
  );
}
