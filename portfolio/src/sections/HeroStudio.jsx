import { useState, useRef, useEffect, useCallback } from "react";
import { PROJECTS, PERSONAL } from "../data/portfolio";
import DetailModal from "../components/DetailModal";

export default function HeroStudio() {
  const [viewMode, setViewMode] = useState("ribbon"); // 'ribbon' | 'grid'
  const [rotation, setRotation] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [hoveredCardId, setHoveredCardId] = useState(null);

  const isDraggingRef = useRef(false);
  const lastXRef = useRef(0);
  const velocityRef = useRef(0);
  const currentRotationRef = useRef(0);
  const animFrameIdRef = useRef(null);
  const dragDistanceRef = useRef(0);

  const cards = PROJECTS;
  const count = cards.length;
  const radius = 640;

  // 3D Ribbon Physics Loop (Inertia + Continuous Drift)
  useEffect(() => {
    let lastTime = performance.now();

    const loop = (now) => {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      if (!isDraggingRef.current && viewMode === "ribbon") {
        velocityRef.current *= Math.pow(0.92, dt * 60);

        if (Math.abs(velocityRef.current) < 0.04) {
          velocityRef.current = -0.15; // smooth auto-drift
        }

        currentRotationRef.current += velocityRef.current;
        setRotation(currentRotationRef.current);
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, [viewMode]);

  const handlePointerDown = (e) => {
    if (viewMode !== "ribbon") return;
    isDraggingRef.current = true;
    setIsDragging(true);
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    lastXRef.current = clientX;
    velocityRef.current = 0;
    dragDistanceRef.current = 0;
  };

  const handlePointerMove = useCallback(
    (e) => {
      if (!isDraggingRef.current || viewMode !== "ribbon") return;
      const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
      const deltaX = clientX - lastXRef.current;
      lastXRef.current = clientX;
      dragDistanceRef.current += Math.abs(deltaX);

      const degDelta = deltaX * 0.22;
      currentRotationRef.current += degDelta;
      velocityRef.current = degDelta;
      setRotation(currentRotationRef.current);
    },
    [viewMode]
  );

  const handlePointerUp = () => {
    isDraggingRef.current = false;
    setIsDragging(false);
  };

  useEffect(() => {
    const onMove = (e) => handlePointerMove(e);
    const onUp = () => handlePointerUp();

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("touchmove", onMove, { passive: true });
    window.addEventListener("touchend", onUp);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [handlePointerMove]);

  const handleCardClick = (card) => {
    if (dragDistanceRef.current < 6) {
      setSelectedItem(card);
    }
  };

  return (
    <section id="hero" className="studio-hero-viewport" aria-label="Reference Showcase">
      <div className="studio-stage-frame">
        {/* Top Control Bar */}
        <div className="studio-stage-header">
          <div className="stage-left-tag">
            <span className="dot-live"></span>
            <span className="tag-text">{PERSONAL.status}</span>
          </div>

          {/* View Mode Switcher (recreating 2-6s Ribbon vs 7-10s Grid from video) */}
          <div className="stage-view-switcher">
            <button
              type="button"
              className={`stage-mode-btn ${viewMode === "ribbon" ? "active" : ""}`}
              onClick={() => setViewMode("ribbon")}
              aria-label="3D Curved Ribbon"
            >
              ◎ 3D Ribbon Arc
            </button>
            <button
              type="button"
              className={`stage-mode-btn ${viewMode === "grid" ? "active" : ""}`}
              onClick={() => setViewMode("grid")}
              aria-label="2x3 Gallery Grid"
            >
              ▦ 2x3 Showcase Grid
            </button>
          </div>

          <div className="stage-right-stats">
            <a
              href="#contact"
              className="stage-cta-pill"
              onClick={(e) => {
                e.preventDefault();
                document.querySelector("#contact")?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <span>Get In Touch</span>
              <span className="cta-arrow">→</span>
            </a>
          </div>
        </div>

        {/* Monumental Editorial Typography Backdrop (Exact Video Match) */}
        <div className="studio-backdrop-text" aria-hidden="true">
          <div className="backdrop-title-row title-left">
            <span>INTELLIGENCE IN</span>
          </div>

          <div className="backdrop-center-annotation">
            <span className="annotation-line">EXPLORING IDEAS THROUGH</span>
            <span className="annotation-line">DAILY ENGINEERING & APPLIED AI</span>
          </div>

          <div className="backdrop-title-row title-right">
            <span>MOTION</span>
          </div>
        </div>

        {/* Perspective Guidelines & Center Crosshair (Exact Video Match) */}
        <div className="studio-geometric-guides" aria-hidden="true">
          <div className="guide-upper-ellipse"></div>
          <div className="guide-lower-ellipse"></div>
          <div className="guide-center-crosshair">+</div>
        </div>

        {/* Main Interactive Stage: 3D Ribbon vs 2x3 Grid */}
        <div className="studio-interactive-canvas">
          {viewMode === "ribbon" ? (
            <div
              className={`ribbon-3d-scene ${isDragging ? "is-dragging" : ""}`}
              onPointerDown={handlePointerDown}
              role="region"
              aria-label="Interactive 3D Ribbon"
            >
              <div
                className="ribbon-cylinder-track"
                style={{
                  transform: `rotateX(10deg) rotateZ(-3.5deg) rotateY(${rotation}deg)`,
                }}
              >
                {cards.map((card, idx) => {
                  const angle = (idx * 360) / count;
                  return (
                    <div
                      key={card.id}
                      className={`ribbon-3d-card ${hoveredCardId === card.id ? "hovered" : ""}`}
                      style={{
                        transform: `rotateY(${angle}deg) translateZ(${radius}px)`,
                      }}
                      onMouseEnter={() => setHoveredCardId(card.id)}
                      onMouseLeave={() => setHoveredCardId(null)}
                      onClick={() => handleCardClick(card)}
                    >
                      <div className="ribbon-card-shell">
                        <div className="ribbon-card-media">
                          <img
                            src={card.image}
                            alt={card.name}
                            draggable={false}
                            loading="lazy"
                          />
                          <div className="ribbon-category-tag">{card.category}</div>
                        </div>
                        <div className="ribbon-card-caption">
                          <h4 className="caption-title">{card.name}</h4>
                          <p className="caption-sub">{card.tagline}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            /* 2x3 Showcase Grid (Exact Match to 8-12s of Reference Video) */
            <div className="studio-showcase-grid" role="region" aria-label="Showcase Grid">
              {cards.map((card) => (
                <article
                  key={card.id}
                  className="showcase-grid-item"
                  onClick={() => setSelectedItem(card)}
                >
                  <div className="grid-item-media">
                    <img referrerPolicy="no-referrer" src={card.image} alt={card.name} loading="lazy" />
                    <span className="grid-item-badge">{card.category}</span>
                  </div>
                  <div className="grid-item-meta">
                    <div className="grid-item-title-bar">
                      <h4 className="grid-item-title">{card.name}</h4>
                      <span className="grid-item-year">{card.year}</span>
                    </div>
                    <p className="grid-item-tagline">{card.tagline}</p>
                    <div className="grid-item-tags">
                      {card.techStack.slice(0, 4).map((tech) => (
                        <span key={tech} className="meta-pill">
                          {tech}
                        </span>
                      ))}
                    </div>
                    <div className="grid-item-actions">
                      {card.live && (
                        <a
                          href={card.live}
                          target="_blank"
                          rel="noreferrer"
                          className="mini-link primary"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Live Demo ↗
                        </a>
                      )}
                      {card.github && (
                        <a
                          href={card.github}
                          target="_blank"
                          rel="noreferrer"
                          className="mini-link"
                          onClick={(e) => e.stopPropagation()}
                        >
                          Code ↗
                        </a>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        {/* Bottom Margins Captions (Exact Video Match) */}
        <div className="studio-stage-footer">
          <div className="footer-lead-text">
            <span>Concepts, software architectures, and machine</span>
            <span>learning pipelines built to ship.</span>
          </div>

          <div className="footer-drag-hint">
            {viewMode === "ribbon" ? "DRAG TO ROTATE 3D RIBBON" : "CLICK CARD TO INSPECT ARCHITECTURE"}
          </div>

          <div className="footer-action-link">
            <a href="#about" className="footer-arrow-link">
              <span>EXPLORE WHO I AM</span>
              <span className="arrow-glyph">→</span>
            </a>
          </div>
        </div>
      </div>

      {selectedItem && (
        <DetailModal item={selectedItem} onClose={() => setSelectedItem(null)} />
      )}
    </section>
  );
}
