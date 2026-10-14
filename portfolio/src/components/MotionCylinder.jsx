import { useState, useRef, useEffect, useCallback } from "react";
import { SHOWCASE_CARDS } from "../data/portfolio";

/**
 * MotionCylinder
 * Recreates the signature 3D motion cylinder & interactive ribbon carousel
 * seen in the reference video (TRIONN / Awwwards creative showcase).
 * Supports continuous auto-drift, inertial drag / swipe, 3D perspective depth,
 * and view toggling between 3D Cylinder and 2x3 Gallery Grid.
 */
export default function MotionCylinder({ onSelectCard }) {
  const [viewMode, setViewMode] = useState("cylinder"); // 'cylinder' | 'grid'
  const [rotation, setRotation] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [hoveredCardId, setHoveredCardId] = useState(null);

  const containerRef = useRef(null);
  const isDraggingRef = useRef(false);
  const startXRef = useRef(0);
  const lastXRef = useRef(0);
  const velocityRef = useRef(0);
  const currentRotationRef = useRef(0);
  const animFrameIdRef = useRef(null);
  const dragDistanceRef = useRef(0);

  const cardCount = SHOWCASE_CARDS.length;
  // Radius of the 3D cylinder
  const radius = 680;

  // Physics animation loop (inertia + auto-drift)
  useEffect(() => {
    let lastTime = performance.now();

    const loop = (now) => {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      if (!isDraggingRef.current) {
        // Apply friction to velocity
        velocityRef.current *= Math.pow(0.92, dt * 60);

        // Constant gentle auto-drift if barely moving
        if (Math.abs(velocityRef.current) < 0.05) {
          velocityRef.current = -0.15;
        }

        currentRotationRef.current += velocityRef.current;
        setRotation(currentRotationRef.current);
      }

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, []);

  // Pointer / Touch Handlers
  const handlePointerDown = (e) => {
    isDraggingRef.current = true;
    setIsDragging(true);
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    startXRef.current = clientX;
    lastXRef.current = clientX;
    velocityRef.current = 0;
    dragDistanceRef.current = 0;
  };

  const handlePointerMove = useCallback((e) => {
    if (!isDraggingRef.current) return;
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    const deltaX = clientX - lastXRef.current;
    lastXRef.current = clientX;
    dragDistanceRef.current += Math.abs(deltaX);

    // Speed factor: drag px to degrees
    const degDelta = deltaX * 0.22;
    currentRotationRef.current += degDelta;
    velocityRef.current = degDelta;
    setRotation(currentRotationRef.current);
  }, []);

  const handlePointerUp = () => {
    isDraggingRef.current = false;
    setIsDragging(false);
  };

  useEffect(() => {
    const onWindowMove = (e) => handlePointerMove(e);
    const onWindowUp = () => handlePointerUp();

    window.addEventListener("pointermove", onWindowMove);
    window.addEventListener("pointerup", onWindowUp);
    window.addEventListener("touchmove", onWindowMove, { passive: true });
    window.addEventListener("touchend", onWindowUp);

    return () => {
      window.removeEventListener("pointermove", onWindowMove);
      window.removeEventListener("pointerup", onWindowUp);
      window.removeEventListener("touchmove", onWindowMove);
      window.removeEventListener("touchend", onWindowUp);
    };
  }, [handlePointerMove]);

  const handleCardClick = (card) => {
    // Only open if it wasn't a significant drag gesture
    if (dragDistanceRef.current < 6) {
      if (onSelectCard) {
        onSelectCard(card);
      }
    }
  };

  return (
    <div className="motion-showcase" ref={containerRef}>
      {/* View Toggle Bar */}
      <div className="showcase-controls">
        <div className="view-mode-tabs">
          <button
            type="button"
            className={`tab-btn ${viewMode === "cylinder" ? "active" : ""}`}
            onClick={() => setViewMode("cylinder")}
            aria-label="Switch to 3D Cylinder view"
          >
            <span className="tab-icon">◎</span> 3D Cylinder Arc
          </button>
          <button
            type="button"
            className={`tab-btn ${viewMode === "grid" ? "active" : ""}`}
            onClick={() => setViewMode("grid")}
            aria-label="Switch to Gallery Grid view"
          >
            <span className="tab-icon">▦</span> Gallery Grid
          </button>
        </div>
        <div className="drag-hint">
          {viewMode === "cylinder" ? "Drag horizontally to rotate 3D ribbon" : "Click card for project deep-dive"}
        </div>
      </div>

      {/* 3D Cylinder View */}
      {viewMode === "cylinder" ? (
        <div
          className={`cylinder-viewport ${isDragging ? "is-dragging" : ""}`}
          onPointerDown={handlePointerDown}
          role="region"
          aria-label="Interactive 3D Project Cylinder"
        >
          {/* Subtle background guides matching reference video */}
          <div className="cylinder-guidelines" aria-hidden="true">
            <div className="guide-arc-top"></div>
            <div className="guide-arc-bottom"></div>
            <div className="guide-crosshair">+</div>
          </div>

          <div
            className="cylinder-track"
            style={{
              transform: `rotateX(11deg) rotateZ(-3deg) rotateY(${rotation}deg)`,
            }}
          >
            {SHOWCASE_CARDS.map((card, idx) => {
              const cardAngle = (idx * 360) / cardCount;
              return (
                <div
                  key={card.id}
                  className={`cylinder-card ${hoveredCardId === card.id ? "hovered" : ""}`}
                  style={{
                    transform: `rotateY(${cardAngle}deg) translateZ(${radius}px)`,
                  }}
                  onMouseEnter={() => setHoveredCardId(card.id)}
                  onMouseLeave={() => setHoveredCardId(null)}
                  onClick={() => handleCardClick(card)}
                >
                  <div className="card-inner">
                    <div className="card-media">
                      <img
                        src={card.image}
                        alt={card.title}
                        loading="lazy"
                        draggable={false}
                      />
                      <div className="card-badge">{card.category}</div>
                      <div className="card-year">{card.year}</div>
                    </div>
                    <div className="card-meta">
                      <h4 className="card-title">{card.title}</h4>
                      <p className="card-sub">{card.subtitle}</p>
                      <div className="card-pills">
                        {card.tags.slice(0, 3).map((tag) => (
                          <span key={tag} className="tag-pill">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        /* Gallery Grid View (Matching Frame 4 in reference video) */
        <div className="gallery-grid" role="region" aria-label="Project Gallery Grid">
          {SHOWCASE_CARDS.map((card) => (
            <div
              key={card.id}
              className="gallery-grid-card"
              onClick={() => onSelectCard && onSelectCard(card)}
            >
              <div className="grid-card-media">
                <img referrerPolicy="no-referrer" src={card.image} alt={card.title} loading="lazy" />
                <span className="grid-category-pill">{card.category}</span>
              </div>
              <div className="grid-card-content">
                <div className="grid-card-header">
                  <h4 className="grid-card-title">{card.title}</h4>
                  <span className="grid-card-year">{card.year}</span>
                </div>
                <p className="grid-card-desc">{card.description}</p>
                <div className="grid-card-tags">
                  {card.tags.map((t) => (
                    <span key={t} className="tag-pill">
                      {t}
                    </span>
                  ))}
                </div>
                <div className="grid-card-actions">
                  <a
                    href={card.live}
                    target="_blank"
                    rel="noreferrer"
                    className="grid-btn primary"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Live Demo ↗
                  </a>
                  <a
                    href={card.github}
                    target="_blank"
                    rel="noreferrer"
                    className="grid-btn secondary"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Code ↗
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Editorial footer captions like video */}
      <div className="showcase-caption-bar">
        <div className="caption-left">
          <span>Concepts, machine learning pipelines, and full-stack software built to ship.</span>
        </div>
        <div className="caption-right">
          <a href="#projects" className="link-arrow">
            Explore Detailed Case Studies <span>→</span>
          </a>
        </div>
      </div>
    </div>
  );
}
