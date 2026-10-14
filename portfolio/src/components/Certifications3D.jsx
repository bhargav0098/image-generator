import { useState, useRef, useEffect, useCallback } from "react";
import { CERTIFICATES } from "../data/portfolio";
import ScrollReveal from "./ScrollReveal";
import DetailModal from "./DetailModal";

export default function Certifications3D() {
  const [viewMode, setViewMode] = useState("cylinder"); // 'cylinder' | 'grid'
  const [rotation, setRotation] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedCert, setSelectedCert] = useState(null);
  const [hoveredCertId, setHoveredCertId] = useState(null);

  const isDraggingRef = useRef(false);
  const lastXRef = useRef(0);
  const velocityRef = useRef(0);
  const currentRotationRef = useRef(0);
  const animFrameIdRef = useRef(null);
  const dragDistanceRef = useRef(0);

  const certCount = CERTIFICATES.length;
  const radius = 640;

  // Inertia + continuous auto-drift loop
  useEffect(() => {
    let lastTime = performance.now();

    const loop = (now) => {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      if (!isDraggingRef.current && viewMode === "cylinder") {
        velocityRef.current *= Math.pow(0.92, dt * 60);

        if (Math.abs(velocityRef.current) < 0.05) {
          velocityRef.current = -0.15; // gentle continuous rotation
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
    if (viewMode !== "cylinder") return;
    isDraggingRef.current = true;
    setIsDragging(true);
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    lastXRef.current = clientX;
    velocityRef.current = 0;
    dragDistanceRef.current = 0;
  };

  const handlePointerMove = useCallback(
    (e) => {
      if (!isDraggingRef.current || viewMode !== "cylinder") return;
      const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
      const deltaX = clientX - lastXRef.current;
      lastXRef.current = clientX;
      dragDistanceRef.current += Math.abs(deltaX);

      const degDelta = deltaX * 0.24;
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

  const handleCertClick = (cert) => {
    if (dragDistanceRef.current < 8) {
      setSelectedCert(cert);
    }
  };

  return (
    <section id="certifications" className="section cert-studio-section" aria-label="Certifications">
      <div className="section-inner">
        <ScrollReveal>
          <div className="section-head text-center">
            <span className="section-label">04 / VERIFIED CREDENTIALS</span>
            <h2 className="section-title">Industry Certifications</h2>
            <p className="section-sub mx-auto">
              Validated expertise in AWS Machine Learning, Cloud Architecture, Microsoft Fabric Analytics, and Oracle Generative AI.
            </p>
          </div>
        </ScrollReveal>

        {/* 3D Certificate Stage replicating 2 to 6 sec of reference video */}
        <div className="cert-stage-container">
          <div className="showcase-controls">
            <div className="view-mode-tabs">
              <button
                type="button"
                className={`tab-btn ${viewMode === "cylinder" ? "active" : ""}`}
                onClick={() => setViewMode("cylinder")}
              >
                <span className="tab-icon">◎</span> 3D Cylinder Ribbon
              </button>
              <button
                type="button"
                className={`tab-btn ${viewMode === "grid" ? "active" : ""}`}
                onClick={() => setViewMode("grid")}
              >
                <span className="tab-icon">▦</span> Certificate Grid
              </button>
            </div>
            <span className="drag-hint">
              {viewMode === "cylinder" ? "DRAG TO ROTATE 3D CERTIFICATES" : "CLICK CERTIFICATE TO INSPECT"}
            </span>
          </div>

          {/* Typography Backdrop behind the 3D rotating certificates */}
          <div className="studio-typography-backdrop" aria-hidden="true">
            <div className="typo-row typo-left">
              <span>CREDENTIALS IN</span>
            </div>
            <div className="typo-center-caption">
              <span className="cert-live-count">6 VERIFIED INDUSTRY CREDENTIALS</span>
              <p className="typo-sub-explainer">AWS · MICROSOFT · ORACLE · SAP</p>
            </div>
            <div className="typo-row typo-right">
              <span>MOTION</span>
            </div>
          </div>

          {viewMode === "cylinder" ? (
            <div
              className={`cylinder-viewport ${isDragging ? "is-dragging" : ""}`}
              onPointerDown={handlePointerDown}
              role="region"
              aria-label="3D Rotating Certificate Ribbon"
            >
              <div className="cylinder-guidelines" aria-hidden="true">
                <div className="guide-arc-top"></div>
                <div className="guide-arc-bottom"></div>
                <div className="guide-crosshair">+</div>
              </div>

              <div
                className="cylinder-track"
                style={{
                  transform: `rotateX(12deg) rotateZ(-3deg) rotateY(${rotation}deg)`,
                }}
              >
                {CERTIFICATES.map((cert, idx) => {
                  const angle = (idx * 360) / certCount;
                  return (
                    <div
                      key={cert.id}
                      className={`cylinder-card cert-card-3d ${hoveredCertId === cert.id ? "hovered" : ""}`}
                      style={{
                        transform: `rotateY(${angle}deg) translateZ(${radius}px)`,
                      }}
                      onMouseEnter={() => setHoveredCertId(cert.id)}
                      onMouseLeave={() => setHoveredCertId(null)}
                      onClick={() => handleCertClick(cert)}
                    >
                      <div className="card-inner cert-inner">
                        <div className="card-media cert-media">
                          <img
                            src={cert.image}
                            alt={cert.title}
                            draggable={false}
                            loading="lazy"
                          />
                          <span className="card-badge">{cert.category}</span>
                        </div>
                        <div className="card-meta cert-meta">
                          <h4 className="card-title cert-title">{cert.title}</h4>
                          <p className="card-sub cert-issuer">{cert.issuer}</p>
                          <span className="cert-date">{cert.date}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="gallery-grid cert-grid">
              {CERTIFICATES.map((cert) => (
                <div
                  key={cert.id}
                  className="gallery-grid-card cert-grid-card"
                  onClick={() => setSelectedCert(cert)}
                >
                  <div className="grid-card-media cert-grid-media">
                    <img referrerPolicy="no-referrer" src={cert.image} alt={cert.title} loading="lazy" />
                    <span className="grid-category-pill">{cert.category}</span>
                  </div>
                  <div className="grid-card-content">
                    <h4 className="grid-card-title">{cert.title}</h4>
                    <p className="cert-grid-issuer">{cert.issuer}</p>
                    <p className="grid-card-desc">{cert.description}</p>
                    <div className="cert-grid-footer">
                      <span className="cert-date-pill">{cert.date}</span>
                      <button className="grid-btn primary">Verify ↗</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="showcase-caption-bar">
            <span className="caption-left">Click any certificate to inspect full resolution and credential ID.</span>
            <span className="caption-right">Verified via Official Certification Portals</span>
          </div>
        </div>
      </div>

      {selectedCert && (
        <DetailModal item={selectedCert} onClose={() => setSelectedCert(null)} />
      )}
    </section>
  );
}
