import { useState, useEffect, useRef } from "react";
import { CERTIFICATES } from "../data/portfolio";
import DetailModal from "../components/DetailModal";

export default function CertificationCylinderScene() {
  const containerRef = useRef(null);
  const [scrollRotation, setScrollRotation] = useState(0);
  const [activeCertIndex, setActiveCertIndex] = useState(0);
  const [selectedCert, setSelectedCert] = useState(null);

  const certCount = CERTIFICATES.length;
  const radius = 580; // 3D cylinder depth radius

  useEffect(() => {
    const handleScroll = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight;
      const totalScrollDistance = rect.height - windowHeight;

      if (totalScrollDistance <= 0) return;

      const currentScroll = -rect.top;
      const progress = Math.max(0, Math.min(1, currentScroll / totalScrollDistance));

      // Scroll controls cylinder rotation (e.g. 540 degrees across the scroll track)
      const deg = progress * 540;
      setScrollRotation(deg);

      // Determine which certificate is currently facing forward
      const currentAngleNorm = ((deg % 360) + 360) % 360;
      const activeIdx = Math.round(currentAngleNorm / (360 / certCount)) % certCount;
      setActiveCertIndex(activeIdx);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [certCount]);

  return (
    <section
      id="certifications"
      ref={containerRef}
      className="scene-cert-cylinder-container"
      aria-label="Scene 3: Scroll-Controlled Certification Cylinder"
    >
      <div className="sticky-cylinder-stage">
        {/* Scene Heading */}
        <div className="cylinder-scene-head">
          <span className="section-roman-tag">03 / VERIFIED CREDENTIALS</span>
          <h2 className="cylinder-main-title">Scroll-Controlled Certification Cylinder</h2>
          <p className="cylinder-sub-text">
            Scroll down to rotate through verified credentials in AWS Machine Learning, Cloud Architecture, Oracle Generative AI, and Microsoft Fabric.
          </p>
        </div>

        {/* 3D Rotating Cylinder Viewport */}
        <div className="cylinder-3d-viewport" role="region" aria-label="3D Cylinder Showcase">
          {/* Subtle background vanishing guidelines */}
          <div className="cylinder-depth-grid" aria-hidden="true">
            <div className="depth-circle-top"></div>
            <div className="depth-circle-bottom"></div>
            <div className="depth-crosshair">+</div>
          </div>

          <div
            className="cylinder-track-ring"
            style={{
              transform: `perspective(1600px) rotateX(10deg) rotateZ(-2deg) rotateY(${-scrollRotation}deg)`,
            }}
          >
            {CERTIFICATES.map((cert, idx) => {
              const cardAngle = (idx * 360) / certCount;
              const isFront = idx === activeCertIndex;

              return (
                <div
                  key={cert.id}
                  className={`cylinder-panel-card ${isFront ? "is-front-focus" : ""}`}
                  style={{
                    transform: `rotateY(${cardAngle}deg) translateZ(${radius}px)`,
                  }}
                  onClick={() => setSelectedCert(cert)}
                >
                  <div className="panel-card-inner">
                    <div className="panel-cert-media">
                      <img
                        src={cert.image}
                        alt={cert.title}
                        loading="lazy"
                        draggable={false}
                      />
                      <span className="panel-badge">{cert.category}</span>
                    </div>

                    <div className="panel-cert-meta">
                      <h3 className="panel-title">{cert.title}</h3>
                      <p className="panel-issuer">{cert.issuer}</p>
                      <div className="panel-foot">
                        <span className="panel-date">{cert.date}</span>
                        <span className="panel-click-hint">Inspect ↗</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dynamic focused certificate readout */}
        <div className="cylinder-focused-bar">
          <div className="focused-info">
            <span className="focused-counter">
              CREDENTIAL {activeCertIndex + 1} OF {certCount}
            </span>
            <span className="focused-name">{CERTIFICATES[activeCertIndex]?.title}</span>
          </div>
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setSelectedCert(CERTIFICATES[activeCertIndex])}
          >
            Inspect Credential Details ↗
          </button>
        </div>
      </div>

      {/* Accessible Alternative List */}
      <div className="accessible-credentials-tray">
        <h3 className="tray-heading">Complete Credential Roster</h3>
        <div className="tray-list-grid">
          {CERTIFICATES.map((c) => (
            <div
              key={c.id}
              className="tray-item-card"
              onClick={() => setSelectedCert(c)}
            >
              <div className="tray-item-text">
                <span className="tray-cat">{c.category}</span>
                <h4 className="tray-title">{c.title}</h4>
                <p className="tray-sub">{c.issuer} · {c.date}</p>
              </div>
              <button className="tray-view-btn">View ↗</button>
            </div>
          ))}
        </div>
      </div>

      {selectedCert && (
        <DetailModal item={selectedCert} onClose={() => setSelectedCert(null)} />
      )}
    </section>
  );
}
