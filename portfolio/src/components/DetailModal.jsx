import { useEffect } from "react";

export default function DetailModal({ item, onClose }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  if (!item) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
          ✕
        </button>

        <div className="modal-media-header">
          <img referrerPolicy="no-referrer" src={item.image} alt={item.title || item.name} className="modal-image" />
          <div className="modal-badge-row">
            <span className="modal-category">{item.category}</span>
            <span className="modal-year">{item.year || item.date}</span>
          </div>
        </div>

        <div className="modal-body">
          <h2 className="modal-title">{item.title || item.name}</h2>
          <p className="modal-subtitle">{item.subtitle || item.issuer || item.tagline}</p>
          <p className="modal-description">{item.description}</p>

          {item.credentialId && (
            <div className="cert-id-badge">
              <span className="id-label">Credential Verification ID:</span>
              <code className="id-code">{item.credentialId}</code>
            </div>
          )}

          {item.techStack && (
            <div className="modal-tags">
              {item.techStack.map((tech) => (
                <span key={tech} className="tag-pill">
                  {tech}
                </span>
              ))}
            </div>
          )}

          <div className="modal-actions">
            {item.live && (
              <a href={item.live} target="_blank" rel="noreferrer" className="btn btn-primary">
                <span>Launch Live Demo</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            {item.github && (
              <a href={item.github} target="_blank" rel="noreferrer" className="btn btn-ghost">
                <span>Source Code</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            {item.verifyUrl && (
              <a href={item.verifyUrl} target="_blank" rel="noreferrer" className="btn btn-primary">
                <span>Verify on Official Portal</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            <button className="btn btn-outline" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
