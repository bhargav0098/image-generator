import { useEffect } from "react";

export default function CertificateModal({ cert, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  if (!cert) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container cert-modal-container" onClick={(e) => e.stopPropagation()}>
        <button
          className="modal-close-btn"
          onClick={onClose}
          aria-label="Close modal"
        >
          ✕
        </button>

        <div className="modal-cert-preview">
          <img referrerPolicy="no-referrer" src={cert.image} alt={cert.title} className="modal-cert-img" />
        </div>

        <div className="modal-body">
          <div className="modal-badge-row">
            <span className="modal-category">{cert.category}</span>
            <span className="modal-year">{cert.date}</span>
          </div>

          <h2 className="modal-title">{cert.title}</h2>
          <p className="modal-subtitle">Issued by {cert.issuer}</p>

          <p className="modal-description">{cert.description}</p>

          {cert.credentialId && (
            <div className="cert-id-badge">
              <span className="id-label">Credential Verification ID:</span>
              <code className="id-code">{cert.credentialId}</code>
            </div>
          )}

          <div className="modal-actions">
            {cert.verifyUrl && (
              <a
                href={cert.verifyUrl}
                target="_blank"
                rel="noreferrer"
                className="btn btn-primary"
              >
                <span>Verify Credential on Official Portal</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            {cert.fileUrl && (
              <a href={cert.fileUrl} target="_blank" rel="noreferrer" className="btn btn-ghost">
                <span>Open Certificate File</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            <button className="btn btn-ghost" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
