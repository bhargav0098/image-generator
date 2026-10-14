import { useEffect } from "react";

export default function ProjectModal({ card, onClose }) {
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

  if (!card) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <button
          className="modal-close-btn"
          onClick={onClose}
          aria-label="Close project modal"
        >
          ✕
        </button>

        <div className="modal-media-header">
          <img referrerPolicy="no-referrer" src={card.image} alt={card.title} className="modal-image" />
          <div className="modal-badge-row">
            <span className="modal-category">{card.category}</span>
            <span className="modal-year">{card.year}</span>
          </div>
        </div>

        <div className="modal-body">
          <h2 className="modal-title">{card.title}</h2>
          <p className="modal-subtitle">{card.subtitle}</p>
          <p className="modal-description">{card.description}</p>

          <div className="modal-tags">
            {card.tags.map((tag) => (
              <span key={tag} className="tag-pill">
                {tag}
              </span>
            ))}
          </div>

          <div className="modal-actions">
            {card.live && (
              <a
                href={card.live}
                target="_blank"
                rel="noreferrer"
                className="btn btn-primary"
              >
                <span>Launch Live Demo</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
            {card.github && (
              <a
                href={card.github}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary"
              >
                <span>Inspect Repository</span>
                <span className="btn-arrow">↗</span>
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
