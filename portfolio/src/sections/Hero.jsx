import { PERSONAL } from "../data/portfolio";
import HeroBackground from "../components/HeroBackground";
import RoleTyper from "../components/RoleTyper";

export default function Hero() {
  return (
    <section id="hero" className="hero-landing" aria-label="Hero Section">
      {/* Cinematic Ambient Canvas Layer */}
      <HeroBackground />

      <div className="hero-inner">
        {/* Status indicator */}
        <div className="hero-status-pill">
          <span className="status-indicator-dot"></span>
          <span className="status-text">{PERSONAL.status}</span>
        </div>

        {/* Greeting & Headline */}
        <p className="hero-salutation">Hi, I'm</p>
        <h1 className="hero-main-title">{PERSONAL.name}</h1>

        {/* Role Typewriter */}
        <div className="hero-typewriter-container">
          <RoleTyper roles={PERSONAL.roles} />
        </div>

        {/* Bio Description */}
        <p className="hero-bio-lead">
          {PERSONAL.bioHero}
        </p>

        {/* Domain Focus Tags */}
        <div className="hero-tech-badges">
          {PERSONAL.heroTags.map((tag) => (
            <span key={tag} className="tech-badge">
              {tag}
            </span>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="hero-cta-group">
          <a
            href="#projects"
            className="btn btn-primary"
            onClick={(e) => {
              e.preventDefault();
              document.querySelector("#projects")?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            <span>View Projects</span>
            <span className="btn-arrow">↓</span>
          </a>

          <a
            href={PERSONAL.resumePath}
            target="_blank"
            rel="noreferrer"
            className="btn btn-ghost"
          >
            <span>Download Resume PDF</span>
            <span className="btn-arrow">↗</span>
          </a>

          <div className="hero-social-links">
            <a href={PERSONAL.github} target="_blank" rel="noreferrer" className="btn-icon" aria-label="GitHub">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.5 0 12.3c0 5.43 3.44 10.03 8.2 11.65.6.12.82-.27.82-.6 0-.3-.01-1.08-.02-2.12-3.34.74-4.04-1.65-4.04-1.65-.55-1.43-1.34-1.82-1.34-1.82-1.1-.77.08-.76.08-.76 1.21.09 1.85 1.27 1.85 1.27 1.08 1.88 2.83 1.34 3.52 1.02.1-.79.42-1.34.76-1.65-2.66-.31-5.47-1.37-5.47-6.1 0-1.35.46-2.45 1.22-3.32-.12-.31-.53-1.56.12-3.25 0 0 1-.33 3.3 1.26a11.2 11.2 0 0 1 6 0c2.28-1.59 3.29-1.26 3.29-1.26.65 1.69.24 2.94.12 3.25.76.87 1.22 1.97 1.22 3.32 0 4.74-2.81 5.78-5.49 6.09.43.38.81 1.13.81 2.28 0 1.65-.02 2.98-.02 3.39 0 .33.22.72.83.6C20.56 22.32 24 17.72 24 12.3 24 5.5 18.63 0 12 0Z"/>
              </svg>
            </a>
            <a href={PERSONAL.linkedin} target="_blank" rel="noreferrer" className="btn-icon" aria-label="LinkedIn">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.38-1.85 3.61 0 4.28 2.38 4.28 5.47v6.27ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45Z"/>
              </svg>
            </a>
          </div>
        </div>

        {/* Hero Stat Counters */}
        <div className="hero-metrics-strip">
          {PERSONAL.heroStats.map((stat) => (
            <div key={stat.label} className="metric-card">
              <span className="metric-number">{stat.value}</span>
              <span className="metric-label">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="hero-scroll-prompt" aria-hidden="true">
        <span>01 / ABOUT ME</span>
        <div className="scroll-arrow-anim">↓</div>
      </div>
    </section>
  );
}
