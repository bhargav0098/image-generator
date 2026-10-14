import { PERSONAL } from "../data/portfolio";

export default function Footer() {
  return (
    <footer className="site-footer" role="contentinfo">
      <div className="footer-inner">
        <div className="footer-brand">{PERSONAL.name}</div>
        <p className="footer-credit">
          © {new Date().getFullYear()} {PERSONAL.name}. Built with React & Vite.
        </p>
        <div className="footer-socials">
          <a href={PERSONAL.github} target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href={PERSONAL.linkedin} target="_blank" rel="noopener noreferrer">
            LinkedIn
          </a>
          <a href={`mailto:${PERSONAL.email}`}>Email</a>
        </div>
      </div>
    </footer>
  );
}
