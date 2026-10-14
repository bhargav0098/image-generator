import { useState, useEffect, useCallback } from "react";
import { PERSONAL } from "../data/portfolio";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 40);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const closeMobile = useCallback(() => setMobileMenuOpen(false), []);

  const handleNavClick = (e, targetId) => {
    e.preventDefault();
    closeMobile();
    const elem = document.querySelector(targetId);
    if (elem) {
      elem.scrollIntoView({ behavior: "smooth" });
      if (window.history && window.history.pushState) {
        window.history.pushState(null, "", targetId);
      }
    }
  };

  const navItems = [
    { label: "About", href: "#about", num: "01" },
    { label: "Certifications", href: "#certifications", num: "02" },
    { label: "Projects", href: "#projects", num: "03" },
    { label: "Skills", href: "#skills", num: "04" },
    { label: "Contact", href: "#contact", num: "05" },
  ];

  return (
    <>
      <header id="navbar" className={scrolled ? "is-scrolled" : ""}>
        <div className="nav-container">
          {/* Brand */}
          <a
            href="#hero"
            className="nav-brand"
            onClick={(e) => handleNavClick(e, "#hero")}
            aria-label="Back to Top"
          >
            <span className="brand-symbol">⟁</span>
            <span className="brand-text">BR.</span>
          </a>

          {/* Desktop Navigation Links */}
          <nav className="nav-desktop-links" aria-label="Main Navigation">
            {navItems.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="nav-anchor"
                onClick={(e) => handleNavClick(e, item.href)}
              >
                <span className="nav-idx">{item.num}</span>
                <span className="nav-title">{item.label}</span>
              </a>
            ))}
          </nav>

          {/* Hire Me Action & Mobile Toggle */}
          <div className="nav-right-actions">
            <a
              href="#contact"
              className="hire-me-pill"
              onClick={(e) => handleNavClick(e, "#contact")}
            >
              <span>Hire Me</span>
              <span className="hire-indicator-dot"></span>
            </a>

            <button
              className={`mobile-hamburger ${mobileMenuOpen ? "active" : ""}`}
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle navigation menu"
              aria-expanded={mobileMenuOpen}
            >
              <span />
              <span />
              <span />
            </button>
          </div>
        </div>
      </header>

      {/* Accessible Mobile Menu */}
      <div
        className={`mobile-nav-backdrop ${mobileMenuOpen ? "open" : ""}`}
        onClick={closeMobile}
        aria-hidden="true"
      />

      <nav
        className={`mobile-nav-drawer ${mobileMenuOpen ? "open" : ""}`}
        aria-label="Mobile Navigation"
      >
        <div className="drawer-header">
          <span className="drawer-brand">⟁BR.</span>
          <button className="drawer-close" onClick={closeMobile} aria-label="Close menu">
            ✕
          </button>
        </div>
        <ul className="drawer-links">
          {navItems.map((item) => (
            <li key={item.href}>
              <a href={item.href} onClick={(e) => handleNavClick(e, item.href)}>
                <span className="drawer-num">{item.num}</span>
                <span className="drawer-label">{item.label}</span>
              </a>
            </li>
          ))}
          <li>
            <a
              href="#contact"
              className="drawer-hire-btn"
              onClick={(e) => handleNavClick(e, "#contact")}
            >
              Hire Me ↗
            </a>
          </li>
        </ul>
        <div className="drawer-footer">
          <a href={PERSONAL.github} target="_blank" rel="noreferrer">GitHub</a>
          <a href={PERSONAL.linkedin} target="_blank" rel="noreferrer">LinkedIn</a>
          <a href={PERSONAL.resumePath} target="_blank" rel="noreferrer">View Resume ↗</a>
        </div>
      </nav>
    </>
  );
}
