import "./styles/index.css";
import Navbar from "./components/Navbar";
import IntroLoader from "./components/IntroLoader";
import BackToTop from "./components/BackToTop";
import HeroCinematic from "./sections/HeroCinematic";
import AboutReveal from "./sections/AboutReveal";
import CertificationRibbon from "./sections/CertificationRibbon";
import ProjectsRibbon from "./sections/ProjectsRibbon";
import SkillsStructuredScene from "./sections/SkillsStructuredScene";
import HireMeClosingScene from "./sections/HireMeClosingScene";
import Footer from "./sections/Footer";

export default function App() {
  return (
    <div className="portfolio-cinematic-app">
      <IntroLoader />
      {/* Restrained Top Navigation */}
      <Navbar />

      <main className="cinematic-exhibition-main">
        {/* Scene 1: First Screen — Interactive 3D Dark Hero */}
        <HeroCinematic />

        {/* Scene 2: About — Dramatic 3D Page-Flip Transition */}
        <AboutReveal />

        {/* Scene 3: Certifications — Scroll-Controlled Rotating 3D Cylinder */}
        <CertificationRibbon />

        {/* Scene 4: Projects — Six-Direction Converging Entrance Animations */}
        <ProjectsRibbon />

        {/* Scene 5: Technical Skills — Structured Architectural Layers */}
        <SkillsStructuredScene />

        {/* Scene 6: Hire Me / Contact — Premium Converging Closing Scene */}
        <HireMeClosingScene />
      </main>

      <Footer />
      <BackToTop />
    </div>
  );
}
