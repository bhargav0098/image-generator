# Telugu Bhargav Ram — Developer Portfolio

A modern, cinematic developer portfolio built with React 19, Vite, and Canvas 2D reflection physics. Designed specifically for software engineering, full-stack development, AI/ML, and Generative AI roles.

Live Link: `https://bhargav0098.github.io/image-generator/`

---

## 🚀 Key Highlights & Architectural Features

- **Cinematic Canvas Hero Background**: Replaces heavy video overlays with a dynamic, low-overhead HTML5 Canvas 2D ambient light simulation over the cinematic dark reflective backdrop (`background.jpg`).
- **Authentic & Credible Content**: All statistics, skills, education, experience, and project metrics are synchronized directly with your verified resume (`Telugu_BhargavRam_Final_SAP.pdf`) — no fabricated stats or arbitrary proficiency percentages.
- **Data-Driven Architecture**: Easily update any information from a single central file at [`portfolio/src/data/portfolio.js`](file:///c:/Users/bharg/Downloads/image-generator-main/image-generator-main/portfolio/src/data/portfolio.js).
- **Responsive & Accessible**: Fully accessible semantic HTML, keyboard-navigable focus outlines, respects `prefers-reduced-motion`, and fluid layouts across mobile, tablet, and desktop screens.
- **Production Build Ready**: Configured for instant deployment via GitHub Pages (`.github/workflows/deploy.yml`) or Vercel.

---

## 📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── deploy.yml              # Automated GitHub Pages CI/CD workflow
├── portfolio/
│   ├── public/
│   │   ├── Telugu_BhargavRam_Final_SAP.pdf  # Resume PDF
│   │   └── favicon.svg             # Custom signal branding icon
│   ├── src/
│   │   ├── assets/                 # Background & static images
│   │   ├── components/             # Reusable UI components
│   │   │   ├── HeroBackground.jsx  # Moving light reflection canvas
│   │   │   ├── Navbar.jsx          # Pill navbar with mobile drawer
│   │   │   ├── RoleTyper.jsx       # Dynamic typewriter role cycling
│   │   │   ├── ScrollReveal.jsx    # IntersectionObserver reveal wrapper
│   │   │   └── BackToTop.jsx       # Floating scroll-to-top button
│   │   ├── data/
│   │   │   └── portfolio.js        # Central content configuration
│   │   ├── sections/               # Page sections
│   │   │   ├── Hero.jsx            # Hero banner & CTAs
│   │   │   ├── Projects.jsx        # Editorial project cards with live & code links
│   │   │   ├── Skills.jsx          # Categorized skill badges
│   │   │   ├── Experience.jsx      # Verified SmartBridge AI/ML internship
│   │   │   ├── Education.jsx       # B.Tech degree, Certifications & Achievements
│   │   │   ├── Contact.jsx         # Direct email & social channels
│   │   │   └── Footer.jsx          # Copyright and branding
│   │   ├── styles/
│   │   │   └── index.css           # Premium dark-theme design system
│   │   ├── App.jsx                 # Main layout
│   │   └── main.jsx                # React root mount
│   ├── vite.config.js              # Vite config (base path configured for GitHub Pages)
│   └── package.json
└── package.json                    # Workspace helper scripts
```

---

## 🛠️ Local Development

### Prerequisites
- Node.js (v18 or higher)
- npm (v9 or higher)

### Quick Start

```bash
# 1. Install dependencies
cd portfolio
npm install

# 2. Start local development server
npm run dev

# Or from the repository root:
npm run dev
```

Visit `http://localhost:5173/` in your browser.

### Production Build & Preview

```bash
# Build the production distribution
npm run build

# Preview the production build locally
npm run preview
```

---

## 🌐 Deployment to GitHub Pages

1. **Push to GitHub**:
   Ensure this project is pushed to your GitHub repository (e.g., `image-generator` or `bhargav0098.github.io`).
2. **Enable GitHub Actions Pages**:
   - Go to your repository **Settings** → **Pages**.
   - Under **Build and deployment** → **Source**, select **GitHub Actions**.
3. **Automated CI/CD**:
   - The included workflow in `.github/workflows/deploy.yml` will automatically build the site and deploy `portfolio/dist` to your live GitHub Pages URL whenever changes are pushed to `main` or `master`.

---

## ✏️ Updating Portfolio Information

To edit any details on the website, simply update [`portfolio/src/data/portfolio.js`](file:///c:/Users/bharg/Downloads/image-generator-main/image-generator-main/portfolio/src/data/portfolio.js):

- **Personal Information**: Name, headline, contact email, GitHub, LinkedIn.
- **Projects**: Project title, tags, description, GitHub repository link, and live demo URL.
- **Skills**: Categorized skills (Programming Languages, AI/ML & GenAI, Web Development, Databases, Cloud & DevOps).
- **Experience**: Role, organization, duration, and bullet points.
- **Education, Certifications & Achievements**: Colleges, credentials, and honors.
