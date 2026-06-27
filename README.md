# Telugu Bhargav Ram — Portfolio

A premium, futuristic personal portfolio built with pure HTML5, CSS3, vanilla JavaScript, Three.js, and GSAP. No frameworks, no build step — deploy straight from the repo root with GitHub Pages.
live link : https://bhargav0098.github.io/image-generator/

## Structure

```
index.html     → markup for every section
style.css      → design system + all styles
script.js      → Three.js neural-network hero, GSAP animation, interactions
assets/        → put your resume PDF and any images here
.nojekyll      → tells GitHub Pages to skip Jekyll processing (keeps things fast & simple)
```

## Run locally

No build tools needed. Just serve the folder, since `fetch`/modules-free vanilla JS works fine over plain HTTP(S):

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Or just double-click `index.html` — it works opened directly via `file://` too, since everything is self-contained (Three.js/GSAP load from CDN).

## Deploy on GitHub Pages

1. Create a new repo (e.g. `bhargav0098.github.io` for a user-root site, or any repo name for a project site).
2. Push these files to the **root** of the `main` branch:
   ```bash
   git init
   git add .
   git commit -m "Launch portfolio"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
3. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from a branch**.
4. Branch: `main`, folder: `/ (root)`. Save.
5. Your site goes live at `https://<your-username>.github.io/<repo-name>/` (or `https://<your-username>.github.io/` if you used the special `username.github.io` repo name).

## Before you publish — fill these in

- **Resume**: drop your actual PDF at `assets/Telugu_Bhargav_Ram_Resume.pdf` (the Download Resume button already points here).
- **Live Demo links**: the three project cards have `href="#"` placeholders for "Live Demo" — swap in your deployed URLs for MediQueueAI, INTENTO, and DevForgeAI.
- **Social links**: GitHub (`github.com/bhargav0098`) and LinkedIn (`linkedin.com/in/telugu-bhargav-ram`) URLs are already wired in the hero, nav, contact section, and footer — update if they change.
- **Contact form**: it currently opens the visitor's email client via a `mailto:` link (no backend, works for free on GitHub Pages). If you want real form submissions later, wire it to a service like Formspree, Web3Forms, or a small serverless function — that's a one-line change in `initContactForm()` in `script.js`.

## What's inside

- **Hero**: a Three.js neural network rendered as five "domain hubs" (AI/ML, GenAI, Cloud, Full Stack, Data) with satellite nodes and traveling signal pulses — a literal diagram of the skill graph, not decorative particles. Mouse-reactive parallax tilt.
- **Loader**: terminal-style boot sequence before reveal.
- **Custom cursor**: dot + lagging ring, expands over links (desktop only — falls back to the system cursor on touch devices).
- **Scroll reveals & tilt cards**: GSAP + ScrollTrigger powers section reveals, animated counters, the timeline progress line, and 3D tilt on project/cert/timeline cards.
- **Magnetic buttons**, **typewriter role text**, **terminal section**, **floating AI assistant widget**, **back-to-top button** — all implemented in vanilla JS, no dependencies beyond Three.js and GSAP (loaded via CDN `<script>` tags, so nothing to install).
- Respects `prefers-reduced-motion` and works without a mouse (touch-friendly, keyboard-focus states included).

## Editing content

Everything is plain markup in `index.html` — section by section, in the same order as the nav (About → Experience → Projects → Skills → Certifications → Contact). No JSX, no templating, just edit the text directly.
