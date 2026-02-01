// AOS init
AOS.init({
  duration: 700,
  once: true,
  offset: 80,
});

// GSAP setup
if (typeof gsap !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);

  gsap.from("#hero h1", {
    y: 24,
    opacity: 0,
    duration: 0.9,
    ease: "power3.out",
  });

  gsap.from("#hero .btn-primary", {
    y: 16,
    opacity: 0,
    duration: 0.8,
    delay: 0.2,
  });

}

// Typing effect
const typingText = document.getElementById("typing-text");
const phrases = [
  "AI/ML & Full Stack Developer.",
  "Generative AI & Cybersecurity Enthusiast.",
  "Building intelligent, production-ready systems.",
];
let currentPhrase = 0;
let currentChar = 0;
let isDeleting = false;

function typeLoop() {
  if (!typingText) return;
  const phrase = phrases[currentPhrase];
  if (!isDeleting) {
    typingText.textContent = phrase.slice(0, currentChar + 1);
    currentChar++;
    if (currentChar === phrase.length) {
      isDeleting = true;
      setTimeout(typeLoop, 1500);
      return;
    }
  } else {
    typingText.textContent = phrase.slice(0, currentChar - 1);
    currentChar--;
    if (currentChar === 0) {
      isDeleting = false;
      currentPhrase = (currentPhrase + 1) % phrases.length;
    }
  }
  const delay = isDeleting ? 50 : 90;
  setTimeout(typeLoop, delay);
}
typeLoop();

// Dark / light theme toggle
const body = document.getElementById("body");
const themeToggle = document.getElementById("theme-toggle");
const themeToggleMobile = document.getElementById("theme-toggle-mobile");
const themeIcon = document.getElementById("theme-icon");
const themeIconMobile = document.getElementById("theme-icon-mobile");

function setTheme(mode) {
  if (mode === "light") {
    body.classList.add("light");
    themeIcon.textContent = "☀️";
    if (themeIconMobile) themeIconMobile.textContent = "☀️ Toggle Theme";
    localStorage.setItem("theme", "light");
  } else {
    body.classList.remove("light");
    themeIcon.textContent = "🌙";
    if (themeIconMobile) themeIconMobile.textContent = "🌙 Toggle Theme";
    localStorage.setItem("theme", "dark");
  }
}

const savedTheme = localStorage.getItem("theme");
if (savedTheme) {
  setTheme(savedTheme);
} else {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(prefersDark ? "dark" : "light");
}

themeToggle.addEventListener("click", () => {
  const newMode = body.classList.contains("light") ? "dark" : "light";
  setTheme(newMode);
});
if (themeToggleMobile) {
  themeToggleMobile.addEventListener("click", () => {
    const newMode = body.classList.contains("light") ? "dark" : "light";
    setTheme(newMode);
  });
}

// Mobile menu
const mobileMenuBtn = document.getElementById("mobile-menu-btn");
const mobileMenu = document.getElementById("mobile-menu");

if (mobileMenuBtn && mobileMenu) {
  mobileMenuBtn.addEventListener("click", () => {
    mobileMenu.classList.toggle("hidden");
  });

  document.querySelectorAll(".nav-link-mobile").forEach((link) => {
    link.addEventListener("click", () => {
      mobileMenu.classList.add("hidden");
    });
  });
}

// Skill bars animation on scroll
const skillBars = document.querySelectorAll(".skill-bar-fill");
const skillObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const percent = el.getAttribute("data-skill");
        el.style.width = percent + "%";
        observer.unobserve(el);
      }
    });
  },
  { threshold: 0.4 },
);
skillBars.forEach((bar) => skillObserver.observe(bar));

// Simple circular skills animation (CSS-only circle, JS for %)
document.querySelectorAll(".circle-wrapper").forEach((wrapper) => {
  const percent = parseInt(wrapper.getAttribute("data-percent"), 10);
  const numberEl = wrapper.querySelector(".circle-number");
  let current = 0;
  const step = () => {
    current += 2;
    if (current > percent) current = percent;
    numberEl.textContent = current + "%";
    if (current < percent) {
      requestAnimationFrame(step);
    }
  };
  const obs = new IntersectionObserver(
    (entries, o) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          step();
          o.unobserve(wrapper);
        }
      });
    },
    { threshold: 0.6 },
  );
  obs.observe(wrapper);
});

// Custom cursor
const cursor = document.getElementById("custom-cursor");
if (cursor) {
  document.addEventListener("mousemove", (e) => {
    cursor.style.left = e.clientX + "px";
    cursor.style.top = e.clientY + "px";
  });

  const hoverables = document.querySelectorAll("a, button, .project-inner");
  hoverables.forEach((el) => {
    el.addEventListener("mouseenter", () => {
      cursor.style.width = "16px";
      cursor.style.height = "16px";
    });
    el.addEventListener("mouseleave", () => {
      cursor.style.width = "24px";
      cursor.style.height = "24px";
    });
  });
}

// Contact form submission logic
const form = document.getElementById("contact-form");
const statusEl = document.getElementById("form-status");

if (form && statusEl) {
  form.addEventListener("submit", () => {
    statusEl.textContent = "Redirecting to activation page...";
  });
}

// Dynamic year
document.getElementById("year").textContent = new Date().getFullYear();
