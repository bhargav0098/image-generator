/* ============================================================
   TELUGU BHARGAV RAM — PORTFOLIO SCRIPT
   Three.js neural network hero + GSAP scroll/motion system
   ============================================================ */

(function(){
"use strict";

/* ============================================================
   EMAILJS CONFIG
   1. Create a free account at https://www.emailjs.com
   2. Add an Email Service (e.g. Gmail) connected to bhargavram085@gmail.com
   3. Create an Email Template with variables: {{from_name}}, {{from_email}}, {{message}}
   4. Paste your Public Key, Service ID, and Template ID below.
   ============================================================ */
const EMAILJS_PUBLIC_KEY  = "6c9m6fbpzkIQlGyZu";
const EMAILJS_SERVICE_ID  = "service_w5xup4d";
const EMAILJS_TEMPLATE_ID = "template_yg5ulqe";

if(typeof emailjs !== 'undefined'){
  emailjs.init({ publicKey: EMAILJS_PUBLIC_KEY });
}

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isTouch = window.matchMedia('(hover: none)').matches;

/* ============================================================ LOADER */
function runLoader(){
  const loader = document.getElementById('loader');
  const loaderText = document.getElementById('loaderText');
  const barFill = document.getElementById('loaderBarFill');
  const lines = [
    'booting neural_core.sys',
    'loading model: bhargav-v2.6',
    'compiling skill graph',
    'ready.'
  ];
  let li = 0, ci = 0;

  function typeNext(){
    if(li >= lines.length){
      gsap.to(barFill, { width: '100%', duration: .3, onComplete: hideLoader });
      return;
    }
    const line = lines[li];
    if(ci <= line.length){
      loaderText.textContent = line.slice(0, ci);
      ci++;
      setTimeout(typeNext, 22);
    } else {
      gsap.to(barFill, { width: ((li+1)/lines.length*100)+'%', duration:.4, ease:'power2.out' });
      li++; ci = 0;
      setTimeout(typeNext, 220);
    }
  }

  function hideLoader(){
    setTimeout(()=>{
      loader.classList.add('is-hidden');
      document.body.style.overflow = '';
      runHeroIntro();
      setTimeout(()=> loader.remove(), 700);
    }, 260);
  }

  if(prefersReducedMotion){
    loader.classList.add('is-hidden');
    setTimeout(()=> loader.remove(), 50);
    runHeroIntro();
    return;
  }

  document.body.style.overflow = 'hidden';
  typeNext();
}

/* ============================================================ CUSTOM CURSOR */
function initCursor(){
  if(isTouch) return;
  const dot = document.getElementById('cursorDot');
  const ring = document.getElementById('cursorRing');
  let mx = window.innerWidth/2, my = window.innerHeight/2;
  let rx = mx, ry = my;

  window.addEventListener('mousemove', (e)=>{
    mx = e.clientX; my = e.clientY;
    dot.style.transform = `translate(${mx}px, ${my}px) translate(-50%,-50%)`;
  });

  function loop(){
    rx += (mx - rx) * 0.18;
    ry += (my - ry) * 0.18;
    ring.style.transform = `translate(${rx}px, ${ry}px) translate(-50%,-50%)`;
    requestAnimationFrame(loop);
  }
  loop();

  document.querySelectorAll('[data-cursor="link"]').forEach(el=>{
    el.addEventListener('mouseenter', ()=> ring.classList.add('is-link'));
    el.addEventListener('mouseleave', ()=> ring.classList.remove('is-link'));
  });

  document.addEventListener('mouseleave', ()=>{
    dot.style.opacity = '0'; ring.style.opacity = '0';
  });
  document.addEventListener('mouseenter', ()=>{
    dot.style.opacity = '1'; ring.style.opacity = '1';
  });
}

/* ============================================================ MAGNETIC BUTTONS */
function initMagnetic(){
  if(isTouch) return;
  document.querySelectorAll('.magnetic').forEach(el=>{
    let bounds;
    el.addEventListener('mouseenter', ()=>{ bounds = el.getBoundingClientRect(); });
    el.addEventListener('mousemove', (e)=>{
      if(!bounds) bounds = el.getBoundingClientRect();
      const relX = e.clientX - bounds.left - bounds.width/2;
      const relY = e.clientY - bounds.top - bounds.height/2;
      gsap.to(el, { x: relX * 0.25, y: relY * 0.35, duration: .4, ease:'power2.out' });
    });
    el.addEventListener('mouseleave', ()=>{
      gsap.to(el, { x:0, y:0, duration:.5, ease:'elastic.out(1, 0.4)' });
    });
  });
}

/* ============================================================ NAV */
function initNav(){
  const navbar = document.getElementById('navbar');
  const menuToggle = document.getElementById('menuToggle');
  const mobileMenu = document.getElementById('mobileMenu');

  window.addEventListener('scroll', ()=>{
    navbar.classList.toggle('is-scrolled', window.scrollY > 30);
  }, { passive:true });

  menuToggle.addEventListener('click', ()=>{
    menuToggle.classList.toggle('is-active');
    mobileMenu.classList.toggle('is-open');
  });

  mobileMenu.querySelectorAll('a').forEach(a=>{
    a.addEventListener('click', ()=>{
      menuToggle.classList.remove('is-active');
      mobileMenu.classList.remove('is-open');
    });
  });
}

/* ============================================================ BACK TO TOP */
function initBackToTop(){
  const btn = document.getElementById('backToTop');
  window.addEventListener('scroll', ()=>{
    btn.classList.toggle('is-visible', window.scrollY > 700);
  }, { passive:true });
  btn.addEventListener('click', ()=>{
    window.scrollTo({ top:0, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  });
}

/* ============================================================ AI ASSISTANT WIDGET */
function initAIAssistant(){
  const fab = document.getElementById('aiAssistant');
  const panel = document.getElementById('aiAssistantPanel');
  const closeBtn = document.getElementById('aiPanelClose');

  fab.addEventListener('click', ()=> panel.classList.toggle('is-open'));
  closeBtn.addEventListener('click', ()=> panel.classList.remove('is-open'));

  panel.querySelectorAll('[data-jump]').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      const target = document.querySelector(btn.dataset.jump);
      if(target) target.scrollIntoView({ behavior: prefersReducedMotion ? 'auto':'smooth', block:'start' });
      panel.classList.remove('is-open');
    });
  });

  document.addEventListener('click', (e)=>{
    if(!panel.contains(e.target) && !fab.contains(e.target)){
      panel.classList.remove('is-open');
    }
  });
}

/* ============================================================ ROLE TYPEWRITER */
function initRoleTyper(){
  const el = document.getElementById('roleTyper');
  const roles = ['AI Engineer', 'Full Stack Developer', 'GenAI Developer', 'Cloud Architect'];
  let ri = 0, ci = 0, deleting = false;

  function tick(){
    const word = roles[ri];
    if(!deleting){
      ci++;
      el.textContent = word.slice(0, ci);
      if(ci === word.length){
        deleting = true;
        setTimeout(tick, 1500);
        return;
      }
      setTimeout(tick, 65);
    } else {
      ci--;
      el.textContent = word.slice(0, ci);
      if(ci === 0){
        deleting = false;
        ri = (ri+1) % roles.length;
        setTimeout(tick, 300);
        return;
      }
      setTimeout(tick, 35);
    }
  }
  tick();
}

/* ============================================================ STAT COUNTERS */
function initCounters(){
  const counters = document.querySelectorAll('[data-count]');
  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
      if(entry.isIntersecting){
        const el = entry.target;
        const target = parseInt(el.dataset.count, 10);
        const suffix = el.dataset.suffix || '';
        const obj = { val: 0 };
        gsap.to(obj, {
          val: target, duration: 1.6, ease: 'power2.out',
          onUpdate: ()=>{ el.textContent = Math.round(obj.val) + suffix; }
        });
        obs.unobserve(el);
      }
    });
  }, { threshold: .5 });
  counters.forEach(c=> obs.observe(c));
}

/* ============================================================ FOCUS BARS */
function initFocusBars(){
  const bars = document.querySelectorAll('.focus-fill');
  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
      if(entry.isIntersecting){
        const el = entry.target;
        el.style.width = el.dataset.width + '%';
        obs.unobserve(el);
      }
    });
  }, { threshold: .4 });
  bars.forEach(b=> obs.observe(b));
}

/* ============================================================ TIMELINE PROGRESS LINE */
function initTimelineLine(){
  const fill = document.querySelector('.timeline-line-fill');
  const timeline = document.querySelector('.timeline');
  if(!fill || !timeline || typeof ScrollTrigger === 'undefined') return;
  gsap.to(fill, {
    height: '100%',
    ease: 'none',
    scrollTrigger: {
      trigger: timeline,
      start: 'top 70%',
      end: 'bottom 70%',
      scrub: true
    }
  });
}

/* ============================================================ TILT CARDS */
function initTilt(){
  if(isTouch) return;
  document.querySelectorAll('[data-tilt]').forEach(card=>{
    card.style.transformStyle = 'preserve-3d';
    let bounds;
    card.addEventListener('mouseenter', ()=>{ bounds = card.getBoundingClientRect(); });
    card.addEventListener('mousemove', (e)=>{
      if(!bounds) bounds = card.getBoundingClientRect();
      const px = (e.clientX - bounds.left) / bounds.width;
      const py = (e.clientY - bounds.top) / bounds.height;
      const rx = (py - 0.5) * -8;
      const ry = (px - 0.5) * 10;
      gsap.to(card, { rotateX: rx, rotateY: ry, duration: .4, ease:'power2.out', transformPerspective: 800 });
      card.style.setProperty('--mx', (px*100)+'%');
      card.style.setProperty('--my', (py*100)+'%');
    });
    card.addEventListener('mouseleave', ()=>{
      gsap.to(card, { rotateX:0, rotateY:0, duration:.6, ease:'power3.out' });
    });
  });
}

/* ============================================================ THREE.JS NEURAL NETWORK HERO BACKGROUND */
function initNeuralNetwork(){
  const canvas = document.getElementById('neuralCanvas');
  if(!canvas || typeof THREE === 'undefined') return;

  const hero = document.getElementById('hero');
  let width = hero.clientWidth, height = hero.clientHeight;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, width/height, 0.1, 100);
  camera.position.set(0, 0, 18);

  const teal = new THREE.Color(0x5eead4);
  const violet = new THREE.Color(0xa78bfa);

  const domainCount = 5;
  const nodesPerDomain = 7;
  const totalNodes = domainCount * nodesPerDomain + domainCount;
  const nodePositions = [];
  const nodeIsHub = [];
  const hubPositions = [];

  const radius = 9.5;
  for(let d=0; d<domainCount; d++){
    const angle = (d / domainCount) * Math.PI * 2;
    const hub = new THREE.Vector3(
      Math.cos(angle) * radius * 0.7,
      Math.sin(angle) * radius * 0.45,
      (Math.random()-0.5) * 3
    );
    hubPositions.push(hub);
    nodePositions.push(hub.clone());
    nodeIsHub.push(true);

    for(let i=0; i<nodesPerDomain; i++){
      const off = new THREE.Vector3(
        (Math.random()-0.5) * 6.5,
        (Math.random()-0.5) * 6.5,
        (Math.random()-0.5) * 6.5
      );
      nodePositions.push(hub.clone().add(off));
      nodeIsHub.push(false);
    }
  }

  const posArray = new Float32Array(totalNodes * 3);
  const colorArray = new Float32Array(totalNodes * 3);
  const sizeArray = new Float32Array(totalNodes);

  nodePositions.forEach((p, i)=>{
    posArray[i*3] = p.x; posArray[i*3+1] = p.y; posArray[i*3+2] = p.z;
    const c = nodeIsHub[i] ? teal : (Math.random() > 0.5 ? teal : violet);
    colorArray[i*3] = c.r; colorArray[i*3+1] = c.g; colorArray[i*3+2] = c.b;
    sizeArray[i] = nodeIsHub[i] ? 0.22 : 0.09 + Math.random()*0.05;
  });

  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  pointsGeo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
  pointsGeo.setAttribute('size', new THREE.BufferAttribute(sizeArray, 1));

  const pointsMat = new THREE.PointsMaterial({
    size: 0.18, vertexColors: true, transparent: true, opacity: 0.95,
    sizeAttenuation: true, depthWrite: false, blending: THREE.AdditiveBlending
  });
  const points = new THREE.Points(pointsGeo, pointsMat);
  scene.add(points);

  const linePositions = [];
  let idx = 0;
  for(let d=0; d<domainCount; d++){
    const hubIndex = idx; idx++;
    for(let i=0; i<nodesPerDomain; i++){
      const satIndex = idx; idx++;
      const hp = nodePositions[hubIndex], sp = nodePositions[satIndex];
      linePositions.push(hp.x, hp.y, hp.z, sp.x, sp.y, sp.z);
    }
  }
  for(let d=0; d<domainCount; d++){
    const a = hubPositions[d], b = hubPositions[(d+1)%domainCount];
    linePositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }

  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(linePositions), 3));
  const lineMat = new THREE.LineBasicMaterial({ color: 0x2c3a4d, transparent: true, opacity: 0.45 });
  const lines = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(lines);

  const pulseCount = 6;
  const pulseGeo = new THREE.SphereGeometry(0.07, 8, 8);
  const pulses = [];
  for(let i=0; i<pulseCount; i++){
    const mat = new THREE.MeshBasicMaterial({ color: i % 2 === 0 ? teal : violet, transparent:true, opacity: 0.9 });
    const mesh = new THREE.Mesh(pulseGeo, mat);
    scene.add(mesh);
    pulses.push({
      mesh,
      from: Math.floor(Math.random()*domainCount),
      progress: Math.random(),
      speed: 0.0035 + Math.random()*0.0025
    });
  }

  const dustCount = 220;
  const dustPos = new Float32Array(dustCount * 3);
  for(let i=0; i<dustCount; i++){
    dustPos[i*3] = (Math.random()-0.5) * 40;
    dustPos[i*3+1] = (Math.random()-0.5) * 24;
    dustPos[i*3+2] = (Math.random()-0.5) * 30 - 8;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  const dustMat = new THREE.PointsMaterial({ size: 0.045, color: 0x4a5568, transparent:true, opacity:0.5, depthWrite:false });
  const dust = new THREE.Points(dustGeo, dustMat);
  scene.add(dust);

  let targetRotX = 0, targetRotY = 0;
  let mouseNX = 0, mouseNY = 0;

  function onPointerMove(e){
    const x = (e.clientX !== undefined) ? e.clientX : (e.touches && e.touches[0].clientX);
    const y = (e.clientY !== undefined) ? e.clientY : (e.touches && e.touches[0].clientY);
    if(x === undefined) return;
    mouseNX = (x / window.innerWidth) * 2 - 1;
    mouseNY = (y / window.innerHeight) * 2 - 1;
    targetRotY = mouseNX * 0.35;
    targetRotX = mouseNY * -0.2;
  }
  window.addEventListener('mousemove', onPointerMove, { passive:true });
  window.addEventListener('touchmove', onPointerMove, { passive:true });

  function onResize(){
    width = hero.clientWidth; height = hero.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width/height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', onResize);

  const group = new THREE.Group();
  group.add(points, lines, dust);
  scene.add(group);
  pulses.forEach(p => group.add(p.mesh));

  const clock = new THREE.Clock();
  let rafId;

  function animate(){
    rafId = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    group.rotation.y += (targetRotY - group.rotation.y) * 0.02 + 0.0009;
    group.rotation.x += (targetRotX - group.rotation.x) * 0.02;

    points.material.size = 0.18 + Math.sin(t*1.5) * 0.015;

    pulses.forEach(p=>{
      p.progress += p.speed;
      if(p.progress >= 1){ p.progress = 0; p.from = (p.from+1) % domainCount; }
      const a = hubPositions[p.from];
      const b = hubPositions[(p.from+1) % domainCount];
      p.mesh.position.lerpVectors(a, b, p.progress);
      p.mesh.material.opacity = 0.5 + Math.sin(p.progress * Math.PI) * 0.5;
    });

    dust.rotation.y = t * 0.01;

    renderer.render(scene, camera);
  }

  if(prefersReducedMotion){
    renderer.render(scene, camera);
  } else {
    animate();
  }

  const obs = new IntersectionObserver((entries)=>{
    entries.forEach(entry=>{
      if(!entry.isIntersecting && rafId){ cancelAnimationFrame(rafId); rafId = null; }
      else if(entry.isIntersecting && !rafId && !prefersReducedMotion){ animate(); }
    });
  }, { threshold: 0 });
  obs.observe(hero);
}

/* ============================================================ HERO INTRO ORCHESTRATION (after loader) */
function runHeroIntro(){
  if(typeof gsap === 'undefined') return;
  const heroEls = document.querySelectorAll('#hero .reveal-up');
  gsap.set(heroEls, { opacity:0, y:36 });
  gsap.to(heroEls, {
    opacity:1, y:0, duration: 1, ease:'power3.out', stagger: 0.09, delay: 0.1
  });
  initRoleTyper();
}

/* ============================================================ SCROLL REVEALS (everything below hero) */
function initScrollReveals(){
  if(typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;
  gsap.registerPlugin(ScrollTrigger);

  document.querySelectorAll('.section .reveal-up, .timeline-item.reveal-up').forEach(el=>{
    gsap.fromTo(el, { opacity:0, y:40 }, {
      opacity:1, y:0, duration: .9, ease:'power3.out',
      scrollTrigger: { trigger: el, start: 'top 86%', toggleActions: 'play none none none' }
    });
  });

  // staggered card grids
  [['.project-grid', '.project-card'], ['.skills-grid', '.skill-block'], ['.about-stats', '.stat-card']].forEach(([parentSel, childSel])=>{
    document.querySelectorAll(parentSel).forEach(parent=>{
      const cards = parent.querySelectorAll(childSel);
      gsap.fromTo(cards, { opacity:0, y:50 }, {
        opacity:1, y:0, duration:.8, ease:'power3.out', stagger:0.1,
        scrollTrigger: { trigger: parent, start:'top 85%' }
      });
    });
  });
}

/* ============================================================ CONTACT FORM (EmailJS — sends to bhargavram085@gmail.com) */
function initContactForm(){
  const form = document.getElementById('contactForm');
  if(!form) return;

  form.addEventListener('submit', async function(e){
    e.preventDefault();

    const submitBtn = document.querySelector('.form-submit');
    const submitText = document.getElementById('formSubmitText');
    const formNote = document.getElementById('formNote');

    const name = document.getElementById('cf-name').value.trim();
    const email = document.getElementById('cf-email').value.trim();
    const message = document.getElementById('cf-message').value.trim();

    if(!name || !email || !message){
      formNote.style.color = '#f5b942';
      formNote.textContent = 'Please fill in every field before sending.';
      return;
    }

    submitBtn.disabled = true;
    submitText.textContent = 'Sending...';
    formNote.textContent = '';

    try{
      await emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, {
        from_name: name,
        from_email: email,
        message: message
      });

      formNote.style.color = '#22c55e';
      formNote.textContent = '✅ Message sent successfully!';
      form.reset();
    } catch(error){
      console.error(error);
      formNote.style.color = '#ef4444';
      formNote.textContent = '❌ Failed to send message.';
    } finally {
      submitBtn.disabled = false;
      submitText.textContent = 'Send Message';
    }
  });
}

/* ============================================================ FOOTER YEAR */
function setYear(){
  const y = document.getElementById('year');
  if(y) y.textContent = new Date().getFullYear();
}

/* ============================================================ INIT */
document.addEventListener('DOMContentLoaded', ()=>{
  setYear();
  initCursor();
  initNav();
  initBackToTop();
  initAIAssistant();
  initMagnetic();
  initTilt();
  initCounters();
  initFocusBars();
  initContactForm();
  initNeuralNetwork();

  if(typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined'){
    gsap.registerPlugin(ScrollTrigger);
  }
  initScrollReveals();
  initTimelineLine();

  runLoader();
});

})();
