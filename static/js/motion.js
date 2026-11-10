/**
 * Motion & Animation Controller Module.
 * Implements smooth scroll-spy, number count-ups, section reveals,
 * toasts, and strict adherence to prefers-reduced-motion.
 */

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  if (!sections.length || !navLinks.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navLinks.forEach(link => {
          const href = link.getAttribute('href');
          if (href === `#${id}`) {
            link.classList.add('active');
          } else {
            link.classList.remove('active');
          }
        });
      }
    });
  }, {
    rootMargin: '-20% 0px -70% 0px',
    threshold: 0.1
  });

  sections.forEach(sec => observer.observe(sec));
}

export function animateCountUp(element, targetValue, duration = 900) {
  if (!element) return;
  if (prefersReducedMotion) {
    element.textContent = targetValue;
    return;
  }

  const start = 0;
  const startTime = performance.now();
  const isFloat = String(targetValue).includes('.');
  const numTarget = parseFloat(targetValue);

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Cubic ease-out
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = start + (numTarget - start) * easeProgress;

    element.textContent = isFloat ? current.toFixed(1) : Math.round(current);

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = targetValue;
    }
  }

  requestAnimationFrame(update);
}

export function initMetricCountUps() {
  const countElements = document.querySelectorAll('[data-count-target]');
  if (!countElements.length) return;

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const target = entry.target.getAttribute('data-count-target');
        animateCountUp(entry.target, target);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  countElements.forEach(el => observer.observe(el));
}

export function showToast(message, type = 'info', duration = 3500) {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.setAttribute('role', 'status');
  toast.setAttribute('aria-live', 'polite');
  toast.innerHTML = `<span>${message}</span>`;

  container.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => toast.classList.add('show'));

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

export function initBackToTop() {
  const btn = document.querySelector('.back-to-top');
  if (!btn) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 400) {
      btn.classList.add('show');
    } else {
      btn.classList.remove('show');
    }
  });

  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
  });
}

export function handleDeepLinking() {
  const hash = window.location.hash;
  if (hash) {
    const target = document.querySelector(hash);
    if (target) {
      setTimeout(() => {
        target.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
      }, 100);
    }
  }
}

