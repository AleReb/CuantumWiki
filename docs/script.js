/* ============================================================
   CuantumWiki — Interactive Script
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Sidebar navigation: active state on scroll ──
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.section[id]');

  function updateActiveNav() {
    const scrollY = window.scrollY + 120;
    let current = '';

    sections.forEach(section => {
      if (section.offsetTop <= scrollY) {
        current = section.id;
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current) {
        link.classList.add('active');
      }
    });
  }

  window.addEventListener('scroll', updateActiveNav);
  updateActiveNav();

  // ── Smooth scroll on nav click ──
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        const offset = 80;
        const y = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: y, behavior: 'smooth' });
      }

      // Close sidebar on mobile
      const sidebar = document.querySelector('.sidebar');
      if (sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  });

  // ── Mobile menu toggle ──
  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');

  if (menuToggle) {
    menuToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (sidebar.classList.contains('open') &&
          !sidebar.contains(e.target) &&
          !menuToggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // ── Copy to clipboard for code blocks ──
  document.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copiar';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      if (code) {
        navigator.clipboard.writeText(code.textContent).then(() => {
          btn.textContent = '✓ Copiado';
          btn.classList.add('copied');
          setTimeout(() => {
            btn.textContent = 'Copiar';
            btn.classList.remove('copied');
          }, 2000);
        });
      }
    });
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });

  // ── Animate sections on scroll (IntersectionObserver) ──
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animationDelay = '0s';
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  sections.forEach(section => observer.observe(section));

  // ── Flow diagram hover effect ──
  document.querySelectorAll('.flow-node').forEach(node => {
    node.addEventListener('mouseenter', () => {
      node.style.transform = 'translateY(-4px) scale(1.02)';
    });
    node.addEventListener('mouseleave', () => {
      node.style.transform = '';
    });
  });

});
