/* ReWorld project page — behavior
   vanilla JS: hero scroll handoff, lazy video wall, lightbox, bibtex copy */
(function () {
  'use strict';

  var $$ = function (sel, el) { return Array.prototype.slice.call((el || document).querySelectorAll(sel)); };
  var reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- nav + hero scroll handoff ---------- */
  var nav = document.getElementById('nav');
  var hero = document.getElementById('hero');
  var heroMedia = document.getElementById('heroMedia');
  var heroVideo = document.getElementById('heroVideo');
  var heroInner = document.getElementById('heroInner');
  var scrollCue = document.getElementById('scrollCue');
  var heroDone = false;
  var ticking = false;

  function heroUpdate() {
    ticking = false;
    var y = window.scrollY || window.pageYOffset;
    var vh = window.innerHeight || 800;
    if (nav) nav.classList.toggle('scrolled', y > 40);

    var p = Math.min(1, y / (vh * 0.85));
    if (!reducedMotion) {
      if (heroInner) {
        heroInner.style.opacity = String(Math.max(0, 1 - p * 1.45));
        heroInner.style.transform = 'translateY(' + (-p * 64) + 'px)';
      }
      if (heroMedia) heroMedia.style.transform = 'scale(' + (1 + p * 0.12) + ')';
    }
    if (hero) hero.style.setProperty('--dim', String(Math.min(0.72, p * 0.72)));
    if (scrollCue) scrollCue.style.opacity = String(Math.max(0, 1 - p * 2.4));

    if (heroVideo) {
      var shouldPause = p >= 0.98 || reducedMotion || document.hidden;
      if (shouldPause && !heroVideo.paused) { heroVideo.pause(); heroDone = true; }
      else if (!shouldPause && heroVideo.paused && heroDone) {
        heroVideo.play().catch(function () {});
        heroDone = false;
      }
    }
  }
  function onScroll() {
    if (!ticking) { ticking = true; window.requestAnimationFrame(heroUpdate); }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  document.addEventListener('visibilitychange', heroUpdate);
  heroUpdate();

  if (heroVideo && !reducedMotion) heroVideo.play().catch(function () {});
  if (heroVideo && reducedMotion) heroVideo.pause();

  /* ---------- lazy video wall with playback cap ---------- */
  var MAX_PLAYING = 8;
  var wallVideos = $$('.js-lazy-video');
  var visibleSet = new Set();

  function syncPlayback() {
    var playing = 0;
    for (var i = 0; i < wallVideos.length; i++) {
      var v = wallVideos[i];
      if (visibleSet.has(v) && playing < MAX_PLAYING) {
        playing++;
        if (v.paused) v.play().catch(function () {});
      } else if (!v.paused) {
        v.pause();
      }
    }
  }

  if ('IntersectionObserver' in window) {
    var vio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) visibleSet.add(e.target);
        else visibleSet.delete(e.target);
      });
      syncPlayback();
    }, { threshold: 0.22 });
    wallVideos.forEach(function (v) { vio.observe(v); });
  } else {
    wallVideos.forEach(function (v) { v.setAttribute('controls', ''); });
  }

  /* ---------- reveal on scroll ---------- */
  var revealEls = $$('.reveal');
  if (reducedMotion || !('IntersectionObserver' in window)) {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  } else {
    var rio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); rio.unobserve(e.target); }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -4% 0px' });
    revealEls.forEach(function (el) { rio.observe(el); });
  }

  /* ---------- lightbox ---------- */
  var lb = document.getElementById('lightbox');
  var lbVideo = document.getElementById('lbVideo');
  var lbTitle = document.getElementById('lbTitle');
  var lbNote = document.getElementById('lbNote');
  var lbPrompt = document.getElementById('lbPrompt');
  var lbPromptText = document.getElementById('lbPromptText');
  var lbClose = document.getElementById('lbClose');
  var lastFocus = null;

  function openLightbox(card, focusEl) {
    if (!lb) return;
    lastFocus = focusEl || card;
    lbTitle.textContent = card.getAttribute('data-title') || '';
    lbNote.textContent = card.getAttribute('data-note') || '';
    var prompt = card.getAttribute('data-prompt') || '';
    if (lbPrompt) {
      lbPrompt.hidden = !prompt;
      if (lbPromptText) lbPromptText.textContent = prompt;
    }
    lbVideo.src = card.getAttribute('data-video');
    lb.classList.add('open');
    document.body.style.overflow = 'hidden';
    lbVideo.play().catch(function () {});
    lbClose.focus();
  }
  function closeLightbox() {
    if (!lb) return;
    lb.classList.remove('open');
    lbVideo.pause();
    lbVideo.removeAttribute('src');
    lbVideo.load();
    document.body.style.overflow = '';
    if (lastFocus) lastFocus.focus();
  }

  /* only the expand button opens the lightbox — the video body is inert */
  $$('.expand-btn').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      var card = btn.closest('[data-video]');
      if (card) openLightbox(card, btn);
    });
  });

  /* ---------- touch: tap a card to toggle its prompt veil ----------
     No hover on phones, so the first tap shows the prompt overlay and
     a second tap hides it. The expand button always wins (lightbox). */
  if (window.matchMedia('(hover: none)').matches) {
    var veilCards = $$('.vcard[data-prompt], .longrow .frame[data-prompt]');
    veilCards.forEach(function (card) {
      card.addEventListener('click', function (e) {
        if (e.target.closest('.expand-btn')) return;
        var wasOn = card.classList.contains('veil-on');
        veilCards.forEach(function (c) { c.classList.remove('veil-on'); });
        if (!wasOn) card.classList.add('veil-on');
      });
    });
  }

  /* ---------- method figure: dismiss the mobile scroll hint ---------- */
  var figScroll = document.getElementById('figScroll');
  var figHint = document.getElementById('figHint');
  if (figScroll && figHint) {
    figScroll.addEventListener('scroll', function () {
      figHint.classList.add('seen');
    }, { passive: true, once: true });
  }
  if (lb) {
    lb.addEventListener('click', function (e) {
      if (e.target === lb) closeLightbox();
    });
    lbClose.addEventListener('click', closeLightbox);
    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && lb.classList.contains('open')) closeLightbox();
    });
  }

  /* ---------- bibtex copy ---------- */
  var copyBtn = document.getElementById('copyBib');
  var bib = document.getElementById('bibtex');
  if (copyBtn && bib) {
    copyBtn.addEventListener('click', function () {
      var text = bib.textContent;
      function done() {
        var old = copyBtn.textContent;
        copyBtn.textContent = 'Copied ✓';
        setTimeout(function () { copyBtn.textContent = old; }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () {});
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (err) {}
        document.body.removeChild(ta);
      }
    });
  }
})();
