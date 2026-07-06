(() => {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  const getTarget = (hash) => {
    if (!hash || hash === "#") return null;

    try {
      return document.getElementById(decodeURIComponent(hash.slice(1)));
    } catch {
      return null;
    }
  };

  const openChapterFor = (target) => {
    const chapter = target.closest("details.chapter-details");
    if (chapter && !chapter.open) {
      chapter.open = true;
    }
  };

  const scrollToHash = (hash) => {
    const target = getTarget(hash);
    if (!target) return;

    openChapterFor(target);
    requestAnimationFrame(() => {
      target.scrollIntoView({
        block: "start",
        behavior: prefersReducedMotion ? "auto" : "smooth",
      });
    });
  };

  document.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-open-target]");
    if (!link) return;

    const url = new URL(link.href, window.location.href);
    if (url.origin !== window.location.origin || url.pathname !== window.location.pathname) {
      return;
    }

    const target = getTarget(url.hash);
    if (!target) return;

    event.preventDefault();
    history.pushState(null, "", url.hash);
    scrollToHash(url.hash);
  });

  window.addEventListener("hashchange", () => scrollToHash(window.location.hash));

  if (window.location.hash) {
    scrollToHash(window.location.hash);
  }
})();
