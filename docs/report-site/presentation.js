(() => {
  const chapterSelector = "main > details.chapter-details";
  const stepSelector = ".story-step[id]";
  const presentationRootId = "presentation-view";

  const body = document.body;
  const header = document.querySelector("body > header");
  const main = document.querySelector("main");
  if (!header || !main) return;

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  const textFrom = (node, selector) => {
    const target = node.querySelector(selector);
    return target ? target.textContent.trim().replace(/\s+/g, " ") : "";
  };

  const stripDuplicateReferences = (root) => {
    const attrsToRemove = [
      "id",
      "for",
      "aria-labelledby",
      "aria-describedby",
      "aria-controls",
    ];
    root.querySelectorAll("*").forEach((node) => {
      attrsToRemove.forEach((attr) => node.removeAttribute(attr));
    });
    attrsToRemove.forEach((attr) => root.removeAttribute(attr));
    return root;
  };

  const prepareMedia = (root) => {
    root.querySelectorAll("video").forEach((video) => {
      video.controls = true;
      video.muted = true;
      video.playsInline = true;
      video.preload = "metadata";
    });
  };

  const cloneClean = (source) => {
    const clone = source.cloneNode(true);
    stripDuplicateReferences(clone);
    prepareMedia(clone);
    return clone;
  };

  const cloneChapterBody = (chapter) => {
    const bodyClone = document.createElement("div");
    bodyClone.className = "presentation-chapter-body";
    Array.from(chapter.children)
      .filter((child) => child.tagName.toLowerCase() !== "summary")
      .forEach((child) => bodyClone.append(cloneClean(child)));
    return bodyClone;
  };

  const makeHeaderSlide = () => ({
    id: "cover",
    kind: "cover",
    number: "0",
    title: textFrom(header, "h1") || document.title,
    source: header,
    render() {
      const clone = cloneClean(header);
      clone.classList.add("presentation-cover-card");
      return clone;
    },
  });

  const makeLeadSlide = () => {
    const lead = document.querySelector(".lead-section.project-overview");
    if (!lead) return null;
    return {
      id: "project-overview",
      kind: "lead",
      number: "Intro",
      title: textFrom(lead, "h2"),
      source: lead,
      render() {
        const clone = cloneClean(lead);
        clone.classList.add("presentation-lead-card");
        return clone;
      },
    };
  };

  const makeChapterTitleSlide = (chapter) => {
    const summary = chapter.querySelector(":scope > summary");
    return {
      id: `${chapter.id || "chapter"}-title`,
      kind: "chapter-title",
      number:
        textFrom(summary || chapter, ".chapter-number") ||
        textFrom(summary || chapter, ".kicker"),
      title: textFrom(summary || chapter, "h2"),
      source: summary || chapter,
      render() {
        const wrapper = document.createElement("section");
        wrapper.className = "presentation-chapter-title";
        if (summary) {
          wrapper.append(cloneClean(summary));
        }
        return wrapper;
      },
    };
  };

  const makeChapterBodySlide = (chapter) => ({
    id: `${chapter.id || "chapter"}-body`,
    kind: "chapter-body",
    number: textFrom(chapter, ".chapter-number") || textFrom(chapter, ".kicker"),
    title: textFrom(chapter, "h2"),
    source: chapter,
    render() {
      return cloneChapterBody(chapter);
    },
  });

  const makeStepSlide = (step, chapter) => ({
    id: step.id,
    kind: "step",
    number: textFrom(step, ".step-index"),
    title: textFrom(step, "h3"),
    chapterTitle: textFrom(chapter, "h2"),
    source: step,
    render() {
      const clone = cloneClean(step);
      clone.classList.add("presentation-step-card");
      return clone;
    },
  });

  const collectSlides = () => {
    const slides = [makeHeaderSlide()];
    const lead = makeLeadSlide();
    if (lead) slides.push(lead);

    document.querySelectorAll(chapterSelector).forEach((chapter) => {
      slides.push(makeChapterTitleSlide(chapter));
      const steps = Array.from(chapter.querySelectorAll(stepSelector));
      if (steps.length > 0) {
        steps.forEach((step) => slides.push(makeStepSlide(step, chapter)));
      } else {
        slides.push(makeChapterBodySlide(chapter));
      }
    });

    return slides.filter(Boolean);
  };

  const slides = collectSlides();
  if (slides.length === 0) return;

  const root = document.createElement("div");
  root.id = presentationRootId;
  root.className = "presentation-view";
  root.hidden = true;
  root.innerHTML = `
    <div class="presentation-shell" role="dialog" aria-modal="true" aria-labelledby="presentation-title">
      <div class="presentation-topbar">
        <button type="button" class="presentation-button" data-presentation-prev aria-label="Diapositiva anterior">Anterior</button>
        <div class="presentation-status" aria-live="polite">
          <span class="presentation-counter"></span>
          <strong id="presentation-title"></strong>
        </div>
        <button type="button" class="presentation-button" data-presentation-next aria-label="Diapositiva siguiente">Siguiente</button>
        <button type="button" class="presentation-close" data-presentation-close aria-label="Salir de presentación">Salir</button>
      </div>
      <div class="presentation-progress" aria-hidden="true"><span></span></div>
      <div class="presentation-slide-frame" tabindex="-1"></div>
    </div>
  `;

  const launchButton = document.createElement("button");
  launchButton.type = "button";
  launchButton.className = "presentation-launch";
  launchButton.textContent = "Presentar";
  launchButton.setAttribute("aria-haspopup", "dialog");
  launchButton.setAttribute("aria-controls", presentationRootId);

  document.body.append(launchButton, root);

  const frame = root.querySelector(".presentation-slide-frame");
  const counter = root.querySelector(".presentation-counter");
  const title = root.querySelector("#presentation-title");
  const progress = root.querySelector(".presentation-progress span");
  const prevButton = root.querySelector("[data-presentation-prev]");
  const nextButton = root.querySelector("[data-presentation-next]");
  const closeButton = root.querySelector("[data-presentation-close]");
  const controls = [prevButton, nextButton, closeButton];

  let currentIndex = 0;
  let lastFocused = null;
  let scrollY = 0;

  const slideIndexForHash = () => {
    if (!window.location.hash) return 0;
    const id = decodeURIComponent(window.location.hash.slice(1));
    const direct = slides.findIndex((slide) => slide.id === id);
    if (direct >= 0) return direct;
    const nested = slides.findIndex(
      (slide) => slide.source && slide.source.id === id,
    );
    return nested >= 0 ? nested : 0;
  };

  const renderSlide = () => {
    const slide = slides[currentIndex];
    frame.replaceChildren();
    frame.scrollTop = 0;
    frame.dataset.slideKind = slide.kind;
    frame.append(slide.render());

    counter.textContent = `${currentIndex + 1} / ${slides.length}`;
    title.textContent = [slide.number, slide.title].filter(Boolean).join(" - ");
    progress.style.inlineSize = `${((currentIndex + 1) / slides.length) * 100}%`;
    prevButton.disabled = currentIndex === 0;
    nextButton.disabled = currentIndex === slides.length - 1;

    frame.querySelectorAll("video[autoplay]").forEach((video) => {
      if (prefersReducedMotion) return;
      video.play().catch(() => {});
    });
  };

  const goTo = (index) => {
    currentIndex = Math.min(Math.max(index, 0), slides.length - 1);
    renderSlide();
    frame.focus({ preventScroll: true });
  };

  const openPresentation = (index = slideIndexForHash()) => {
    lastFocused = document.activeElement;
    scrollY = window.scrollY;
    currentIndex = index;
    root.hidden = false;
    body.classList.add("presentation-open");
    renderSlide();
    frame.focus({ preventScroll: true });
  };

  const closePresentation = () => {
    root.hidden = true;
    body.classList.remove("presentation-open");
    window.scrollTo({ top: scrollY, behavior: "auto" });
    if (lastFocused && typeof lastFocused.focus === "function") {
      lastFocused.focus({ preventScroll: true });
    }
  };

  const focusAdjacentControl = (direction) => {
    const enabledControls = controls.filter((control) => !control.disabled);
    const activeIndex = enabledControls.indexOf(document.activeElement);
    const fallback = direction > 0 ? 0 : enabledControls.length - 1;
    const nextIndex =
      activeIndex >= 0
        ? (activeIndex + direction + enabledControls.length) % enabledControls.length
        : fallback;
    enabledControls[nextIndex].focus();
  };

  launchButton.addEventListener("click", () => openPresentation());
  prevButton.addEventListener("click", () => goTo(currentIndex - 1));
  nextButton.addEventListener("click", () => goTo(currentIndex + 1));
  closeButton.addEventListener("click", closePresentation);

  root.addEventListener("click", (event) => {
    if (event.target === root) closePresentation();
  });

  document.addEventListener("keydown", (event) => {
    if (root.hidden) return;
    const tagName = document.activeElement?.tagName?.toLowerCase();
    const isTyping =
      tagName === "input" || tagName === "textarea" || tagName === "select";
    if (isTyping) return;

    if (event.key === "Escape") {
      event.preventDefault();
      closePresentation();
    } else if (
      event.key === "ArrowRight" ||
      event.key === "PageDown" ||
      event.key === " "
    ) {
      event.preventDefault();
      goTo(currentIndex + 1);
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      goTo(currentIndex - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      goTo(0);
    } else if (event.key === "End") {
      event.preventDefault();
      goTo(slides.length - 1);
    } else if (event.key === "Tab") {
      event.preventDefault();
      focusAdjacentControl(event.shiftKey ? -1 : 1);
    }
  });
})();
