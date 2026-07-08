(() => {
  const chapterSelector = "main > details.chapter-details";
  const stepSelector = ".story-step[id]";
  const presentationRootId = "presentation-view";
  const storageKey = "cool-antz:presentation-edits:v2";

  const body = document.body;
  const header = document.querySelector("body > header");
  const main = document.querySelector("main");
  if (!header || !main) return;

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  const normalizeText = (value) => (value || "").replace(/\s+/g, " ").trim();

  const textFrom = (node, selector) => {
    if (!node) return "";
    const target = node.querySelector(selector);
    return target ? normalizeText(target.textContent) : "";
  };

  const shorten = (value, maxLength = 150) => {
    const text = normalizeText(value).replace(/[.;:]$/, "");
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength).replace(/\s+\S*$/, "")}...`;
  };

  const uniqueTexts = (values) => {
    const seen = new Set();
    return values.filter((value) => {
      const normalized = normalizeText(value);
      const key = normalized.toLowerCase();
      if (!normalized || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
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

  const splitIntoSentences = (text) => {
    const protectedText = normalizeText(text).replace(
      /(\d)\.(\d)/g,
      "$1__decimal_dot__$2",
    );
    return (protectedText.match(/[^.!?]+[.!?]?/g) || [])
      .map((sentence) =>
        normalizeText(sentence)
          .replace(/__decimal_dot__/g, ".")
          .replace(/[.!?]$/, ""),
      )
      .filter(Boolean);
  };

  const candidateSelectors = [
    ".step-copy > p",
    ".experiment-note p",
    ".summary",
    ".abstract p",
    ".takeaway-card p",
    ".metric-card",
    ".result-card",
    ".finding-card p",
    ".runner-panel p",
    "figcaption",
  ];

  const collectTextCandidates = (source) => {
    if (!source) return [];
    const nodes = [];
    candidateSelectors.forEach((selector) => {
      source.querySelectorAll(selector).forEach((node) => nodes.push(node));
    });

    if (nodes.length === 0) {
      source.querySelectorAll("p, li, dd").forEach((node) => nodes.push(node));
    }

    return uniqueTexts(
      nodes
        .map((node) => normalizeText(node.textContent))
        .filter((text) => text.length >= 24),
    );
  };

  const makeDefaultBullets = (slide, limit = 3) => {
    const rejected = new Set(
      [slide.title, slide.chapterTitle, slide.number]
        .filter(Boolean)
        .map((value) => normalizeText(value).toLowerCase()),
    );
    const candidates = collectTextCandidates(slide.source);
    const bullets = [];

    candidates.forEach((candidate) => {
      splitIntoSentences(candidate).forEach((sentence) => {
        const bullet = shorten(sentence, 145);
        const key = bullet.toLowerCase();
        if (bullet.length < 18 || rejected.has(key)) return;
        bullets.push(bullet);
      });
    });

    return uniqueTexts(bullets).slice(0, limit);
  };

  const makeVisual = (slide) => {
    if (!slide.source || slide.kind === "cover" || slide.kind === "chapter-title") {
      return null;
    }

    const visual = slide.source.querySelector(
      [
        ".step-media",
        ".media-frame",
        ".evidence-pair",
        ".large-map-videos",
        ".figure-grid",
        "figure",
        "video",
        "img",
      ].join(", "),
    );
    if (!visual) return null;

    const wrapper = document.createElement("aside");
    wrapper.className = "presentation-visual";
    wrapper.append(cloneClean(visual));
    return wrapper;
  };

  const readStoredEdits = () => {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(storageKey));
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  };

  const writeStoredEdits = (edits) => {
    try {
      if (Object.keys(edits).length === 0) {
        window.localStorage.removeItem(storageKey);
        return;
      }
      window.localStorage.setItem(storageKey, JSON.stringify(edits));
    } catch {
      // Local storage can be unavailable in restricted browsing modes.
    }
  };

  const makeHeaderSlide = () => ({
    id: "cover",
    kind: "cover",
    number: "0",
    title: textFrom(header, "h1") || document.title,
    source: header,
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
    };
  };

  const makeChapterBodySlide = (chapter) => ({
    id: `${chapter.id || "chapter"}-body`,
    kind: "chapter-body",
    number: textFrom(chapter, ".chapter-number") || textFrom(chapter, ".kicker"),
    title: textFrom(chapter, "h2"),
    source: chapter,
  });

  const makeStepSlide = (step, chapter) => ({
    id: step.id,
    kind: "step",
    number: textFrom(step, ".step-index"),
    title: textFrom(step, "h3"),
    chapterTitle: textFrom(chapter, "h2"),
    source: step,
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

  const makeDefaultContent = (slide) => {
    const authors = textFrom(header, ".authors");
    const subtitle = textFrom(header, ".subtitle");
    const candidates = collectTextCandidates(slide.source);
    const fallbackLede = candidates.length > 0 ? shorten(candidates[0], 135) : "";
    const bullets = makeDefaultBullets(slide, slide.kind === "chapter-title" ? 2 : 3);

    if (slide.kind === "cover") {
      return {
        kicker: subtitle || "Proyecto final de aprendizaje por refuerzo",
        title: slide.title,
        lede: authors,
        bullets: [],
      };
    }

    if (slide.kind === "chapter-title") {
      return {
        kicker: slide.number,
        title: slide.title,
        lede: fallbackLede,
        bullets,
      };
    }

    return {
      kicker: [slide.number, slide.chapterTitle].filter(Boolean).join(" / "),
      title: slide.title,
      lede: slide.kind === "lead" || slide.kind === "chapter-body" ? fallbackLede : "",
      bullets,
    };
  };

  const slides = collectSlides();
  if (slides.length === 0) return;

  const defaultContent = new Map(
    slides.map((slide) => [slide.id, makeDefaultContent(slide)]),
  );
  const storedEdits = readStoredEdits();

  const contentForSlide = (slide) => {
    const defaults = defaultContent.get(slide.id) || makeDefaultContent(slide);
    const edits = storedEdits[slide.id] || {};
    return {
      kicker: edits.kicker ?? defaults.kicker,
      title: edits.title ?? defaults.title,
      lede: edits.lede ?? defaults.lede,
      bullets: Array.isArray(edits.bullets) ? edits.bullets : defaults.bullets,
    };
  };

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
        <button type="button" class="presentation-button" data-presentation-edit>Editar</button>
        <button type="button" class="presentation-button presentation-edit-only" data-presentation-add-bullet hidden>+ bullet</button>
        <button type="button" class="presentation-button presentation-edit-only" data-presentation-reset hidden>Restaurar</button>
        <button type="button" class="presentation-button" data-presentation-next aria-label="Diapositiva siguiente">Siguiente</button>
        <button type="button" class="presentation-close" data-presentation-close aria-label="Salir de presentacion">Salir</button>
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
  const editButton = root.querySelector("[data-presentation-edit]");
  const addBulletButton = root.querySelector("[data-presentation-add-bullet]");
  const resetButton = root.querySelector("[data-presentation-reset]");
  const closeButton = root.querySelector("[data-presentation-close]");
  const controls = [
    prevButton,
    editButton,
    addBulletButton,
    resetButton,
    nextButton,
    closeButton,
  ];

  let currentIndex = 0;
  let lastFocused = null;
  let scrollY = 0;
  let editMode = false;

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

  const editableText = (tagName, className, value, field, placeholder) => {
    const node = document.createElement(tagName);
    node.className = className;
    node.dataset.editField = field;
    node.dataset.placeholder = placeholder;
    node.textContent = value;
    node.contentEditable = editMode ? "true" : "false";
    node.spellcheck = editMode;
    return node;
  };

  const editableBullet = (value = "") => {
    const item = document.createElement("li");
    item.dataset.editBullet = "true";
    item.dataset.placeholder = "Nuevo bullet";
    item.textContent = value;
    item.contentEditable = editMode ? "true" : "false";
    item.spellcheck = editMode;
    return item;
  };

  const renderSlideContent = (slide) => {
    const content = contentForSlide(slide);
    const card = document.createElement("article");
    card.className = `presentation-deck-card presentation-deck-${slide.kind}`;
    if (editMode) card.classList.add("is-editing");

    const copy = document.createElement("div");
    copy.className = "presentation-deck-copy";

    copy.append(
      editableText("p", "presentation-deck-kicker", content.kicker, "kicker", "Kicker"),
      editableText("h2", "presentation-deck-title", content.title, "title", "Titulo"),
    );

    if (content.lede || editMode) {
      copy.append(
        editableText(
          "p",
          "presentation-deck-lede",
          content.lede,
          "lede",
          "Frase de apertura",
        ),
      );
    }

    const bullets = editMode && content.bullets.length === 0 ? [""] : content.bullets;
    if (bullets.length > 0) {
      const list = document.createElement("ul");
      list.className = "presentation-bullet-list";
      bullets.forEach((bullet) => list.append(editableBullet(bullet)));
      copy.append(list);
    }

    const visual = makeVisual(slide);
    if (visual) card.classList.add("has-visual");
    card.append(copy);
    if (visual) card.append(visual);
    return card;
  };

  const getEditedContentFromFrame = () => {
    const field = (name) =>
      normalizeText(frame.querySelector(`[data-edit-field="${name}"]`)?.textContent);
    const bullets = Array.from(frame.querySelectorAll("[data-edit-bullet]"))
      .map((item) => normalizeText(item.textContent))
      .filter(Boolean);
    return {
      kicker: field("kicker"),
      title: field("title"),
      lede: field("lede"),
      bullets,
    };
  };

  const contentMatchesDefault = (slide, content) => {
    const defaults = defaultContent.get(slide.id) || makeDefaultContent(slide);
    const sameText =
      normalizeText(content.kicker) === normalizeText(defaults.kicker) &&
      normalizeText(content.title) === normalizeText(defaults.title) &&
      normalizeText(content.lede) === normalizeText(defaults.lede);
    const contentBullets = content.bullets.map(normalizeText);
    const defaultBullets = defaults.bullets.map(normalizeText);
    const sameBullets =
      contentBullets.length === defaultBullets.length &&
      contentBullets.every((bullet, index) => bullet === defaultBullets[index]);
    return sameText && sameBullets;
  };

  const saveCurrentSlide = () => {
    if (!editMode || frame.children.length === 0) return;
    const slide = slides[currentIndex];
    const content = getEditedContentFromFrame();
    if (contentMatchesDefault(slide, content)) {
      delete storedEdits[slide.id];
    } else {
      storedEdits[slide.id] = content;
    }
    writeStoredEdits(storedEdits);
  };

  const syncEditControls = () => {
    root.classList.toggle("is-editing", editMode);
    editButton.textContent = editMode ? "Listo" : "Editar";
    addBulletButton.hidden = !editMode;
    resetButton.hidden = !editMode;
  };

  const renderSlide = () => {
    const slide = slides[currentIndex];
    const content = contentForSlide(slide);
    frame.replaceChildren();
    frame.scrollTop = 0;
    frame.dataset.slideKind = slide.kind;
    frame.append(renderSlideContent(slide));

    counter.textContent = `${currentIndex + 1} / ${slides.length}`;
    title.textContent = [content.kicker, content.title].filter(Boolean).join(" - ");
    progress.style.inlineSize = `${((currentIndex + 1) / slides.length) * 100}%`;
    prevButton.disabled = currentIndex === 0;
    nextButton.disabled = currentIndex === slides.length - 1;
    syncEditControls();

    frame.querySelectorAll("video[autoplay]").forEach((video) => {
      if (prefersReducedMotion) return;
      video.play().catch(() => {});
    });
  };

  const goTo = (index) => {
    saveCurrentSlide();
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
    saveCurrentSlide();
    root.hidden = true;
    body.classList.remove("presentation-open");
    window.scrollTo({ top: scrollY, behavior: "auto" });
    if (lastFocused && typeof lastFocused.focus === "function") {
      lastFocused.focus({ preventScroll: true });
    }
  };

  const focusAdjacentControl = (direction) => {
    const enabledControls = controls.filter((control) => !control.hidden && !control.disabled);
    const activeIndex = enabledControls.indexOf(document.activeElement);
    const fallback = direction > 0 ? 0 : enabledControls.length - 1;
    const nextIndex =
      activeIndex >= 0
        ? (activeIndex + direction + enabledControls.length) % enabledControls.length
        : fallback;
    enabledControls[nextIndex].focus();
  };

  const appendBullet = () => {
    let list = frame.querySelector(".presentation-bullet-list");
    if (!list) {
      list = document.createElement("ul");
      list.className = "presentation-bullet-list";
      frame.querySelector(".presentation-deck-copy")?.append(list);
    }
    const item = editableBullet("");
    list.append(item);
    item.focus();
    saveCurrentSlide();
  };

  launchButton.addEventListener("click", () => openPresentation());
  prevButton.addEventListener("click", () => goTo(currentIndex - 1));
  nextButton.addEventListener("click", () => goTo(currentIndex + 1));
  closeButton.addEventListener("click", closePresentation);

  editButton.addEventListener("click", () => {
    saveCurrentSlide();
    editMode = !editMode;
    renderSlide();
    if (editMode) {
      frame.querySelector("[data-edit-field='title']")?.focus();
    } else {
      frame.focus({ preventScroll: true });
    }
  });

  addBulletButton.addEventListener("click", appendBullet);

  resetButton.addEventListener("click", () => {
    const slide = slides[currentIndex];
    delete storedEdits[slide.id];
    writeStoredEdits(storedEdits);
    renderSlide();
  });

  frame.addEventListener("input", () => {
    if (editMode) saveCurrentSlide();
  });

  frame.addEventListener("keydown", (event) => {
    if (!editMode || !document.activeElement?.matches("[data-edit-bullet]")) return;
    const item = document.activeElement;
    if (event.key === "Enter") {
      event.preventDefault();
      const next = editableBullet("");
      item.after(next);
      next.focus();
      saveCurrentSlide();
    } else if (event.key === "Backspace" && normalizeText(item.textContent) === "") {
      const previous = item.previousElementSibling || item.nextElementSibling;
      if (previous) {
        event.preventDefault();
        item.remove();
        previous.focus();
        saveCurrentSlide();
      }
    }
  });

  root.addEventListener("click", (event) => {
    if (event.target === root) closePresentation();
  });

  document.addEventListener("keydown", (event) => {
    if (root.hidden) return;
    const active = document.activeElement;
    const tagName = active?.tagName?.toLowerCase();
    const isTyping =
      active?.isContentEditable ||
      tagName === "input" ||
      tagName === "textarea" ||
      tagName === "select";
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
    } else if (event.key === "Tab" && !editMode) {
      event.preventDefault();
      focusAdjacentControl(event.shiftKey ? -1 : 1);
    }
  });
})();
