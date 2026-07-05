(function () {
  const modal = document.querySelector("#experiment-modal");
  const modalTitle = document.querySelector("#modal-title");
  const modalStatus = document.querySelector("#modal-status");
  const modalBody = document.querySelector("#modal-body");
  const modalPanel = document.querySelector(".library-modal-panel");
  const closeButtons = Array.from(document.querySelectorAll("[data-modal-close]"));
  const detailRoot = document.querySelector(".library-details");

  if (!modal || !modalTitle || !modalStatus || !modalBody || !modalPanel || !detailRoot) {
    return;
  }

  let lastFocused = null;
  const pageRoots = Array.from(document.body.children).filter(
    (element) => element !== modal && element.tagName !== "SCRIPT",
  );

  function getExperiment(id) {
    if (!id) return null;
    return detailRoot.querySelector(`#${CSS.escape(id)}`);
  }

  function setPageInert(isInert) {
    pageRoots.forEach((element) => {
      if (isInert) {
        element.setAttribute("aria-hidden", "true");
        element.inert = true;
      } else {
        element.removeAttribute("aria-hidden");
        element.inert = false;
      }
    });
  }

  function focusModalTitle() {
    modalTitle.focus({ preventScroll: true });
  }

  function pauseModalVideos() {
    modal.querySelectorAll("video").forEach((video) => {
      video.pause();
    });
  }

  function focusableElements() {
    return Array.from(
      modalPanel.querySelectorAll(
        "a[href], button:not([disabled]), select:not([disabled]), textarea:not([disabled]), input:not([disabled]), video[controls], [tabindex]:not([tabindex='-1'])",
      ),
    ).filter((element) => element.offsetParent !== null || element === document.activeElement);
  }

  function trapFocus(event) {
    if (event.key !== "Tab" || modal.hidden) return;
    const focusable = focusableElements();
    if (focusable.length === 0) {
      event.preventDefault();
      focusModalTitle();
      return;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function buildBrief(article) {
    const items = [
      ["Resultado corto", article.dataset.summary],
      ["Pregunta", article.dataset.question],
      ["Setup", article.dataset.setup],
      ["Qué medimos", article.dataset.measured],
    ].filter(([, value]) => value);

    if (items.length === 0) return null;

    const brief = document.createElement("div");
    brief.className = "detail-brief";
    items.forEach(([title, value]) => {
      const section = document.createElement("section");
      const heading = document.createElement("h3");
      const text = document.createElement("p");
      heading.textContent = title;
      text.textContent = value;
      section.append(heading, text);
      brief.append(section);
    });
    return brief;
  }

  function renderStatus(article) {
    const status = article.dataset.status || "";
    modalStatus.className = "status-chip";
    modalStatus.textContent = "";
    modalStatus.hidden = true;
    if (!status) return;

    modalStatus.textContent = status;
    modalStatus.classList.add(article.dataset.statusClass || "status-diagnostic");
    modalStatus.hidden = false;
  }

  function renderArticle(article) {
    modalBody.replaceChildren();
    const brief = buildBrief(article);
    if (brief) modalBody.append(brief);

    const content = document.createElement("div");
    content.innerHTML = article.innerHTML;
    const layout = content.querySelector(".detail-layout");
    if (layout && !layout.querySelector(".detail-media")) {
      layout.classList.add("no-media");
    }
    modalBody.append(...Array.from(content.childNodes));
  }

  function openExperiment(id, pushHash) {
    const article = getExperiment(id);
    if (!article) {
      return;
    }

    lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    modalTitle.textContent = article.dataset.title || "Experimento";
    renderStatus(article);
    renderArticle(article);
    modal.hidden = false;
    document.body.classList.add("modal-open");
    setPageInert(true);

    if (pushHash && window.location.hash !== `#${id}`) {
      history.pushState(null, "", `#${id}`);
    }

    window.setTimeout(focusModalTitle, 0);
  }

  function closeModal(clearHash) {
    if (modal.hidden) {
      return;
    }

    pauseModalVideos();
    modal.hidden = true;
    modalBody.replaceChildren();
    document.body.classList.remove("modal-open");
    setPageInert(false);

    if (clearHash && window.location.hash) {
      history.pushState(null, "", `${window.location.pathname}${window.location.search}`);
    }

    if (lastFocused) {
      lastFocused.focus();
    }
  }

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-experiment-card], [data-detail-link]");
    if (!trigger) return;

    const explicitId =
      trigger.getAttribute("data-experiment-card") || trigger.getAttribute("data-detail-link") || "";
    const href = trigger.getAttribute("href") || "";
    const hashId = trigger.hash ? trigger.hash.slice(1) : href.replace(/^#/, "");
    const id = explicitId || hashId;

    if (!getExperiment(id)) return;
    event.preventDefault();
    openExperiment(id, true);
  });

  closeButtons.forEach((button) => {
    button.addEventListener("click", () => closeModal(true));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal(true);
      return;
    }
    trapFocus(event);
  });

  window.addEventListener("popstate", () => {
    const id = window.location.hash.slice(1);
    if (id && getExperiment(id)) {
      openExperiment(id, false);
    } else {
      closeModal(false);
    }
  });

  const initialId = window.location.hash.slice(1);
  if (initialId && getExperiment(initialId)) {
    openExperiment(initialId, false);
  }
})();
