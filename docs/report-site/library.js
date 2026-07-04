(function () {
  const modal = document.querySelector("#experiment-modal");
  const modalTitle = document.querySelector("#modal-title");
  const modalBody = document.querySelector("#modal-body");
  const closeButtons = Array.from(document.querySelectorAll("[data-modal-close]"));
  const detailRoot = document.querySelector(".library-details");

  if (!modal || !modalTitle || !modalBody || !detailRoot) {
    return;
  }

  let lastFocused = null;

  function getExperiment(id) {
    if (!id) return null;
    return detailRoot.querySelector(`#${CSS.escape(id)}`);
  }

  function focusFirstControl() {
    const target = modal.querySelector("button, [href], video, [tabindex]:not([tabindex='-1'])");
    if (target) {
      target.focus();
      return;
    }
    modal.focus();
  }

  function pauseModalVideos() {
    modal.querySelectorAll("video").forEach((video) => {
      video.pause();
    });
  }

  function openExperiment(id, pushHash) {
    const article = getExperiment(id);
    if (!article) {
      return;
    }

    lastFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    modalTitle.textContent = article.dataset.title || "Experimento";
    modalBody.innerHTML = article.innerHTML;
    modal.hidden = false;
    document.body.classList.add("modal-open");

    if (pushHash && window.location.hash !== `#${id}`) {
      history.pushState(null, "", `#${id}`);
    }

    window.setTimeout(focusFirstControl, 0);
  }

  function closeModal(clearHash) {
    if (modal.hidden) {
      return;
    }

    pauseModalVideos();
    modal.hidden = true;
    modalBody.innerHTML = "";
    document.body.classList.remove("modal-open");

    if (clearHash && window.location.hash) {
      history.pushState(null, "", `${window.location.pathname}${window.location.search}`);
    }

    if (lastFocused) {
      lastFocused.focus();
    }
  }

  document.addEventListener("click", (event) => {
    const trigger = event.target.closest("[data-experiment-card], a[href^='#']");
    if (!trigger) return;

    const explicitId = trigger.dataset ? trigger.dataset.experimentCard : "";
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
    }
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
