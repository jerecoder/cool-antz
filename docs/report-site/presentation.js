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

  const asset = (path) => `report-site/assets/${path}`;

  const bulletStepContent = {
    "vision-radius-curriculum": {
      groups: [
        {
          bullets: [
            "Usar el radio de vision como curriculo: empezar con una politica que observa casi todo el tablero y despues exigirle operar con observacion local.",
            "La hipotesis inicial era que ver todas las galletas hacia mas facil aprender foraging antes de achicar la ventana.",
            "Radio <code>25</code> implica una observacion <code>51x51</code>: <code>2601</code> celdas con comida, hormigas, bits, hub, borde/obstaculos y senales propias.",
            "El critic centralizado estima valor durante entrenamiento, pero las acciones ejecutadas salen del actor local de cada hormiga.",
          ],
        },
        {
          heading: "Transferencia entre radios",
          bullets: [
            "En la version densa, al pasar de <code>51x51</code> a <code>41x41</code>, se copiaba el sub-tensor central de la primera capa.",
            "Las capas internas, heads de accion y critic quedaban como estaban; solo cambiaba la entrada compatible con el nuevo radio.",
            "Tambien se probo un actor convolucional para leer la grilla con filtros compartidos antes del MLP.",
          ],
        },
        {
          heading: "Lectura",
          bullets: [
            "La ventana <code>51x51</code> no era una pista pequena: agrandaba el espacio visual sin resolver exploracion ni asignacion de credito.",
            "Incluso con entorno denso en galletas, no aparecio una conducta greedy de pickup estable ni retorno confiable al hub.",
            "La etapa de vision completa no produjo una base estable para transferir.",
          ],
        },
      ],
      media: [
        {
          type: "image",
          src: asset("plots/fig_vision_range_51x51_reward.png"),
          alt: "Retorno real durante el entrenamiento con radio de vision 25",
          caption:
            "<strong>Radio 25 / ventana 51x51.</strong> El retorno queda cerca de cero; los picos tardios no forman una meseta estable.",
        },
        {
          type: "video",
          src: asset("videos/vision-range-51x51.mp4"),
          caption:
            "<strong>Rollout greedy 51x51.</strong> La politica no muestra pickups claros ni retorno confiable al hub.",
        },
      ],
    },
    "lethal-mechanism": {
      groups: [
        {
          bullets: [
            "La galleta letal no esta marcada en la observacion: para la politica se ve como comida comun hasta que una hormiga muere.",
            "Internamente el entorno guarda comida segura y letal en grillas separadas.",
            "En la observacion ambas se proyectan al mismo canal visible: el actor no recibe un canal que diga <em>esto es veneno</em>.",
            "La novedad observable es <code>dead_ants_count</code>, un canal espacial local con la misma geometria que comida, hormigas, bytes, hub y obstaculos.",
          ],
        },
        {
          heading: "Que cambia en la dinamica",
          bullets: [
            "Un pickup sobre comida letal mata a la hormiga, congela su cuerpo y la remueve de la poblacion activa.",
            "Esa muerte deja una senal espacial local que otras hormigas pueden ver desde su propia orientacion.",
            "El actor sigue eligiendo las mismas dos acciones: <code>move</code> y <code>write_value</code>.",
            "La diferencia no es otro MAPPO: es una fuente visualmente identica que puede matar y dejar evidencia social.",
          ],
        },
      ],
    },
    "lethal-open-results": {
      groups: [
        {
          bullets: [
            "El mapa abierto no se resolvio de forma robusta, pero mostro una senal interesante.",
            "El primer escenario tenia mapa <code>50x50</code>, una fuente normal de <code>50</code> cookies y una fuente letal de <code>50</code> cookies.",
            "Con <code>death_penalty=1.0</code>, la corrida larga tendia a una solucion conservadora: no agarrar galletas para no arriesgar muerte.",
            "Subir el <code>pickup_bonus</code> recupero riesgo: la politica volvio a intentar pickups, aumento entregas y mejoro el porcentaje entregado.",
          ],
        },
        {
          heading: "Lectura",
          bullets: [
            "El mejor caso abierto queda en <code>4.75/50</code> entregas, alrededor de <code>9.5%</code>.",
            "En el anillo de galletas se ve que puede ignorar la fuente letal asociada a cadaveres y seguir tomando comida segura.",
            "Aun asi, la curva sigue irregular: es una politica parcialmente util, no una convergencia limpia.",
          ],
        },
      ],
      metrics: [
        ["mejor caso abierto", "4.75 entregas"],
        ["porcentaje entregado", "9.5%"],
        ["retorno evaluacion", "4.375"],
      ],
      media: [
        {
          type: "video",
          src: asset("presentation/lethal-open-two-source.mp4"),
          caption:
            "<strong>Mapa abierto, dos fuentes.</strong> Una fuente segura de 50 cookies y una letal de 50 cookies.",
        },
        {
          type: "video",
          src: asset("lethal-cookies/media/hub_ring.mp4"),
          poster: asset("lethal-cookies/media/hub_ring_poster.jpg"),
          caption:
            "<strong>Anillo seguro con fuente letal cercana.</strong> Explota comida segura y evita la zona con cadaveres.",
        },
        {
          type: "image",
          src: asset("presentation/lethal-open-gray-return.png"),
          alt: "Reward evolution del mapa abierto con fuente segura y letal",
          caption:
            "<strong>Reward evolution.</strong> Retornos ruidosos: hay mejora parcial, pero no meseta robusta.",
        },
      ],
    },
    "lethal-random-walls": {
      groups: [
        {
          heading: "Falla al salir del mapa abierto",
          bullets: [
            "Al evaluar la politica abierta en laberintos fijos aparecio una falla doble: navegacion inconsistente y asociacion fragil entre cadaveres y fuente letal.",
            "La politica habia visto veneno en una geometria demasiado especifica, normalmente cerca del hormiguero y lejos de paredes.",
            "Una cookie letal cerca de pasillos o cuartos laterales degradaba tanto la navegacion como la lectura de riesgo.",
          ],
        },
        {
          heading: "Random walls",
          bullets: [
            "La solucion siguiente fue incluir paredes en la distribucion de entrenamiento.",
            "Random walls mezcla segmentos de pared, comida segura y comida letal en el mismo mapa.",
            "Vuelve a aparecer una senal positiva: la colonia puede dejar zonas de exclusion alrededor de cadaveres.",
            "Pero no resuelve bien navegacion: con galletas cerca del hormiguero todavia puede conseguir reward sin aprender recorridos largos alrededor de paredes.",
          ],
        },
        {
          heading: "Near-nest walls",
          bullets: [
            "Near-nest walls reduce comida segura a <code>12</code> fuentes y fuerza trayectorias mas largas alrededor de paredes.",
            "La fuente letal se mantiene presente para que la politica no olvide el problema original.",
            "En evaluacion completa <code>12/12</code> entregas y conserva la fuente letal sin explotar.",
            "Es el resultado mas fuerte de paredes, pero sigue acotado a una distribucion guiada.",
          ],
        },
      ],
      metrics: [
        ["random walls", "2.5 entregas / 5%"],
        ["near-nest walls", "12/12 entregas"],
        ["near-nest retorno", "11"],
      ],
      media: [
        {
          type: "video",
          src: asset("lethal-cookies/media/random_walls.mp4"),
          poster: asset("lethal-cookies/media/random_walls_poster.jpg"),
          caption:
            "<strong>Random walls.</strong> Peligro social mas obstaculos aleatorios.",
        },
        {
          type: "image",
          src: asset("presentation/random-walls-blue-return.png"),
          alt: "Curvas azules de retorno en random walls",
          caption:
            "<strong>Random walls.</strong> Senal positiva, pero retornos bajos e inestables.",
        },
        {
          type: "video",
          src: asset("lethal-cookies/media/near_nest.mp4"),
          poster: asset("lethal-cookies/media/near_nest_poster.jpg"),
          caption:
            "<strong>Near-nest walls.</strong> Rutas activas alrededor de paredes con fuente letal presente.",
        },
        {
          type: "image",
          src: asset("presentation/near-nest-red-return.png"),
          alt: "Curvas rojas de retorno en near-nest walls",
          caption:
            "<strong>Near-nest walls.</strong> Salto a retorno alto y entregas completas en la distribucion controlada.",
        },
      ],
    },
    "lethal-transfer": {
      groups: [
        {
          bullets: [
            "La idea era combinar navegacion ya convergida con aprendizaje de paredes y cookies letales.",
            "Se partio de un checkpoint de mapa abierto sin lethal cookies que entregaba el <code>100%</code> y recorria el mapa con conducta reconocible.",
            "La hipotesis era que esa navegacion sobreviviria al fine-tuning con paredes y veneno.",
            "En las curvas la continuacion parece prometedora, pero visualmente aparece un modo de falla: evitar pickups y sesgarse espacialmente para reducir riesgo.",
          ],
        },
        {
          heading: "Comparacion temporal",
          bullets: [
            "Con <code>100</code> updates todavia conserva parte del comportamiento util: <code>8.125</code> entregas y <code>67.7%</code>.",
            "Ese video temprano muestra una trampa: hay alta tasa de exito, pero las hormigas todavia agarran cookies letales pese a los cadaveres.",
            "Con <code>1000</code> updates cae a <code>3.875</code> entregas, <code>32.3%</code> y retorno negativo.",
            "La metrica agregada no alcanza para describir la perdida de navegacion.",
          ],
        },
        {
          heading: "Trabajo futuro",
          bullets: [
            "La penalizacion por muerte deberia entrar como curriculum, no como castigo rigido desde el comienzo.",
            "Tambien conviene elegir checkpoints con metricas de cobertura o diversidad de trayectorias, no solo retorno agregado.",
          ],
        },
      ],
      metrics: [
        ["100 updates", "8.125 entregas / 67.7%"],
        ["1000 updates", "3.875 entregas / 32.3%"],
        ["retorno 1000", "-46.125"],
      ],
      media: [
        {
          type: "image",
          src: asset("presentation/transfer-60-orange-return.png"),
          alt: "Curvas naranjas de retorno para Transfer 60",
          caption:
            "<strong>Transfer 60.</strong> El retorno mejora, pero no garantiza que la politica haya aprendido peligro correctamente.",
        },
        {
          type: "video",
          src: asset("presentation/transfer-60-early-100-updates.mp4"),
          caption:
            "<strong>100 updates.</strong> Buen desempeno agregado, pero todavia toma cookies letales.",
        },
        {
          type: "video",
          src: asset("lethal-cookies/media/random_walls_active.mp4"),
          poster: asset("lethal-cookies/media/random_walls_active_poster.jpg"),
          caption:
            "<strong>Transferencia tardia.</strong> Hay navegacion parcial, pero tambien evitacion que reduce foraging.",
        },
      ],
    },
  };

  const fromHTML = (template) => {
    const wrapper = document.createElement("template");
    wrapper.innerHTML = template.trim();
    return wrapper.content.firstElementChild;
  };

  const renderBulletGroups = (groups) =>
    groups
      .map(
        (group) => `
          <section class="presentation-bullet-group">
            ${group.heading ? `<h4>${group.heading}</h4>` : ""}
            <ul>
              ${group.bullets.map((bullet) => `<li>${bullet}</li>`).join("")}
            </ul>
          </section>
        `,
      )
      .join("");

  const renderBulletMetrics = (metrics = []) => {
    if (metrics.length === 0) return "";
    return `
      <dl class="presentation-bullet-metrics">
        ${metrics
          .map(
            ([label, value]) => `
              <div>
                <dt>${label}</dt>
                <dd>${value}</dd>
              </div>
            `,
          )
          .join("")}
      </dl>
    `;
  };

  const renderBulletMedia = (media = []) => {
    if (media.length === 0) return "";
    return `
      <div class="presentation-bullet-media">
        ${media
          .map((item) => {
            const visual =
              item.type === "video"
                ? `<video autoplay loop muted playsinline controls preload="metadata"${item.poster ? ` poster="${item.poster}"` : ""}><source src="${item.src}" type="video/mp4"></video>`
                : `<img src="${item.src}" alt="${item.alt || ""}">`;
            return `
              <figure>
                ${visual}
                <figcaption>${item.caption || ""}</figcaption>
              </figure>
            `;
          })
          .join("")}
      </div>
    `;
  };

  const renderBulletStep = (step, content) => {
    const article = document.createElement("article");
    article.className = "presentation-step-card presentation-bullet-step";
    const header = step.querySelector(":scope > .step-header");
    if (header) article.append(cloneClean(header));
    article.append(
      fromHTML(`
        <div class="presentation-bullet-layout">
          <div class="presentation-bullet-copy">
            ${renderBulletGroups(content.groups)}
            ${renderBulletMetrics(content.metrics)}
          </div>
          ${renderBulletMedia(content.media)}
        </div>
      `),
    );
    return article;
  };

  const makeStepSlide = (step, chapter) => ({
    id: step.id,
    kind: "step",
    number: textFrom(step, ".step-index"),
    title: textFrom(step, "h3"),
    chapterTitle: textFrom(chapter, "h2"),
    source: step,
    render() {
      const customContent = bulletStepContent[step.id];
      if (customContent) {
        return renderBulletStep(step, customContent);
      }
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

  if (new URLSearchParams(window.location.search).get("presentation") === "1") {
    window.requestAnimationFrame(() => openPresentation());
  }
})();
