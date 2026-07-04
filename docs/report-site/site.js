(() => {
  const policies = {
    frontier50: {
      label: "50x50",
      title: "Política frontera 50x50",
      src: "report-site/assets/videos/frontier-50x50.mp4",
      caption:
        "Checkpoint 50x50 de 60 hormigas renderizado como MP4 de sprites 800x800 desde la política real.",
      metrics: [
        ["grilla de tarea", "50x50"],
        ["video codificado", "800x800"],
        ["hormigas", "60"],
        ["comida", "125 bocados / 2 fuentes"],
        ["bits", "8"],
        ["crítico", "strided_cnn"],
        ["entregado", "123.906 / 125"],
        ["éxito", "0.906 en 64 episodios de evaluación"],
        ["modo de acción", "movimiento muestreado / escritura greedy"],
        ["temp. movimiento", "0.525 render, 0.5 confirmación"],
        ["escritura no cero", "0.998"],
      ],
      caveat:
        "Comportamiento fuerte, pero mezclado con cantidad de hormigas, bits de escritura, rasgos de identidad, continuación seleccionada, temperatura y escrituras saturadas.",
    },
    unlock25: {
      label: "25x25",
      title: "Rollout 25x25 del currículum de cantidad de hormigas",
      src: "report-site/assets/videos/forage-25x25-4ants.mp4",
      caption:
        "Rollout de cuatro hormigas y 3 bits desde el artefacto del currículum de cantidad de hormigas, renderizado como MP4 800x800.",
      metrics: [
        ["grilla de tarea", "25x25"],
        ["video codificado", "800x800"],
        ["hormigas", "4"],
        ["comida", "23 bocados / 12 fuentes"],
        ["bits", "3"],
        ["checkpoint", "ant_count_25x25_3_bits/4_ants"],
        ["obs. del actor", "151 rasgos legacy"],
        ["crítico", "MLP"],
        ["retorno del entorno en entrenamiento", "7.8125"],
        ["retorno de episodio en entrenamiento", "12.545"],
      ],
      caveat:
        "Este video es el artefacto del currículum de cantidad de hormigas. El resultado DISTANCE_CAP4 23/23 viene de archivos locales de evaluación separados con una tarea de 6 fuentes.",
    },
    bridge100: {
      label: "puente 100x100",
      title: "Puente 100x100 en render 1000x1000",
      src: "report-site/assets/videos/bridge-100x100.mp4",
      caption:
        "Rollout solo actor en una grilla de simulación 1000x1000, codificado como MP4 1008x1008.",
      metrics: [
        ["grilla de render", "1000x1000"],
        ["video codificado", "1008x1008"],
        ["ventana activa", "250x250 dentro de [375, 624]"],
        ["hormigas en render", "500"],
        ["comida en render", "5000 bocados / 6 fuentes"],
        ["tarea del checkpoint", "continuación hard375 100x100"],
        ["configuración de evaluación", "120 hormigas, 375 comida, 6 fuentes"],
        ["crítico", "linaje set_cnn"],
        ["entregado en evaluación", "372 / 375"],
        ["éxito en evaluación", "0.625 en 24 episodios"],
        ["temperatura", "movimiento 0.525"],
        ["tasa de evaluación", "0.803 entregas / 1000 pasos-hormiga"],
      ],
      caveat:
        "La geometría del video es 1000x1000 con 500 hormigas y 5000 comida. La etiqueta 100x100 pertenece al linaje del checkpoint y a la tarea de evaluación.",
    },
    reset250: {
      label: "reset 250",
      title: "Checkpoint reset-boundary 250x250",
      src: "report-site/assets/videos/reset-boundary-250x250.mp4",
      caption:
        "Video local del checkpoint reset-boundary de la rama diagnóstica fixed8 250x250, codificado como 1008x1008.",
      metrics: [
        ["grilla de tarea", "250x250"],
        ["video codificado", "1008x1008"],
        ["fotogramas", "600 a 8 fps"],
        ["hormigas", "500"],
        ["comida", "5000 bocados / 1 fuente"],
        ["familia del checkpoint", "fixed8-reset-boundary256"],
        ["crítico", "set_cnn"],
        ["entregas finales", "654"],
        ["recogidas finales", "869"],
        ["mejor entrega de entrenamiento", "alrededor de 1003"],
        ["punto diagnóstico", "entrega cruda separada del retorno moldeado"],
      ],
      caveat:
        "Esta rama muestra progreso local real de entrega dentro de la configuración reset/distancia 250x250.",
    },
    random: {
      label: "azar",
      title: "Rollout línea base aleatoria",
      src: "report-site/assets/videos/random-rollout.mp4",
      caption:
        "Línea base temprana de política aleatoria, codificada como MP4 576x576 con 301 fotogramas a 12 fps.",
      metrics: [
        ["video codificado", "576x576"],
        ["fotogramas", "301 a 12 fps"],
        ["propósito", "línea base de comportamiento"],
        ["lectura del resultado", "el movimiento importa cuando crea bucles de entrega"],
      ],
      caveat:
        "Este ancla visual de línea base muestra la tarea antes de que el entrenamiento entre en la historia.",
    },
    bigmap50: {
      label: "1000 50x50",
      title: "Política 50x50 en render 1000x1000",
      src: "report-site/assets/videos/bigmap-50x50-policy-1000x1000.mp4",
      caption:
        "Política 50x50 renderizada en una grilla de simulación 1000x1000, codificada como MP4 1008x1008.",
      metrics: [
        ["grilla de render", "1000x1000"],
        ["video codificado", "1008x1008"],
        ["fotogramas", "8109 a 160 fps"],
        ["política fuente", "frontera 50x50"],
        ["hormigas en render", "500"],
        ["comida en render", "5000 bocados"],
        ["primera entrega", "paso 14574"],
        ["entregado", "147 al paso 120000"],
      ],
      caveat:
        "Despliegue solo actor: evidencia visual de transferencia de comportamiento, no entrenamiento de punta a punta en 1000x1000.",
    },
  };

  const video = document.querySelector("#policy-video");
  const source = document.querySelector("#policy-source");
  const title = document.querySelector("#policy-title");
  const caption = document.querySelector("#policy-caption");
  const metrics = document.querySelector("#policy-metrics");
  const caveat = document.querySelector("#policy-caveat");
  const buttons = Array.from(document.querySelectorAll("[data-policy]"));

  function renderMetrics(rows) {
    metrics.replaceChildren();
    rows.forEach(([key, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      metrics.append(dt, dd);
    });
  }

  function selectPolicy(id) {
    const policy = policies[id];
    if (!policy) return;

    title.textContent = policy.title;
    caption.textContent = policy.caption;
    caveat.textContent = policy.caveat;
    renderMetrics(policy.metrics);

    if (source.getAttribute("src") !== policy.src) {
      source.setAttribute("src", policy.src);
      video.load();
      video.play().catch(() => {});
    }

    buttons.forEach((button) => {
      button.classList.toggle("active", button.dataset.policy === id);
    });
  }

  buttons.forEach((button) => {
    const policy = policies[button.dataset.policy];
    if (policy && policy.label) {
      button.textContent = policy.label;
    }
    button.addEventListener("click", () => selectPolicy(button.dataset.policy));
  });

  selectPolicy("frontier50");
})();
