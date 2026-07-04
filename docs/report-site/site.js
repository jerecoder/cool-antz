(() => {
  const experimentTree = {
    id: "world",
    href: "#world",
    label: "Mundo y métrica",
    small: "entrega, fuentes, bytes",
    poster: "report-site/assets/posters/tree-world.jpg",
    alt: "Fotograma de la línea base aleatoria",
    kind: "root",
    children: [
      {
        id: "size-curriculum",
        href: "#size-curriculum",
        label: "Currículo de tamaño",
        small: "8x8 a 50x50 desde pickles",
        poster: "report-site/assets/posters/tree-size-curriculum.jpg",
        alt: "Fotograma del currículo temprano de tamaño de mapa",
        kind: "main",
        children: [
          {
            id: "small-scale",
            href: "#small-scale",
            label: "Primer cuello",
            small: "una colonia chica no cubre",
            poster: "report-site/assets/posters/tree-small-scale.jpg",
            alt: "Fotograma 25x25 con 2 hormigas",
            kind: "main",
            children: [
              {
                id: "unlock25",
                href: "#unlock25",
                label: "Desbloqueo 25x25",
                small: "4 hormigas + moldeado",
                poster: "report-site/assets/posters/tree-unlock25.jpg",
                alt: "Fotograma 25x25 con 4 hormigas",
                kind: "main",
                children: [
                  {
                    id: "bits-ants",
                    href: "#bits-ants",
                    label: "Bytes vs cobertura",
                    small: "hipótesis, no conclusión",
                    poster: "report-site/assets/posters/tree-bits-ants.jpg",
                    alt: "Fotograma del currículo de cantidad de hormigas",
                    kind: "main",
                    children: [
                      {
                        id: "source-layouts",
                        href: "#source-layouts",
                        label: "Fuentes dispersas",
                        small: "de muchas posiciones a 2 fuentes",
                        poster: "report-site/assets/posters/tree-source-layouts.jpg",
                        alt: "Fotograma de una política entrenada con pocas posiciones de comida",
                        kind: "main",
                        children: [
                          {
                            id: "rare50",
                            href: "#rare50",
                            label: "50x50 raro",
                            small: "descubrir fuentes escasas",
                            poster: "report-site/assets/posters/tree-rare50.jpg",
                            alt: "Fotograma de una política 50x50 rara con 4 hormigas",
                            kind: "main",
                            children: [
                              {
                                id: "critic50",
                                href: "#critic50",
                                label: "Cambio de crítico",
                                small: "MLP a strided_cnn",
                                poster: "report-site/assets/posters/tree-critic50.jpg",
                                alt: "Fotograma de una política 50x50 con crítico strided_cnn",
                                kind: "main",
                                children: [
                                  {
                                    id: "frontier50",
                                    href: "#frontier50",
                                    label: "Frontera 60 hormigas",
                                    small: "123.90625/125",
                                    poster: "report-site/assets/posters/tree-frontier50.jpg",
                                    alt: "Fotograma de la política 50x50 de 60 hormigas",
                                    kind: "frontier",
                                    children: [
                                      {
                                        id: "large-scale-100",
                                        href: "#large-scale-100",
                                        label: "Puente 100x100",
                                        small: "372-373/375",
                                        poster: "report-site/assets/posters/tree-large-scale-100.jpg",
                                        alt: "Fotograma del render 1000x1000 del puente 100x100",
                                        kind: "scale",
                                      },
                                      {
                                        id: "large-scale-250",
                                        href: "#large-scale-250",
                                        label: "250x250 reset",
                                        small: "falló, luego entregó",
                                        poster: "report-site/assets/posters/tree-large-scale-250.jpg",
                                        alt: "Fotograma del reset-boundary 250x250",
                                        kind: "scale",
                                      },
                                      {
                                        id: "bigmap",
                                        href: "#bigmap",
                                        label: "1000x1000 solo actor",
                                        small: "política 50x50 en estrés",
                                        poster: "report-site/assets/posters/tree-bigmap.jpg",
                                        alt: "Fotograma del despliegue solo actor 1000x1000",
                                        kind: "scale",
                                      },
                                    ],
                                  },
                                ],
                              },
                            ],
                          },
                        ],
                      },
                      {
                        id: "autocurriculum",
                        href: "#autocurriculum",
                        label: "Autocurrículos",
                        small: "etapas y moldeado no bastaron",
                        poster: "report-site/assets/posters/tree-autocurriculum.jpg",
                        alt: "Fotograma real del autocurrículo 250x250 con entrega cero",
                        kind: "side",
                      },
                      {
                        id: "maze-pipeline",
                        href: "#maze-pipeline",
                        label: "Laberinto",
                        small: "laberinto W&B 50x50",
                        poster: "report-site/assets/posters/tree-maze.jpg",
                        alt: "Fotograma del currículo de laberintos 50x50 entrenado en W&B",
                        kind: "side",
                      },
                      {
                        id: "memory-autoresearch",
                        href: "#memory-autoresearch",
                        label: "Memoria R8-R12",
                        small: "ablaciones antes que videos",
                        poster: "report-site/assets/posters/tree-memory.jpg",
                        alt: "Gráfico de entregas en pruebas normal, sin lectura de bytes y sin escritura",
                        kind: "side",
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
  };

  function createSvg(tagName) {
    return document.createElementNS("http://www.w3.org/2000/svg", tagName);
  }

  function splitLabel(text, maxChars, maxLines) {
    const words = text.split(/\s+/);
    const lines = [];
    let current = "";
    words.forEach((word) => {
      const candidate = current ? `${current} ${word}` : word;
      if (candidate.length <= maxChars) {
        current = candidate;
        return;
      }
      if (current) lines.push(current);
      current = word;
    });
    if (current) lines.push(current);
    if (lines.length <= maxLines) return lines;
    const trimmed = lines.slice(0, maxLines);
    trimmed[maxLines - 1] = `${trimmed[maxLines - 1].replace(/\.*$/, "")}...`;
    return trimmed;
  }

  function collectTreeLayout(root) {
    const nodeWidth = 220;
    const nodeHeight = 104;
    const levelGap = 54;
    const rowGap = 34;
    const padding = 26;
    const nodes = [];
    const edges = [];

    function primaryChild(node) {
      return (node.children || []).find((child) => child.kind === "main" || child.kind === "frontier") || null;
    }

    function makeLayoutNode(node, depth, row) {
      const layoutNode = {
        ...node,
        depth,
        row,
        width: nodeWidth,
        height: nodeHeight,
        x: padding + depth * (nodeWidth + levelGap),
        y: padding + row * (nodeHeight + rowGap),
      };
      nodes.push(layoutNode);
      return layoutNode;
    }

    const mainChain = [];
    let current = root;
    while (current) {
      mainChain.push(current);
      current = primaryChild(current);
    }

    const layoutById = new Map();
    mainChain.forEach((node, depth) => {
      layoutById.set(node.id, makeLayoutNode(node, depth, 0));
    });
    for (let index = 1; index < mainChain.length; index += 1) {
      edges.push({
        parent: layoutById.get(mainChain[index - 1].id),
        child: layoutById.get(mainChain[index].id),
      });
    }

    mainChain.forEach((node, depth) => {
      const parent = layoutById.get(node.id);
      const primary = primaryChild(node);
      const branches = (node.children || []).filter((child) => child !== primary);
      const scaleBranch = parent.kind === "frontier" || branches.every((child) => child.kind === "scale");
      branches.forEach((branch, index) => {
        const branchDepth = depth + (scaleBranch ? index + 1 : index);
        const row = 1;
        const child = makeLayoutNode(branch, branchDepth, row);
        edges.push({ parent, child });
      });
    });

    const maxX = Math.max(...nodes.map((node) => node.x)) + nodeWidth + padding;
    const maxY = Math.max(...nodes.map((node) => node.y)) + nodeHeight + padding;
    return { nodes, edges, width: maxX, height: maxY };
  }

  function appendWrappedText(parent, text, x, y, maxChars, maxLines, className, lineHeight) {
    splitLabel(text, maxChars, maxLines).forEach((line, index) => {
      const tspan = createSvg("text");
      tspan.setAttribute("x", x);
      tspan.setAttribute("y", y + index * lineHeight);
      tspan.setAttribute("class", className);
      tspan.textContent = line;
      parent.append(tspan);
    });
  }

  function renderExperimentTree() {
    const container = document.querySelector("#experiment-lineage");
    if (!container) return;

    const { nodes, edges, width, height } = collectTreeLayout(experimentTree);
    const svg = createSvg("svg");
    svg.setAttribute("class", "tree-svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-labelledby", "tree-svg-title tree-svg-desc");

    const title = createSvg("title");
    title.setAttribute("id", "tree-svg-title");
    title.textContent = "Árbol navegable de experimentos";
    const desc = createSvg("desc");
    desc.setAttribute("id", "tree-svg-desc");
    desc.textContent =
      "Linaje desde la definición del mundo y los primeros currículos hasta la frontera 50x50, las ramas exploratorias y los despliegues grandes.";
    svg.append(title, desc);

    const edgeLayer = createSvg("g");
    edgeLayer.setAttribute("class", "tree-edges");
    edges.forEach(({ parent, child }) => {
      const edge = createSvg("path");
      edge.setAttribute("class", `tree-edge ${child.kind || "main"}`);
      if (child.x > parent.x) {
        const x1 = parent.x + parent.width;
        const y1 = parent.y + parent.height / 2;
        const x2 = child.x;
        const y2 = child.y + child.height / 2;
        const mid = x1 + (x2 - x1) * 0.5;
        edge.setAttribute("d", `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`);
      } else {
        const x1 = parent.x + parent.width / 2;
        const y1 = parent.y + parent.height;
        const x2 = child.x + child.width / 2;
        const y2 = child.y;
        const mid = y1 + (y2 - y1) * 0.5;
        edge.setAttribute("d", `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`);
      }
      edgeLayer.append(edge);
    });
    svg.append(edgeLayer);

    const nodeLayer = createSvg("g");
    nodeLayer.setAttribute("class", "tree-nodes");
    nodes.forEach((node) => {
      const link = createSvg("a");
      link.setAttribute("href", node.href);
      link.setAttribute("class", `tree-link ${node.kind || "main"}`);
      link.setAttribute("aria-label", `${node.label}: ${node.small}`);
      link.addEventListener("click", (event) => {
        const target = document.querySelector(node.href);
        if (!target) return;
        event.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        history.pushState(null, "", node.href);
      });

      const group = createSvg("g");
      group.setAttribute("transform", `translate(${node.x} ${node.y})`);

      const frame = createSvg("rect");
      frame.setAttribute("class", "tree-node-frame");
      frame.setAttribute("width", node.width);
      frame.setAttribute("height", node.height);
      frame.setAttribute("rx", "7");

      const rail = createSvg("rect");
      rail.setAttribute("class", "tree-node-rail");
      rail.setAttribute("width", node.width);
      rail.setAttribute("height", "5");
      rail.setAttribute("rx", "5");

      const image = createSvg("image");
      image.setAttribute("class", "tree-node-image");
      image.setAttribute("href", node.poster);
      image.setAttribute("x", "10");
      image.setAttribute("y", "14");
      image.setAttribute("width", "74");
      image.setAttribute("height", "56");
      image.setAttribute("preserveAspectRatio", "xMidYMid slice");

      const imageTitle = createSvg("title");
      imageTitle.textContent = node.alt;
      image.append(imageTitle);

      group.append(frame, rail, image);
      appendWrappedText(group, node.label, 94, 31, 17, 2, "tree-node-label", 17);
      appendWrappedText(group, node.small, 94, 75, 22, 2, "tree-node-small", 15);
      link.append(group);
      nodeLayer.append(link);
    });
    svg.append(nodeLayer);

    container.replaceChildren(svg);
  }

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
      label: "25x25 4h",
      title: "Rollout 25x25 con 4 hormigas",
      src: "report-site/assets/videos/forage-25x25-4ants.mp4",
      caption:
        "Rollout de cuatro hormigas y 3 bits desde el artefacto del currículum de cantidad de hormigas, renderizado como MP4 800x800.",
      metrics: [
        ["grilla de tarea", "25x25"],
        ["video codificado", "800x800"],
        ["hormigas", "4"],
        ["comida", "23 bocados / 12 fuentes"],
        ["bits", "3"],
        ["punto guardado", "ant_count_25x25_3_bits/4_ants"],
        ["obs. del actor", "151 rasgos legacy"],
        ["crítico", "MLP"],
        ["retorno del entorno en entrenamiento", "7.8125"],
        ["retorno de episodio en entrenamiento", "12.545"],
      ],
      caveat:
        "Este video es el artefacto del currículum de cantidad de hormigas. El resultado DISTANCE_CAP4 23/23 viene de archivos locales de evaluación separados con una tarea de 6 fuentes.",
    },
    ant2: {
      label: "25x25 2h",
      title: "Rollout 25x25 con 2 hormigas",
      src: "report-site/assets/videos/ant-count-25x25-2ants.mp4",
      caption:
        "Rollout preservado del currículum de cantidad de hormigas: dos hormigas y 3 bits, renderizado como MP4 800x800.",
      metrics: [
        ["grilla de tarea", "25x25"],
        ["video codificado", "800x800"],
        ["fotogramas", "2501 a 8 fps"],
        ["hormigas", "2"],
        ["comida", "23 bocados / 12 fuentes"],
        ["bits", "3"],
        ["punto guardado", "ant_count_25x25_3_bits/2_ants"],
        ["retorno del entorno en entrenamiento", "4.8125"],
        ["retorno de episodio en entrenamiento", "7.7875"],
      ],
      caveat:
        "Este video muestra el currículo de conteo de hormigas, no el primer avance de una sola hormiga. Para esa etapa no hay MP4 preservado en el checkout.",
    },
    ant8: {
      label: "25x25 8h",
      title: "Rollout 25x25 con 8 hormigas",
      src: "report-site/assets/videos/ant-count-25x25-8ants.mp4",
      caption:
        "Rollout preservado del currículum de cantidad de hormigas: ocho hormigas y 3 bits, renderizado como MP4 800x800.",
      metrics: [
        ["grilla de tarea", "25x25"],
        ["video codificado", "800x800"],
        ["fotogramas", "233 a 8 fps"],
        ["hormigas", "8"],
        ["comida", "23 bocados / 12 fuentes"],
        ["bits", "3"],
        ["punto guardado", "ant_count_25x25_3_bits/8_ants"],
        ["retorno del entorno en entrenamiento", "10.875"],
        ["retorno de episodio en entrenamiento", "18.2294"],
      ],
      caveat:
        "La mejora en este currículo apunta a cobertura multi-agente. No prueba por sí sola comunicación causal mediante bytes.",
    },
    bridge100: {
      label: "100x100 en 1000",
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
        ["tarea del punto guardado", "continuación hard375 100x100"],
        ["configuración de evaluación", "120 hormigas, 375 comida, 6 fuentes"],
        ["crítico", "linaje set_cnn"],
        ["entregado en evaluación", "372 / 375"],
        ["éxito en evaluación", "0.625 en 24 episodios"],
        ["temperatura", "movimiento 0.525"],
        ["tasa de evaluación", "0.803 entregas / 1000 pasos-hormiga"],
      ],
      caveat:
        "La geometría del video es 1000x1000 con 500 hormigas y 5000 comida. La etiqueta 100x100 pertenece al linaje del punto guardado y a la tarea de evaluación.",
    },
    maze50: {
      label: "laberinto 50",
      title: "Currículo de laberintos 50x50 entrenado",
      src: "report-site/assets/videos/maze-exploration-50x50-trained-policy.mp4",
      caption:
        "Video W&B de la etapa 50x50 del currículo entrenado para laberintos generados. Es una política de exploración en laberinto, no el actor abierto 50x50 de 60 hormigas.",
      metrics: [
        ["run W&B", "8ebnjq9f / maze_exploration_curriculum"],
        ["política fuente", "maze_exploration_curriculum, etapa 50x50"],
        ["entrenada para", "laberintos generados con obstáculos"],
        ["grilla", "50x50"],
        ["video codificado", "800x800"],
        ["fotogramas", "600 a 8 fps"],
        ["duración", "75 s"],
        ["hormigas", "1"],
        ["comida", "48 bocados / 12 fuentes"],
        ["bits", "2"],
        ["radio actor", "1"],
        ["crítico", "MLP"],
        ["objetivo", "reward_mode=explore"],
        ["laberinto", "pasillos 3, paredes 1, semilla 17"],
        ["summary W&B", "episode_return=26.125"],
        ["comida restante media", "45.5 / 48"],
        ["episodios completados", "0"],
        ["checkpoint recuperable", "jax_mappo_maze_explore_24x24.pkl"],
      ],
      caveat:
        "La colección de modelos preserva checkpoints de laberinto hasta 24x24; el artefacto 50x50 disponible es el video W&B del currículo entrenado. No debe evaluarse como entrega de comida porque esa rama optimizaba exploración.",
    },
    maze100stress: {
      label: "stress laberinto 100",
      title: "Stress 100x100: actor 50x50 de 60 hormigas en laberinto",
      src: "report-site/assets/videos/maze-100x100-60ants-one-far-source.mp4",
      caption:
        "Prueba actor-only corregida con la temperatura estándar: política abierta 50x50 de 60 hormigas, una fuente lejana y paredes de laberinto visibles por el canal local borde/obstáculo.",
      metrics: [
        ["política", "best estabilizado 50x50 / 60 hormigas / 8 bits"],
        ["entrenada para", "forrajeo abierto 50x50, no laberinto"],
        ["grilla", "100x100"],
        ["laberinto", "layout efectivo seed 111"],
        ["hormigas", "60"],
        ["comida", "125 bocados / 1 fuente"],
        ["hormiguero", "[6, 5]"],
        ["fuente efectiva", "[81, 70]"],
        ["distancia por pasillos", "172 pasos"],
        ["acción", "sampled_move_greedy_write"],
        ["temperatura", "movimiento 0.525, escritura greedy"],
        ["horizonte", "60000 pasos simulados"],
        ["video codificado", "800x800, 3001 fotogramas, 60 fps"],
        ["entregado", "0 / 125"],
        ["pickups", "0"],
        ["celdas visitadas", "887"],
        ["bytes no nulos", "886 tiles"],
      ],
      caveat:
        "Las paredes sí entran al actor por el mismo canal local de borde/obstáculo, pero son paredes internas fuera de la distribución de entrenamiento del actor abierto. Este video muestra una falla de transferencia directa, no un fallo de renderizado ni un éxito de laberinto.",
    },
    reset250: {
      label: "reset 250",
      title: "Punto guardado reset-boundary 250x250",
      src: "report-site/assets/videos/reset-boundary-250x250.mp4",
      caption:
        "Video local del punto guardado reset-boundary de la rama diagnóstica fixed8 250x250, codificado como 1008x1008.",
      metrics: [
        ["grilla de tarea", "250x250"],
        ["video codificado", "1008x1008"],
        ["fotogramas", "600 a 8 fps"],
        ["hormigas", "500"],
        ["comida", "5000 bocados / 1 fuente"],
        ["familia del punto guardado", "fixed8-reset-boundary256"],
        ["crítico", "set_cnn"],
        ["entregas finales", "654"],
        ["recogidas finales", "869"],
        ["mejor entrega de entrenamiento", "alrededor de 1003"],
        ["punto diagnóstico", "entrega cruda separada del retorno moldeado"],
      ],
      caveat:
        "Esta rama muestra progreso local real de entrega dentro de la configuración reset/distancia 250x250; no demuestra que el autocurrículo 250x250 general haya quedado resuelto.",
    },
    auto250fail: {
      label: "auto 250 fallo",
      title: "Autocurrículo 250x250: fallo diagnóstico",
      src: "report-site/assets/videos/autocurriculum-250x250-failure.mp4",
      caption:
        "Checkpoint-video de la diagnosis de autocurrículo 250x250: actividad, shaping y bytes sin evidencia suficiente de entrega real.",
      metrics: [
        ["grilla de tarea", "250x250"],
        ["video codificado", "1008x1008"],
        ["fotogramas", "600 a 8 fps"],
        ["hormigas", "500"],
        ["comida", "5000 bocados / 1 fuente"],
        ["familia", "distance_autocurriculum_source_teacher"],
        ["crítico", "set_cnn"],
        ["entregas en diagnosis", "0 en corridas clave"],
        ["lectura", "actividad visual no equivale a resolver entrega"],
      ],
      caveat:
        "Este video se incluye como evidencia negativa: ayuda a ver por qué no alcanza con retorno moldeado, agentes cargando o bytes activos.",
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
      label: "stress 1000",
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

  renderExperimentTree();
  selectPolicy("frontier50");
})();
