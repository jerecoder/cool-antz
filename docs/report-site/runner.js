(() => {
  const data = window.CoolAntzPolicyRunnerData;
  const canvas = document.querySelector("#runner-canvas");

  if (!data || !canvas) {
    return;
  }

  const env = data.env;
  const actor = data.actor;
  const ctx = canvas.getContext("2d");
  const metrics = {
    step: document.querySelector("#runner-step"),
    delivered: document.querySelector("#runner-delivered"),
    remaining: document.querySelector("#runner-remaining"),
    carrying: document.querySelector("#runner-carrying"),
    bytes: document.querySelector("#runner-bytes"),
    sources: document.querySelector("#runner-sources"),
    status: document.querySelector("#runner-status"),
  };
  const runButton = document.querySelector("#runner-run");
  const stepButton = document.querySelector("#runner-step-once");
  const resetRunButton = document.querySelector("#runner-reset-run");
  const resetLayoutButton = document.querySelector("#runner-reset-layout");
  const modeButtons = Array.from(document.querySelectorAll("[data-place-mode]"));
  const actionMode = document.querySelector("#runner-action-mode");
  const speedInput = document.querySelector("#runner-speed");
  const uiText = {
    run: "Ejecutar",
    pause: "Pausar",
    step: "Paso",
    resetRun: "Reiniciar ejecución",
    resetLayout: "Reiniciar mapa",
    tiles: "celdas",
    loading: "cargando",
    statuses: {
      complete: "completo",
      maxSteps: "pasos máximos",
      running: "en ejecución",
      ready: "listo",
    },
    placement: {
      hub: "Hormiguero",
      food: "Comida",
      erase: "Borrar",
    },
    actionModes: {
      sampled_move_greedy_write: "movimiento muestreado / escritura greedy",
      greedy_move_greedy_write: "movimiento greedy / escritura greedy",
      sampled_move_sampled_write: "movimiento muestreado / escritura muestreada",
    },
  };

  const ACTION_STAY = 0;
  const ACTION_UP = 1;
  const ACTION_RIGHT = 2;
  const ACTION_DOWN = 3;
  const ACTION_LEFT = 4;
  const DEFAULT_FACING = ACTION_RIGHT;
  const spritePaths = {
    ant: "report-site/assets/sprites/ant.png",
    food: "report-site/assets/sprites/food.png",
    hub: "report-site/assets/sprites/hub.png",
    tile: "report-site/assets/sprites/tile.png",
  };

  let state = null;
  let running = false;
  let placementMode = "food";
  const defaultFoodSources = buildDefaultFoodSources();
  let foodSources = defaultFoodSources.slice(0, env.food_sources).map(copyPosition);
  let hub = [Math.floor(env.width / 2), Math.floor(env.height / 2)];
  let rngState = 0x5eed1234;
  let animationFrame = 0;
  let canvasCssSize = 0;
  const sprites = {};

  function copyPosition(position) {
    return [Number(position[0]), Number(position[1])];
  }

  function buildDefaultFoodSources() {
    const candidates = [
      [0.22, 0.22],
      [0.78, 0.78],
      [0.22, 0.78],
      [0.78, 0.22],
      [0.5, 0.18],
      [0.18, 0.5],
      [0.82, 0.5],
      [0.5, 0.82],
      [0.35, 0.35],
      [0.65, 0.35],
      [0.35, 0.65],
      [0.65, 0.65],
    ];
    return candidates.slice(0, env.food_sources).map(([xRatio, yRatio]) => [
      Math.max(0, Math.min(env.width - 1, Math.round((env.width - 1) * xRatio))),
      Math.max(0, Math.min(env.height - 1, Math.round((env.height - 1) * yRatio))),
    ]);
  }

  function makeGrid(fillValue) {
    return Array.from({ length: env.height }, () => Array(env.width).fill(fillValue));
  }

  function cloneGrid(grid) {
    return grid.map((row) => row.slice());
  }

  function samePosition(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }

  function inBounds(x, y) {
    return 0 <= x && x < env.width && 0 <= y && y < env.height;
  }

  function positionKey(position) {
    return `${position[0]},${position[1]}`;
  }

  function uniqueSources(sources) {
    const seen = new Set();
    const result = [];
    sources.forEach((position) => {
      const x = Math.max(0, Math.min(env.width - 1, Math.round(position[0])));
      const y = Math.max(0, Math.min(env.height - 1, Math.round(position[1])));
      const clean = [x, y];
      const key = positionKey(clean);
      if (!samePosition(clean, hub) && !seen.has(key)) {
        seen.add(key);
        result.push(clean);
      }
    });
    return result.slice(0, env.food_sources);
  }

  function buildFoodGrid() {
    const grid = makeGrid(0);
    const sources = uniqueSources(foodSources);
    if (sources.length === 0) {
      return grid;
    }
    const base = Math.floor(env.food_count / sources.length);
    const extra = env.food_count % sources.length;
    sources.forEach(([x, y], index) => {
      grid[y][x] += base + (index < extra ? 1 : 0);
    });
    return grid;
  }

  function resetRun() {
    const food = buildFoodGrid();
    state = {
      hub: copyPosition(hub),
      food,
      initialFood: cloneGrid(food),
      bytes: makeGrid(0),
      ants: Array.from({ length: env.num_ants }, () => copyPosition(hub)),
      facing: Array(env.num_ants).fill(DEFAULT_FACING),
      carrying: Array(env.num_ants).fill(false),
      delivered: 0,
      step: 0,
      numWrites: 0,
    };
    rngState = 0x5eed1234;
    render();
    updateMetrics();
  }

  function resetLayout() {
    running = false;
    hub = [Math.floor(env.width / 2), Math.floor(env.height / 2)];
    foodSources = defaultFoodSources.slice(0, env.food_sources).map(copyPosition);
    resetRun();
  }

  function setRunning(nextRunning) {
    running = nextRunning;
    runButton.textContent = running ? uiText.pause : uiText.run;
    if (state) {
      updateMetrics();
    }
    if (running) {
      animationFrame = window.requestAnimationFrame(tick);
    } else {
      window.cancelAnimationFrame(animationFrame);
    }
  }

  function tick() {
    if (!running) {
      return;
    }
    const steps = Math.max(1, Math.min(6, Number(speedInput.value) || 1));
    for (let index = 0; index < steps && running; index += 1) {
      stepPolicy();
    }
    render();
    updateMetrics();
    animationFrame = window.requestAnimationFrame(tick);
  }

  function rand() {
    rngState |= 0;
    rngState = (rngState + 0x6d2b79f5) | 0;
    let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function linear(layer, input) {
    const output = layer.bias.slice();
    const weight = layer.weight;
    for (let i = 0; i < input.length; i += 1) {
      const inputValue = input[i];
      if (inputValue === 0) {
        continue;
      }
      const row = weight[i];
      for (let j = 0; j < output.length; j += 1) {
        output[j] += inputValue * row[j];
      }
    }
    return output;
  }

  function tanhVector(values) {
    return values.map((value) => Math.tanh(value));
  }

  function forwardActor(obs) {
    const hidden0 = tanhVector(linear(actor.actor_body[0], obs));
    const hidden1 = tanhVector(linear(actor.actor_body[1], hidden0));
    return {
      move: linear(actor.move_head, hidden1),
      write: linear(actor.write_head, hidden1),
    };
  }

  function argmax(values) {
    let bestIndex = 0;
    let bestValue = values[0];
    for (let index = 1; index < values.length; index += 1) {
      if (values[index] > bestValue) {
        bestValue = values[index];
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  function sampleCategorical(logits, temperature) {
    const temp = Math.max(0.05, Number(temperature) || 1);
    const maxLogit = Math.max(...logits);
    const weights = logits.map((value) => Math.exp((value - maxLogit) / temp));
    const total = weights.reduce((sum, value) => sum + value, 0);
    let threshold = rand() * total;
    for (let index = 0; index < weights.length; index += 1) {
      threshold -= weights[index];
      if (threshold <= 0) {
        return index;
      }
    }
    return weights.length - 1;
  }

  function gridValue(grid, x, y) {
    return inBounds(x, y) ? grid[y][x] : 0;
  }

  function legacyLocalPatch(position, valueAt, invalidValue = 0) {
    const radius = env.actor_vision_radius;
    const values = [];
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        const x = position[0] + dx;
        const y = position[1] + dy;
        values.push(inBounds(x, y) ? valueAt(x, y) : invalidValue);
      }
    }
    return values;
  }

  function localOffsets(facing) {
    const radius = env.actor_vision_radius;
    const offsets = [];
    for (let dy = -radius; dy <= radius; dy += 1) {
      for (let dx = -radius; dx <= radius; dx += 1) {
        if (facing === ACTION_DOWN) {
          offsets.push([-dy, dx]);
        } else if (facing === ACTION_LEFT) {
          offsets.push([-dx, -dy]);
        } else if (facing === ACTION_UP) {
          offsets.push([dy, -dx]);
        } else {
          offsets.push([dx, dy]);
        }
      }
    }
    return offsets;
  }

  function localPatch(position, facing, valueAt, invalidValue = 0) {
    return localOffsets(facing).map(([dx, dy]) => {
      const x = position[0] + dx;
      const y = position[1] + dy;
      return inBounds(x, y) ? valueAt(x, y) : invalidValue;
    });
  }

  function antsCountGrid() {
    const grid = makeGrid(0);
    state.ants.forEach(([x, y]) => {
      grid[y][x] += 1;
    });
    return grid;
  }

  function identityFeatures(antIndex) {
    if (env.num_ants <= 1) {
      return [];
    }
    const width = Number(env.agent_identity_types || env.num_ants);
    return Array.from({ length: width }, (_, index) =>
      index === antIndex % width ? 1 : 0,
    );
  }

  function facingOneHot(facing) {
    const index = Math.max(0, Math.min(3, facing - 1));
    return Array.from({ length: 4 }, (_, itemIndex) => (itemIndex === index ? 1 : 0));
  }

  function buildLegacyActorObs(antIndex) {
    const position = state.ants[antIndex];
    const values = [];
    values.push(
      ...legacyLocalPatch(position, (x, y) => gridValue(state.food, x, y) / env.food_scale),
    );
    for (let bit = 0; bit < env.write_bits; bit += 1) {
      values.push(
        ...legacyLocalPatch(position, (x, y) => (gridValue(state.bytes, x, y) >> bit) & 1),
      );
    }
    values.push(
      ...legacyLocalPatch(
        position,
        (x, y) => (x === state.hub[0] && y === state.hub[1] ? 1 : 0),
      ),
    );
    values.push(...legacyLocalPatch(position, () => 0, 1));
    values.push(state.carrying[antIndex] ? 1 : 0);
    return values;
  }

  function buildActorObs(antIndex) {
    if (actor.observation_layout[0].startsWith("legacy")) {
      return buildLegacyActorObs(antIndex);
    }
    const position = state.ants[antIndex];
    const facing = state.facing[antIndex];
    const antCounts = state.currentAntCounts || antsCountGrid();
    const values = [];
    values.push(
      ...localPatch(position, facing, (x, y) => gridValue(state.food, x, y) / env.food_scale),
    );
    values.push(
      ...localPatch(position, facing, (x, y) => gridValue(antCounts, x, y) / env.num_ants),
    );
    for (let bit = 0; bit < env.write_bits; bit += 1) {
      values.push(
        ...localPatch(position, facing, (x, y) => (gridValue(state.bytes, x, y) >> bit) & 1),
      );
    }
    values.push(
      ...localPatch(
        position,
        facing,
        (x, y) => (x === state.hub[0] && y === state.hub[1] ? 1 : 0),
      ),
    );
    values.push(...localPatch(position, facing, () => 0, 1));
    values.push(...identityFeatures(antIndex));
    values.push(state.carrying[antIndex] ? 1 : 0);
    values.push(...facingOneHot(facing));
    return values;
  }

  function chooseAction(antIndex) {
    const obs = buildActorObs(antIndex);
    if (obs.length !== actor.actor_obs_dim) {
      throw new Error(
        `runner observation dim ${obs.length} does not match actor dim ${actor.actor_obs_dim}`,
      );
    }
    const logits = forwardActor(obs);
    const mode = actionMode.value;
    const move =
      mode === "greedy_move_greedy_write"
        ? argmax(logits.move)
        : sampleCategorical(logits.move, 1.0);
    const write =
      mode === "sampled_move_sampled_write"
        ? sampleCategorical(logits.write, 1.0)
        : argmax(logits.write);
    return [move, write];
  }

  function nextPosition(position, action) {
    let dx = 0;
    let dy = 0;
    if (action === ACTION_RIGHT) dx = 1;
    if (action === ACTION_LEFT) dx = -1;
    if (action === ACTION_DOWN) dy = 1;
    if (action === ACTION_UP) dy = -1;
    if (action === ACTION_STAY) {
      return copyPosition(position);
    }
    return [
      Math.max(0, Math.min(env.width - 1, position[0] + dx)),
      Math.max(0, Math.min(env.height - 1, position[1] + dy)),
    ];
  }

  function updateFacing(currentFacing, move) {
    return move === ACTION_UP ||
      move === ACTION_RIGHT ||
      move === ACTION_DOWN ||
      move === ACTION_LEFT
      ? move
      : currentFacing;
  }

  function applyAntAction(antIndex, action) {
    const [move, writeValue] = action;
    const next = nextPosition(state.ants[antIndex], move);
    state.facing[antIndex] = updateFacing(state.facing[antIndex], move);
    state.ants[antIndex] = next;

    const [x, y] = next;
    const hadFood = state.food[y][x] > 0;
    const isHub = x === state.hub[0] && y === state.hub[1];
    const wasCarrying = state.carrying[antIndex];
    const pickedUp = !wasCarrying && hadFood;
    if (pickedUp) {
      state.food[y][x] -= 1;
    }
    const carryingAfterPickup = wasCarrying || pickedUp;
    const delivered = carryingAfterPickup && isHub;
    state.carrying[antIndex] = carryingAfterPickup && !delivered;
    if (delivered) {
      state.delivered += 1;
    }

    const wantsWrite = move === ACTION_STAY || env.write_while_moving;
    const canWrite = wantsWrite && !hadFood && !isHub;
    if (canWrite) {
      state.bytes[y][x] = writeValue;
      state.numWrites += 1;
    }
  }

  function stepPolicy() {
    if (remainingFood() <= 0 || state.step >= env.max_steps) {
      setRunning(false);
      return;
    }
    state.currentAntCounts = antsCountGrid();
    const actions = state.ants.map((_, antIndex) => chooseAction(antIndex));
    state.currentAntCounts = null;
    actions.forEach((action, antIndex) => applyAntAction(antIndex, action));
    state.step += 1;
    if (remainingFood() <= 0 || state.step >= env.max_steps) {
      setRunning(false);
    }
  }

  function remainingFood() {
    return state.food.reduce(
      (total, row) => total + row.reduce((rowTotal, value) => rowTotal + value, 0),
      0,
    );
  }

  function carryingCount() {
    return state.carrying.filter(Boolean).length;
  }

  function nonzeroBytes() {
    return state.bytes.reduce(
      (total, row) => total + row.filter((value) => value > 0).length,
      0,
    );
  }

  function foodSourceCount() {
    return uniqueSources(foodSources).length;
  }

  function setStatus(message) {
    metrics.status.textContent = message;
  }

  function updateMetrics() {
    metrics.step.textContent = `${state.step} / ${env.max_steps}`;
    metrics.delivered.textContent = `${state.delivered} / ${env.food_count}`;
    metrics.remaining.textContent = `${remainingFood()}`;
    metrics.carrying.textContent = `${carryingCount()} / ${env.num_ants}`;
    metrics.bytes.textContent = `${nonzeroBytes()} ${uiText.tiles}`;
    metrics.sources.textContent = `${foodSourceCount()} / ${env.food_sources}`;
    if (remainingFood() <= 0) {
      setStatus(uiText.statuses.complete);
    } else if (state.step >= env.max_steps) {
      setStatus(uiText.statuses.maxSteps);
    } else {
      setStatus(running ? uiText.statuses.running : uiText.statuses.ready);
    }
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const size = Math.max(260, Math.round(rect.width));
    const dpr = window.devicePixelRatio || 1;
    if (size !== canvasCssSize || canvas.width !== Math.round(size * dpr)) {
      canvasCssSize = size;
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  function drawSprite(name, x, y, cell, alpha = 1, rotation = 0) {
    const image = sprites[name];
    const px = x * cell;
    const py = y * cell;
    ctx.save();
    ctx.globalAlpha = alpha;
    if (image && image.complete && image.naturalWidth > 0) {
      ctx.translate(px + cell / 2, py + cell / 2);
      ctx.rotate(rotation);
      ctx.drawImage(image, -cell / 2, -cell / 2, cell, cell);
    } else {
      ctx.fillStyle = name === "hub" ? "#5b4e9c" : name === "food" ? "#e8b44e" : "#c47a2c";
      ctx.beginPath();
      ctx.arc(px + cell / 2, py + cell / 2, cell * 0.35, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function antRotation(facing) {
    if (facing === ACTION_UP) return -Math.PI / 2;
    if (facing === ACTION_DOWN) return Math.PI / 2;
    if (facing === ACTION_LEFT) return Math.PI;
    return 0;
  }

  function render() {
    if (!state) {
      return;
    }
    resizeCanvas();
    const size = canvasCssSize;
    const cell = size / env.width;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#d7cfb5";
    ctx.fillRect(0, 0, size, size);

    for (let y = 0; y < env.height; y += 1) {
      for (let x = 0; x < env.width; x += 1) {
        if (sprites.tile && sprites.tile.complete && sprites.tile.naturalWidth > 0) {
          ctx.drawImage(sprites.tile, x * cell, y * cell, cell, cell);
        }
        const byteValue = state.bytes[y][x];
        if (byteValue > 0) {
          const ratio = byteValue / Math.max((1 << env.write_bits) - 1, 1);
          ctx.fillStyle = `rgba(${Math.round(40 + 180 * ratio)}, 92, ${Math.round(
            255 - 120 * ratio,
          )}, 0.38)`;
          ctx.fillRect(x * cell, y * cell, cell, cell);
          if (cell >= 18) {
            ctx.fillStyle = "#181f24";
            ctx.font = `${Math.max(8, Math.floor(cell * 0.45))}px Georgia, serif`;
            ctx.fillText(String(byteValue), x * cell + 2, y * cell + cell * 0.55);
          }
        }
      }
    }

    ctx.strokeStyle = "rgba(80, 70, 52, 0.16)";
    ctx.lineWidth = 1;
    for (let index = 0; index <= env.width; index += 1) {
      const pos = index * cell;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(size, pos);
      ctx.stroke();
    }

    drawSprite("hub", state.hub[0], state.hub[1], cell);
    for (let y = 0; y < env.height; y += 1) {
      for (let x = 0; x < env.width; x += 1) {
        const amount = state.food[y][x];
        if (amount <= 0) {
          continue;
        }
        const initial = Math.max(state.initialFood[y][x], 1);
        drawSprite("food", x, y, cell, Math.max(0.22, Math.min(1, amount / initial)));
        if (amount > 1 && cell >= 14) {
          ctx.fillStyle = "#fff7db";
          ctx.font = `bold ${Math.max(9, Math.floor(cell * 0.48))}px Georgia, serif`;
          ctx.fillText(String(amount), x * cell + cell * 0.52, y * cell + cell * 0.74);
        }
      }
    }

    state.ants.forEach(([x, y], index) => {
      drawSprite("ant", x, y, cell, 1, antRotation(state.facing[index]));
      if (state.carrying[index]) {
        ctx.fillStyle = "#bc702d";
        ctx.beginPath();
        ctx.arc(x * cell + cell * 0.74, y * cell + cell * 0.25, Math.max(3, cell * 0.13), 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#efa752";
        ctx.beginPath();
        ctx.arc(x * cell + cell * 0.74, y * cell + cell * 0.25, Math.max(1, cell * 0.06), 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  function cellFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * env.width);
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * env.height);
    return [Math.max(0, Math.min(env.width - 1, x)), Math.max(0, Math.min(env.height - 1, y))];
  }

  function applyPlacement(position) {
    setRunning(false);
    if (placementMode === "hub") {
      hub = copyPosition(position);
      foodSources = foodSources.filter((source) => !samePosition(source, hub));
    } else if (placementMode === "food") {
      if (!samePosition(position, hub)) {
        const exists = foodSources.some((source) => samePosition(source, position));
        if (!exists) {
          foodSources = uniqueSources([...foodSources, position]);
        }
      }
    } else if (placementMode === "erase") {
      foodSources = foodSources.filter((source) => !samePosition(source, position));
    }
    resetRun();
  }

  function setPlacementMode(mode) {
    placementMode = mode;
    modeButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.placeMode === mode);
    });
  }

  function loadSprites() {
    return Promise.all(
      Object.entries(spritePaths).map(([name, src]) => {
        const image = new Image();
        image.src = src;
        sprites[name] = image;
        return new Promise((resolve) => {
          image.addEventListener("load", resolve, { once: true });
          image.addEventListener("error", resolve, { once: true });
        });
      }),
    );
  }

  function setText(selector, text) {
    const element = document.querySelector(selector);
    if (element) {
      element.textContent = text;
    }
  }

  function setDefinitionLabel(valueSelector, text) {
    const value = document.querySelector(valueSelector);
    const label = value ? value.previousElementSibling : null;
    if (label && label.tagName === "DT") {
      label.textContent = text;
    }
  }

  function localizeUi() {
    setText(".runner-panel h3", "50x50 / 60 hormigas / 8 bits");
    setText(
      ".runner-panel .caption",
      "La entrada exportada del actor tiene 313 dimensiones: parche de comida orientado por dirección, parche de cantidad de hormigas, parches de bits de bytes, hormiguero, borde, rasgos de identidad repetidos, bandera de carga y dirección one-hot.",
    );
    setText(".control-label", "colocar");
    setText("label[for='runner-action-mode']", "modo de acción");
    setText("label[for='runner-speed']", "pasos por cuadro");
    setDefinitionLabel("#runner-step", "paso");
    setDefinitionLabel("#runner-delivered", "entregado");
    setDefinitionLabel("#runner-remaining", "restante");
    setDefinitionLabel("#runner-carrying", "transportando");
    setDefinitionLabel("#runner-bytes", "bytes no cero");
    setDefinitionLabel("#runner-sources", "fuentes de comida");
    setDefinitionLabel("#runner-status", "estado");
    canvas.setAttribute("aria-label", "Sandbox interactivo de la política de hormigas 50x50");
    const placementTabs = document.querySelector(".policy-tabs.runner-tabs");
    if (placementTabs) {
      placementTabs.setAttribute("aria-label", "Modo de colocación");
    }
    modeButtons.forEach((button) => {
      const label = uiText.placement[button.dataset.placeMode];
      if (label) {
        button.textContent = label;
      }
    });
    Array.from(actionMode.options).forEach((option) => {
      const label = uiText.actionModes[option.value];
      if (label) {
        option.textContent = label;
      }
    });
    runButton.textContent = uiText.run;
    stepButton.textContent = uiText.step;
    resetRunButton.textContent = uiText.resetRun;
    resetLayoutButton.textContent = uiText.resetLayout;
    setStatus(uiText.loading);
  }

  localizeUi();
  modeButtons.forEach((button) => {
    button.addEventListener("click", () => setPlacementMode(button.dataset.placeMode));
  });
  canvas.addEventListener("click", (event) => applyPlacement(cellFromEvent(event)));
  runButton.addEventListener("click", () => setRunning(!running));
  stepButton.addEventListener("click", () => {
    setRunning(false);
    stepPolicy();
    render();
    updateMetrics();
  });
  resetRunButton.addEventListener("click", () => {
    setRunning(false);
    resetRun();
  });
  resetLayoutButton.addEventListener("click", resetLayout);
  window.addEventListener("resize", render);

  setPlacementMode("food");
  loadSprites().then(() => {
    resetRun();
  });
})();
