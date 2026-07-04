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
    obstacles: document.querySelector("#runner-obstacles"),
    sources: document.querySelector("#runner-sources"),
    status: document.querySelector("#runner-status"),
  };
  const runButton = document.querySelector("#runner-run");
  const stepButton = document.querySelector("#runner-step-once");
  const resetRunButton = document.querySelector("#runner-reset-run");
  const resetLayoutButton = document.querySelector("#runner-reset-layout");
  const modeButtons = Array.from(document.querySelectorAll("[data-place-mode]"));
  const actionMode = document.querySelector("#runner-action-mode");
  const paletteSelect = document.querySelector("#runner-palette");
  const speedInput = document.querySelector("#runner-speed");
  const configControls = {
    gridSize: document.querySelector("#runner-grid-size"),
    antCount: document.querySelector("#runner-ant-count"),
    foodCount: document.querySelector("#runner-food-count"),
    sourceCount: document.querySelector("#runner-source-count"),
    maxSteps: document.querySelector("#runner-max-steps"),
  };
  const configLimits = {
    gridSize: { min: 15, max: Number.MAX_SAFE_INTEGER },
    antCount: { min: 1, max: Infinity },
    foodCount: { min: 0, max: Infinity },
    sourceCount: { min: 0, max: Number.MAX_SAFE_INTEGER },
    maxSteps: { min: 1, max: Number.MAX_SAFE_INTEGER },
  };
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
      food: "Cookie +1",
      obstacle: "Pared",
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
  const LARGE_MAP_THRESHOLD = 100;
  const spritePaths = {
    ant: "report-site/assets/sprites/ant.png",
    food: "report-site/assets/sprites/food.png",
    hub: "report-site/assets/sprites/hub.png",
    tile: "report-site/assets/sprites/tile.png",
  };
  const palettes = {
    natural: {
      floor: "#d7cfb5",
      largeFloor: "#d3ccb4",
      grid: "rgba(80, 70, 52, 0.16)",
      byteLow: [88, 112, 190],
      byteHigh: [222, 76, 90],
      byteAlpha: 0.38,
      byteText: "#181f24",
      hub: "#5b4e9c",
      food: "#e8b44e",
      ant: "#a8452e",
      obstacle: "#46515a",
      carryingOuter: "#bc702d",
      carryingInner: "#efa752",
      useTileSprite: true,
    },
    paper: {
      floor: "#f1e7cf",
      largeFloor: "#e6dcc5",
      grid: "rgba(91, 70, 40, 0.14)",
      byteLow: [74, 123, 138],
      byteHigh: [156, 78, 52],
      byteAlpha: 0.42,
      byteText: "#2f2a23",
      hub: "#466b73",
      food: "#c9912e",
      ant: "#7c3529",
      obstacle: "#5d625f",
      carryingOuter: "#a46d2d",
      carryingInner: "#e7bd68",
      useTileSprite: false,
    },
    night: {
      floor: "#172027",
      largeFloor: "#111920",
      grid: "rgba(226, 214, 180, 0.12)",
      byteLow: [71, 129, 201],
      byteHigh: [235, 150, 72],
      byteAlpha: 0.5,
      byteText: "#f7efd8",
      hub: "#9a8cff",
      food: "#f6c85f",
      ant: "#e4674f",
      obstacle: "#6f7a86",
      carryingOuter: "#d58b42",
      carryingInner: "#ffd37a",
      useTileSprite: false,
    },
    contrast: {
      floor: "#f5f1e4",
      largeFloor: "#ece6d5",
      grid: "rgba(22, 22, 18, 0.24)",
      byteLow: [0, 94, 154],
      byteHigh: [190, 38, 51],
      byteAlpha: 0.52,
      byteText: "#171511",
      hub: "#2f3a8f",
      food: "#d89b00",
      ant: "#b62222",
      obstacle: "#2d3439",
      carryingOuter: "#8f5a00",
      carryingInner: "#ffca45",
      useTileSprite: false,
    },
  };

  let state = null;
  let running = false;
  let placementMode = "food";
  let hub = [Math.floor(env.width / 2), Math.floor(env.height / 2)];
  let obstacleLayout = makeGrid(0);
  let obstacleKeys = new Set();
  let foodSources = buildDefaultFoodSources();
  let foodAmounts = distributeFoodAmounts(foodSources, env.food_count);
  let rngState = 0x5eed1234;
  let animationFrame = 0;
  let canvasCssSize = 0;
  let activePointerId = null;
  let lastPaintedKey = null;
  let obstaclePaintValue = 1;
  const sprites = {};

  function copyPosition(position) {
    return [Number(position[0]), Number(position[1])];
  }

  function clampInteger(value, min, max, fallback) {
    const parsed = Number.isFinite(value) ? Math.round(value) : fallback;
    return Math.max(min, Math.min(max, parsed));
  }

  function maxSourceCountFor(gridSize, blockedCells = 0) {
    return Math.max(0, gridSize * gridSize - 1 - blockedCells);
  }

  function readIntegerControl(control, limits, fallback, maxOverride = limits.max) {
    const raw = control ? Number.parseInt(control.value, 10) : fallback;
    return clampInteger(raw, limits.min, maxOverride, fallback);
  }

  function writeIntegerControl(control, value, maxOverride = null) {
    if (!control) {
      return;
    }
    if (maxOverride !== null) {
      control.max = String(maxOverride);
    } else {
      control.removeAttribute("max");
    }
    control.value = String(value);
  }

  function syncConfigControls() {
    const sourceMax = maxSourceCountFor(env.width, obstacleKeys.size);
    writeIntegerControl(configControls.gridSize, env.width, null);
    writeIntegerControl(configControls.antCount, env.num_ants);
    writeIntegerControl(configControls.foodCount, env.food_count);
    writeIntegerControl(configControls.sourceCount, env.food_sources, sourceMax);
    writeIntegerControl(configControls.maxSteps, env.max_steps);
    updateSandboxTitle();
  }

  function scalePosition(position, fromWidth, fromHeight, toWidth, toHeight) {
    const xScale = fromWidth > 1 ? (toWidth - 1) / (fromWidth - 1) : 0;
    const yScale = fromHeight > 1 ? (toHeight - 1) / (fromHeight - 1) : 0;
    return [
      Math.max(0, Math.min(toWidth - 1, Math.round(position[0] * xScale))),
      Math.max(0, Math.min(toHeight - 1, Math.round(position[1] * yScale))),
    ];
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
    const gridSide = Math.ceil(Math.sqrt(env.food_sources + 1));
    for (let yIndex = 1; yIndex <= gridSide; yIndex += 1) {
      for (let xIndex = 1; xIndex <= gridSide; xIndex += 1) {
        candidates.push([xIndex / (gridSide + 1), yIndex / (gridSide + 1)]);
      }
    }
    const scaled = candidates.map(([xRatio, yRatio]) => [
      Math.max(0, Math.min(env.width - 1, Math.round((env.width - 1) * xRatio))),
      Math.max(0, Math.min(env.height - 1, Math.round((env.height - 1) * yRatio))),
    ]);
    const scaledSources = uniqueSources(scaled, env.food_sources);
    if (scaledSources.length >= env.food_sources) {
      return scaledSources;
    }
    const fallback = [];
    for (let y = 0; y < env.height; y += 1) {
      for (let x = 0; x < env.width; x += 1) {
        fallback.push([x, y]);
      }
    }
    return uniqueSources([...scaledSources, ...fallback], env.food_sources);
  }

  function completeFoodSources(sources) {
    const cleaned = uniqueSources(sources, env.food_sources);
    if (cleaned.length >= env.food_sources) {
      return cleaned;
    }
    return uniqueSources([...cleaned, ...buildDefaultFoodSources()], env.food_sources);
  }

  function distributeFoodAmounts(sources, totalFood) {
    const cleanSources = uniqueSources(sources);
    const amounts = new Map();
    if (cleanSources.length === 0) {
      return amounts;
    }
    const normalizedTotal = Math.max(cleanSources.length, Math.round(totalFood));
    const base = Math.floor(normalizedTotal / cleanSources.length);
    const extra = normalizedTotal % cleanSources.length;
    cleanSources.forEach((source, index) => {
      amounts.set(positionKey(source), base + (index < extra ? 1 : 0));
    });
    return amounts;
  }

  function totalFoodAmount() {
    return Array.from(foodAmounts.values()).reduce((sum, amount) => sum + amount, 0);
  }

  function setDistributedFoodLayout(sources, totalFood) {
    foodSources = uniqueSources(sources);
    env.food_sources = foodSources.length;
    env.food_count = foodSources.length === 0 ? 0 : Math.max(foodSources.length, totalFood);
    foodAmounts = distributeFoodAmounts(foodSources, env.food_count);
  }

  function syncFoodLayoutFromAmounts() {
    const cleanSources = uniqueSources(foodSources).filter((source) => {
      const amount = foodAmounts.get(positionKey(source)) || 0;
      return amount > 0;
    });
    const cleanAmounts = new Map();
    cleanSources.forEach((source) => {
      cleanAmounts.set(positionKey(source), Math.max(1, Math.round(foodAmounts.get(positionKey(source)))));
    });
    foodSources = cleanSources;
    foodAmounts = cleanAmounts;
    env.food_sources = foodSources.length;
    env.food_count = totalFoodAmount();
  }

  function addFoodAt(position) {
    if (samePosition(position, hub)) {
      return;
    }
    removeObstacleAt(position);
    const source = copyPosition(position);
    const key = positionKey(source);
    if (!foodSources.some((candidate) => samePosition(candidate, source))) {
      foodSources = uniqueSources([...foodSources, source]);
    }
    foodAmounts.set(key, (foodAmounts.get(key) || 0) + 1);
    syncFoodLayoutFromAmounts();
  }

  function removeFoodAt(position) {
    const key = positionKey(position);
    foodAmounts.delete(key);
    foodSources = foodSources.filter((source) => !samePosition(source, position));
    syncFoodLayoutFromAmounts();
  }

  function setObstaclePlacement(position, blocked) {
    if (blocked && samePosition(position, hub)) {
      return;
    }
    if (blocked) {
      removeFoodAt(position);
    }
    setObstacleAt(position, blocked);
  }

  function updateSandboxTitle() {
    const antWord = env.num_ants === 1 ? "hormiga" : "hormigas";
    setText(
      ".runner-panel h3",
      `Actor 50x50 en sandbox ${env.width}x${env.height} / ${env.num_ants} ${antWord}`,
    );
    canvas.setAttribute(
      "aria-label",
      `Sandbox interactivo de la política de hormigas en grilla ${env.width}x${env.height}`,
    );
  }

  function applySandboxConfig() {
    setRunning(false);
    const previousWidth = env.width;
    const previousHeight = env.height;
    const gridSize = readIntegerControl(
      configControls.gridSize,
      configLimits.gridSize,
      env.width,
    );
    const antCount = readIntegerControl(
      configControls.antCount,
      configLimits.antCount,
      env.num_ants,
    );
    let foodCount = readIntegerControl(
      configControls.foodCount,
      configLimits.foodCount,
      env.food_count,
    );
    const sourceMax = maxSourceCountFor(gridSize);
    const sourceCount = readIntegerControl(
      configControls.sourceCount,
      configLimits.sourceCount,
      env.food_sources,
      sourceMax,
    );
    const maxSteps = readIntegerControl(
      configControls.maxSteps,
      configLimits.maxSteps,
      env.max_steps,
    );
    foodCount = sourceCount === 0 ? 0 : Math.max(foodCount, sourceCount);

    env.width = gridSize;
    env.height = gridSize;
    env.num_ants = antCount;
    env.food_count = foodCount;
    env.food_sources = sourceCount;
    env.max_steps = maxSteps;

    if (previousWidth !== env.width || previousHeight !== env.height) {
      hub = scalePosition(hub, previousWidth, previousHeight, env.width, env.height);
      scaleObstacleLayout(previousWidth, previousHeight, env.width, env.height);
      foodSources = foodSources.map((source) =>
        scalePosition(source, previousWidth, previousHeight, env.width, env.height),
      );
    }
    removeObstacleAt(hub);
    setDistributedFoodLayout(completeFoodSources(foodSources), foodCount);
    syncConfigControls();
    resetRun();
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

  function positionFromKey(key) {
    return key.split(",").map(Number);
  }

  function isLayoutObstacle(position) {
    const [x, y] = position;
    return inBounds(x, y) && obstacleLayout[y][x] > 0;
  }

  function isStateObstacle(position) {
    const [x, y] = position;
    return state && inBounds(x, y) && state.obstacles[y][x] > 0;
  }

  function setObstacleAt(position, blocked) {
    const [x, y] = copyPosition(position);
    if (!inBounds(x, y)) {
      return;
    }
    const key = positionKey([x, y]);
    const shouldBlock = Boolean(blocked) && !samePosition([x, y], hub);
    obstacleLayout[y][x] = shouldBlock ? 1 : 0;
    if (shouldBlock) {
      obstacleKeys.add(key);
    } else {
      obstacleKeys.delete(key);
    }
  }

  function removeObstacleAt(position) {
    setObstacleAt(position, false);
  }

  function obstaclePositions() {
    return Array.from(obstacleKeys, positionFromKey);
  }

  function rebuildObstacleLayout(positions) {
    obstacleLayout = makeGrid(0);
    obstacleKeys = new Set();
    positions.forEach((position) => setObstacleAt(position, true));
  }

  function scaleObstacleLayout(fromWidth, fromHeight, toWidth, toHeight) {
    const scaled = obstaclePositions().map((position) =>
      scalePosition(position, fromWidth, fromHeight, toWidth, toHeight),
    );
    rebuildObstacleLayout(scaled);
    removeObstacleAt(hub);
  }

  function obstacleCount() {
    return obstacleKeys.size;
  }

  function uniqueSources(sources, limit = Infinity) {
    const seen = new Set();
    const result = [];
    sources.forEach((position) => {
      if (result.length >= limit) {
        return;
      }
      const x = Math.max(0, Math.min(env.width - 1, Math.round(position[0])));
      const y = Math.max(0, Math.min(env.height - 1, Math.round(position[1])));
      const clean = [x, y];
      const key = positionKey(clean);
      if (!samePosition(clean, hub) && !isLayoutObstacle(clean) && !seen.has(key)) {
        seen.add(key);
        result.push(clean);
      }
    });
    return result;
  }

  function buildFoodGrid() {
    const grid = makeGrid(0);
    const sources = uniqueSources(foodSources);
    if (sources.length === 0) {
      return grid;
    }
    sources.forEach(([x, y]) => {
      const amount = Math.max(0, Math.round(foodAmounts.get(positionKey([x, y])) || 0));
      if (amount > 0) {
        grid[y][x] += amount;
      }
    });
    return grid;
  }

  function resetRun() {
    const food = buildFoodGrid();
    state = {
      hub: copyPosition(hub),
      food,
      initialFood: cloneGrid(food),
      obstacles: cloneGrid(obstacleLayout),
      obstacleKeys: new Set(obstacleKeys),
      bytes: makeGrid(0),
      writtenKeys: new Set(),
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
    setRunning(false);
    hub = [Math.floor(env.width / 2), Math.floor(env.height / 2)];
    rebuildObstacleLayout([]);
    setDistributedFoodLayout(buildDefaultFoodSources(), env.food_count);
    syncConfigControls();
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
    const width = Number(env.agent_identity_types || 0);
    if (width <= 0) {
      return [];
    }
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
    values.push(
      ...legacyLocalPatch(position, (x, y) => (gridValue(state.obstacles, x, y) ? 1 : 0), 1),
    );
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
    values.push(
      ...localPatch(position, facing, (x, y) => (gridValue(state.obstacles, x, y) ? 1 : 0), 1),
    );
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
    const next = [
      Math.max(0, Math.min(env.width - 1, position[0] + dx)),
      Math.max(0, Math.min(env.height - 1, position[1] + dy)),
    ];
    return isStateObstacle(next) ? copyPosition(position) : next;
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
      const key = positionKey([x, y]);
      if (writeValue > 0) {
        state.writtenKeys.add(key);
      } else {
        state.writtenKeys.delete(key);
      }
      state.numWrites += 1;
    }
  }

  function stepPolicy() {
    if (allFoodDelivered() || state.step >= env.max_steps) {
      setRunning(false);
      return;
    }
    state.currentAntCounts = antsCountGrid();
    const actions = state.ants.map((_, antIndex) => chooseAction(antIndex));
    state.currentAntCounts = null;
    actions.forEach((action, antIndex) => applyAntAction(antIndex, action));
    state.step += 1;
    if (allFoodDelivered() || state.step >= env.max_steps) {
      setRunning(false);
    }
  }

  function sumGrid(grid) {
    return grid.reduce(
      (total, row) => total + row.reduce((rowTotal, value) => rowTotal + value, 0),
      0,
    );
  }

  function remainingFood() {
    return sumGrid(state.food);
  }

  function deliveryTarget() {
    return state ? sumGrid(state.initialFood) : env.food_count;
  }

  function allFoodDelivered() {
    return state.delivered >= deliveryTarget();
  }

  function carryingCount() {
    return state.carrying.filter(Boolean).length;
  }

  function nonzeroBytes() {
    if (state.writtenKeys) {
      return state.writtenKeys.size;
    }
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
    const target = deliveryTarget();
    metrics.step.textContent = `${state.step} / ${env.max_steps}`;
    metrics.delivered.textContent = `${state.delivered} / ${target}`;
    metrics.remaining.textContent = `${remainingFood()}`;
    metrics.carrying.textContent = `${carryingCount()} / ${env.num_ants}`;
    metrics.bytes.textContent = `${nonzeroBytes()} ${uiText.tiles}`;
    metrics.obstacles.textContent = `${obstacleCount()} ${uiText.tiles}`;
    metrics.sources.textContent = `${foodSourceCount()} posiciones`;
    if (allFoodDelivered()) {
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

  function drawSprite(
    name,
    x,
    y,
    cell,
    alpha = 1,
    rotation = 0,
    palette = currentPalette(),
    largeMap = false,
  ) {
    const image = sprites[name];
    const px = x * cell;
    const py = y * cell;
    const canUseSprite =
      !largeMap &&
      palette.useTileSprite &&
      image &&
      image.complete &&
      image.naturalWidth > 0 &&
      cell >= 5;
    if (canUseSprite) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(px + cell / 2, py + cell / 2);
      ctx.rotate(rotation);
      ctx.drawImage(image, -cell / 2, -cell / 2, cell, cell);
      ctx.restore();
    } else {
      drawMarker(name, x, y, cell, palette, alpha, rotation);
    }
  }

  function antRotation(facing) {
    if (facing === ACTION_UP) return -Math.PI / 2;
    if (facing === ACTION_DOWN) return Math.PI / 2;
    if (facing === ACTION_LEFT) return Math.PI;
    return 0;
  }

  function currentPalette() {
    const key = paletteSelect ? paletteSelect.value : "natural";
    return palettes[key] || palettes.natural;
  }

  function isLargeMap() {
    return env.width > LARGE_MAP_THRESHOLD || env.height > LARGE_MAP_THRESHOLD;
  }

  function rgba(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function mixRgb(start, end, ratio) {
    return start.map((value, index) =>
      Math.round(value + (end[index] - value) * ratio),
    );
  }

  function drawByteCell(x, y, value, cell, palette, largeMap) {
    if (value <= 0) {
      return;
    }
    const ratio = value / Math.max((1 << env.write_bits) - 1, 1);
    const color = mixRgb(palette.byteLow, palette.byteHigh, ratio);
    const px = x * cell;
    const py = y * cell;
    const marker = largeMap ? Math.max(cell, 1.4) : cell;
    ctx.fillStyle = rgba(color, palette.byteAlpha);
    ctx.fillRect(px + (cell - marker) / 2, py + (cell - marker) / 2, marker, marker);
    if (!largeMap && cell >= 18) {
      ctx.fillStyle = palette.byteText;
      ctx.font = `${Math.max(8, Math.floor(cell * 0.45))}px Georgia, serif`;
      ctx.fillText(String(value), px + 2, py + cell * 0.55);
    }
  }

  function drawObstacleCell(x, y, cell, palette, largeMap) {
    const px = x * cell;
    const py = y * cell;
    const inset = largeMap ? 0 : Math.max(0.5, cell * 0.08);
    ctx.fillStyle = palette.obstacle;
    ctx.fillRect(px + inset, py + inset, Math.max(1, cell - inset * 2), Math.max(1, cell - inset * 2));
    if (!largeMap && cell >= 10) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
      ctx.lineWidth = Math.max(1, cell * 0.06);
      ctx.beginPath();
      ctx.moveTo(px + cell * 0.22, py + cell * 0.74);
      ctx.lineTo(px + cell * 0.78, py + cell * 0.26);
      ctx.stroke();
    }
  }

  function drawMarker(name, x, y, cell, palette, alpha = 1, rotation = 0) {
    const px = x * cell + cell / 2;
    const py = y * cell + cell / 2;
    const radius = Math.max(2, cell * 0.38);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(px, py);
    if (name === "hub") {
      const size = Math.max(3, cell * 0.72);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = palette.hub;
      ctx.fillRect(-size / 2, -size / 2, size, size);
    } else if (name === "food") {
      ctx.fillStyle = palette.food;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fill();
    } else {
      const size = Math.max(3, cell * 0.72);
      ctx.rotate(rotation);
      ctx.fillStyle = palette.ant;
      ctx.beginPath();
      ctx.moveTo(size * 0.62, 0);
      ctx.lineTo(-size * 0.45, -size * 0.38);
      ctx.lineTo(-size * 0.3, 0);
      ctx.lineTo(-size * 0.45, size * 0.38);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  function render() {
    if (!state) {
      return;
    }
    resizeCanvas();
    const size = canvasCssSize;
    const cell = size / env.width;
    const palette = currentPalette();
    const largeMap = isLargeMap();
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = largeMap ? palette.largeFloor : palette.floor;
    ctx.fillRect(0, 0, size, size);

    if (!largeMap) {
      for (let y = 0; y < env.height; y += 1) {
        for (let x = 0; x < env.width; x += 1) {
          if (
            palette.useTileSprite &&
            sprites.tile &&
            sprites.tile.complete &&
            sprites.tile.naturalWidth > 0
          ) {
            ctx.drawImage(sprites.tile, x * cell, y * cell, cell, cell);
          }
          drawByteCell(x, y, state.bytes[y][x], cell, palette, false);
          if (state.obstacles[y][x] > 0) {
            drawObstacleCell(x, y, cell, palette, false);
          }
        }
      }
    } else {
      state.writtenKeys.forEach((key) => {
        const [x, y] = key.split(",").map(Number);
        if (inBounds(x, y)) {
          drawByteCell(x, y, state.bytes[y][x], cell, palette, true);
        }
      });
      state.obstacleKeys.forEach((key) => {
        const [x, y] = positionFromKey(key);
        if (inBounds(x, y)) {
          drawObstacleCell(x, y, cell, palette, true);
        }
      });
    }

    if (!largeMap) {
      ctx.strokeStyle = palette.grid;
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
    }

    drawSprite("hub", state.hub[0], state.hub[1], cell, 1, 0, palette, largeMap);
    const drawFoodAt = (x, y) => {
      const amount = state.food[y][x];
      if (amount <= 0) {
        return;
      }
      const initial = Math.max(state.initialFood[y][x], 1);
      drawSprite("food", x, y, cell, Math.max(0.22, Math.min(1, amount / initial)), 0, palette, largeMap);
      if (!largeMap && amount > 1 && cell >= 14) {
        ctx.fillStyle = "#fff7db";
        ctx.font = `bold ${Math.max(9, Math.floor(cell * 0.48))}px Georgia, serif`;
        ctx.fillText(String(amount), x * cell + cell * 0.52, y * cell + cell * 0.74);
      }
    };
    if (largeMap) {
      foodSources.forEach(([x, y]) => {
        if (inBounds(x, y)) {
          drawFoodAt(x, y);
        }
      });
    } else {
      for (let y = 0; y < env.height; y += 1) {
        for (let x = 0; x < env.width; x += 1) {
          drawFoodAt(x, y);
        }
      }
    }

    state.ants.forEach(([x, y], index) => {
      drawSprite("ant", x, y, cell, 1, antRotation(state.facing[index]), palette, largeMap);
      if (state.carrying[index]) {
        const carryRadius = largeMap ? Math.max(1.4, cell * 0.22) : Math.max(3, cell * 0.13);
        const shineRadius = largeMap ? Math.max(0.8, cell * 0.1) : Math.max(1, cell * 0.06);
        ctx.fillStyle = palette.carryingOuter;
        ctx.beginPath();
        ctx.arc(x * cell + cell * 0.74, y * cell + cell * 0.25, carryRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = palette.carryingInner;
        ctx.beginPath();
        ctx.arc(x * cell + cell * 0.74, y * cell + cell * 0.25, shineRadius, 0, Math.PI * 2);
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

  function applyPlacement(position, options = {}) {
    setRunning(false);
    if (placementMode === "hub") {
      hub = copyPosition(position);
      removeObstacleAt(hub);
      removeFoodAt(hub);
    } else if (placementMode === "food") {
      addFoodAt(position);
    } else if (placementMode === "obstacle") {
      const blocked = options.obstacleValue ?? !isLayoutObstacle(position);
      setObstaclePlacement(position, blocked);
    } else if (placementMode === "erase") {
      removeFoodAt(position);
      removeObstacleAt(position);
    }
    syncConfigControls();
    resetRun();
  }

  function beginPlacement(event) {
    event.preventDefault();
    const position = cellFromEvent(event);
    activePointerId = event.pointerId;
    lastPaintedKey = positionKey(position);
    if (placementMode === "obstacle") {
      obstaclePaintValue = isLayoutObstacle(position) ? 0 : 1;
      applyPlacement(position, { obstacleValue: obstaclePaintValue });
    } else {
      applyPlacement(position);
    }
    if (canvas.setPointerCapture) {
      canvas.setPointerCapture(event.pointerId);
    }
  }

  function continuePlacement(event) {
    if (event.pointerId !== activePointerId) {
      return;
    }
    if (placementMode !== "obstacle" && placementMode !== "erase") {
      return;
    }
    event.preventDefault();
    const position = cellFromEvent(event);
    const key = positionKey(position);
    if (key === lastPaintedKey) {
      return;
    }
    lastPaintedKey = key;
    if (placementMode === "obstacle") {
      applyPlacement(position, { obstacleValue: obstaclePaintValue });
    } else {
      applyPlacement(position);
    }
  }

  function endPlacement(event) {
    if (event.pointerId !== activePointerId) {
      return;
    }
    if (canvas.releasePointerCapture) {
      canvas.releasePointerCapture(event.pointerId);
    }
    activePointerId = null;
    lastPaintedKey = null;
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
    updateSandboxTitle();
    setText(
      ".runner-panel .caption",
      "Los pesos no cambian: es el actor best 50x50 de 60 hormigas exportado desde el checkpoint estabilizado. Los controles modifican el entorno simulado para explorar generalización local, no para declarar una política reentrenada en otro tamaño.",
    );
    setText(".control-label", "colocar");
    setText("label[for='runner-action-mode']", "modo de acción");
    setText("label[for='runner-palette']", "paleta");
    setText("label[for='runner-speed']", "pasos por cuadro");
    const configLabels = [
      [configControls.gridSize, "tamaño de grilla"],
      [configControls.antCount, "hormigas"],
      [configControls.foodCount, "cookies en el mapa"],
      [configControls.sourceCount, "posiciones de cookies"],
      [configControls.maxSteps, "límite de truncación"],
    ];
    configLabels.forEach(([control, label]) => {
      const labelElement = control ? control.closest("label") : null;
      const span = labelElement ? labelElement.querySelector("span") : null;
      if (span) {
        span.textContent = label;
      }
    });
    setDefinitionLabel("#runner-step", "paso");
    setDefinitionLabel("#runner-delivered", "entregado");
    setDefinitionLabel("#runner-remaining", "en mapa");
    setDefinitionLabel("#runner-carrying", "transportando");
    setDefinitionLabel("#runner-bytes", "bytes no cero");
    setDefinitionLabel("#runner-obstacles", "paredes");
    setDefinitionLabel("#runner-sources", "fuentes de comida");
    setDefinitionLabel("#runner-status", "estado");
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
    if (paletteSelect) {
      const paletteLabels = {
        natural: "natural",
        paper: "papel",
        night: "nocturna",
        contrast: "contraste",
      };
      Array.from(paletteSelect.options).forEach((option) => {
        const label = paletteLabels[option.value];
        if (label) {
          option.textContent = label;
        }
      });
    }
    runButton.textContent = uiText.run;
    stepButton.textContent = uiText.step;
    resetRunButton.textContent = uiText.resetRun;
    resetLayoutButton.textContent = uiText.resetLayout;
    syncConfigControls();
    setStatus(uiText.loading);
  }

  localizeUi();
  modeButtons.forEach((button) => {
    button.addEventListener("click", () => setPlacementMode(button.dataset.placeMode));
  });
  canvas.addEventListener("pointerdown", beginPlacement);
  canvas.addEventListener("pointermove", continuePlacement);
  canvas.addEventListener("pointerup", endPlacement);
  canvas.addEventListener("pointercancel", endPlacement);
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
  Object.values(configControls).forEach((control) => {
    if (!control) {
      return;
    }
    control.addEventListener("change", applySandboxConfig);
    control.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        control.blur();
        applySandboxConfig();
      }
    });
  });
  if (paletteSelect) {
    paletteSelect.addEventListener("change", render);
  }
  window.addEventListener("resize", render);

  setPlacementMode("food");
  loadSprites().then(() => {
    resetRun();
  });
})();
