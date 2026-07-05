(() => {
  const canvas = document.querySelector("#ambient-ant-field");
  const data = window.CoolAntzPolicyRunnerData;

  if (!canvas || !canvas.getContext || !data || !data.actor || !data.env) {
    return;
  }

  const ctx = canvas.getContext("2d", { alpha: true });
  const actor = data.actor;
  const env = {
    ...data.env,
    width: 50,
    height: 50,
    num_ants: 4,
    food_count: 36,
    food_sources: 9,
    max_steps: Number.MAX_SAFE_INTEGER,
  };
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const maxDpr = 2;
  const policyStepSeconds = 0.04;
  const messageLifetime = 32;
  const cookieRefillThreshold = 10;
  const maxLayerOpacity = 0.72;
  const minLayerOpacity = 0;
  const hubCornerMarginCells = 0;
  const foodSpawnMarginCells = 3;
  const ACTION_STAY = 0;
  const ACTION_UP = 1;
  const ACTION_RIGHT = 2;
  const ACTION_DOWN = 3;
  const ACTION_LEFT = 4;
  const DEFAULT_FACING = ACTION_RIGHT;
  const sprites = {
    ant: loadImage("report-site/assets/sprites/ant.png"),
    food: loadImage("report-site/assets/sprites/food.png"),
    hub: loadImage("report-site/assets/sprites/hub.png"),
  };
  const palette = {
    byteLow: [88, 112, 190],
    byteHigh: [222, 76, 90],
    byteAlpha: 0.38,
    hub: "#5b4e9c",
    food: "#e8b44e",
    ant: "#a8452e",
  };
  const rand = mulberry32(randomSeed());
  let bounds = readBounds();
  let state = makeInitialState();
  let raf = 0;
  let scrollRaf = 0;
  let lastTimestamp = 0;
  let stepClock = 0;

  function loadImage(src) {
    const image = new Image();
    image.decoding = "async";
    image.src = src;
    image.addEventListener("load", draw);
    return image;
  }

  function readBounds() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width || window.innerWidth);
    const height = Math.max(1, rect.height || window.innerHeight);
    const stageSize = Math.max(220, Math.min(width, height));
    return {
      width,
      height,
      dpr: Math.min(window.devicePixelRatio || 1, maxDpr),
      stageSize,
      stageX: width - stageSize,
      stageY: 0,
      cell: stageSize / env.width,
    };
  }

  function mulberry32(seed) {
    return () => {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randomSeed() {
    if (window.crypto && window.crypto.getRandomValues) {
      const values = new Uint32Array(1);
      window.crypto.getRandomValues(values);
      return values[0];
    }
    return (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
  }

  function makeGrid(fillValue) {
    return Array.from({ length: env.height }, () => Array(env.width).fill(fillValue));
  }

  function cloneGrid(grid) {
    return grid.map((row) => row.slice());
  }

  function clonePosition(position) {
    return [position[0], position[1]];
  }

  function positionKey(position) {
    return `${position[0]},${position[1]}`;
  }

  function inBounds(x, y) {
    return 0 <= x && x < env.width && 0 <= y && y < env.height;
  }

  function samePosition(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }

  function cornerHubPosition() {
    return [
      env.width - hubCornerMarginCells - 1,
      hubCornerMarginCells,
    ];
  }

  function distanceSquared(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  function randomFreePosition(hub, existing = []) {
    for (let attempt = 0; attempt < 240; attempt += 1) {
      const position = [
        foodSpawnMarginCells +
          Math.floor(rand() * (env.width - foodSpawnMarginCells * 2)),
        foodSpawnMarginCells +
          Math.floor(rand() * (env.height - foodSpawnMarginCells * 2)),
      ];
      const farFromHub = distanceSquared(position, hub) >= 36;
      const distinct = !existing.some((other) => samePosition(other, position));
      if (farFromHub && distinct) {
        return position;
      }
    }
    return [Math.min(env.width - 4, hub[0] + 7), Math.max(3, hub[1] - 7)];
  }

  function randomFoodSources(hub) {
    const sources = [];
    while (sources.length < env.food_sources) {
      sources.push(randomFreePosition(hub, sources));
    }
    return sources;
  }

  function isHubPosition(position, hub = state?.hub) {
    return hub ? samePosition(hub, position) : false;
  }

  function makeFoodGrid(sources) {
    const food = makeGrid(0);
    sources.forEach(([x, y], index) => {
      food[y][x] = 3 + (index % 3);
    });
    return food;
  }

  function makeInitialState() {
    const hub = cornerHubPosition();
    const foodSources = randomFoodSources(hub);
    const food = makeFoodGrid(foodSources);
    return {
      hub,
      foodSources,
      food,
      initialFood: cloneGrid(food),
      bytes: makeGrid(0),
      messages: new Map(),
      ants: Array.from({ length: env.num_ants }, () => ({
        position: clonePosition(hub),
        previousPosition: clonePosition(hub),
        facing: DEFAULT_FACING,
        carrying: false,
        transition: 1,
      })),
      step: 0,
    };
  }

  function gridValue(grid, x, y) {
    return inBounds(x, y) ? grid[y][x] : 0;
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
    state.ants.forEach((ant) => {
      const [x, y] = ant.position;
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

  function buildActorObs(antIndex, antCounts) {
    const ant = state.ants[antIndex];
    const values = [];
    values.push(
      ...localPatch(
        ant.position,
        ant.facing,
        (x, y) => gridValue(state.food, x, y) / env.food_scale,
      ),
    );
    values.push(
      ...localPatch(
        ant.position,
        ant.facing,
        (x, y) => gridValue(antCounts, x, y) / env.num_ants,
      ),
    );
    for (let bit = 0; bit < env.write_bits; bit += 1) {
      values.push(
        ...localPatch(
          ant.position,
          ant.facing,
          (x, y) => (gridValue(state.bytes, x, y) >> bit) & 1,
        ),
      );
    }
    values.push(
      ...localPatch(
        ant.position,
        ant.facing,
        (x, y) => (x === state.hub[0] && y === state.hub[1] ? 1 : 0),
      ),
    );
    values.push(...localPatch(ant.position, ant.facing, () => 0, 1));
    values.push(...identityFeatures(antIndex));
    values.push(ant.carrying ? 1 : 0);
    values.push(...facingOneHot(ant.facing));
    return values;
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

  function sampleCategorical(logits, temperature = 0.8) {
    const maxLogit = Math.max(...logits);
    const weights = logits.map((value) => Math.exp((value - maxLogit) / temperature));
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

  function chooseAction(antIndex, antCounts) {
    const obs = buildActorObs(antIndex, antCounts);
    if (obs.length !== actor.actor_obs_dim) {
      throw new Error(
        `ambient observation dim ${obs.length} does not match actor dim ${actor.actor_obs_dim}`,
      );
    }
    const logits = forwardActor(obs);
    return [sampleCategorical(logits.move, 0.75), argmax(logits.write)];
  }

  function updateFacing(facing, move) {
    return move === ACTION_UP ||
      move === ACTION_RIGHT ||
      move === ACTION_DOWN ||
      move === ACTION_LEFT
      ? move
      : facing;
  }

  function nextPosition(position, move) {
    let dx = 0;
    let dy = 0;
    if (move === ACTION_RIGHT) dx = 1;
    if (move === ACTION_LEFT) dx = -1;
    if (move === ACTION_DOWN) dy = 1;
    if (move === ACTION_UP) dy = -1;
    return [
      Math.max(0, Math.min(env.width - 1, position[0] + dx)),
      Math.max(0, Math.min(env.height - 1, position[1] + dy)),
    ];
  }

  function addMessage(position, value) {
    const key = positionKey(position);
    if (value <= 0) {
      state.messages.delete(key);
      state.bytes[position[1]][position[0]] = 0;
      return;
    }
    state.messages.set(key, { value, age: 0 });
    state.bytes[position[1]][position[0]] = value;
  }

  function applyAntAction(antIndex, action) {
    const ant = state.ants[antIndex];
    const [move, writeValue] = action;
    const next = nextPosition(ant.position, move);
    const [x, y] = next;
    const hadFood = state.food[y][x] > 0;
    const delivered = ant.carrying && samePosition(next, state.hub);
    const pickedUp = !ant.carrying && hadFood;
    if (pickedUp) {
      state.food[y][x] -= 1;
    }
    if (!hadFood && !isHubPosition(next)) {
      addMessage(next, writeValue);
    }
    state.ants[antIndex] = {
      ...ant,
      previousPosition: clonePosition(ant.position),
      position: next,
      facing: updateFacing(ant.facing, move),
      carrying: pickedUp || (ant.carrying && !delivered),
      transition: 0,
    };
  }

  function ageMessages(dt) {
    state.messages.forEach((message, key) => {
      const nextAge = message.age + dt;
      if (nextAge >= messageLifetime) {
        const [x, y] = key.split(",").map(Number);
        state.bytes[y][x] = 0;
        state.messages.delete(key);
      } else {
        state.messages.set(key, { ...message, age: nextAge });
      }
    });
  }

  function remainingFood() {
    return state.food.reduce(
      (total, row) => total + row.reduce((rowTotal, value) => rowTotal + value, 0),
      0,
    );
  }

  function refillCookies() {
    if (remainingFood() > cookieRefillThreshold) {
      return;
    }
    state.foodSources.forEach(([x, y], index) => {
      if (state.food[y][x] === 0 && rand() > 0.35) {
        state.food[y][x] = 2 + ((state.step + index) % 4);
      }
    });
  }

  function policyStep() {
    const antCounts = antsCountGrid();
    const actions = state.ants.map((_, antIndex) => chooseAction(antIndex, antCounts));
    actions.forEach((action, antIndex) => applyAntAction(antIndex, action));
    state.step += 1;
    refillCookies();
  }

  function resizeCanvas() {
    bounds = readBounds();
    canvas.width = Math.ceil(bounds.width * bounds.dpr);
    canvas.height = Math.ceil(bounds.height * bounds.dpr);
    ctx.setTransform(bounds.dpr, 0, 0, bounds.dpr, 0, 0);
    updateScrollFade();
    draw();
  }

  function smoothstep(edge0, edge1, value) {
    const x = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
    return x * x * (3 - 2 * x);
  }

  function scrollTop() {
    return window.scrollY || document.documentElement.scrollTop || 0;
  }

  function updateScrollFade() {
    const fadeEnd = Math.max(180, bounds.height * 0.72);
    const fade = 1 - smoothstep(0, fadeEnd, scrollTop());
    const baseOpacity = motionQuery.matches ? 0.24 : maxLayerOpacity;
    const opacity = minLayerOpacity + (baseOpacity - minLayerOpacity) * fade;
    canvas.style.setProperty("--ambient-ant-opacity", opacity.toFixed(3));
  }

  function requestScrollFade() {
    if (scrollRaf) {
      return;
    }
    scrollRaf = window.requestAnimationFrame(() => {
      scrollRaf = 0;
      updateScrollFade();
    });
  }

  function worldToScreen(position) {
    const x = bounds.stageX + (position[0] + 0.5) * bounds.cell;
    const y = bounds.stageY + (position[1] + 0.5) * bounds.cell;
    return [x, y];
  }

  function interpolatedPosition(ant) {
    const ratio = Math.min(1, ant.transition);
    return [
      ant.previousPosition[0] + (ant.position[0] - ant.previousPosition[0]) * ratio,
      ant.previousPosition[1] + (ant.position[1] - ant.previousPosition[1]) * ratio,
    ];
  }

  function rgba(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function mixRgb(start, end, ratio) {
    return start.map((value, index) =>
      Math.round(value + (end[index] - value) * ratio),
    );
  }

  function drawSprite(name, position, alpha = 1, rotation = 0) {
    const image = sprites[name];
    const [x, y] = worldToScreen(position);
    const size = bounds.cell;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(x, y);
    ctx.rotate(rotation);
    if (image.complete && image.naturalWidth > 0) {
      ctx.drawImage(image, -size / 2, -size / 2, size, size);
    } else {
      drawFallbackMarker(name, size);
    }
    ctx.restore();
  }

  function drawFallbackMarker(name, size) {
    const radius = Math.max(2, size * 0.38);
    if (name === "hub") {
      ctx.fillStyle = palette.hub;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fill();
    } else if (name === "food") {
      ctx.fillStyle = palette.food;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fill();
    } else if (name === "ant") {
      ctx.fillStyle = palette.ant;
      ctx.beginPath();
      ctx.moveTo(size * 0.36, 0);
      ctx.lineTo(-size * 0.28, -size * 0.28);
      ctx.lineTo(-size * 0.18, 0);
      ctx.lineTo(-size * 0.28, size * 0.28);
      ctx.closePath();
      ctx.fill();
    }
  }

  function drawMessage(position, message) {
    const [gridX, gridY] = position;
    const x = bounds.stageX + gridX * bounds.cell;
    const y = bounds.stageY + gridY * bounds.cell;
    const fade = Math.max(0, 1 - message.age / messageLifetime);
    const bitTone = message.value / Math.max(1, (1 << env.write_bits) - 1);
    const color = mixRgb(palette.byteLow, palette.byteHigh, bitTone);
    ctx.save();
    ctx.fillStyle = rgba(color, palette.byteAlpha * fade * 0.78);
    ctx.fillRect(x, y, bounds.cell, bounds.cell);
    ctx.restore();
  }

  function antRotation(facing) {
    if (facing === ACTION_UP) return -Math.PI / 2;
    if (facing === ACTION_DOWN) return Math.PI / 2;
    if (facing === ACTION_LEFT) return Math.PI;
    return 0;
  }

  function drawAnt(ant) {
    const position = interpolatedPosition(ant);
    drawSprite("ant", position, ant.carrying ? 0.66 : 0.56, antRotation(ant.facing));
  }

  function drawHub(hub) {
    drawSprite("hub", hub, 0.56);
  }

  function drawCookies() {
    for (let y = 0; y < env.height; y += 1) {
      for (let x = 0; x < env.width; x += 1) {
        if (state.food[y][x] <= 0) {
          continue;
        }
        const initial = Math.max(state.initialFood[y][x], 1);
        const alpha = Math.max(0.2, Math.min(0.46, state.food[y][x] / initial));
        drawSprite("food", [x, y], alpha);
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, bounds.width, bounds.height);
    state.messages.forEach((message, key) => {
      drawMessage(key.split(",").map(Number), message);
    });
    drawCookies();
    drawHub(state.hub);
    state.ants.forEach(drawAnt);
  }

  function tick(timestamp) {
    const dt = lastTimestamp
      ? Math.min(0.05, (timestamp - lastTimestamp) / 1000)
      : 0;
    lastTimestamp = timestamp;
    stepClock += dt;
    ageMessages(dt);
    state.ants = state.ants.map((ant) => ({
      ...ant,
      transition: Math.min(1, ant.transition + dt / policyStepSeconds),
    }));
    while (stepClock >= policyStepSeconds) {
      policyStep();
      stepClock -= policyStepSeconds;
    }
    draw();
    raf = window.requestAnimationFrame(tick);
  }

  function stop() {
    if (raf) {
      window.cancelAnimationFrame(raf);
      raf = 0;
    }
  }

  function start() {
    stop();
    lastTimestamp = 0;
    updateScrollFade();
    draw();
    if (!motionQuery.matches && !document.hidden) {
      raf = window.requestAnimationFrame(tick);
    }
  }

  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("scroll", requestScrollFade, { passive: true });
  document.addEventListener("visibilitychange", start);
  if (motionQuery.addEventListener) {
    motionQuery.addEventListener("change", start);
  } else {
    motionQuery.addListener(start);
  }
  resizeCanvas();
  start();
})();
