(() => {
  const policies = {
    frontier50: {
      title: "50x50 frontier policy",
      src: "report-site/assets/videos/frontier-50x50.mp4",
      caption:
        "60-ant 50x50 checkpoint rendered at 800px from the actual policy.",
      metrics: [
        ["grid", "50x50"],
        ["ants", "60"],
        ["food", "125 bites / 2 sources"],
        ["bits", "8"],
        ["critic", "strided_cnn"],
        ["delivered", "123.906 / 125"],
        ["success", "0.906 over 64 eval episodes"],
        ["write nonzero", "0.998"],
      ],
      caveat:
        "Strong behavior, but confounded by ant count, write bits, identity features, selected continuation, temperature, and saturated writes.",
    },
    unlock25: {
      title: "25x25 sampled-movement unlock",
      src: "report-site/assets/videos/forage-25x25-4ants.mp4",
      caption:
        "Actual 25x25 rollout from the ant-count branch; the matching evaluation reaches full delivery under sampled movement.",
      metrics: [
        ["grid", "25x25"],
        ["ants", "4"],
        ["food", "23 bites / 6 sources"],
        ["bits", "1 in DISTANCE_CAP4"],
        ["critic", "MLP"],
        ["delivered", "23 / 23 sampled"],
        ["success", "1.0 sampled"],
        ["greedy result", "2.75 / 23"],
      ],
      caveat:
        "The clean claim is sparse-foraging learnability under sampled movement, not causal byte communication.",
    },
    bridge100: {
      title: "100x100 bridge continuation",
      src: "report-site/assets/videos/bridge-100x100.mp4",
      caption:
        "1000px actor-only bigmap render from the hard 375-food / 6-source continuation branch.",
      metrics: [
        ["grid", "100x100 bridge / 1000px render"],
        ["ants", "120 policy branch, rendered actor-only"],
        ["food", "375 eval task / 6 sources"],
        ["critic", "set_cnn lineage"],
        ["delivered", "372 / 375"],
        ["success", "0.625"],
        ["temperature", "move 0.525"],
        ["rate", "0.803 delivered / 1000 ant-steps"],
      ],
      caveat:
        "This is impressive continuation evidence, but it is selected and temperature-sensitive.",
    },
    frontier250: {
      title: "250x250 diagnostic frontier",
      src: "report-site/assets/videos/frontier-250x250.mp4",
      caption:
        "1000px actor-only bigmap render from the half-scale 250x250 branch.",
      metrics: [
        ["grid", "250x250 inner / 1000px render"],
        ["ants", "500"],
        ["food", "5000 bites / one rich source"],
        ["bits", "4"],
        ["critic", "set_cnn branch"],
        ["best local train", "about 1003 deliveries"],
        ["frontier eval", "685-best lineage"],
        ["byte occupancy", "low in reset-boundary branch"],
      ],
      caveat:
        "250x250 remains diagnostic: raw delivery matters more than shaped return or byte activity.",
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
    button.addEventListener("click", () => selectPolicy(button.dataset.policy));
  });

  selectPolicy("frontier50");
})();
