(() => {
  const policies = {
    frontier50: {
      title: "50x50 frontier policy",
      src: "report-site/assets/videos/frontier-50x50.mp4",
      caption:
        "60-ant 50x50 checkpoint rendered as an 800x800 sprite MP4 from the actual policy.",
      metrics: [
        ["task grid", "50x50"],
        ["encoded frame", "800x800"],
        ["ants", "60"],
        ["food", "125 bites / 2 sources"],
        ["bits", "8"],
        ["critic", "strided_cnn"],
        ["delivered", "123.906 / 125"],
        ["success", "0.906 over 64 eval episodes"],
        ["action mode", "sampled move / greedy write"],
        ["move temp", "0.525 render, 0.5 confirmation"],
        ["write nonzero", "0.998"],
      ],
      caveat:
        "Strong behavior, but confounded by ant count, write bits, identity features, selected continuation, temperature, and saturated writes.",
    },
    unlock25: {
      title: "25x25 ant-count curriculum rollout",
      src: "report-site/assets/videos/forage-25x25-4ants.mp4",
      caption:
        "Four-ant, 3-bit rollout from the ant-count curriculum vault, rendered as an 800x800 MP4.",
      metrics: [
        ["task grid", "25x25"],
        ["encoded frame", "800x800"],
        ["ants", "4"],
        ["food", "23 bites / 12 sources"],
        ["bits", "3"],
        ["checkpoint", "ant_count_25x25_3_bits/4_ants"],
        ["actor obs", "151 legacy features"],
        ["critic", "MLP"],
        ["train env return", "7.8125"],
        ["train episode return", "12.545"],
      ],
      caveat:
        "This video is not the DISTINCT DISTANCE_CAP4 row in the results table; that 23/23 result is reported from local evaluation files, not this MP4 artifact.",
    },
    bridge100: {
      title: "1000x1000 bridge bigmap render",
      src: "report-site/assets/videos/bridge-100x100.mp4",
      caption:
        "Actor-only bigmap rollout on a 1000x1000 simulation grid, encoded as a 1008x1008 MP4.",
      metrics: [
        ["render grid", "1000x1000"],
        ["encoded frame", "1008x1008"],
        ["active window", "250x250 inside [375, 624]"],
        ["render ants", "500"],
        ["render food", "5000 bites / 6 sources"],
        ["checkpoint task", "100x100 hard375 continuation"],
        ["eval contract", "120 ants, 375 food, 6 sources"],
        ["critic", "set_cnn lineage"],
        ["eval delivered", "372 / 375"],
        ["eval success", "0.625 over 24 episodes"],
        ["temperature", "move 0.525"],
        ["eval rate", "0.803 delivered / 1000 ant-steps"],
      ],
      caveat:
        "The video geometry is 1000x1000 with 500 ants and 5000 food. The 100x100 label belongs only to the checkpoint lineage and evaluation task.",
    },
    reset250: {
      title: "250x250 reset-boundary checkpoint",
      src: "report-site/assets/videos/reset-boundary-250x250.mp4",
      caption:
        "Local reset-boundary checkpoint video from the fixed8 250x250 diagnostic branch, encoded as 1008x1008.",
      metrics: [
        ["task grid", "250x250"],
        ["encoded frame", "1008x1008"],
        ["frames", "600 at 8 fps"],
        ["ants", "500"],
        ["food", "5000 bites / 1 source"],
        ["checkpoint family", "fixed8-reset-boundary256"],
        ["critic", "set_cnn"],
        ["final deliveries", "654"],
        ["final pickups", "869"],
        ["best train delivery", "around 1003"],
        ["diagnostic point", "raw delivery, not shaped return"],
      ],
      caveat:
        "This branch shows real local delivery progress, but it is still not a solved general 250x250 task.",
    },
    frontier250: {
      title: "1000x1000 half-scale bigmap render",
      src: "report-site/assets/videos/frontier-250x250.mp4",
      caption:
        "Actor-only bigmap rollout from the half-scale branch on a 1000x1000 simulation grid, encoded as 1008x1008.",
      metrics: [
        ["render grid", "1000x1000"],
        ["encoded frame", "1008x1008"],
        ["active window", "250x250 inside [375, 624]"],
        ["render ants", "500"],
        ["render food", "5000 bites / 1 source"],
        ["checkpoint", "best_training_delivery.pkl"],
        ["actor obs", "705"],
        ["bits", "4"],
        ["critic", "set_cnn branch"],
        ["checkpoint metric", "685.0 best training delivery"],
        ["render move temp", "0.9"],
      ],
      caveat:
        "250x250 remains diagnostic: raw delivery matters more than shaped return or byte activity.",
    },
    random: {
      title: "Random baseline rollout",
      src: "report-site/assets/videos/random-rollout.mp4",
      caption:
        "Early random-policy baseline, encoded as a 576x576 MP4 with 301 frames at 12 fps.",
      metrics: [
        ["encoded frame", "576x576"],
        ["frames", "301 at 12 fps"],
        ["purpose", "behavior baseline"],
        ["result meaning", "motion without delivery is not success"],
      ],
      caveat:
        "This is a baseline visual anchor, not a trained policy result.",
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
