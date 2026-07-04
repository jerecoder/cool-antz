#!/usr/bin/env python3
"""Build report figures from local experiment artifacts.

The report intentionally separates comparable held-out evaluations from
engineering milestones with changed tasks, critics, or checkpoint lineage.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "figures"
DATA = ROOT / "report" / "data"

BLUE = "#3566a8"
GREEN = "#3b8b65"
ORANGE = "#d08a2d"
RED = "#b94b4b"
PURPLE = "#7c5aa6"
GRAY = "#6f7785"
DARK = "#20242b"


def rel(path: str | Path) -> Path:
    return ROOT / path


def load_json(path: str | Path) -> Any:
    with rel(path).open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with rel(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def thin_rows(rows: list[dict[str, Any]], max_points: int = 260) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    step = math.ceil(len(rows) / max_points)
    thinned = rows[::step]
    if thinned[-1] is not rows[-1]:
        thinned.append(rows[-1])
    return thinned


def savefig(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def style_axes(ax: plt.Axes, ylabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dde6", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def eval_metric(run: str, mode: str, key: str) -> float:
    data = load_json(f"runs/autoresearch/forage_loop/{run}/evaluation.json")
    return float(data[mode][key])


def best_eval_mode(run: str, preferred: list[str]) -> tuple[str, dict[str, Any]]:
    data = load_json(f"runs/autoresearch/forage_loop/{run}/evaluation.json")
    for mode in preferred:
        if mode in data:
            return mode, data[mode]
    candidates = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    return max(candidates, key=lambda kv: kv[1].get("eval_mean_delivered_food", -math.inf))


def wandb(path: str) -> dict[str, Any]:
    return load_json(path)


def nested_metric(path: str, section: str, key: str, default: float | None = None) -> float | None:
    data = load_json(path)
    value = data.get(section, {}).get(key, default)
    return None if value is None else float(value)


def figure_25x25_modes() -> list[dict[str, Any]]:
    runs = [
        ("DISTANCE_SHAPE", "1 ant + distancia"),
        ("DISTANCE_CAP4", "4 ants"),
        ("DISTANCE_CAP4_SHARP", "sharp"),
        ("DISTANCE_CAP4_LONG_CREDIT_TUNE", "long credit"),
        ("DISTANCE_CAP4_LONG_CREDIT_GENTLE_GREEDY", "gentle greedy"),
        ("DISTANCE_CAP4_NO_WRITE", "sin escritura"),
    ]
    rows: list[dict[str, Any]] = []
    for run, label in runs:
        for mode in ["deterministic", "sampled"]:
            rows.append(
                {
                    "run": run,
                    "label": label,
                    "mode": "greedy" if mode == "deterministic" else "sampled",
                    "delivered_food": eval_metric(run, mode, "eval_mean_delivered_food"),
                    "delivered_fraction": eval_metric(run, mode, "eval_mean_delivered_fraction"),
                    "success_rate": eval_metric(run, mode, "eval_success_rate"),
                    "food_total": 23,
                }
            )
    write_csv(DATA / "figure_25x25_modes.csv", rows)

    df = pd.DataFrame(rows)
    labels = [label for _, label in runs]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    for offset, mode, color in [(-width / 2, "greedy", ORANGE), (width / 2, "sampled", BLUE)]:
        ys = [df[(df.label == label) & (df["mode"] == mode)].delivered_fraction.iloc[0] for label in labels]
        bars = ax.bar([i + offset for i in x], ys, width, label=mode, color=color)
        for bar, y, label in zip(bars, ys, labels):
            food = df[(df.label == label) & (df["mode"] == mode)].delivered_food.iloc[0]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(y + 0.035, 1.04),
                f"{food:.1f}/23",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if food >= 20 else 0,
            )
    ax.axhline(1.0, color=DARK, linewidth=1.1, linestyle="--", alpha=0.8)
    ax.set_ylim(0, 1.13)
    ax.set_xticks(list(x), labels, rotation=22, ha="right")
    style_axes(ax, "fracción entregada en evaluación")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    savefig(fig, "fig_25x25_modes")
    return rows


def figure_bits_vs_ants() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bits in [2, 3, 5, 8]:
        for base, task in [
            ("communication_bits", "15x15 aprox."),
            ("communication_bits_25x25", "25x25 anchor"),
        ]:
            metrics = load_json(f"runs/notebooks/{base}/{bits}_bits/summary.json")["metrics"]
            rows.append(
                {
                    "family": "bits",
                    "setting": bits,
                    "task": task,
                    "episode_return": float(metrics["episode_return"]),
                    "env_return": float(metrics["env_return"]),
                }
            )
    for ants in [2, 3, 4, 6, 8]:
        metrics = load_json(f"runs/notebooks/ant_count_25x25_3_bits/{ants}_ants/summary.json")["metrics"]
        rows.append(
            {
                "family": "ants",
                "setting": ants,
                "task": "25x25, 3 bits",
                "episode_return": float(metrics["episode_return"]),
                "env_return": float(metrics["env_return"]),
            }
        )
    write_csv(DATA / "figure_bits_vs_ants.csv", rows)

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    ax = axes[0]
    for task, color, marker in [("15x15 aprox.", GREEN, "o"), ("25x25 anchor", ORANGE, "s")]:
        sub = df[(df.family == "bits") & (df.task == task)].sort_values("setting")
        ax.plot(sub.setting, sub.episode_return, marker=marker, linewidth=2.4, color=color, label=task)
    ax.set_xlabel("bits de escritura")
    style_axes(ax, "retorno de entrenamiento")
    ax.legend(frameon=False)

    ax = axes[1]
    sub = df[df.family == "ants"].sort_values("setting")
    ax.bar(sub.setting.astype(str), sub.episode_return, color=BLUE)
    ax.set_xlabel("hormigas")
    style_axes(ax, "retorno de entrenamiento")
    savefig(fig, "fig_bits_vs_ants")
    return rows


def figure_logged_metrics_story() -> list[dict[str, Any]]:
    sources = [
        ("50x50 largo", "runs/notebooks/exploration_to_forage_50x50/metrics.jsonl"),
        ("50x50 eficiente", "runs/notebooks/exploration_to_forage_50x50_efficient/metrics.jsonl"),
    ]
    metric_keys = [
        "update",
        "global_step",
        "episode_return",
        "env_return",
        "eval_mean_delivered_food",
        "eval_mean_delivered_food_per_1000_ant_steps",
        "delivery_events",
        "pickup_events",
        "mean_carrying_ants",
        "final_mean_visited_cell_fraction",
        "write_action_nonzero_rate",
        "entropy",
        "approx_kl",
        "grad_norm",
        "value_loss",
    ]
    rows: list[dict[str, Any]] = []
    loaded: dict[str, list[dict[str, Any]]] = {}
    for label, path in sources:
        data = load_jsonl(path)
        loaded[label] = data
        for row in data:
            out = {"source": label}
            for key in metric_keys:
                value = row.get(key)
                out[key] = "" if value is None else value
            rows.append(out)
    write_csv(DATA / "figure_logged_metrics_story.csv", rows)

    long_rows = thin_rows(loaded["50x50 largo"])
    efficient_rows = loaded["50x50 eficiente"]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), constrained_layout=True)

    ax = axes[0][0]
    xs = [r["update"] for r in long_rows]
    ax.plot(xs, [r.get("pickup_events", 0.0) for r in long_rows], color=ORANGE, linewidth=1.9, label="pickups")
    ax.plot(xs, [r.get("delivery_events", 0.0) for r in long_rows], color=BLUE, linewidth=1.9, label="entregas")
    ax.set_xlabel("actualización")
    style_axes(ax, "eventos por resumen")
    ax.legend(frameon=False)

    ax = axes[0][1]
    xs = [r["update"] for r in efficient_rows]
    ax.plot(xs, [r.get("eval_mean_delivered_food", 0.0) for r in efficient_rows], color=GREEN, marker="o", linewidth=2.0, label="comida eval.")
    ax.plot(
        xs,
        [r.get("eval_mean_delivered_food_per_1000_ant_steps", 0.0) for r in efficient_rows],
        color=PURPLE,
        marker="s",
        linewidth=2.0,
        label="entregas / 1000 pasos-hormiga",
    )
    ax.set_xlabel("actualización")
    style_axes(ax, "evaluación held-out")
    ax.legend(frameon=False)

    ax = axes[1][0]
    xs = [r["update"] for r in long_rows]
    ax.plot(xs, [r.get("final_mean_visited_cell_fraction", 0.0) for r in long_rows], color=GREEN, linewidth=1.9, label="cobertura final")
    ax.plot(xs, [r.get("write_action_nonzero_rate", 0.0) for r in long_rows], color=PURPLE, linewidth=1.9, label="escritura no-cero")
    ax.set_xlabel("actualización")
    ax.set_ylim(0, 1.05)
    style_axes(ax, "fracción / tasa")
    ax.legend(frameon=False)

    ax = axes[1][1]
    xs = [r["update"] for r in long_rows]
    for key, color, label in [
        ("entropy", BLUE, "entropía"),
        ("approx_kl", RED, "KL aprox."),
        ("grad_norm", ORANGE, "norma grad."),
        ("value_loss", GRAY, "loss valor"),
    ]:
        ys = [max(float(r.get(key, 0.0)), 1e-6) for r in long_rows]
        ax.plot(xs, ys, linewidth=1.6, color=color, label=label)
    ax.set_xlabel("actualización")
    ax.set_yscale("log")
    style_axes(ax, "escala log")
    ax.legend(frameon=False, ncols=2)

    savefig(fig, "fig_logged_metrics_story")
    return rows


def figure_rare_50x50() -> list[dict[str, Any]]:
    configs = [
        ("DISTANCE_CAP8_BIGMAP_RARE_RANDOM_SPAWN", "base raro", ["sampled_move_t100_greedy_write"]),
        ("DISTANCE_CAP12_BIGMAP_RARE_VISION2_RANDOM_SPAWN", "vision r=2", ["sampled_move_t100_greedy_write"]),
        ("DISTANCE_CAP8_BIGMAP_RARE_HUBVECTOR_RANDOM_SPAWN", "hub vector", ["sampled_move_t100_greedy_write"]),
        ("DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_RANDOM_SPAWN", "hub+food vector", ["sampled_move_t090_greedy_write"]),
        ("DISTANCE_CAP8_BIGMAP_RARE_NAVVECTOR_HELDOUT_SELECT", "seleccion held-out", ["sampled_move_t090_greedy_write"]),
    ]
    rows = []
    for run, label, preferred in configs:
        mode, metrics = best_eval_mode(run, preferred)
        rows.append(
            {
                "run": run,
                "label": label,
                "mode": mode,
                "delivered_food": float(metrics["eval_mean_delivered_food"]),
                "delivered_fraction": float(metrics["eval_mean_delivered_fraction"]),
                "success_rate": float(metrics["eval_success_rate"]),
                "food_total": 23,
            }
        )
    write_csv(DATA / "figure_rare_50x50.csv", rows)

    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    bars = ax.bar([r["label"] for r in rows], [r["delivered_fraction"] for r in rows], color=[GRAY, GRAY, ORANGE, GREEN, BLUE])
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["delivered_fraction"] + 0.025,
            f"{row['delivered_food']:.1f}/23\nsucc {row['success_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 0.82)
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows], rotation=18, ha="right")
    style_axes(ax, "fracción entregada")
    savefig(fig, "fig_rare_50x50")
    return rows


def figure_critic_50x50() -> list[dict[str, Any]]:
    confirmation_path = "runs/overnight_efficiency_sweep/sweep_20260702_003029/confirmation_64/summary.json"
    entries = [
        {
            "label": "eficiente 50x50",
            "critic": "MLP",
            "ants": 4,
            "path": "runs/notebooks/exploration_to_forage_50x50_efficient/wandb/run-20260620_235136-g8u36g2a/files/wandb-summary.json",
            "note": "default critic",
        },
        {
            "label": "proximidad positiva",
            "critic": "strided CNN",
            "ants": 4,
            "path": "runs/notebooks/exploration_to_forage_proximity_sources_positive_only_50x50_outer_30x30_inner/wandb/run-20260622_164029-fhnv5vt7/files/wandb-summary.json",
            "note": "spatial critic",
        },
        {
            "label": "full layout",
            "critic": "strided CNN",
            "ants": 4,
            "path": "runs/notebooks/exploration_to_forage_proximity_sources_full_layout_50x50_from_best/wandb/run-20260623_020535-38olo9wl/files/wandb-summary.json",
            "note": "final, peak was higher",
        },
        {
            "label": "8 ants long3",
            "critic": "strided CNN",
            "ants": 8,
            "path": "runs/notebooks/exploration_to_forage_proximity_sources_full_layout_50x50_8ants_half_food_2src_long3_from_long2_latest/wandb/run-20260624_142948-6v9swr1f/files/wandb-summary.json",
            "note": "half food",
        },
        {
            "label": "8 ants shared write",
            "critic": "strided CNN",
            "ants": 8,
            "path": "runs/notebooks/exploration_to_forage_proximity_sources_full_layout_50x50_8ants_half_food_2src_shared_writes_from_64env_best/wandb/run-20260625_003652-kf4d52yf/files/wandb-summary.json",
            "note": "shared writes",
        },
        {
            "label": "8 ants write cost",
            "critic": "strided CNN",
            "ants": 8,
            "path": "runs/notebooks/exploration_to_forage_proximity_sources_full_layout_50x50_8ants_half_food_2src_shared_writes_write_cost_from_shared_best/wandb/run-20260625_160730-77z1vhpr/files/wandb-summary.json",
            "note": "best 8-ant local result",
        },
        {
            "label": "60 ants confirmado",
            "critic": "strided CNN",
            "ants": 60,
            "path": None,
            "note": "64 eval episodes, saturated writes",
        },
    ]
    required_paths = [rel(entry["path"]) for entry in entries if entry["path"]] + [rel(confirmation_path)]
    missing_sources = [path for path in required_paths if not path.exists()]
    if missing_sources:
        rows = load_csv_rows(DATA / "figure_critic_50x50.csv")
        print(
            f"using {DATA / 'figure_critic_50x50.csv'} for critic 50x50 "
            f"because {len(missing_sources)} source artifact(s) are unavailable"
        )
    else:
        rows = []
        for entry in entries:
            if entry["path"]:
                data = wandb(entry["path"])
                row = {
                    "label": entry["label"],
                    "critic": entry["critic"],
                    "ants": entry["ants"],
                    "delivered_food": float(data["eval_mean_delivered_food"]),
                    "food_total": int(data["food_count"]),
                    "delivered_fraction": float(data["eval_mean_delivered_fraction"]),
                    "success_rate": float(data["eval_success_rate"]),
                    "write_nonzero_rate": float(data.get("write_action_nonzero_rate", 0.0)),
                    "note": entry["note"],
                }
            else:
                confirmation = load_json(confirmation_path)
                baseline = next(item for item in confirmation if item["label"] == "baseline_cool_temp050")
                metrics = baseline["metrics"]
                row = {
                    "label": entry["label"],
                    "critic": entry["critic"],
                    "ants": entry["ants"],
                    "delivered_food": float(metrics["eval_mean_delivered_food"]),
                    "food_total": 125,
                    "delivered_fraction": float(metrics["eval_mean_delivered_fraction"]),
                    "success_rate": float(metrics["eval_success_rate"]),
                    "write_nonzero_rate": 0.9980143904685974,
                    "note": entry["note"],
                }
            rows.append(row)
        write_csv(DATA / "figure_critic_50x50.csv", rows)

    fig, ax = plt.subplots(figsize=(11.8, 4.9), constrained_layout=True)
    colors = [GRAY if r["critic"] == "MLP" else (BLUE if r["ants"] < 60 else GREEN) for r in rows]
    bars = ax.bar([r["label"] for r in rows], [r["delivered_fraction"] for r in rows], color=colors)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(row["delivered_fraction"] + 0.035, 1.05),
            f"{row['delivered_food']:.1f}/{row['food_total']}\nsucc {row['success_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axvline(0.5, color=DARK, linestyle="--", linewidth=1, alpha=0.55)
    ax.text(0.52, 0.98, "cambio de crítico", ha="left", va="top", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.13)
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows], rotation=22, ha="right")
    style_axes(ax, "fracción entregada")
    savefig(fig, "fig_critic_50x50")
    return rows


def figure_250x250() -> list[dict[str, Any]]:
    paths = [
        (
            "resnet one-food",
            "resnet CNN",
            "runs/training/half_scale_one_food_resnet_actor_warmstart_250x250/online-systemd-20260628T192132Z/summary.json",
        ),
        (
            "NPC teacher",
            "strided CNN",
            "runs/training/half_scale_one_food_npc_teacher_from_source_250x250/online-systemd-source-teacher-20260628T204020/summary.json",
        ),
        (
            "distance auto",
            "set CNN",
            "runs/training/half_scale_distance_autocurriculum_source_teacher_250x250/diagnosis-20260629T180833Z/summary.json",
        ),
        (
            "reset boundary",
            "set CNN",
            "runs/training/half_scale_distance_fixed8_source_reset_boundary_256_250x250/fixed8-reset-boundary256-20260629T214228Z/summary.json",
        ),
        (
            "byte decay",
            "set CNN",
            "runs/training/fixed8-rb256-byte-decay-hl256-from-rbfinal-250x250/fixed8-rb256-byte-decay-hl256-from-rbfinal-20260630T035740Z/summary.json",
        ),
    ]
    rows = []
    for label, critic, path in paths:
        m = load_json(path)["metrics"]
        rows.append(
            {
                "label": label,
                "critic": critic,
                "delivery_events": float(m.get("delivery_events", 0.0)),
                "pickup_events": float(m.get("pickup_events", 0.0)),
                "env_return": float(m.get("env_return", 0.0)),
                "episode_return": float(m.get("episode_return", 0.0)),
                "write_nonzero_rate": float(m.get("write_action_nonzero_rate", m.get("applied_write_action_nonzero_rate", 0.0))),
                "nonzero_byte_fraction": float(m.get("final_mean_nonzero_byte_fraction", 0.0)),
            }
        )
    write_csv(DATA / "figure_250x250.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
    x = range(len(rows))
    width = 0.36
    ax = axes[0]
    ax.bar([i - width / 2 for i in x], [r["delivery_events"] for r in rows], width, color=BLUE, label="entregas")
    ax.bar([i + width / 2 for i in x], [r["pickup_events"] for r in rows], width, color=ORANGE, label="pickups")
    ax.set_xticks(list(x), [r["label"] for r in rows], rotation=22, ha="right")
    style_axes(ax, "eventos en resumen final")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], [r["write_nonzero_rate"] for r in rows], width, color=PURPLE, label="write nonzero")
    ax.bar([i + width / 2 for i in x], [r["nonzero_byte_fraction"] for r in rows], width, color=GREEN, label="fracción bytes no-cero")
    ax.set_xticks(list(x), [r["label"] for r in rows], rotation=22, ha="right")
    ax.set_ylim(0, 1.05)
    style_axes(ax, "tasa / fracción")
    ax.legend(frameon=False)
    savefig(fig, "fig_250x250_diagnostics")
    return rows


@dataclass
class VideoItem:
    label: str
    path: str
    fraction: float = 0.65


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        return float(out)
    except Exception:
        return None


def extract_frame(video: Path, out: Path, fraction: float) -> bool:
    duration = ffprobe_duration(video)
    if duration is None or duration <= 0:
        return False
    t = max(0.0, min(duration * fraction, duration - 0.05))
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out),
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_behavior_montage() -> list[dict[str, Any]]:
    items = [
        VideoItem("Azar: sin retorno estable", "docs/report-site/assets/videos/random-rollout.mp4", 0.55),
        VideoItem("25x25: 2 hormigas", "docs/report-site/assets/videos/ant-count-25x25-2ants.mp4", 0.52),
        VideoItem("25x25: 4 hormigas", "docs/report-site/assets/videos/forage-25x25-4ants.mp4", 0.62),
        VideoItem("25x25: 8 hormigas", "docs/report-site/assets/videos/ant-count-25x25-8ants.mp4", 0.58),
        VideoItem("50x50: frontera de 60 hormigas", "docs/report-site/assets/videos/frontier-50x50.mp4", 0.55),
        VideoItem("1000x1000: despliegue solo actor", "docs/report-site/assets/videos/bigmap-50x50-policy-1000x1000.mp4", 0.45),
    ]
    missing_videos = [item for item in items if not rel(item.path).exists()]
    if missing_videos:
        missing = ", ".join(item.path for item in missing_videos)
        raise FileNotFoundError(f"missing behavior montage source video(s): {missing}")
    frame_dir = OUT / "behavior_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cell_w, cell_h = 540, 420
    pad = 18
    panel_font = font(28)
    montage = Image.new("RGB", (2 * cell_w, 3 * cell_h), "white")
    draw = ImageDraw.Draw(montage)
    for idx, item in enumerate(items):
        video = rel(item.path)
        frame_path = frame_dir / f"frame_{idx:02d}.jpg"
        ok = video.exists() and extract_frame(video, frame_path, item.fraction)
        if not ok:
            raise RuntimeError(f"could not extract montage frame from {item.path}")
        x0 = (idx % 2) * cell_w
        y0 = (idx // 2) * cell_h
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill="white")
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((cell_w - 2 * pad, cell_h - 2 * pad), Image.Resampling.LANCZOS)
        ix = x0 + (cell_w - image.width) // 2
        iy = y0 + (cell_h - image.height) // 2
        montage.paste(image, (ix, iy))
        draw.rectangle([ix + 8, iy + 8, ix + 44, iy + 44], fill="white", outline="#b8c0cc")
        draw.text((ix + 18, iy + 10), chr(ord("A") + idx), fill=DARK, font=panel_font)
        rows.append({"label": item.label, "path": item.path, "frame_fraction": item.fraction, "extracted": ok})
    out = OUT / "fig_behavior_montage.jpg"
    montage.save(out, quality=92)
    write_csv(DATA / "figure_behavior_sources.csv", rows)
    return rows


def write_source_note(all_rows: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# Fuentes de figuras",
        "",
        "Generado por `report/scripts/build_figures.py` desde artefactos locales.",
        "Las barras mezclan denominadores solo cuando la figura lo declara: por eso cada barra muestra entregas/total.",
        "",
    ]
    for name, rows in all_rows.items():
        lines.append(f"## {name}")
        lines.append(f"- filas: {len(rows)}")
    (DATA / "figure_sources.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    all_rows = {
        "25x25 action modes": figure_25x25_modes(),
        "bits vs ants": figure_bits_vs_ants(),
        "logged metrics story": figure_logged_metrics_story(),
        "rare 50x50": figure_rare_50x50(),
        "critic 50x50": figure_critic_50x50(),
        "250x250 diagnostics": figure_250x250(),
        "behavior montage": make_behavior_montage(),
    }
    write_source_note(all_rows)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
