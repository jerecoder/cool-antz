"""Render the strided CNN critic architecture figure for the report site."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "plots"

PAPER = "#fbf7ef"
PANEL = "#fffdf8"
INK = "#1f2d35"
MUTED = "#64706a"
LINE = "#303030"
CNN_EDGE = "#3f7dac"
CNN_FACE = "#a9d8f3"
CNN_TOP = "#78bee5"
FC_EDGE = "#4f9a58"
FC_FACE = "#c9efc1"
ENTITY_EDGE = "#a97331"
ENTITY_FACE = "#f2dfbf"
VALUE_FACE = "#eeeeee"
FLOW_Y = 3.78


def add_text(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 9,
    weight: str = "normal",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.15,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    lw: float = 1.25,
    mutation_scale: float = 12,
    alpha: float = 1.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=2,
            shrinkB=2,
        )
    )


def dense_box(
    ax,
    *,
    center: tuple[float, float],
    width: float,
    height: float,
    label: str,
    fill: str,
    edge: str,
    size: float = 8.7,
) -> tuple[float, float, float, float]:
    x = center[0] - width / 2
    y = center[1] - height / 2
    ax.add_patch(
        Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=1.25)
    )
    add_text(ax, center[0], center[1], label, size=size)
    return (x, y, width, height)


def box_anchor(box: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, y, width, height = box
    if side == "left":
        return (x, y + height / 2)
    if side == "right":
        return (x + width, y + height / 2)
    if side == "top":
        return (x + width / 2, y + height)
    if side == "bottom":
        return (x + width / 2, y)
    raise ValueError(f"Unknown side: {side}")


def flow_anchor(box: tuple[float, float, float, float], side: str) -> tuple[float, float]:
    x, _, width, _ = box
    if side == "left":
        return (x, FLOW_Y)
    if side == "right":
        return (x + width, FLOW_Y)
    raise ValueError(f"Unknown side: {side}")


def conv_block(
    ax,
    *,
    x: float,
    bottom: float,
    width: float,
    height: float,
    depth: float,
    label: str,
) -> tuple[float, float, float, float]:
    front = Rectangle(
        (x, bottom),
        width,
        height,
        facecolor=CNN_FACE,
        edgecolor=CNN_EDGE,
        linewidth=1.25,
    )
    ax.add_patch(front)
    ax.add_patch(
        Polygon(
            [
                (x, bottom + height),
                (x + depth, bottom + height + depth),
                (x + width + depth, bottom + height + depth),
                (x + width, bottom + height),
            ],
            closed=True,
            facecolor=CNN_TOP,
            edgecolor=CNN_EDGE,
            linewidth=1.05,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (x + width, bottom),
                (x + width + depth, bottom + depth),
                (x + width + depth, bottom + height + depth),
                (x + width, bottom + height),
            ],
            closed=True,
            facecolor="#7fbce0",
            edgecolor=CNN_EDGE,
            linewidth=1.05,
        )
    )
    add_text(ax, x + width / 2, 1.34, label, size=8.0)
    return (x, bottom, width + depth, height + depth)


def draw_cnn(ax) -> None:
    ax.add_patch(
        Rectangle((0.35, 0.85), 13.72, 5.05, facecolor=PANEL, edgecolor="#8fa574", linewidth=1.15)
    )

    y_base = 2.0
    blocks = [
        conv_block(
            ax,
            x=1.0,
            bottom=y_base,
            width=0.34,
            height=3.55,
            depth=0.18,
            label="spatial planes\n50x50x4",
        ),
        conv_block(
            ax,
            x=2.7,
            bottom=y_base + 0.35,
            width=0.56,
            height=2.85,
            depth=0.22,
            label="conv 5x5 s2\n25x25x32",
        ),
        conv_block(
            ax,
            x=4.45,
            bottom=y_base + 0.66,
            width=0.56,
            height=2.23,
            depth=0.22,
            label="conv 3x3 s2\n13x13x64",
        ),
        conv_block(
            ax,
            x=6.1,
            bottom=y_base + 0.98,
            width=0.56,
            height=1.6,
            depth=0.22,
            label="conv 3x3 s2\n7x7x64",
        ),
    ]

    for left, right in zip(blocks, blocks[1:]):
        arrow(ax, flow_anchor(left, "right"), flow_anchor(right, "left"), color=CNN_EDGE)

    flatten = dense_box(
        ax,
        center=(7.85, FLOW_Y),
        width=0.78,
        height=0.72,
        label="flatten\n3136",
        fill="#edf6ff",
        edge=CNN_EDGE,
    )
    spatial_fc = dense_box(
        ax,
        center=(9.25, FLOW_Y),
        width=0.76,
        height=0.72,
        label="FC\n256",
        fill=FC_FACE,
        edge=FC_EDGE,
    )
    entity = dense_box(
        ax,
        center=(7.95, 5.12),
        width=1.18,
        height=0.62,
        label="entity/global\n44",
        fill=ENTITY_FACE,
        edge=ENTITY_EDGE,
    )
    entity_fc = dense_box(
        ax,
        center=(9.45, 5.12),
        width=0.72,
        height=0.62,
        label="FC\n128",
        fill=ENTITY_FACE,
        edge=ENTITY_EDGE,
    )
    concat = dense_box(
        ax,
        center=(10.8, FLOW_Y),
        width=0.78,
        height=0.86,
        label="concat\n384",
        fill="#dcefd1",
        edge="#789c6d",
    )
    fusion = dense_box(
        ax,
        center=(12.25, FLOW_Y),
        width=0.72,
        height=0.72,
        label="FC\n256",
        fill=FC_FACE,
        edge=FC_EDGE,
    )
    value = dense_box(
        ax,
        center=(13.35, FLOW_Y),
        width=0.54,
        height=0.72,
        label="V(s)\n1",
        fill=VALUE_FACE,
        edge="#777777",
    )

    arrow(ax, flow_anchor(blocks[-1], "right"), flow_anchor(flatten, "left"), color=CNN_EDGE)
    arrow(ax, flow_anchor(flatten, "right"), flow_anchor(spatial_fc, "left"), color=FC_EDGE)
    arrow(ax, box_anchor(entity, "right"), box_anchor(entity_fc, "left"), color=ENTITY_EDGE)
    arrow(ax, flow_anchor(spatial_fc, "right"), flow_anchor(concat, "left"), color=FC_EDGE)
    arrow(ax, box_anchor(entity_fc, "right"), box_anchor(concat, "top"), color=ENTITY_EDGE)
    arrow(ax, flow_anchor(concat, "right"), flow_anchor(fusion, "left"), color="#789c6d")
    arrow(ax, flow_anchor(fusion, "right"), flow_anchor(value, "left"), color=FC_EDGE)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
        }
    )
    fig, ax = plt.subplots(figsize=(12.2, 5.0), dpi=160)
    ax.set_xlim(0, 14.25)
    ax.set_ylim(0.7, 6.05)
    ax.axis("off")

    draw_cnn(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_critic_architecture_comparison.svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / "fig_critic_architecture_comparison.png", bbox_inches="tight", pad_inches=0.06)


if __name__ == "__main__":
    main()
