from pathlib import Path
import os
import random
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "llm_tutorial_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_channel(ax, x, y, color, label, value):
    square = patches.Rectangle(
        (x, y),
        1.25,
        1.25,
        facecolor=color,
        edgecolor="#1f2937",
        linewidth=1.4,
    )
    ax.add_patch(square)
    ax.text(x + 0.62, y + 0.72, label, ha="center", va="center", fontsize=18, fontweight="bold", color="#111827")
    ax.text(x + 0.62, y + 0.34, value, ha="center", va="center", fontsize=11, color="#374151")


def make_channel_grids(size=7):
    rng = random.Random(8)
    red = []
    green = []
    blue = []
    center = (size - 1) / 2

    for i in range(size):
        red_row = []
        green_row = []
        blue_row = []
        for j in range(size):
            radial = max(0.0, 1.0 - ((i - center) ** 2 + (j - center) ** 2) / 18.0)
            diagonal = (i + j) / (2 * (size - 1))
            wave = 0.5 + 0.5 * ((i * 2 + j * 3) % 7) / 6.0
            noise = rng.uniform(-0.12, 0.12)

            red_row.append(min(1.0, max(0.0, 0.25 + 0.65 * radial + noise)))
            green_row.append(min(1.0, max(0.0, 0.15 + 0.65 * diagonal + noise * 0.7)))
            blue_row.append(min(1.0, max(0.0, 0.2 + 0.55 * wave - 0.2 * radial + noise * 0.5)))
        red.append(red_row)
        green.append(green_row)
        blue.append(blue_row)

    return red, green, blue


def draw_intensity_grid(ax, x, y, values, channel, label):
    size = len(values)
    cell = 0.17
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if channel == "r":
                color = (value, 0.08, 0.08)
            elif channel == "g":
                color = (0.08, value, 0.08)
            else:
                color = (0.08, 0.08, value)
            ax.add_patch(
                patches.Rectangle(
                    (x + j * cell, y + (size - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor="#f3f4f6",
                    linewidth=0.25,
                )
            )

    width = size * cell
    ax.add_patch(patches.Rectangle((x, y), width, width, fill=False, edgecolor="#1f2937", linewidth=1.0))
    ax.text(x + width / 2, y - 0.18, label, ha="center", fontsize=8.8, color="#374151")


def draw_composed_rgb_grid(ax, x, y, red, green, blue):
    size = len(red)
    cell = 0.22
    for i in range(size):
        for j in range(size):
            color = (red[i][j], green[i][j], blue[i][j])
            ax.add_patch(
                patches.Rectangle(
                    (x + j * cell, y + (size - 1 - i) * cell),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor="#f3f4f6",
                    linewidth=0.25,
                )
            )

    width = size * cell
    ax.add_patch(patches.Rectangle((x, y), width, width, fill=False, edgecolor="#1f2937", linewidth=1.2))
    ax.text(x + width / 2, y - 0.22, "color image", ha="center", fontsize=9.5, color="#374151")
    ax.text(x + width / 2, y + width + 0.18, "$H \\times W \\times 3$", ha="center", fontsize=10.5, color="#111827")


def main() -> None:
    out_path = Path(__file__).resolve().with_name("rgb_channels.png")

    fig, ax = plt.subplots(figsize=(15.6, 4.2))
    ax.set_xlim(0, 15.8)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.text(3.4, 3.8, "One RGB pixel is three channel values", ha="center", fontsize=13, fontweight="bold")

    draw_channel(ax, 0.55, 1.55, "#fecaca", "R", "red = 220")
    draw_channel(ax, 2.05, 1.55, "#bbf7d0", "G", "green = 120")
    draw_channel(ax, 3.55, 1.55, "#bfdbfe", "B", "blue = 60")

    ax.text(1.95, 0.78, "channel values", ha="center", fontsize=10, color="#4b5563")
    ax.text(1.92, 2.18, "+", ha="center", va="center", fontsize=22, color="#374151")
    ax.text(3.42, 2.18, "+", ha="center", va="center", fontsize=22, color="#374151")

    ax.annotate(
        "",
        xy=(5.75, 2.18),
        xytext=(5.02, 2.18),
        arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#374151"},
    )

    mixed_color = (220 / 255, 120 / 255, 60 / 255)
    pixel = patches.Rectangle(
        (6.05, 1.55),
        1.25,
        1.25,
        facecolor=mixed_color,
        edgecolor="#1f2937",
        linewidth=1.4,
    )
    ax.add_patch(pixel)
    ax.text(6.68, 0.78, "displayed color", ha="center", fontsize=10, color="#4b5563")

    ax.text(6.68, 3.0, "$(220, 120, 60)$", ha="center", fontsize=12, color="#111827")
    ax.text(6.68, 1.18, "one pixel", ha="center", fontsize=11, color="#374151")

    ax.text(
        3.85,
        0.28,
        "For an image, every spatial position has an RGB triple, so the image has shape $H \\times W \\times 3$.",
        ha="center",
        fontsize=9.5,
    )

    ax.plot([7.65, 7.65], [0.35, 3.75], color="#d1d5db", linewidth=1.0)

    ax.text(11.25, 3.8, "RGB channel grids compose one image", ha="center", fontsize=13, fontweight="bold")

    red, green, blue = make_channel_grids()
    draw_intensity_grid(ax, 8.0, 1.5, red, "r", "R channel")
    draw_intensity_grid(ax, 9.55, 1.5, green, "g", "G channel")
    draw_intensity_grid(ax, 11.1, 1.5, blue, "b", "B channel")
    ax.text(9.35, 2.1, "+", ha="center", va="center", fontsize=20, color="#374151")
    ax.text(10.9, 2.1, "+", ha="center", va="center", fontsize=20, color="#374151")

    ax.annotate(
        "",
        xy=(12.88, 2.1),
        xytext=(12.45, 2.1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#374151"},
    )

    draw_composed_rgb_grid(ax, 13.05, 1.28, red, green, blue)
    ax.text(
        11.25,
        0.72,
        "Each output pixel uses the same grid location from the R, G, and B channels.",
        ha="center",
        fontsize=9.5,
        color="#111827",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
