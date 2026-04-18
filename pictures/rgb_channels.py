from pathlib import Path
import os
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


def main() -> None:
    out_path = Path(__file__).resolve().with_name("rgb_channels.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    ax.text(4.4, 3.45, "One RGB pixel is three channel values", ha="center", fontsize=14, fontweight="bold")

    draw_channel(ax, 0.65, 1.55, "#fecaca", "R", "red = 220")
    draw_channel(ax, 2.15, 1.55, "#bbf7d0", "G", "green = 120")
    draw_channel(ax, 3.65, 1.55, "#bfdbfe", "B", "blue = 60")

    ax.text(1.95, 0.78, "channel values", ha="center", fontsize=10, color="#4b5563")
    ax.text(2.0, 2.18, "+", ha="center", va="center", fontsize=22, color="#374151")
    ax.text(3.5, 2.18, "+", ha="center", va="center", fontsize=22, color="#374151")

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
        4.4,
        0.28,
        "For an image, every spatial position has an RGB triple, so the image has shape $H \\times W \\times 3$.",
        ha="center",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
