from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, xy, width, height, label, facecolor="#f8fafc", edgecolor="#1f2937"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=12, zorder=4)


def draw_arrow(ax, src, dst, color="#64748b", lw=1.4, rad=0.0):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": lw,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=2,
    )


def main() -> None:
    out_path = Path(__file__).resolve().with_name("rnn_structure.png")

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.18, 0.39, 0.60, 0.81]
    cell_w, cell_h = 0.11, 0.13
    y_cell = 0.48
    y_input = 0.18
    y_output = 0.76

    for idx, x in enumerate(xs, start=1):
        suffix = str(idx) if idx < 4 else "T"
        draw_box(ax, (x - cell_w / 2, y_cell), cell_w, cell_h, f"$h_{suffix}$", facecolor="#eef2ff")
        draw_box(ax, (x - 0.045, y_input), 0.09, 0.10, f"$x_{suffix}$", facecolor="#ecfeff")
        draw_box(ax, (x - 0.045, y_output), 0.09, 0.10, f"$y_{suffix}$", facecolor="#fef3c7")

        draw_arrow(ax, (x, y_input + 0.10), (x, y_cell), color="#0891b2")
        draw_arrow(ax, (x, y_cell + cell_h), (x, y_output), color="#d97706")

    for i in range(len(xs) - 1):
        x0 = xs[i] + cell_w / 2
        x1 = xs[i + 1] - cell_w / 2
        draw_arrow(ax, (x0, y_cell + cell_h / 2), (x1, y_cell + cell_h / 2), color="#4f46e5")

    ax.text(0.50, 0.94, "same recurrent parameters $W_x, W_h, W_y$ are reused at every time step", ha="center", fontsize=11)
    ax.text(0.50, 0.095, "$h_t = \\phi(W_xx_t + W_hh_{t-1} + b_h)$", ha="center", fontsize=13)
    ax.text(0.50, 0.035, "$y_t = \\operatorname{softmax}(W_yh_t + b_y)$", ha="center", fontsize=13)

    ax.text(0.07, 0.84, "output", ha="center", fontsize=11, color="#92400e")
    ax.text(0.05, 0.54, "hidden state", ha="center", fontsize=11, color="#3730a3")
    ax.text(0.07, 0.23, "input", ha="center", fontsize=11, color="#0e7490")

    draw_arrow(ax, (0.10, y_cell + cell_h / 2), (xs[0] - cell_w / 2, y_cell + cell_h / 2), color="#4f46e5")
    ax.text(0.055, y_cell + cell_h / 2 + 0.075, "$h_0$", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
