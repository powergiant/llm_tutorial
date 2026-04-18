from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    out_path = Path(__file__).resolve().with_name("mlp_graph.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layers = [
        ("Input", 0.12, [0.75, 0.5, 0.25], ["$x_1$", "$x_2$", "$x_3$"]),
        ("Hidden 1", 0.38, [0.82, 0.62, 0.38, 0.18], ["", "", "", ""]),
        ("Hidden 2", 0.64, [0.7, 0.5, 0.3], ["", "", ""]),
        ("Output", 0.88, [0.5], ["$y$"]),
    ]

    radius = 0.035

    for i in range(len(layers) - 1):
        _, x0, ys0, _ = layers[i]
        _, x1, ys1, _ = layers[i + 1]
        for y0 in ys0:
            for y1 in ys1:
                ax.plot([x0, x1], [y0, y1], color="#9aa0a6", linewidth=1.0, zorder=1)

        # Add a few representative weight labels directly on edges.
        weight_pairs = list(zip(ys0[: min(2, len(ys0))], ys1[: min(2, len(ys1))]))
        for j, (y0, y1) in enumerate(weight_pairs, start=1):
            xm = 0.58 * x0 + 0.42 * x1
            ym = 0.58 * y0 + 0.42 * y1
            y_offset = 0.025
            if i == 0 and j == 2:
                y_offset = 0.11
            elif j == 2:
                y_offset = 0.08
            ax.text(
                xm,
                ym + y_offset,
                fr"$w_{{{i+1},{j}}}$",
                fontsize=9,
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
                zorder=2,
            )

    for name, x, ys, labels in layers:
        for y, label in zip(ys, labels):
            circle = plt.Circle((x, y), radius, facecolor="#f8f9fa", edgecolor="#1f2937", linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            if label:
                ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)
            else:
                ax.text(x, y, "$\\sigma$", ha="center", va="center", fontsize=12, zorder=4)
        ax.text(x, 0.94, name, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.text(0.25, 0.9, "weights on edges", color="#4b5563", fontsize=10)
    ax.annotate("", xy=(0.31, 0.75), xytext=(0.23, 0.88), arrowprops={"arrowstyle": "->", "color": "#4b5563"})

    ax.text(0.56, 0.12, "activation at nodes", color="#4b5563", fontsize=10)
    ax.annotate("", xy=(0.64, 0.3), xytext=(0.58, 0.15), arrowprops={"arrowstyle": "->", "color": "#4b5563"})

    ax.text(0.5, 0.03, "A multi-layer perceptron: affine maps on edges, nonlinear activations at nodes", ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
