from pathlib import Path

import matplotlib.pyplot as plt


def draw_node(ax, xy, label, facecolor="#f8fafc"):
    x, y = xy
    circle = plt.Circle((x, y), 0.055, facecolor=facecolor, edgecolor="#1f2937", linewidth=1.4, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)


def draw_arrow(ax, src, dst, color="#9ca3af", lw=1.4, text=None, text_offset=(0, 0)):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": lw},
        zorder=1,
    )
    if text is not None:
        xm = 0.5 * (src[0] + dst[0]) + text_offset[0]
        ym = 0.5 * (src[1] + dst[1]) + text_offset[1]
        ax.text(xm, ym, text, fontsize=9, color=color, ha="center", va="center")


def main() -> None:
    out_path = Path(__file__).resolve().with_name("computational_graph_backprop.png")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Left: forward graph with a branching variable.
    ax = axes[0]
    ax.set_title("Forward computational graph", fontsize=13, pad=10)

    nodes = {
        "x": (0.12, 0.68),
        "w": (0.12, 0.38),
        "u": (0.34, 0.53),
        "b": (0.34, 0.20),
        "s": (0.56, 0.68),
        "t": (0.56, 0.38),
        "L": (0.82, 0.53),
    }

    draw_node(ax, nodes["x"], "$x$")
    draw_node(ax, nodes["w"], "$w$")
    draw_node(ax, nodes["u"], "$u$")
    draw_node(ax, nodes["b"], "$b$")
    draw_node(ax, nodes["s"], "$s$")
    draw_node(ax, nodes["t"], "$t$")
    draw_node(ax, nodes["L"], "$\\mathcal{L}$", facecolor="#eef2ff")

    draw_arrow(ax, nodes["x"], nodes["u"])
    draw_arrow(ax, nodes["w"], nodes["u"])
    draw_arrow(ax, nodes["u"], nodes["s"])
    draw_arrow(ax, nodes["u"], nodes["t"])
    draw_arrow(ax, nodes["b"], nodes["t"])
    draw_arrow(ax, nodes["s"], nodes["L"])
    draw_arrow(ax, nodes["t"], nodes["L"])

    ax.text(0.34, 0.63, "$u = wx$", fontsize=10, ha="center")
    ax.text(0.56, 0.79, "$s = u^2$", fontsize=10, ha="center")
    ax.text(0.56, 0.27, "$t = u + b$", fontsize=10, ha="center")
    ax.text(0.82, 0.63, "$\\mathcal{L} = s + t$", fontsize=10, ha="center")
    ax.text(0.5, 0.06, "Node $u$ branches into two children $s$ and $t$.", ha="center", fontsize=10)

    # Right: reverse-mode accumulation.
    ax = axes[1]
    ax.set_title("Backward propagation", fontsize=13, pad=10)

    for key, xy in nodes.items():
        label = {
            "x": "$\\bar x$",
            "w": "$\\bar w$",
            "u": "$\\bar u$",
            "b": "$\\bar b$",
            "s": "$\\bar s$",
            "t": "$\\bar t$",
            "L": "$\\bar{\\mathcal{L}}=1$",
        }[key]
        draw_node(ax, xy, label, facecolor="#fff7ed" if key != "L" else "#fee2e2")

    draw_arrow(ax, nodes["L"], nodes["s"], color="#dc2626", text="$\\bar s = \\bar{\\mathcal{L}}\\, \\frac{\\partial \\mathcal{L}}{\\partial s}$", text_offset=(0.0, 0.07))
    draw_arrow(ax, nodes["L"], nodes["t"], color="#dc2626", text="$\\bar t = \\bar{\\mathcal{L}}\\, \\frac{\\partial \\mathcal{L}}{\\partial t}$", text_offset=(0.0, -0.07))
    draw_arrow(ax, nodes["s"], nodes["u"], color="#2563eb", text="$\\bar s\\, \\frac{\\partial s}{\\partial u}$", text_offset=(0.0, 0.07))
    draw_arrow(ax, nodes["t"], nodes["u"], color="#2563eb", text="$\\bar t\\, \\frac{\\partial t}{\\partial u}$", text_offset=(0.0, -0.07))
    draw_arrow(ax, nodes["t"], nodes["b"], color="#2563eb")
    draw_arrow(ax, nodes["u"], nodes["x"], color="#059669")
    draw_arrow(ax, nodes["u"], nodes["w"], color="#059669")

    ax.text(
        0.54,
        0.9,
        "$\\bar u = \\bar s\\,\\frac{\\partial s}{\\partial u} + \\bar t\\,\\frac{\\partial t}{\\partial u}$",
        fontsize=11,
        ha="center",
    )
    ax.text(0.5, 0.06, "At a branching node, reverse-mode AD sums contributions from all children.", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
