from pathlib import Path

import matplotlib.pyplot as plt


def draw_node(ax, xy, label, facecolor="#f8fafc"):
    x, y = xy
    circle = plt.Circle((x, y), 0.05, facecolor=facecolor, edgecolor="#1f2937", linewidth=1.4, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)


def draw_arrow(ax, src, dst, color="#9ca3af", lw=1.3):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": lw},
        zorder=1,
    )


def main() -> None:
    out_path = Path(__file__).resolve().with_name("computational_graph_general.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "p1": (0.12, 0.72),
        "p2": (0.12, 0.50),
        "pr": (0.12, 0.28),
        "vj": (0.40, 0.50),
        "k1": (0.68, 0.72),
        "k2": (0.68, 0.50),
        "ks": (0.68, 0.28),
        "L": (0.90, 0.50),
    }

    draw_node(ax, nodes["p1"], "$v_{p_1(j)}$")
    draw_node(ax, nodes["p2"], "$v_{p_2(j)}$")
    draw_node(ax, nodes["pr"], "$v_{p_{k_j}(j)}$")
    draw_node(ax, nodes["vj"], "$v_j$", facecolor="#eef2ff")
    draw_node(ax, nodes["k1"], "$v_{k_1}$")
    draw_node(ax, nodes["k2"], "$v_{k_2}$")
    draw_node(ax, nodes["ks"], "$v_{k_s}$")
    draw_node(ax, nodes["L"], "$\\mathcal{L}$", facecolor="#fee2e2")

    for parent in ["p1", "p2", "pr"]:
        draw_arrow(ax, nodes[parent], nodes["vj"])
    for child in ["k1", "k2", "ks"]:
        draw_arrow(ax, nodes["vj"], nodes[child])
        draw_arrow(ax, nodes[child], nodes["L"])

    ax.text(0.40, 0.61, "$v_j = \\phi_j(v_{p_1(j)},\\dots,v_{p_{k_j}(j)})$", ha="center", fontsize=10)
    ax.text(0.68, 0.86, "$\\mathrm{child}(j)=\\{k_1,k_2,\\dots,k_s\\}$", ha="center", fontsize=10)
    ax.text(
        0.52,
        0.10,
        "$\\bar v_j = \\sum_{k \\in \\mathrm{child}(j)} \\bar v_k \\, \\frac{\\partial v_k}{\\partial v_j}$",
        ha="center",
        fontsize=12,
    )
    ax.text(0.52, 0.04, "Abstract reverse-mode rule: sum contributions from all children of $v_j$.", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
