from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ground_truth(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vstack([x**p for p in range(degree + 1)]).T


def fit_polynomial_features(
    x_train: np.ndarray, y_train: np.ndarray, degree: int, x_eval: np.ndarray
) -> np.ndarray:
    x_scaled = 2.0 * x_train - 1.0
    x_eval_scaled = 2.0 * x_eval - 1.0
    features = design_matrix(x_scaled, degree)
    theta, *_ = np.linalg.lstsq(features, y_train, rcond=None)
    return design_matrix(x_eval_scaled, degree) @ theta


def main() -> None:
    rng = np.random.default_rng(7)

    root = Path(__file__).resolve().parent
    output_path = root / "overfitting_polynomial_linear_model.png"

    # Use a small, unevenly spaced sample so a high-degree polynomial fit
    # becomes visibly unstable between noisy observations.
    x_train = np.array([0.03, 0.08, 0.16, 0.27, 0.43, 0.61, 0.78, 0.92])
    y_train = ground_truth(x_train) + rng.normal(0.0, 0.14, size=x_train.shape)

    x_plot = np.linspace(0.0, 1.0, 1200)
    y_true = ground_truth(x_plot)

    underfit_degree = 1
    overfit_degree = 14

    y_underfit = fit_polynomial_features(x_train, y_train, underfit_degree, x_plot)
    y_overfit = fit_polynomial_features(x_train, y_train, overfit_degree, x_plot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    panels = [
        (axes[0], underfit_degree, y_underfit, "Small model: underfit"),
        (axes[1], overfit_degree, y_overfit, "Large model: overfit"),
    ]

    for ax, degree, prediction, title in panels:
        ax.plot(x_plot, y_true, color="black", linewidth=2.2, label="ground truth $f(x)$")
        ax.plot(
            x_plot,
            prediction,
            color="#d62728",
            linewidth=2.0,
            label=fr"linear model on $K(x)=(1,x,\dots,x^{{{degree}}})$",
        )
        ax.scatter(x_train, y_train, color="#1f77b4", s=34, zorder=3, label="training samples")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-4.0, 4.0)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Underfitting and overfitting for linear models in polynomial features", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
