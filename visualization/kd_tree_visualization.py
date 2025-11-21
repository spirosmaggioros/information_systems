from typing import Any, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree


def plot_kdtree_2d(
    tree: KDTree, labels: Optional[list] = None, figsize: tuple[int, int] = (12, 8)
) -> KDTree:
    """
    Plot a KD-tree in 2D space showing the recursive partitioning.
    """
    # Create figure and axis
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    points = tree.get_arrays()[0]
    plot_partitions(ax1, points, labels=labels, max_depth=None)
    ax1.set_title("KD-Tree Partitioning")
    ax1.grid(True, alpha=0.3)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[i],
                    markersize=8,
                    label=f"Label {label}",
                )
            )
        ax1.legend(handles=legend_elements, loc="best")

    plot_tree_structure(ax2, points, max_depth=None)
    ax2.set_title("Tree Structure")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()

    return tree


def plot_partitions(
    ax: Any, points: np.ndarray, labels: Optional[list] = None, max_depth: Any = None
) -> None:
    """Plot the recursive partitioning of KD-tree with optional colored labels"""

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                c=[colors[i]],
                s=50,
                zorder=5,
                label=f"Label {label}",
                alpha=0.8,
            )
    else:
        ax.scatter(points[:, 0], points[:, 1], c="red", s=50, zorder=5)

    x_min, x_max = points[:, 0].min() - 0.1, points[:, 0].max() + 0.1
    y_min, y_max = points[:, 1].min() - 0.1, points[:, 1].max() + 0.1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    draw_partitions_recursive(
        ax, points, x_min, x_max, y_min, y_max, depth=0, max_depth=max_depth
    )


def draw_partitions_recursive(
    ax: Any,
    points: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    depth: int = 0,
    max_depth: Any = None,
) -> None:
    """Recursively draw the partitioning lines"""

    if len(points) <= 1 or (max_depth is not None and depth >= max_depth):
        return

    axis = depth % 2

    sorted_points = points[points[:, axis].argsort()]

    median_idx = len(sorted_points) // 2
    median_point = sorted_points[median_idx]
    median_value = median_point[axis]

    if axis == 0:
        ax.axvline(
            x=median_value,
            ymin=(y_min - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
            ymax=(y_max - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
            color="blue",
            alpha=0.7,
            linewidth=2,
        )

        left_points = sorted_points[:median_idx]
        right_points = sorted_points[median_idx + 1 :]

        if len(left_points) > 0:
            draw_partitions_recursive(
                ax, left_points, x_min, median_value, y_min, y_max, depth + 1, max_depth
            )
        if len(right_points) > 0:
            draw_partitions_recursive(
                ax,
                right_points,
                median_value,
                x_max,
                y_min,
                y_max,
                depth + 1,
                max_depth,
            )
    else:
        ax.axhline(
            y=median_value,
            xmin=(x_min - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
            xmax=(x_max - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
            color="green",
            alpha=0.7,
            linewidth=2,
        )

        bottom_points = sorted_points[:median_idx]
        top_points = sorted_points[median_idx + 1 :]

        if len(bottom_points) > 0:
            draw_partitions_recursive(
                ax,
                bottom_points,
                x_min,
                x_max,
                y_min,
                median_value,
                depth + 1,
                max_depth,
            )
        if len(top_points) > 0:
            draw_partitions_recursive(
                ax, top_points, x_min, x_max, median_value, y_max, depth + 1, max_depth
            )


def plot_tree_structure(ax: Any, points: np.ndarray, max_depth: Any = None) -> None:
    """Plot the tree structure as a hierarchical diagram"""

    tree_data = build_tree_data(points, max_depth)
    if tree_data:
        draw_tree_node(ax, tree_data, 0.5, 0.9, 0.4, 0)


def build_tree_data(
    points: np.ndarray, max_depth: Any = None, depth: int = 0
) -> Optional[dict]:
    """Build tree data structure for visualization"""

    if len(points) <= 1 or (max_depth is not None and depth >= max_depth):
        return None

    axis = depth % 2
    sorted_points = points[points[:, axis].argsort()]
    median_idx = len(sorted_points) // 2
    median_point = sorted_points[median_idx]

    left_points = sorted_points[:median_idx]
    right_points = sorted_points[median_idx + 1 :]

    return {
        "point": median_point,
        "axis": axis,
        "left": (
            build_tree_data(left_points, max_depth, depth + 1)
            if len(left_points) > 0
            else None
        ),
        "right": (
            build_tree_data(right_points, max_depth, depth + 1)
            if len(right_points) > 0
            else None
        ),
        "depth": depth,
    }


def draw_tree_node(
    ax: Any, node: Optional[dict], x: float, y: float, width: float, depth: float
) -> None:
    """Draw a node in the tree structure"""

    if node is None:
        return

    axis_label = "X" if node["axis"] == 0 else "Y"
    node_text = f"{axis_label}: {node['point'][node['axis']]:.2f}"

    color = "lightblue" if node["axis"] == 0 else "lightgreen"

    rect = patches.Rectangle(
        (x - 0.08, y - 0.03),
        0.16,
        0.06,
        linewidth=1,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(rect)

    ax.text(x, y, node_text, ha="center", va="center", fontsize=8, weight="bold")

    child_y = y - 0.15
    left_x = x - width / 2
    right_x = x + width / 2

    if node["left"]:
        ax.plot([x, left_x], [y - 0.03, child_y + 0.03], "k-", alpha=0.6)
        draw_tree_node(ax, node["left"], left_x, child_y, width / 2, depth + 1)

    if node["right"]:
        ax.plot([x, right_x], [y - 0.03, child_y + 0.03], "k-", alpha=0.6)
        draw_tree_node(ax, node["right"], right_x, child_y, width / 2, depth + 1)
