from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection

from ellipse_rcnn.utils.conics import ellipse_angle, conic_center, ellipse_axes


def plot_ellipses(
    ellipses: torch.Tensor,
    figsize: tuple[float, float] = (15., 15.),
    plot_centers: bool = False,
    ax: Axes | None = None,
    rim_color: str = "r",
    alpha: float = 1.0,
):
    a_proj, b_proj = ellipse_axes(ellipses)
    psi_proj = ellipse_angle(ellipses)
    x_pix_proj, y_pix_proj = conic_center(ellipses)

    a_proj, b_proj, psi_proj, x_pix_proj, y_pix_proj = map(
        lambda t: t.detach().cpu().numpy(),
        (a_proj, b_proj, psi_proj, x_pix_proj, y_pix_proj)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})

    ec = EllipseCollection(
        a_proj * 2,
        b_proj * 2,
        psi_proj,
        units="xy",
        offsets=(y_pix_proj, x_pix_proj),
        transOffset=ax.transData,
        facecolors="None",
        edgecolors=rim_color,
        alpha=alpha,
    )
    ax.add_collection(ec)

    if plot_centers:
        crater_centers = conic_center(ellipses)
        for k, c_i in enumerate(crater_centers):
            x, y = c_i[0], c_i[1]
            ax.text(x.item(), y.item(), str(k))
