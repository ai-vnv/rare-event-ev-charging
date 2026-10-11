"""Figures of the paper.

The maps reproduce the figure of the paper, which the original
`EVChargeSim <https://github.com/mansurarief/EVChargeSim>`_ code drew with
``plotly.express.scatter_mapbox`` on the Carto Positron basemap: demand nodes sized by
population, opened stations sized by the number of connectors, candidates that stay
closed at a fixed small size, and a black line for every demand node served by a
station.  Everything is redrawn here in Matplotlib against the cached basemap of
:mod:`evcharging.basemap`, so the figure is vector, offline and deterministic, and the
geometry, colours and sizes of the published figure are kept exactly; see
``tests/test_plots.py`` for the measurements they were taken from.

The remaining figures use the Okabe-Ito palette and vary line style as well as colour,
so they stay readable for colour-blind readers and in black-and-white print.  Fonts are
embedded as TrueType (``pdf.fonttype = 42``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import basemap as basemap_module  # noqa: E402
from .data import Case  # noqa: E402
from .model import Design  # noqa: E402

BLUE = "#0072B2"         # control variates
VERMILLION = "#D55E00"   # Monte Carlo
INK = "#222222"
MUTED = "#555555"
GRID = "#C8C8C8"

MILLION = 1e6

# ----------------------------------------------------------------------------------
# The published map, measured
# ----------------------------------------------------------------------------------
# Plotly's qualitative palette, in the order Plotly Express assigns it to the three
# node types of the figure.
DEMAND_COLOUR = "#636EFA"
SELECTED_COLOUR = "#EF553B"
CLOSED_COLOUR = "#00CC96"
NODE_ALPHA = 0.70

DEMAND_LABEL = "Demand"
SELECTED_LABEL = "Station (Selected)"
CLOSED_LABEL = "Station (Not Selected)"

# Marker areas, as the original code sets them: a demand node carries its population in
# units of ten thousand people, a station that stays closed a fixed 3, and an opened
# station five per connector.
DEMAND_PER_PERSON = 1e-4
CLOSED_SIZE = 3.0
SELECTED_PER_CONNECTOR = 5.0

# Plotly scales bubbles by area, so a marker of area ``a`` is drawn with diameter
# ``MARKER_MAX * sqrt(a / a_max)``.  The figure of the paper is 400 x 260 points, and
# there the largest marker comes out 23.6 points across.  Lines and type are given in
# points at that size too; everything scales with the width of the map.
FIGURE_SIZE = basemap_module.FIGURE_SIZE
MARKER_MAX = 23.6
LINE_WIDTH = 0.2
LEGEND_FONT = 9.6
LEGEND_MARKER = 12.0
ATTRIBUTION_FONT = 10.0


def use_paper_style() -> None:
    """Matplotlib settings shared by all figures."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True, "savefig.dpi": 300,
    })


def save(fig, path: Path, tight: bool = True, dpi: int = 200) -> None:
    """Write ``path`` as PDF and a PNG next to it.

    Maps are saved with ``tight=False``: they are laid out to fill the figure exactly,
    the way the published figure bleeds to its edges.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(bbox_inches="tight", pad_inches=0.02) if tight else {}
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None},  # stable PDF bytes
                **extra)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, **extra)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------

def node_table(case: Case, design: Optional[Design] = None) -> pd.DataFrame:
    """Every node of the map with the colour, label and marker area of the paper.

    Demand nodes are sized by population in units of ten thousand people, a station
    that stays closed has a fixed small size, and an opened station is sized by its
    number of connectors -- exactly as ``create_plot`` does in EVChargeSim.
    """
    demand = pd.DataFrame({
        "node": case.demand["subdistrict"], "lat": case.demand["lat"], "lon": case.demand["lon"],
        "size": np.round(case.demand["population"] * DEMAND_PER_PERSON),
        "kind": DEMAND_LABEL, "colour": DEMAND_COLOUR,
    })
    stations = pd.DataFrame({
        "node": case.stations["station"], "lat": case.stations["lat"], "lon": case.stations["lon"],
        "size": CLOSED_SIZE, "kind": CLOSED_LABEL, "colour": CLOSED_COLOUR,
    })
    if design is not None:
        opened = stations["node"].isin(design.opened)
        stations.loc[opened, "size"] = [SELECTED_PER_CONNECTOR * design.u[j]
                                        for j in stations.loc[opened, "node"]]
        stations.loc[opened, ["kind", "colour"]] = [SELECTED_LABEL, SELECTED_COLOUR]
    return pd.concat([demand, stations], ignore_index=True)


def map_window(case: Case, size: Tuple[float, float] = FIGURE_SIZE,
               zoom: float = basemap_module.MAP_ZOOM) -> basemap_module.Window:
    """The Web Mercator view of the published map."""
    nodes = node_table(case)
    return basemap_module.window_for(nodes["lat"], nodes["lon"], size=size, zoom=zoom)


def _draw_basemap(ax, window: basemap_module.Window) -> str:
    """Paint the cached basemap behind ``ax`` and return its attribution."""
    canvas = basemap_module.load()
    if canvas is None:
        ax.set_facecolor("#F2F2F0")
        return ""
    ax.imshow(canvas.image, extent=canvas.extent, zorder=0, interpolation="bilinear",
              aspect="auto")
    return canvas.attribution


def plot_map(ax, case: Case, design: Optional[Design] = None, title: str = "",
             legend: bool = True, attribution: bool = True, width: Optional[float] = None,
             size_ref: Optional[float] = None) -> None:
    """Draw the map of the paper on ``ax``.

    With a ``design``, opened stations turn red and a line is drawn from every demand
    node to the station that serves it.  ``width`` is the width of ``ax`` in points and
    defaults to the width of the figure; every size on the map scales with it.
    ``size_ref`` is the marker area that is drawn at full size, the largest area on this
    map unless given -- pass the same value to several panels to make them comparable.
    """
    if width is None:
        width = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    scale = width / FIGURE_SIZE[0]
    nodes = node_table(case, design)
    window = map_window(case)
    credit = _draw_basemap(ax, window)

    if design is not None:
        # Some stations share a name with a sub-district, so keep the two lookups apart.
        here = case.demand.set_index("subdistrict")
        there = case.stations.set_index("station")
        for i in case.I:
            for j in case.J:
                if design.flow(case, i, j) > 0:              # links that carry vehicles
                    ax.plot([here.lon[i], there.lon[j]], [here.lat[i], there.lat[j]], color="black",
                            lw=LINE_WIDTH * scale, zorder=3, solid_capstyle="butt")

    reference = float(nodes["size"].max()) if size_ref is None else float(size_ref)
    for kind in (DEMAND_LABEL, SELECTED_LABEL, CLOSED_LABEL):
        group = nodes[nodes["kind"] == kind]
        if group.empty:
            continue
        diameter = MARKER_MAX * scale * np.sqrt(group["size"].to_numpy() / reference)
        ax.scatter(group["lon"], group["lat"], s=diameter**2, c=group["colour"].iloc[0],
                   alpha=NODE_ALPHA, linewidths=0, zorder=1 if kind == DEMAND_LABEL else 2)

    south, north, west, east = window.bounds
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, pad=3 * scale, fontsize=LEGEND_FONT * scale, color=INK)
    if attribution and credit:
        ax.text(0.995, 0.012, credit, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=ATTRIBUTION_FONT * scale, color="black", zorder=5)
    if legend:
        add_legend(ax, design, scale)


def add_legend(ax, design: Optional[Design] = None, scale: float = 1.0):
    """The horizontal white legend that sits at the top of the published map."""
    kinds = [(DEMAND_LABEL, DEMAND_COLOUR)]
    kinds += ([(SELECTED_LABEL, SELECTED_COLOUR), (CLOSED_LABEL, CLOSED_COLOUR)]
              if design is not None else [("Station", CLOSED_COLOUR)])
    handles = [Line2D([], [], marker="o", ls="", color=colour, alpha=NODE_ALPHA,
                      markersize=LEGEND_MARKER * scale, label=label) for label, colour in kinds]
    font = LEGEND_FONT * scale
    legend = ax.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=len(handles),
        fontsize=font, frameon=True, facecolor="white", edgecolor="#D0D0D0", framealpha=1.0,
        borderpad=0.67, handlelength=LEGEND_MARKER * scale / font, handletextpad=1.58,
        columnspacing=2.12, borderaxespad=0.25,
    )
    legend.get_frame().set_linewidth(0.5 * scale)
    legend.set_zorder(6)
    return legend


def figure_map(case: Case, design: Optional[Design] = None, width: float = FIGURE_SIZE[0],
               legend: bool = True, title: str = "", size_ref: Optional[float] = None):
    """One map, laid out exactly like the figure of the paper."""
    height = width * FIGURE_SIZE[1] / FIGURE_SIZE[0]
    fig = plt.figure(figsize=(width / 72.0, height / 72.0), facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    plot_map(ax, case, design, legend=legend, title=title, width=width, size_ref=size_ref)
    return fig


def figure_candidates(case: Case, width: float = FIGURE_SIZE[0]):
    """Fig. 2: demand nodes and the candidate stations, before any design."""
    return figure_map(case, None, width=width)


def figure_designs(case: Case, designs: Dict[str, Design], ncols: int = 2,
                   width: float = FIGURE_SIZE[0]):
    """One map per design, ``ncols`` per row, each panel laid out like the paper's.

    Every panel is drawn to one common marker scale, so the designs can be compared.
    """
    names = list(designs)
    nrows = int(np.ceil(len(names) / ncols))
    size_ref = max(node_table(case, d)["size"].max() for d in designs.values())
    panel_h = width * FIGURE_SIZE[1] / FIGURE_SIZE[0]
    title = 0.055 * panel_h
    fig = plt.figure(figsize=(width * ncols / 72.0, (panel_h + title) * nrows / 72.0),
                     facecolor="white")
    for k, name in enumerate(names):
        row, col = divmod(k, ncols)
        # Each row is a band; the panel sits at the bottom of it and the title above.
        ax = fig.add_axes((col / ncols, 1.0 - (row + 1) / nrows,
                           1.0 / ncols, panel_h / (panel_h + title) / nrows))
        plot_map(ax, case, designs[name], title=name, legend=(k == 0), width=width,
                 size_ref=size_ref)
    return fig


# ----------------------------------------------------------------------------------
# Simulated objective, Monte Carlo against control variates
# ----------------------------------------------------------------------------------

def figure_objective(curves: Dict[str, dict], exact: Dict[str, float]):
    """Running mean and 95% interval of the simulated objective for each design.

    ``curves[name]`` holds ``{"mc": running, "cv": running}`` from
    :func:`evcharging.running_estimate`; ``exact[name]`` is the exact expectation.
    """
    names = list(curves)
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.15), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        mc, cv = curves[name]["mc"], curves[name]["cv"]
        ax.fill_between(mc["n"], mc["lo"] / MILLION, mc["hi"] / MILLION, color=VERMILLION, alpha=0.16, lw=0)
        ax.plot(mc["n"], mc["mean"] / MILLION, color=VERMILLION, lw=1.0, ls=(0, (4, 2)))
        ax.fill_between(cv["n"], cv["lo"] / MILLION, cv["hi"] / MILLION, color=BLUE, alpha=0.30, lw=0)
        ax.plot(cv["n"], cv["mean"] / MILLION, color=BLUE, lw=1.2)
        ax.axhline(exact[name] / MILLION, color=INK, lw=0.6, ls=(0, (1, 2)))
        ax.set_title(name, pad=2.5)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        half = 1.96 * cv["se"][-1] / MILLION
        ax.text(0.97, 0.06, f"CV at n = {int(cv['n'][-1]):,}:\n{cv['mean'][-1] / MILLION:.2f} \u00b1 {half:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=INK)
    lo = min(exact.values()) / MILLION
    hi = max(exact.values()) / MILLION
    axes[0, 0].set_ylim(lo - 7.5, hi + 5.0)
    axes[0, 0].set_yticks([120, 125, 130])
    axes[0, 0].set_xlim(0, curves[names[0]]["mc"]["n"][-1])
    n_max = int(curves[names[0]]["mc"]["n"][-1])
    for ax in axes[1]:
        ax.set_xticks([0, n_max // 2, n_max])
        labels = ax.set_xticklabels(["0", f"{n_max // 2:,}", f"{n_max:,}"])
        labels[0].set_ha("left")
        labels[-1].set_ha("right")
    fig.supxlabel("Sample size $n$ (simulated days)", fontsize=7.5, y=0.105)
    fig.supylabel("Simulated objective (million IDR per day)", fontsize=7.5, x=0.005)
    handles = [Line2D([], [], color=VERMILLION, lw=1.0, ls=(0, (4, 2)), label="Monte Carlo"),
               Line2D([], [], color=BLUE, lw=1.2, label="Control variates"),
               Line2D([], [], color=INK, lw=0.6, ls=(0, (1, 2)), label="Exact expectation")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, -0.018),
               handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.145, right=0.975, top=0.925, bottom=0.245, wspace=0.08, hspace=0.36)
    return fig


def figure_cv_sensitivity(rows: Sequence[dict], default: float):
    """Standard-error ratio (Monte Carlo over control variates) against the proxy correlation."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.array([1.0 - r["proxy_corr"] for r in rows])
    ax.plot(x, [r["se_ratio"] for r in rows], color=BLUE, lw=1.4, marker="o", ms=3.5,
            markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.axvline(1.0 - default, color=INK, lw=0.6, ls=(0, (1, 2)))
    ax.text(1.0 - default, ax.get_ylim()[1] * 0.97, f" default {default}", ha="left", va="top", fontsize=6.5)
    ax.set_xlabel("1 \u2212 correlation between the proxy and the standardised demand")
    ax.set_ylabel("Standard error, MC / CV")
    ax.grid(color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.2)
    return fig
