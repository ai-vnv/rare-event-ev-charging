"""Checks that the maps still match the published figure.

The constants in :mod:`evcharging.plots` and :mod:`evcharging.basemap` were read back
off the map in the paper, an 800 x 520 pixel raster.  The pixel positions below are the
centres of the eleven candidate stations in that raster, measured by segmenting the
markers on their colour; fitting a Web Mercator view to them recovers the Plotly figure
that drew it, and these tests keep the code tied to that fit.
"""
import numpy as np
import pytest

from evcharging import basemap, load_case, solve_design
from evcharging import plots

# Station centres in the published raster, in pixels from its top-left corner.
PUBLISHED_PIXELS = {
    "Embong Wungu": (462.0, 281.5),
    "Indrapura": (438.0, 151.0),
    "Ploso": (549.5, 214.0),
    "Tandes": (311.5, 158.5),
    "Perak": (421.0, 127.5),
    "Kenjeran": (558.0, 117.5),
    "Darmo Permai": (310.0, 266.0),
    "Dukuh Kupang": (392.0, 302.0),
    "Ngagel": (489.0, 301.5),
    "Rungkut": (490.0, 431.0),
    "Karang Pilang": (365.5, 428.0),
}
PUBLISHED_RASTER = (800, 520)        # the figure is 400 x 260 points, written at scale 2


@pytest.fixture(scope="module")
def case():
    return load_case()


def test_window_reproduces_the_published_figure(case):
    """Every station lands within a pixel of where the paper puts it."""
    window = plots.map_window(case)
    scale = PUBLISHED_RASTER[0] / window.width
    south, north, west, east = window.bounds
    y_north, y_south = basemap.mercator_y(north), basemap.mercator_y(south)

    worst = 0.0
    for station in case.stations.itertuples():
        x = (station.lon - west) / (east - west) * PUBLISHED_RASTER[0]
        y = (basemap.mercator_y(station.lat) - y_north) / (y_south - y_north) * PUBLISHED_RASTER[1]
        want = PUBLISHED_PIXELS[station.station]
        worst = max(worst, np.hypot(x - want[0], y - want[1]))
    assert worst < 1.0, f"worst station is {worst:.2f} px off the published figure"
    assert scale == pytest.approx(2.0)


def test_window_is_centred_on_the_mean_node(case):
    """Plotly Express centres a scatter_mapbox on the mean of the coordinates."""
    window = plots.map_window(case)
    lat = np.r_[case.demand.lat.values, case.stations.lat.values]
    lon = np.r_[case.demand.lon.values, case.stations.lon.values]
    assert window.lat == pytest.approx(lat.mean())
    assert window.lon == pytest.approx(lon.mean())
    assert window.zoom == pytest.approx(9.9)


def test_marker_areas_follow_the_original_code(case):
    design = solve_design(case, {j: 1.0 for j in case.J}, tie_break="best",
                          reference=case.reliability)
    nodes = plots.node_table(case, design)
    # some stations share a name with a sub-district, so split the table by kind first
    demand = nodes[nodes["kind"] == plots.DEMAND_LABEL].set_index("node")
    stations = nodes[nodes["kind"] != plots.DEMAND_LABEL].set_index("node")

    # a demand node carries its population in units of ten thousand people
    biggest = case.demand.loc[case.demand.population.idxmax()]
    assert demand.loc[biggest.subdistrict, "size"] == round(biggest.population / 10_000)

    # an opened station five per connector, a closed one a fixed three
    for j in case.J:
        if j in design.opened:
            assert stations.loc[j, "size"] == 5.0 * design.u[j]
            assert stations.loc[j, "kind"] == plots.SELECTED_LABEL
        else:
            assert stations.loc[j, "size"] == 3.0
            assert stations.loc[j, "kind"] == plots.CLOSED_LABEL


def test_published_marker_diameters(case):
    """The largest marker is 23.6 points across, and the rest scale by square root."""
    design = solve_design(case, {j: 1.0 for j in case.J}, tie_break="best",
                          reference=case.reliability)
    nodes = plots.node_table(case, design)
    reference = nodes["size"].max()
    diameter = plots.MARKER_MAX * np.sqrt(nodes["size"] / reference)
    assert diameter.max() == pytest.approx(plots.MARKER_MAX)
    # a closed station against the opened one with the most connectors
    closed = diameter[nodes["kind"] == plots.CLOSED_LABEL].iloc[0]
    assert closed / diameter.max() == pytest.approx(np.sqrt(3.0 / reference))


def test_basemap_cache_covers_the_window(case):
    canvas = basemap.load()
    if canvas is None:
        pytest.skip("no basemap cached; run scripts/fetch_basemap.py")
    south, north, west, east = plots.map_window(case).bounds
    assert canvas.west <= west and canvas.east >= east
    assert canvas.south <= south and canvas.north >= north
    assert canvas.attribution


def test_figures_build(case, tmp_path):
    design = solve_design(case, case.reliability)
    plots.use_paper_style()
    plots.save(plots.figure_map(case, design), tmp_path / "map", tight=False, dpi=72)
    plots.save(plots.figure_designs(case, {"Robust": design}), tmp_path / "designs",
               tight=False, dpi=72)
    assert (tmp_path / "map.png").exists() and (tmp_path / "map.pdf").exists()
    assert (tmp_path / "designs.png").exists()
