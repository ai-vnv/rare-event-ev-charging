"""Cached light-canvas basemap for the maps of the paper.

The maps of the paper are drawn on the Carto *Positron* basemap, the tile set that
Plotly serves as ``mapbox_style="carto-positron"`` in the original Plotly
implementation.  Two
providers are supported:

``carto``
    Positron itself.  Carto has retired its raster tiles, so the vector tiles are
    rendered once with MapLibre in headless Chromium.  Needs a (free) Carto API key and
    the optional ``playwright`` dependency.
``esri``
    Esri *World Light Gray Canvas*, base plus reference layer, stitched from raster
    tiles.  No key and no browser, and close to Positron in look.

Either way the result is a single image committed to ``data/basemap/``, so every later
run of :mod:`evcharging.plots` is offline and deterministic.  Run
``python scripts/fetch_basemap.py`` to rebuild the cache.

The basemap is rendered at the map window of the paper and at the *zoom* of the paper,
which is what decides how many place labels Positron draws; only the pixel density is
raised, so the committed image is sharp where the published figure is soft.

Each provider carries its own attribution, which :mod:`evcharging.plots` prints in the
corner of every map.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .data import default_data_dir


@dataclass(frozen=True)
class Provider:
    """A basemap service."""

    name: str
    attribution: str
    base: str = ""                           # raster URL template with {z}, {x}, {y}
    reference: Optional[str] = None          # transparent label layer drawn on top
    tile_size: int = 256
    needs_key: bool = False
    vector_style: Optional[str] = None       # rendered with MapLibre instead of stitched


PROVIDERS: Dict[str, Provider] = {
    "esri": Provider(
        name="esri",
        base="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
             "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        reference="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                  "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
        attribution="© Esri © OpenStreetMap contributors",
    ),
    "carto": Provider(
        name="carto",
        vector_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        attribution="© Carto © OpenStreetMap contributors",
        needs_key=True,
    ),
    "osm": Provider(
        name="osm",
        base="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap contributors",
    ),
}

MAPLIBRE_JS = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"
MAPLIBRE_CSS = "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"

DEFAULT_PROVIDER = "carto"

# The map of the paper is a 400 x 260 point Plotly figure at zoom 9.9, centred on the
# mean of the node coordinates, written out at scale 2.  Fitting those three numbers to
# the eleven candidate stations of the published figure places every one of them within
# 0.7 pixels; see ``tests/test_plots.py``.
FIGURE_SIZE = (400.0, 260.0)
MAP_ZOOM = 9.9

# Pixel density of the cached render.  MapLibre picks labels by zoom and draws them at
# this ratio, so a larger number only makes the same map sharper.
DEFAULT_SCALE = 4

# Margin around the map window, so small changes of framing stay covered.
MARGIN = 1.06


@dataclass(frozen=True)
class Window:
    """A Web Mercator view: where it is centred, and how large it is on paper."""

    lat: float
    lon: float
    zoom: float                              # MapLibre zoom (512-point tiles)
    width: float                             # points
    height: float                            # points

    @property
    def points_per_degree(self) -> float:
        return 512.0 * 2.0**self.zoom / 360.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(south, north, west, east)`` in degrees."""
        half_lon = self.width / 2.0 / self.points_per_degree
        half_y = self.height / 2.0 / (512.0 * 2.0**self.zoom)
        y = mercator_y(self.lat)
        return (inverse_mercator_y(y + half_y), inverse_mercator_y(y - half_y),
                self.lon - half_lon, self.lon + half_lon)

    def grown(self, factor: float) -> "Window":
        """The same view, ``factor`` times larger in each direction."""
        return Window(self.lat, self.lon, self.zoom, self.width * factor, self.height * factor)

    def at_zoom(self, zoom: float) -> "Window":
        """The same ground area seen at a different zoom, and so at a different size.

        The zoom is what a vector style uses to decide which place labels to draw and
        how large to set them, so this is the knob that matches the label density of
        the published figure.
        """
        factor = 2.0**(zoom - self.zoom)
        return Window(self.lat, self.lon, zoom, self.width * factor, self.height * factor)


@dataclass(frozen=True)
class Basemap:
    """A rendered basemap and the geographic window it covers."""

    image: np.ndarray                       # (H, W, 3) float in [0, 1]
    south: float
    north: float
    west: float
    east: float
    attribution: str = ""

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """``(left, right, bottom, top)``, as :meth:`matplotlib.axes.Axes.imshow` wants it."""
        return self.west, self.east, self.south, self.north


# ------------------------------------------------------------------------------------
# Web Mercator
# ------------------------------------------------------------------------------------

def mercator_y(lat: float) -> float:
    """Web Mercator ordinate in ``[0, 1]``, north to south."""
    return (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0


def inverse_mercator_y(y: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y))))


def lon_to_tile(lon: float, zoom: int) -> float:
    return (lon + 180.0) / 360.0 * 2**zoom


def lat_to_tile(lat: float, zoom: int) -> float:
    return mercator_y(lat) * 2**zoom


def tile_to_lon(x: float, zoom: int) -> float:
    return x / 2**zoom * 360.0 - 180.0


def tile_to_lat(y: float, zoom: int) -> float:
    return inverse_mercator_y(y / 2**zoom)


def window_for(lat: Sequence[float], lon: Sequence[float], size: Tuple[float, float] = FIGURE_SIZE,
               zoom: float = MAP_ZOOM) -> Window:
    """The map window of the paper.

    Plotly Express centres a ``scatter_mapbox`` on the *mean* of the coordinates, which
    is what leaves the empty sea on the right of the published figure.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    return Window(lat=float(lat.mean()), lon=float(lon.mean()), zoom=zoom,
                  width=size[0], height=size[1])


# ------------------------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------------------------

def cache_paths(data_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    folder = (Path(data_dir) if data_dir is not None else default_data_dir()) / "basemap"
    return folder / "surabaya_canvas.jpg", folder / "surabaya_canvas.json"


def load(data_dir: Optional[Path] = None) -> Optional[Basemap]:
    """Read the committed basemap, or ``None`` when it has not been fetched yet."""
    image_path, meta_path = cache_paths(data_dir)
    if not (image_path.exists() and meta_path.exists()):
        return None
    import matplotlib.pyplot as plt

    with open(meta_path) as fh:
        meta = json.load(fh)
    image = plt.imread(image_path)
    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32) / 255.0
    return Basemap(image=image[..., :3], south=meta["south"], north=meta["north"],
                   west=meta["west"], east=meta["east"], attribution=meta.get("attribution", ""))


def _write(image, window: Window, service: Provider, data_dir: Optional[Path],
           quality: int) -> Basemap:
    south, north, west, east = window.bounds
    image_path, meta_path = cache_paths(data_dir)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, quality=quality, subsampling=0, optimize=True)
    meta = {"south": south, "north": north, "west": west, "east": east,
            "centre_lat": window.lat, "centre_lon": window.lon, "zoom": window.zoom,
            "provider": service.name, "attribution": service.attribution,
            "width_px": image.width, "height_px": image.height}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return Basemap(image=np.asarray(image, dtype=np.float32) / 255.0, south=south, north=north,
                   west=west, east=east, attribution=service.attribution)


def fetch(window: Window, provider: str = DEFAULT_PROVIDER, api_key: Optional[str] = None,
          scale: int = DEFAULT_SCALE, data_dir: Optional[Path] = None, quality: int = 92,
          verbose: bool = True) -> Basemap:
    """Render ``window`` with ``provider`` and write the cache."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; pick one of {sorted(PROVIDERS)}")
    service = PROVIDERS[provider]
    key = api_key or os.environ.get("CARTO_API_KEY")
    if service.needs_key and not key:
        raise ValueError(f"the {provider!r} basemap needs an API key: pass --api-key or set "
                         "CARTO_API_KEY, or use --provider esri")

    if service.vector_style is not None:
        image = _render_vector(service, window, key, scale=scale, verbose=verbose)
    else:
        image = _stitch_raster(service, window, scale=scale, verbose=verbose)
    return _write(image, window, service, data_dir, quality)


# ------------------------------------------------------------------------------------
# MapLibre (vector styles)
# ------------------------------------------------------------------------------------

_MAPLIBRE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="{css}" rel="stylesheet"><script src="{js}"></script>
<style>html,body{{margin:0;padding:0;background:#fff}}
#map{{width:{w}px;height:{h}px}}
.maplibregl-ctrl-bottom-left,.maplibregl-ctrl-bottom-right{{display:none !important}}</style>
</head><body><div id="map"></div><script>
window.rendered = false;
const key = {key!r};
const map = new maplibregl.Map({{
  container: 'map', style: {style!r} + '?api_key=' + key,
  center: [{lon!r}, {lat!r}], zoom: {zoom!r}, attributionControl: false,
  interactive: false, fadeDuration: 0, preserveDrawingBuffer: true,
  transformRequest: (url) => ({{url: url + (url.includes('?') ? '&' : '?') + 'api_key=' + key}}),
}});
map.on('idle', () => {{ window.rendered = true; }});
</script></body></html>"""


def _render_vector(service: Provider, window: Window, key: str, scale: int = DEFAULT_SCALE,
                   timeout: float = 120.0, verbose: bool = True):
    """Screenshot a MapLibre map of ``service.vector_style`` (needs ``playwright``)."""
    from io import BytesIO

    from PIL import Image

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:                                     # pragma: no cover - optional
        raise ImportError(
            "the 'carto' basemap renders vector tiles with MapLibre and needs playwright:\n"
            "    pip install playwright && python -m playwright install chromium\n"
            "or use the key-free fallback: python scripts/fetch_basemap.py --provider esri"
        ) from exc

    width, height = int(round(window.width)), int(round(window.height))
    if verbose:
        print(f"  rendering {width * scale}x{height * scale} px with MapLibre "
              f"at zoom {window.zoom:.4f} ...", flush=True)
    html = _MAPLIBRE_PAGE.format(css=MAPLIBRE_CSS, js=MAPLIBRE_JS, w=width, h=height, key=key,
                                 style=service.vector_style, lon=window.lon, lat=window.lat,
                                 zoom=window.zoom)
    with sync_playwright() as play:
        browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=scale)
        page.set_content(html)
        page.wait_for_function("window.rendered === true", timeout=timeout * 1000)
        page.wait_for_timeout(2000)                                # let the last labels settle
        shot = page.locator("#map").screenshot()
        browser.close()
    image = Image.open(BytesIO(shot)).convert("RGB")
    if image.size != (width * scale, height * scale):              # pragma: no cover - defensive
        image = image.resize((width * scale, height * scale), Image.LANCZOS)
    return image


# ------------------------------------------------------------------------------------
# Raster tiles
# ------------------------------------------------------------------------------------

def _download(url: str, timeout: float = 30.0):
    import urllib.request

    from PIL import Image

    request = urllib.request.Request(url, headers={
        "User-Agent": "evcharging/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return Image.open(response).convert("RGBA")


def _stitch_raster(service: Provider, window: Window, scale: int = DEFAULT_SCALE,
                   verbose: bool = True):
    """Download the tiles that cover ``window`` and crop the mosaic to it."""
    from PIL import Image

    south, north, west, east = window.bounds
    # 256-point tiles: one zoom level above MapLibre's, rounded up to the pixel density.
    zoom = int(min(19, math.ceil(window.zoom + 1 + math.log2(scale))))
    x0, x1 = int(math.floor(lon_to_tile(west, zoom))), int(math.ceil(lon_to_tile(east, zoom)))
    y0, y1 = int(math.floor(lat_to_tile(north, zoom))), int(math.ceil(lat_to_tile(south, zoom)))
    size, total = service.tile_size, (x1 - x0) * (y1 - y0)
    if verbose:
        print(f"  stitching {total} tiles at zoom {zoom} ...", flush=True)

    mosaic = Image.new("RGBA", ((x1 - x0) * size, (y1 - y0) * size), (255, 255, 255, 255))
    done = 0
    for x in range(x0, x1):
        for y in range(y0, y1):
            tile = None
            for template in [service.base] + ([service.reference] if service.reference else []):
                layer = _download(template.format(z=zoom, x=x, y=y))
                if layer.size != (size, size):
                    layer = layer.resize((size, size), Image.LANCZOS)
                tile = layer if tile is None else Image.alpha_composite(tile, layer)
            mosaic.paste(tile, ((x - x0) * size, (y - y0) * size))
            done += 1
            if verbose and done % 25 == 0:
                print(f"  {done}/{total} tiles", flush=True)

    # Crop the mosaic to the window, in the tile grid's own pixel coordinates.
    world = size * 2**zoom
    left = lon_to_tile(west, zoom) * size - x0 * size
    right = lon_to_tile(east, zoom) * size - x0 * size
    top = mercator_y(north) * world - y0 * size
    bottom = mercator_y(south) * world - y0 * size
    crop = mosaic.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    target = (int(round(window.width * scale)), int(round(window.height * scale)))
    return crop.convert("RGB").resize(target, Image.LANCZOS)
