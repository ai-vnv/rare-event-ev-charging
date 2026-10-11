"""Download the basemap that the maps of the paper are drawn on.

    python scripts/fetch_basemap.py --api-key $CARTO_API_KEY   # Carto Positron, as in the paper
    python scripts/fetch_basemap.py --provider esri            # no key, no browser

Writes ``data/basemap/surabaya_canvas.jpg`` and a small JSON file with the window the
image covers and the attribution to print on the figures.  Both are committed, so this
script only has to run when the provider or the framing changes -- every other run of
the project works offline.

The Positron tiles are vector tiles, so they are rendered once with MapLibre in
headless Chromium::

    pip install playwright && python -m playwright install chromium

Never commit the API key; pass it on the command line or in ``CARTO_API_KEY``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evcharging import basemap, load_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=basemap.DEFAULT_PROVIDER, choices=sorted(basemap.PROVIDERS))
    parser.add_argument("--api-key", default=None, help="Carto API key (or set CARTO_API_KEY)")
    parser.add_argument("--scale", type=int, default=basemap.DEFAULT_SCALE,
                        help="pixel density of the cached render (default 4)")
    parser.add_argument("--margin", type=float, default=basemap.MARGIN,
                        help="how much wider than the map window to render (default 1.06)")
    args = parser.parse_args()

    case = load_case()
    lat = np.r_[case.demand.lat.values, case.stations.lat.values]
    lon = np.r_[case.demand.lon.values, case.stations.lon.values]
    window = basemap.window_for(lat, lon).grown(args.margin)
    south, north, west, east = window.bounds

    print(f"fetching the {args.provider} basemap at zoom {window.zoom:.4f}, "
          f"centred on ({window.lat:.5f}, {window.lon:.5f})")
    print(f"  window: {south:.5f}..{north:.5f} lat, {west:.5f}..{east:.5f} lon")
    result = basemap.fetch(window, provider=args.provider, api_key=args.api_key, scale=args.scale)

    image_path, meta_path = basemap.cache_paths()
    print(f"wrote {image_path.relative_to(ROOT)} ({image_path.stat().st_size / 1e6:.1f} MB, "
          f"{result.image.shape[1]}x{result.image.shape[0]} px)")
    print(f"wrote {meta_path.relative_to(ROOT)}   attribution: {result.attribution}")


if __name__ == "__main__":
    main()
