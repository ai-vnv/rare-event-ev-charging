"""Build ``results/interactive.html`` from the travel-time sweep.

The page is self-contained: the sweep results and a downscaled copy of the cached
basemap are embedded, so it opens offline in any browser.

    python scripts/sweep_travel_time.py
    python scripts/build_interactive.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "interactive_template.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", type=Path, default=Path("results/travel_time_sweep.json"))
    ap.add_argument("--basemap", type=Path, default=Path("data/basemap/surabaya_canvas"))
    ap.add_argument("--out", type=Path, default=Path("results/interactive.html"))
    ap.add_argument("--width", type=int, default=1696, help="basemap width embedded, px")
    ap.add_argument("--bare", action="store_true", help="omit the document skeleton (for hosts that add their own)")
    args = ap.parse_args()

    sweep = json.loads(args.sweep.read_text())
    window = json.loads(args.basemap.with_suffix(".json").read_text())

    img = Image.open(args.basemap.with_suffix(".jpg"))
    h = round(img.height * args.width / img.width)
    buf = io.BytesIO()
    img.resize((args.width, h), Image.LANCZOS).save(buf, "JPEG", quality=78, optimize=True)
    basemap_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    html = TEMPLATE.read_text()
    html = html.replace("__SWEEP_JSON__", json.dumps(sweep, separators=(",", ":")))
    html = html.replace("__WINDOW_JSON__", json.dumps(window))
    html = html.replace("__BASEMAP_URI__", basemap_uri)
    if not args.bare:
        html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"></head><body>\n"
                + html + "\n</body></html>\n")
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
