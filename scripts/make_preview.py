"""Render the README previews of ``results/interactive.html``.

Drives the page in headless Chromium with "Show assignments" ticked, moves the slider
over the maximum travel time and writes ``results/figures/interactive_preview.gif``
(and a static ``.png`` at 35 min).  Needs Playwright, which the package itself does not:

    uv run --with playwright --with pillow python scripts/make_preview.py
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

# 10 -> 75 -> 35 min, with extra frames at the infeasible edge (28/29) and where the
# Robust design drops Tandes (40/41).
UP = [10, 15, 20, 25, 28, 29, 30, 35, 40, 41, 45, 50, 55, 60, 65, 70, 75]
DOWN = [70, 65, 60, 55, 50, 45, 41, 40, 35]
HOLD = {10: 3, 29: 2, 35: 3, 41: 2, 75: 3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page", type=Path, default=Path("results/interactive.html"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/interactive_preview"))
    ap.add_argument("--width", type=int, default=800, help="GIF width, px")
    ap.add_argument("--ms", type=int, default=260, help="frame duration, ms")
    ap.add_argument("--chromium", help="path to a Chromium binary, if Playwright's own is not installed")
    args = ap.parse_args()

    frames: list[Image.Image] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(viewport={"width": 1000, "height": 900}, device_scale_factor=2)
        page.goto(args.page.resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.check("#showFlows")
        # controls strip down to the bottom of the map, full page width
        top = page.eval_on_selector(".controls", "el => el.getBoundingClientRect().top + scrollY")
        bottom = page.eval_on_selector(".map", "el => el.getBoundingClientRect().bottom + scrollY")
        clip = {"x": 0, "y": top - 16, "width": 1000, "height": bottom - top + 32}

        def shot(d: int) -> Image.Image:
            page.eval_on_selector(
                "#dmax", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }", d)
            page.wait_for_timeout(120)
            return Image.open(io.BytesIO(page.screenshot(clip=clip))).convert("RGB")

        for d in UP + DOWN:
            im = shot(d)
            frames += [im] * HOLD.get(d, 1)
        still = shot(35)
        browser.close()

    w0, h0 = still.size
    still.resize((1000, round(h0 * 1000 / w0)), Image.LANCZOS).save(args.out.with_suffix(".png"), optimize=True)
    h = round(h0 * args.width / w0)
    small = [f.resize((args.width, h), Image.LANCZOS) for f in frames]
    pal = small[len(small) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in small]
    gif[0].save(args.out.with_suffix(".gif"), save_all=True, append_images=gif[1:],
                duration=args.ms, loop=0, optimize=True)
    print(f"wrote {args.out}.gif ({len(gif)} frames) and {args.out}.png")


if __name__ == "__main__":
    main()
