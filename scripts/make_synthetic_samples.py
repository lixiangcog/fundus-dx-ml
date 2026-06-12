"""Generate synthetic fundus-like images for demos and tests.

These are procedurally drawn (orange disc, optic-disc highlight, branching
vessels) — no patient data. They exercise the full upload/predict pipeline;
the model's prediction on them is not clinically meaningful.

Usage:
    python scripts/make_synthetic_samples.py            # writes samples/synthetic/
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "synthetic"
SIZE = 512


def draw_fundus(seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    center = SIZE // 2
    radius = SIZE // 2 - 10
    base = (rng.randint(150, 200), rng.randint(60, 95), rng.randint(20, 45))
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=base,
    )

    # Optic disc: bright yellowish circle off-center
    disc_x = center + rng.randint(-120, -60)
    disc_y = center + rng.randint(-40, 40)
    disc_r = rng.randint(35, 50)
    draw.ellipse(
        [disc_x - disc_r, disc_y - disc_r, disc_x + disc_r, disc_y + disc_r],
        fill=(230, 190, 120),
    )

    # Branching vessels radiating from the optic disc
    vessel_color = (rng.randint(90, 120), rng.randint(20, 40), rng.randint(15, 30))
    for _ in range(rng.randint(6, 10)):
        x, y = disc_x, disc_y
        angle = rng.uniform(0, 6.28)
        for _ in range(rng.randint(15, 30)):
            nx = x + rng.uniform(5, 15) * math.cos(angle)
            ny = y + rng.uniform(5, 15) * math.sin(angle)
            draw.line([x, y, nx, ny], fill=vessel_color, width=rng.randint(2, 4))
            angle += rng.uniform(-0.3, 0.3)
            x, y = nx, ny

    return img.filter(ImageFilter.GaussianBlur(1.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        path = OUT_DIR / f"synthetic_fundus_{i + 1}.png"
        draw_fundus(seed=i).save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
