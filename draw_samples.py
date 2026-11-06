"""Draw YOLO labels on images and tile them into a contact sheet.
usage: python draw_samples.py <dataset_dir> <out.jpg> [n=6]"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from yolo_labels import CLASSES, yolo_to_box

ds, out = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
tiles = []
for p in sorted((ds / "images").glob("*.jpg"))[:n]:
    im = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(im)
    for line in (ds / "labels" / (p.stem + ".txt")).read_text().split("\n"):
        if line:
            c, *v = line.split()
            box = yolo_to_box(*map(float, v), *im.size)
            d.rectangle(box, outline=(255, 40, 40), width=2)
            d.text((box[0] + 3, box[1] + 2), CLASSES[int(c)], fill=(255, 255, 0))
    tiles.append(im)
cols = 2 if n <= 4 else 3
w, h = tiles[0].size
sheet = Image.new("RGB", (cols * w, ((len(tiles) + cols - 1) // cols) * h))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * w, (i // cols) * h))
sheet.save(out, quality=88)
