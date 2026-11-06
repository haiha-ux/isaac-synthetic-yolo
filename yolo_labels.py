"""Pixel boxes <-> YOLO normalized labels. No Isaac dependency, so it can be unit tested anywhere."""

CLASSES = ["cracker_box", "sugar_box", "soup_can", "mustard_bottle", "bowl"]


def box_to_yolo(x_min, y_min, x_max, y_max, w, h):
    """Pixel corner box -> (cx, cy, bw, bh) normalized to [0, 1]. Clips to the image."""
    x_min, x_max = max(0.0, min(x_min, w)), max(0.0, min(x_max, w))
    y_min, y_max = max(0.0, min(y_min, h)), max(0.0, min(y_max, h))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("empty box after clipping")
    return ((x_min + x_max) / 2 / w, (y_min + y_max) / 2 / h, (x_max - x_min) / w, (y_max - y_min) / h)


def yolo_to_box(cx, cy, bw, bh, w, h):
    """Inverse of box_to_yolo: normalized -> pixel corners (x_min, y_min, x_max, y_max)."""
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def label_line(cls, box, w, h):
    return "%d %.6f %.6f %.6f %.6f" % ((cls,) + box_to_yolo(*box, w, h))
