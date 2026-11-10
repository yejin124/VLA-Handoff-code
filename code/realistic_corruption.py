#!/usr/bin/env python3
"""Realistic visual degradations for the closed-loop LIBERO runs (replaces the centered mask).

Three families, each parameterised by a severity s in [0, 1] and a per-episode seed, so that
the degradation is a fixed physical situation that persists for the whole episode rather than
a per-frame random pattern:

  occluder  an out-of-focus object or hand near the lens: a blurred patch of a real LIBERO
            frame (the donor bank) composited through an irregular, feathered silhouette at a
            seeded location. Severity sets the covered area (about 0.35*s of the frame).
  smudge    dirt or a fingerprint on the lens: several soft blobs in which the image is blurred
            and hazed (contrast pulled towards a bright grey). Severity sets blob area, blur
            radius and haze strength.
  lighting  a dim, warm room: brightness and contrast drop with a slight colour-temperature
            shift on the whole frame. Severity sets the amount of dimming; s may exceed 1 (up
            to 1.5, a nearly dark room), added 2026-10-08 after the s=1 pilot broke no policy.

Camera rule: by default occluder and smudge touch the agent-view camera only and lighting touches both;
the LIBERO cohorts of the article pass `apply_to_wrist=True` after the pilot, so both cameras are degraded
(Fig. 3 of the article).

Everything is deterministic in (family, severity, seed). Run the module to write an example
strip: python code/realistic_corruption.py --out <png> (Fig. 3 of the article is drawn by code/plot_perturbation_examples.py)
"""
import argparse
import os
import glob
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FAMILIES = ("occluder", "smudge", "lighting")
WRIST_FAMILIES = ("lighting",)          # families that also touch the wrist camera by default
DONOR_BANK = Path(os.environ.get("VLA_DONOR_BANK", Path(__file__).resolve().parent.parent / "assets" / "occluder"))


def _blob_mask(height, width, area_fraction, seed, vertices=18, feather_scale=0.018,
               aspect_range=(0.72, 1.28), at_edge=False):
    """Soft-edged irregular polygon covering about `area_fraction` of the frame.

    With at_edge the blob is anchored on a seeded frame border (part of it lies outside), the
    way an object close to the lens enters the view; otherwise it sits inside the frame."""
    if area_fraction <= 0:
        return np.zeros((height, width), dtype=np.float32)
    rng = np.random.default_rng(seed)
    base_radius = math.sqrt(area_fraction * height * width / math.pi)
    if at_edge:
        base_radius *= 1.35                       # roughly 45% of the blob is clipped by the border
        side = int(rng.integers(4))               # 0 left, 1 right, 2 top, 3 bottom
        along = float(rng.uniform(0.15, 0.85))
        cx, cy = {0: (0.0, along * height), 1: (float(width), along * height),
                  2: (along * width, 0.0), 3: (along * width, float(height))}[side]
        cx, cy = (cx + (base_radius * 0.15 if side == 0 else -base_radius * 0.15 if side == 1 else 0),
                  cy + (base_radius * 0.15 if side == 2 else -base_radius * 0.15 if side == 3 else 0))
    else:
        margin_x = min(base_radius * 1.25, width * 0.38)
        margin_y = min(base_radius * 1.25, height * 0.38)
        cx = float(rng.uniform(margin_x, width - margin_x))
        cy = float(rng.uniform(margin_y, height - margin_y))
    angles = np.linspace(0, 2 * math.pi, vertices, endpoint=False)
    radial = rng.uniform(0.72, 1.22, size=vertices)
    aspect = float(rng.uniform(*aspect_range))
    points = [(cx + base_radius * aspect * r * math.cos(a), cy + base_radius / aspect * r * math.sin(a))
              for r, a in zip(radial, angles)]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255 if at_edge else 242)   # a solid object is opaque at its core
    feather = max(2.0, min(height, width) * feather_scale)
    return np.asarray(canvas.filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255.0


def load_donor_bank(path=DONOR_BANK):
    files = sorted(glob.glob(str(Path(path) / "*.png")))
    if not files:
        raise FileNotFoundError(f"donor bank is empty: {path}")
    return [np.asarray(Image.open(f).convert("RGB")) for f in files]


def occluder(image, severity, seed, donors):
    """Blurred real-frame texture through an irregular silhouette at a seeded location."""
    if severity <= 0:
        return image.copy()
    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)
    donor = donors[int(rng.integers(len(donors)))]
    # A real object a few centimetres from the lens is far out of focus, under-exposed and
    # low in saturation: heavy blur, darkening and desaturation of the donor texture.
    texture = Image.fromarray(donor).resize((width, height), Image.BILINEAR)
    texture = texture.filter(ImageFilter.GaussianBlur(max(6, min(height, width) // 14)))
    texture = np.asarray(texture, dtype=np.float32)
    grey = texture.mean(axis=2, keepdims=True)
    texture = grey + (texture - grey) * 0.35
    texture = np.clip(texture * rng.uniform(0.25, 0.50) + rng.uniform(-8, 8), 0, 255)
    alpha = _blob_mask(height, width, 0.35 * severity, seed, feather_scale=0.03, at_edge=True)[..., None]
    out = image.astype(np.float32) * (1 - alpha) + texture * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def smudge(image, severity, seed):
    """Lens dirt: blurred, hazy blobs; count, area, blur and haze all grow with severity."""
    if severity <= 0:
        return image.copy()
    height, width = image.shape[:2]
    rng = np.random.default_rng(seed)
    count = 2 + int(round(4 * severity))
    alpha = np.zeros((height, width), dtype=np.float32)
    for k in range(count):
        area = rng.uniform(0.03, 0.10) * (0.4 + 0.6 * severity)
        alpha = np.maximum(alpha, _blob_mask(height, width, area, seed * 7919 + k, vertices=14,
                                             feather_scale=0.05, aspect_range=(1.6, 3.2)))
    blur_radius = 2.0 + 7.0 * severity
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.float32)
    # Grease scatters light: contrast collapses towards a light grey and highlights bloom.
    haze = 0.20 + 0.45 * severity
    hazed = blurred * (1 - haze) + 190.0 * haze
    bloom = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(10)), dtype=np.float32)
    hazed = np.clip(hazed + 0.25 * severity * np.maximum(bloom - 150, 0), 0, 255)
    a = alpha[..., None]
    out = image.astype(np.float32) * (1 - a) + hazed * a
    return np.clip(out, 0, 255).astype(np.uint8)


def lighting(image, severity, seed):
    """Dim, warm light: brightness and contrast fall, colour temperature shifts slightly."""
    if severity <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32)
    mean = x.mean(axis=(0, 1), keepdims=True)
    s1, extra = min(severity, 1.0), max(severity - 1.0, 0.0)   # s > 1 (up to 1.5): a nearly dark room
    contrast = 1.0 - 0.35 * s1 - 0.50 * extra                   # 0.65 at s=1, 0.40 at s=1.5
    x = mean + (x - mean) * contrast
    tint = np.array([1.0 + 0.10 * s1, 1.0, 1.0 - 0.14 * s1], dtype=np.float32)
    x = x * tint * (1.0 - 0.70 * s1 - 0.44 * extra)             # brightness 0.30 at s=1, 0.08 at s=1.5
    x += rng.normal(0, 2.0 * severity, x.shape)                 # sensor noise grows in low light
    return np.clip(x, 0, 255).astype(np.uint8)


def corrupt(image, family, severity, seed, camera=0, donors=None, apply_to_wrist=False):
    """Degrade one camera image. camera 0 is the agent view, 1 the wrist view."""
    if family == "clean" or severity <= 0:
        return image.copy()
    if camera == 1 and family not in WRIST_FAMILIES and not apply_to_wrist:
        return image.copy()
    if family == "occluder":
        return occluder(image, severity, seed, donors or load_donor_bank())
    if family == "smudge":
        return smudge(image, severity, seed)
    if family == "lighting":
        return lighting(image, severity, seed)
    raise ValueError(f"unknown degradation family {family!r}")


def example_strip(images, out, seed=20261008):
    """Example strip of the camera situations (not the article figure; see plot_perturbation_examples.py): rows are the camera situations as used in the study (occluder on both cameras, dim light on
    both); columns are severities, with the rightmost column the severity the study uses (occluder 1.0,
    dim light 1.25). Each cell is labelled with its severity."""
    donors = load_donor_bank()
    agent, wrist = images
    rows = [("occluder, agent view", agent, 0, (0, 0.5, 0.75, 1.0)), ("occluder, wrist view", wrist, 1, (0, 0.5, 0.75, 1.0)),
            ("dim light, agent view", agent, 0, (0, 0.5, 1.0, 1.25)), ("dim light, wrist view", wrist, 1, (0, 0.5, 1.0, 1.25))]
    h, w = agent.shape[:2]
    pad, label_w = 6, 230
    try:
        font = ImageFont.truetype(os.environ.get("EXAMPLE_FONT", "DejaVuSans.ttf"), 26)   # ~8 pt at column width
    except OSError:
        font = ImageFont.load_default()
    ncol = max(len(r[3]) for r in rows)
    canvas = Image.new("RGB", (ncol * (w + pad) + label_w, len(rows) * (h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, img, cam, sevs) in enumerate(rows):
        fam = "occluder" if name.startswith("occluder") else "lighting"
        fam_label, view = name.split(", ")
        draw.text((6, i * (h + pad) + h // 2 - 30), fam_label, fill="black", font=font)
        draw.text((6, i * (h + pad) + h // 2 + 2), view, fill="black", font=font)
        for j, s in enumerate(sevs):
            y = corrupt(img, fam, s, seed + 1000 * i, camera=cam, donors=donors, apply_to_wrist=True)
            x0, y0 = label_w + j * (w + pad), i * (h + pad)
            canvas.paste(Image.fromarray(y), (x0, y0))
            tag = f"s = {s:g}" + (" (used)" if j == len(sevs) - 1 else "")
            tw = draw.textlength(tag, font=font)
            draw.rectangle([x0, y0, x0 + tw + 12, y0 + 32], fill="white")
            draw.text((x0 + 6, y0 + 2), tag, fill="black", font=font)
    canvas.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None, help="npz with agent_image and wrist_image (default: first clean main-run query)")
    ap.add_argument("--out", required=True, help="output PNG")
    args = ap.parse_args()
    frame = args.frame or sorted(glob.glob("data/icassp2027/realistic/spatial_main/pi05/shard*/t3_*_clean/q0000.npz"))[0]
    z = np.load(frame)
    print("wrote", example_strip((z["agent_image"], z["wrist_image"]), args.out))
