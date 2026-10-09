"""All verification logic lives here.

Every complaint photo is turned into a SEVERITY (0..1) by blending independent signals:
  - YOLO detector per category (backend/weights/<category>.pt), if you trained one:
        measures how much of the frame the problem covers and how many there are
  - CLIP "ordinal" scoring (no training needed): compares the photo with 4-5 graded
        descriptions (none / minor / moderate / large / severe) and takes the expected level
  - streetlights only: how dark the scene is (needs a photo taken after dark)
If no model can be loaded the app falls back to DEMO mode, which decides from the photo's
FILE NAME so the whole flow can still be shown. The UI says so clearly.

Everything else (hash, distance, priority) is plain rules, no ML.
"""
import math
import os
import re
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

CATEGORIES = ["pothole", "garbage", "streetlight", "waterlogging"]
THRESHOLD = 0.5            # below this we say "uncertain"
HASH_SAME = 5              # perceptual-hash distance that counts as "the same photo"
MODE = os.getenv("FIXCHECK_MODE", "auto").lower()   # auto | demo
CLIP_NAME = "openai/clip-vit-base-patch32"
YOLO_CONF = 0.25
WEIGHTS_DIR = Path(os.getenv("FIXCHECK_WEIGHTS", Path(__file__).parent / "weights"))
HERE = Path(__file__).parent


def _weight_paths(category):
    paths = [WEIGHTS_DIR / f"{category}.pt"]
    if category == "pothole":                       # older setups
        paths += [HERE / "best.pt"]
        if os.getenv("POTHOLE_WEIGHTS"):
            paths.insert(0, Path(os.environ["POTHOLE_WEIGHTS"]))
    return paths


# Per-category settings. All numbers here are STARTING POINTS: run eval_severity.py on
# photos you rate by hand and adjust the anchors until the scores match your judgement.
#  coverage: (fraction of the photo covered by the problem, severity) points, joined by lines
#  levels:   graded descriptions for CLIP: (severity, is_the_problem, [prompts])
#  bad:      YOLO class names that count as the problem (None = every class)
CFG = {
    "pothole": {
        "label": "Pothole",
        "bad": None,
        "coverage": [(0, 0), (0.002, 0.1), (0.02, 0.35), (0.06, 0.6), (0.15, 0.85), (0.3, 1.0)],
        "levels": [
            (0.0, False, ["a smooth road surface", "an intact asphalt road"]),
            (0.25, True, ["a small crack or shallow hole in a road"]),
            (0.55, True, ["a pothole in a road"]),
            (0.8, True, ["a large deep pothole in a road"]),
            (1.0, True, ["a huge dangerous crater in the road"]),
        ],
    },
    "garbage": {
        "label": "Garbage",
        "bad": None,
        "coverage": [(0, 0), (0.01, 0.1), (0.08, 0.4), (0.25, 0.7), (0.5, 1.0)],
        "levels": [
            (0.0, False, ["a clean street", "a clear empty roadside"]),
            (0.2, True, ["a few pieces of litter on a street"]),
            (0.55, True, ["a pile of garbage on a street"]),
            (0.8, True, ["a large heap of uncollected garbage"]),
            (1.0, True, ["overflowing garbage spilling onto the road"]),
        ],
    },
    "waterlogging": {
        "label": "Waterlogging",
        "bad": None,
        "coverage": [(0, 0), (0.05, 0.15), (0.2, 0.4), (0.45, 0.7), (0.7, 1.0)],
        "levels": [
            (0.0, False, ["a dry road", "a dry street"]),
            (0.2, True, ["a small puddle on a road"]),
            (0.5, True, ["a road covered with shallow standing water"]),
            (0.8, True, ["a flooded street with deep water"]),
            (1.0, True, ["a heavily flooded road with vehicles stuck in water"]),
        ],
    },
    "streetlight": {
        "label": "Streetlight",
        "bad": re.compile(r"off|broken|dark|unlit|faulty|fault|dead|out", re.I),
        "coverage": None,
        "levels": [
            (0.0, False, ["a working streetlight that is lit", "a brightly lit street at night"]),
            (0.35, True, ["a street with a dim flickering streetlight"]),
            (0.65, True, ["a streetlight that is switched off on a dim street"]),
            (1.0, True, ["a completely dark street at night with an unlit streetlight"]),
        ],
    },
}
# Streetlights: how dark the photo is (mean brightness 0-255) -> severity. Only used for
# night photos; a daytime photo cannot show whether a lamp is switched off.
NIGHT_BELOW = 110
DARKNESS = [(15, 1.0), (45, 0.85), (80, 0.5), (110, 0.15)]

_state = {"yolo": {}, "clip": None, "proc": None, "notes": []}


def load_models():
    if MODE == "demo":
        _state["notes"].append("FIXCHECK_MODE=demo: models skipped.")
        return
    try:
        from transformers import CLIPModel, CLIPProcessor
        _state["clip"] = CLIPModel.from_pretrained(CLIP_NAME).eval()
        _state["proc"] = CLIPProcessor.from_pretrained(CLIP_NAME)
    except Exception as e:  # missing packages, no internet, etc.
        _state["notes"].append(f"CLIP not loaded ({type(e).__name__}).")
    for cat in CATEGORIES:
        path = next((p for p in _weight_paths(cat) if p.exists()), None)
        if path is None:
            _state["notes"].append(f"No detector for {cat} (backend/weights/{cat}.pt): using CLIP only.")
            continue
        try:
            from ultralytics import YOLO
            _state["yolo"][cat] = YOLO(str(path))
        except Exception as e:
            _state["notes"].append(f"YOLO for {cat} not loaded ({type(e).__name__}).")


def engine_for(category):
    y, c = category in _state["yolo"], _state["clip"] is not None
    return "yolo+clip" if y and c else "yolo" if y else "clip" if c else "demo"


def info():
    engines = {c: engine_for(c) for c in CATEGORIES}
    demo = [c for c, e in engines.items() if e == "demo"]
    mode = "demo" if len(demo) == len(CATEGORIES) else ("live" if not demo else "partial")
    return {"mode": mode, "engines": engines, "notes": _state["notes"]}


# ---------- signal 1: CLIP ordinal scoring ----------
def ordinal_scores(logits, levels):
    """logits: one CLIP similarity per prompt, in the order the prompts appear in `levels`.
    Softmax over the LEVELS (not over yes/no), then:
      prob     = chance the photo shows the problem at any level
      severity = expected severity given that it does
    """
    level_logits, i = [], 0
    for _, _, prompts in levels:
        level_logits.append(float(np.mean(logits[i:i + len(prompts)])))
        i += len(prompts)
    z = np.array(level_logits)
    p = np.exp(z - z.max())
    p /= p.sum()
    prob = float(sum(pi for pi, (_, bad, _) in zip(p, levels) if bad))
    if prob < THRESHOLD:        # the problem is not judged present: nothing to be severe about
        return {"prob": prob, "severity": 0.0}
    sev = sum(pi * s for pi, (s, bad, _) in zip(p, levels) if bad) / prob
    return {"prob": prob, "severity": float(sev)}


def _clip_logits(path, texts):
    import torch
    img = Image.open(path).convert("RGB")
    inputs = _state["proc"](text=texts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        return _state["clip"](**inputs).logits_per_image[0].tolist()


def _clip_signal(path, category):
    levels = CFG[category]["levels"]
    texts = [t for _, _, prompts in levels for t in prompts]
    return ordinal_scores(_clip_logits(path, texts), levels)


# ---------- signal 2: YOLO detections (boxes or segmentation masks) ----------
GRID = (96, 128)


def _union_fraction(masks, boxes, shape):
    """Fraction of the photo covered by the detections, counting overlaps once."""
    h, w = shape
    grid = np.zeros(GRID, bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if masks is not None and i < len(masks):
            m = np.asarray(masks[i]) > 0.5
            ys = np.arange(GRID[0]) * m.shape[0] // GRID[0]
            xs = np.arange(GRID[1]) * m.shape[1] // GRID[1]
            grid |= m[np.ix_(ys, xs)]
        else:
            grid[int(y1 / h * GRID[0]):int(math.ceil(y2 / h * GRID[0])),
                 int(x1 / w * GRID[1]):int(math.ceil(x2 / w * GRID[1]))] = True
    return float(grid.mean())


def _yolo_detect(category, path):
    r = _state["yolo"][category](str(path), conf=YOLO_CONF, verbose=False)[0]
    bad_re = CFG[category]["bad"]
    boxes, confs, good = [], [], 0
    xyxy, cf, cl = r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
    keep = []
    for i, (b, c, k) in enumerate(zip(xyxy, cf, cl)):
        if bad_re is None or bad_re.search(str(r.names[int(k)])):
            keep.append(i)
            boxes.append(b)
            confs.append(c)
        else:
            good += 1
    masks = None
    if getattr(r, "masks", None) is not None:
        allm = r.masks.data.cpu().numpy()
        masks = [allm[i] for i in keep if i < len(allm)]
    cov = _union_fraction(masks, boxes, r.orig_shape) if boxes else 0.0
    return {"count": len(boxes), "good": good, "conf": max(confs) if confs else 0.0, "coverage": cov}


def _coverage_severity(category, coverage, count):
    pts = CFG[category]["coverage"]
    s = float(np.interp(coverage, [p[0] for p in pts], [p[1] for p in pts]))
    return min(1.0, s + 0.05 * min(count - 1, 3))     # several separate spots: a little worse


# ---------- signal 3: darkness (streetlights) ----------
def brightness(path):
    return float(np.asarray(Image.open(path).convert("L").resize((160, 120)), dtype=float).mean())


def _darkness_severity(lum):
    xs, ys = zip(*DARKNESS)
    return float(np.interp(lum, xs, ys))


def band(sev):
    return "minor" if sev < 0.25 else "moderate" if sev < 0.5 else "serious" if sev < 0.75 else "severe"


# ---------- put the signals together ----------
def _demo_check(path, category):
    name = Path(path).name.lower()
    if any(k in name for k in ("fixed", "repaired", "clean", "after")):
        p = 0.08          # looks like a repaired spot
    elif any(k in name for k in ("blurry", "random", "notpothole", "unclear")):
        p = 0.20          # looks like not-the-problem
    else:
        p = 0.90
    sev = [0.3, 0.6, 1.0][int(photo_hash(path), 16) % 3]
    return {"prob": p, "severity": sev, "engine": "demo",
            "measures": {"summary": "Demo mode: severity is a placeholder", "signals": [], "notes": []}}


def verify(path, category):
    """Returns {prob, severity, engine, measures{summary, signals, notes, ...}}."""
    engine = engine_for(category)
    if engine == "demo":
        return _demo_check(path, category)
    label = CFG[category]["label"]
    m = {"signals": [], "notes": []}
    sigs = []                               # (name, severity, weight)
    det = None
    if "yolo" in engine:
        det = _yolo_detect(category, path)
        m.update(count=det["count"], coverage=round(det["coverage"], 3))
        if det["count"] == 0:
            m["summary"] = f"{label}: none detected"
            return {"prob": 0.0, "severity": 0.0, "engine": engine, "measures": m}
        prob = det["conf"]
        if category == "streetlight":
            off_frac = det["count"] / (det["count"] + det["good"])
            sigs.append(("detector", 0.4 + 0.6 * off_frac, 0.5))
            m["lamps_off"], m["lamps_on"] = det["count"], det["good"]
        else:
            sigs.append(("detector", _coverage_severity(category, det["coverage"], det["count"]), 0.7))
    clip = _clip_signal(path, category) if "clip" in engine else None
    if clip is not None:
        if det is None:
            prob = clip["prob"]
        sigs.append(("visual model", clip["severity"], 0.3 if det else 1.0))
    if category == "streetlight":
        lum = brightness(path)
        m["brightness"] = round(lum)
        if lum < NIGHT_BELOW:
            sigs.append(("darkness", _darkness_severity(lum), 0.4))
        else:
            m["notes"].append("Daytime photo: cannot tell if the lamp is off. Retake after dark.")
    total = sum(w for _, _, w in sigs)
    sev = round(min(1.0, max(0.0, sum(s * w for _, s, w in sigs) / total)), 2) if total else 0.0
    m["signals"] = [{"name": n, "severity": round(s, 2)} for n, s, _ in sigs]
    if det and category == "streetlight":
        m["summary"] = f"{label}: {det['count']} lamp(s) off of {det['count'] + det['good']} seen, {band(sev)}"
    elif det:
        m["summary"] = f"{label}: {det['count']} found, {det['coverage']:.0%} of the photo, {band(sev)}"
    else:
        m["summary"] = f"{label}: {band(sev)} (visual model)"
    return {"prob": float(prob), "severity": sev, "engine": engine, "measures": m}


# ---------- rule-based checks ----------
def photo_hash(path):
    return str(imagehash.phash(Image.open(path).convert("RGB")))


def hash_distance(a, b):
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name or "photo.jpg")[-60:]


SLA_DAYS = 7


def priority(severity, near_school, dups, age_days, rain, ward_factor=1.0):
    """Transparent score 0..100 with the reasons, so officers can see why."""
    score = (40 * severity + 20 * (1 if near_school else 0) + 15 * min(dups, 5) / 5
             + 15 * min(age_days / SLA_DAYS, 1) + 10 * (1 if rain else 0))
    reasons = ["High severity" if severity >= 0.8 else "Medium severity" if severity >= 0.5 else "Low severity"]
    if near_school:
        reasons.append("Near a school or hospital")
    if dups:
        reasons.append(f"{dups} duplicate report{'s' if dups > 1 else ''} merged")
    reasons.append(f"Open {age_days:.1f} days (SLA {SLA_DAYS})")
    if rain:
        reasons.append("Rain expected")
    if ward_factor > 1.05:
        reasons.append("Quiet-ward adjustment (+)")
    elif ward_factor < 0.95:
        reasons.append("Busy-ward adjustment (-)")
    return round(min(100.0, score * ward_factor), 1), reasons
