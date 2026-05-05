"""
inference.py  —  IMPROVED
===========================
Drop-in replacement for abc/inference.py.
Changes vs original:
  - extract_features: 128 → 256 dims (Gabor + Hough + scratch mask + fine LBP)
  - predict() handles multi-class damage (not binary) — scratch sub-type preserved
  - severity mapping updated to include scratch-specific logic
"""

import os, json, io, base64
import numpy as np
import cv2
import joblib
from PIL import Image, ImageDraw

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── confidence thresholds ──────────────────────────────────────────────────────
# Must beat these to report damage / confident part — prevents false positives
DAMAGE_CONF_THRESHOLD = 0.65   # below → report no_damage
PART_CONF_THRESHOLD   = 0.35   # below → part label shown as "uncertain"

# ── marketplace links (unchanged) ─────────────────────────────────────────────
PART_LINKS = {
    "bumper":     [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+bumper+replacement"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+bumper"},
                   {"site": "CarDekho Parts", "url": "https://parts.cardekho.com/bumper"},
                   {"site": "Moglix",         "url": "https://www.moglix.com/search?q=car+bumper"}],
    "door":       [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+door+panel"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+door+panel"},
                   {"site": "IndiaMart",      "url": "https://www.indiamart.com/search.mp?ss=car+door+panel"},
                   {"site": "Spare20",        "url": "https://www.spare20.com/car-door"}],
    "hood":       [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+hood+bonnet"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+bonnet"},
                   {"site": "IndiaMart",      "url": "https://www.indiamart.com/search.mp?ss=car+bonnet"},
                   {"site": "Moglix",         "url": "https://www.moglix.com/search?q=car+bonnet"}],
    "headlight":  [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+headlight+assembly"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+headlight"},
                   {"site": "Spare20",        "url": "https://www.spare20.com/headlights"}],
    "windshield": [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+windshield+glass"},
                   {"site": "GlassFit",       "url": "https://www.glassfit.in"},
                   {"site": "IndiaMart",      "url": "https://www.indiamart.com/search.mp?ss=windshield+glass"}],
    "tyre":       [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+tyre"},
                   {"site": "TyrePlex",       "url": "https://www.tyreplex.com"},
                   {"site": "Apollo Tyres",   "url": "https://www.apollotyres.com/in-en/buy-tyres/"}],
    "fender":     [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+fender"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+fender"},
                   {"site": "IndiaMart",      "url": "https://www.indiamart.com/search.mp?ss=car+fender"}],
    "mirror":     [{"site": "Amazon India",   "url": "https://www.amazon.in/s?k=car+side+mirror"},
                   {"site": "Flipkart",       "url": "https://www.flipkart.com/search?q=car+side+mirror"},
                   {"site": "Spare20",        "url": "https://www.spare20.com/mirrors"}],
}

REPAIR_TIPS = {
    "dent":      "Panel beating or PDR (Paintless Dent Repair) recommended.",
    "scratch":   "Surface sanding + spot repaint. Deep scratches need primer coat first.",
    "crack":     "Structural filler + repaint. Metal parts may need welding.",
    "shatter":   "Full part replacement required. Do not drive with shattered glass.",
    "no_damage": "No significant damage detected. Routine maintenance only.",
}

SEVERITY_TIPS = {
    "minor":    "Can be addressed at a local garage. No urgent repair needed.",
    "moderate": "Repair within 2-4 weeks recommended to prevent further damage.",
    "severe":   "Repair urgently — structural integrity may be compromised.",
    "critical": "Do not drive. Immediate professional assessment required.",
    "no_damage": "Vehicle appears undamaged.",
}

# ── model cache ───────────────────────────────────────────────────────────────
_part_model   = None
_damage_model = None
_cost_model   = None
_meta         = None


def load_models():
    global _part_model, _damage_model, _cost_model, _meta
    if _part_model is None:
        _part_model   = joblib.load(os.path.join(MODELS_DIR, "part_classifier.pkl"))
        _damage_model = joblib.load(os.path.join(MODELS_DIR, "damage_classifier.pkl"))
        _cost_model   = joblib.load(os.path.join(MODELS_DIR, "cost_regressor.pkl"))
        with open(os.path.join(MODELS_DIR, "meta.json")) as f:
            _meta = json.load(f)
    return _part_model, _damage_model, _cost_model, _meta


# ── 256-dim feature extractor ─────────────────────────────────────────────────

def _gabor_features(gray: np.ndarray) -> list:
    feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel   = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                      lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
        feats.append(float(filtered.mean()) / 255.0)
        feats.append(float(filtered.std())  / 255.0)
    return feats


def _hough_line_density(gray: np.ndarray) -> list:
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=20, minLineLength=10, maxLineGap=5)
    if lines is None:
        return [0.0, 0.0, 0.0]
    lengths, angles = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lengths.append(np.hypot(x2-x1, y2-y1))
        angles.append(np.arctan2(y2-y1, x2-x1+1e-6))
    return [
        min(1.0, len(lines) / 80.0),
        min(1.0, float(np.mean(lengths)) / gray.shape[1]),
        min(1.0, float(np.std(angles)) / np.pi),
    ]


def _scratch_mask_ratio(gray: np.ndarray) -> float:
    k_h     = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    k_v     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    lines_h = cv2.subtract(gray, cv2.erode(gray, k_h))
    lines_v = cv2.subtract(gray, cv2.erode(gray, k_v))
    combined = cv2.add(lines_h, lines_v)
    _, mask  = cv2.threshold(combined, 15, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(mask)) / float(gray.size)


def _lbp_variance_grid(gray: np.ndarray, grid: int = 8) -> list:
    h, w = gray.shape
    bh, bw = h // grid, w // grid
    feats = []
    for r in range(grid):
        for c in range(grid):
            feats.append(float(gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw].std()) / 128.0)
    return feats


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """256-dim scratch-aware feature vector. Must match train_models.py."""
    img  = cv2.resize(img_bgr, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    for ch in range(3):
        hist, _ = np.histogram(img[:, :, ch], bins=16, range=(0, 256))
        feats.extend(hist / (hist.sum() + 1e-8))

    edges = cv2.Canny(gray, 50, 150)
    feats.append(edges.mean() / 255.0)

    bs = 32
    for r in range(0, 128, bs):
        for c in range(0, 128, bs):
            feats.append(float(gray[r:r+bs, c:c+bs].std()) / 128.0)

    feats.extend(_gabor_features(gray))
    feats.extend(_hough_line_density(gray))
    feats.append(_scratch_mask_ratio(gray))
    feats.extend(_lbp_variance_grid(gray, grid=8))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gm = np.sqrt(gx**2 + gy**2)
    feats.extend([gm.mean() / 255.0, gm.std() / 255.0])
    feats.append(float((gray < 60).mean()))

    feats = feats[:256]
    while len(feats) < 256:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── severity mapping (scratch-aware) ─────────────────────────────────────────

def confidence_to_severity(damage: str, conf: float) -> str:
    """Scratch uses lower thresholds — surface scratches are minor by default."""
    if damage == "scratch":
        if conf >= 0.85: return "severe"
        if conf >= 0.65: return "moderate"
        return "minor"
    # All other damages
    if conf >= 0.88: return "critical"
    if conf >= 0.75: return "severe"
    if conf >= 0.60: return "moderate"
    return "minor"


# ── cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(part, damage, severity, segment="sedan", age=3, panels=1):
    pm = _cost_model["le_part"]
    dm = _cost_model["le_damage"]
    sm = _cost_model["le_segment"]
    vm = _cost_model["le_severity"]
    try:
        row = np.array([[
            pm.transform([part])[0],
            dm.transform([damage])[0],
            sm.transform([segment])[0],
            vm.transform([severity])[0],
            age, panels
        ]], dtype=float)
    except ValueError:
        row = np.array([[0, 0, 1, 1, age, panels]], dtype=float)

    point = float(_cost_model["model"].predict(row)[0])
    lo  = int(round(point * 0.78 / 500) * 500)
    hi  = int(round(point * 1.28 / 500) * 500)
    mid = int(round(point / 500) * 500)
    return {"low": lo, "mid": mid, "high": hi,
            "formatted": f"Rs.{lo:,} - Rs.{hi:,}",
            "midformatted": f"Rs.{mid:,}"}


# ── annotated image ───────────────────────────────────────────────────────────

def annotate_image(pil_img: Image.Image, part: str, damage: str,
                   severity: str, conf: float) -> str:
    img = pil_img.copy().convert("RGB")
    img = img.resize((480, 360), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    colour = {"minor": "#27ae60", "moderate": "#f39c12",
              "severe": "#e74c3c", "critical": "#8e44ad",
              "no_damage": "#2980b9"}.get(severity, "#e74c3c")
    for t in range(4):
        draw.rectangle([t, t, img.width-1-t, img.height-1-t], outline=colour)
    label = f" {part.upper()} | {damage.upper()} | {severity.upper()} ({conf*100:.0f}%) "
    bx1 = min(len(label)*8+12, img.width-8)
    draw.rectangle([8, 8, bx1, 36], fill=colour)
    draw.text((14, 12), label, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── main predict ──────────────────────────────────────────────────────────────

def predict(pil_img: Image.Image, segment="sedan", age=3, panels=1):
    load_models()
    bgr  = pil_to_bgr(pil_img)
    feat = extract_features(bgr)   # shape (1, 256)

    # ── Part prediction ────────────────────────────────────────────────────
    part_probs = _part_model["model"].predict_proba(feat)[0]
    part_idx   = int(np.argmax(part_probs))
    part       = _part_model["classes"][part_idx]
    part_conf  = float(part_probs[part_idx])

    # Low part confidence → label as uncertain (don't mislead user)
    if part_conf < PART_CONF_THRESHOLD:
        part = "uncertain"

    top3_idx   = np.argsort(part_probs)[-3:][::-1]
    top3_parts = [{"part": _part_model["classes"][i],
                   "prob": round(float(part_probs[i])*100, 1)} for i in top3_idx]

    # ── Damage prediction (multi-class) ───────────────────────────────────
    dmg_probs  = _damage_model["model"].predict_proba(feat)[0]
    dmg_idx    = int(np.argmax(dmg_probs))
    raw_damage = _damage_model["classes"][dmg_idx]
    dmg_conf   = float(dmg_probs[dmg_idx])

    # ── THRESHOLD GATE ────────────────────────────────────────────────────
    # If model isn't confident enough, or top class is no_damage → clean car
    no_damage_idx = list(_damage_model["classes"]).index("no_damage") \
                    if "no_damage" in _damage_model["classes"] else -1
    no_damage_prob = float(dmg_probs[no_damage_idx]) if no_damage_idx >= 0 else 0.0

    if raw_damage == "no_damage" or dmg_conf < DAMAGE_CONF_THRESHOLD or no_damage_prob > 0.45:
        # Treat as undamaged — prevents false alarms on clean cars
        damage     = "no_damage"
        dmg_conf   = no_damage_prob if raw_damage == "no_damage" else (1.0 - dmg_conf)
        is_damaged = False
        severity   = "no_damage"
        cost       = None
    else:
        damage     = raw_damage
        is_damaged = True
        severity   = confidence_to_severity(damage, dmg_conf)
        cost       = estimate_cost(part if part != "uncertain" else "door",
                                   damage, severity, segment, age, panels)

    # All damage scores for UI — cast keys to plain str
    all_dmg_scores = {str(_damage_model["classes"][i]): round(float(dmg_probs[i])*100, 1)
                      for i in range(len(_damage_model["classes"]))}

    ann_b64 = annotate_image(pil_img, part, damage, severity,
                             dmg_conf if is_damaged else part_conf)

    return {
        "part":           part,
        "part_conf":      round(part_conf * 100, 1),
        "top3_parts":     top3_parts,
        "damage":         damage,
        "damage_conf":    round(dmg_conf * 100, 1),
        "all_dmg_scores": all_dmg_scores,
        "is_damaged":     is_damaged,
        "severity":       severity,
        "cost":           cost,
        "repair_tip":     REPAIR_TIPS.get(damage, ""),
        "severity_tip":   SEVERITY_TIPS.get(severity, ""),
        "part_links":     PART_LINKS.get(part, []),
        "annotated_img":  ann_b64,
    }