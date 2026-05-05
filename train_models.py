"""
train_models.py  —  IMPROVED
==============================
Key upgrades over original:
  1. extract_features: 128 → 256 dims
       + Gabor filters (4 orientations) — directional scratch detection
       + Hough line density            — scratch line count
       + LBP proxy (variance grid)     — fine-texture irregularity
       + Scratch-line mask ratio       — morphological thin-line detector
  2. make_synthetic_image:
       + Realistic multi-width curved scratch bundles
       + Metallic shimmer base texture
       + Depth variation (surface / metal-exposed scratches)
       + Scratch clusters at random angles
  3. Damage classifier now multi-class (dent/scratch/crack/shatter/no_damage)
     instead of binary — retains scratch sub-type through prediction
  4. n_per_class bumped to 350, RF n_estimators=400
"""

import os, json, warnings
import numpy as np
import joblib
import cv2
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

PARTS   = ["bumper", "door", "hood", "headlight", "windshield", "tyre", "fender", "mirror"]
DAMAGES = ["dent", "scratch", "crack", "shatter", "no_damage"]

PART_COLOUR = {
    "bumper":     ([180, 180, 175], [30, 30, 30]),
    "door":       ([160, 165, 160], [35, 35, 35]),
    "hood":       ([150, 155, 150], [40, 40, 40]),
    "headlight":  ([220, 215, 200], [20, 20, 20]),
    "windshield": ([200, 210, 215], [25, 25, 25]),
    "tyre":       ([ 50,  50,  50], [20, 20, 20]),
    "fender":     ([170, 170, 165], [35, 35, 35]),
    "mirror":     ([190, 195, 192], [30, 30, 30]),
}

DAMAGE_TEXTURE = {
    "dent":      dict(edge_boost=1.2, noise=0.05, dark_patch=True,  crack_lines=0,  scratch_lines=0),
    "scratch":   dict(edge_boost=1.8, noise=0.04, dark_patch=False, crack_lines=0,  scratch_lines=1),
    "crack":     dict(edge_boost=2.0, noise=0.06, dark_patch=False, crack_lines=5,  scratch_lines=0),
    "shatter":   dict(edge_boost=2.5, noise=0.10, dark_patch=True,  crack_lines=10, scratch_lines=0),
    "no_damage": dict(edge_boost=0.6, noise=0.02, dark_patch=False, crack_lines=0,  scratch_lines=0),
}

# Confidence thresholds — below these, fall back to no_damage
# Prevents false positives on clean cars
DAMAGE_CONF_THRESHOLD = 0.65   # must beat 65% to call damage
PART_CONF_THRESHOLD   = 0.35   # below this, report "uncertain" part


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPROVED SYNTHETIC IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _add_metallic_texture(img: np.ndarray, rng) -> np.ndarray:
    """Subtle horizontal metallic shimmer — makes scratch detection non-trivial."""
    h, w = img.shape[:2]
    shimmer = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.integers(3, 8)):
        y = rng.integers(0, h)
        shimmer[max(0,y-2):y+2, :] += rng.uniform(5, 20)
    shimmer = cv2.GaussianBlur(shimmer, (1, 21), 0)
    img = img.astype(np.float32)
    img[:, :, 0] += shimmer
    img[:, :, 1] += shimmer * 0.9
    img[:, :, 2] += shimmer * 0.8
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_realistic_scratches(img: np.ndarray, rng, mean_color) -> np.ndarray:
    """
    Realistic scratch bundle:
      - 1-5 scratches per cluster, 1-3 clusters
      - Random angle, slight curve via polyline
      - Width 1-3px
      - Two depths: surface (paint smear) and deep (dark metal exposed)
    """
    h, w = img.shape[:2]
    n_clusters = rng.integers(1, 4)

    for _ in range(n_clusters):
        cx = rng.integers(w // 5, 4 * w // 5)
        cy = rng.integers(h // 5, 4 * h // 5)
        angle = rng.uniform(0, np.pi)            # dominant scratch angle
        n_lines = rng.integers(1, 6)

        for j in range(n_lines):
            length = rng.integers(20, 90)
            width  = int(rng.integers(1, 4))
            offset = rng.integers(-8, 9)         # parallel offset within cluster
            deep   = rng.random() < 0.40         # deep scratch exposes metal

            # Build polyline with slight random curve (3-5 points)
            n_pts = rng.integers(3, 6)
            t_vals = np.linspace(0, 1, n_pts)
            dx = np.cos(angle) * length
            dy = np.sin(angle) * length
            perp_x = -np.sin(angle)
            perp_y =  np.cos(angle)

            pts = []
            for t in t_vals:
                jitter = rng.uniform(-4, 4)
                px = int(cx + dx * (t - 0.5) + offset * perp_x + jitter * perp_x)
                py = int(cy + dy * (t - 0.5) + offset * perp_y + jitter * perp_y)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                pts.append([px, py])

            pts_arr = np.array(pts, dtype=np.int32)

            if deep:
                # Dark metal-exposed scratch
                color = [max(0, m - 60) for m in mean_color]
            else:
                # Paint-smear: lighter streak
                color = [min(255, m + 40) for m in mean_color]

            cv2.polylines(img, [pts_arr], isClosed=False,
                          color=color, thickness=width, lineType=cv2.LINE_AA)

            # Add faint bright highlight alongside deep scratch
            if deep and width >= 2:
                highlight = [min(255, m + 60) for m in mean_color]
                shift_pts = pts_arr.copy()
                shift_pts[:, 0] += 1
                cv2.polylines(img, [shift_pts], isClosed=False,
                              color=highlight, thickness=1, lineType=cv2.LINE_AA)
    return img


def _make_clean_panel(part: str, size: int, rng) -> np.ndarray:
    """
    Photorealistic clean panel — no damage.
    Varies lighting gradient, paint sheen, panel edges so the model
    learns that smooth / gradual variation = no damage.
    """
    mean, std = PART_COLOUR[part]

    # Smooth gradient base (simulates lighting across panel)
    img = np.zeros((size, size, 3), dtype=np.float32)
    grad_dir = rng.choice(["h", "v", "d"])
    for i in range(size):
        t = i / size
        for ch in range(3):
            if grad_dir == "h":
                img[:, i, ch] = mean[ch] + rng.uniform(-8, 8)
            elif grad_dir == "v":
                img[i, :, ch] = mean[ch] + rng.uniform(-8, 8) * (1 - t)
            else:
                img[i, :, ch] = mean[ch] + rng.uniform(-6, 6) * t

    # Fine uniform noise (paint texture, not damage)
    noise = rng.normal(0, std[0] * 0.4, (size, size, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Metallic shimmer (clean car has this)
    if part not in ("headlight", "windshield", "tyre"):
        img = _add_metallic_texture(img, rng)

    # Subtle panel edge shadow (not a scratch — wide soft shadow)
    if rng.random() < 0.5:
        edge_side = rng.choice(["left", "right", "top", "bottom"])
        shadow = np.zeros((size, size), dtype=np.float32)
        w = rng.integers(10, 25)
        if edge_side == "left":   shadow[:, :w] = np.linspace(30, 0, w)
        elif edge_side == "right": shadow[:, -w:] = np.linspace(0, 30, w)
        elif edge_side == "top":  shadow[:w, :] = np.linspace(30, 0, w).reshape(-1, 1)
        else:                     shadow[-w:, :] = np.linspace(0, 30, w).reshape(-1, 1)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch].astype(np.float32) - shadow, 0, 255).astype(np.uint8)

    return img


def make_synthetic_image(part: str, damage: str, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng()

    # no_damage gets dedicated clean-panel generator
    if damage == "no_damage":
        return _make_clean_panel(part, size, rng)

    mean, std = PART_COLOUR[part]
    img = rng.normal(mean, std, (size, size, 3)).clip(0, 255).astype(np.uint8)

    # Metallic base texture for all metal parts
    if part not in ("headlight", "windshield", "tyre"):
        img = _add_metallic_texture(img, rng)

    tex = DAMAGE_TEXTURE[damage]

    # Random noise
    noise = (rng.random((size, size, 3)) * tex["noise"] * 255).astype(np.uint8)
    img = cv2.add(img, noise)

    # Dark elliptical patch (dent / shatter)
    if tex["dark_patch"]:
        cx, cy = rng.integers(30, size - 30, 2)
        r = rng.integers(15, 35)
        cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r // 2)),
                    int(rng.integers(0, 180)), 0, 360,
                    [max(0, m - 40) for m in mean], -1)

    # Crack lines (jagged)
    for _ in range(tex["crack_lines"]):
        pt1 = tuple(rng.integers(5, size - 5, 2).tolist())
        pt2 = tuple(rng.integers(5, size - 5, 2).tolist())
        cv2.line(img, pt1, pt2, (30, 30, 30), int(rng.integers(1, 3)))

    # Realistic scratch lines
    if tex["scratch_lines"] > 0:
        img = _add_realistic_scratches(img, rng, mean)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IMPROVED FEATURE EXTRACTOR  (128 → 256 dims)
# ─────────────────────────────────────────────────────────────────────────────

def _gabor_features(gray: np.ndarray) -> list:
    """
    4-orientation Gabor filter bank.
    Scratches are directional — Gabor captures orientation energy.
    Returns 8 values (mean + std per orientation).
    """
    feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        kernel = cv2.getGaborKernel(
            (21, 21), sigma=4.0, theta=theta,
            lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
        feats.append(float(filtered.mean()) / 255.0)
        feats.append(float(filtered.std())  / 255.0)
    return feats  # 8 values


def _hough_line_density(gray: np.ndarray) -> list:
    """
    Probabilistic Hough — count detected lines (scratch indicator).
    Returns [line_count_norm, mean_line_length_norm, angle_std_norm].
    """
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=20, minLineLength=10, maxLineGap=5)
    if lines is None:
        return [0.0, 0.0, 0.0]

    count = len(lines)
    lengths, angles = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lengths.append(np.hypot(x2 - x1, y2 - y1))
        angles.append(np.arctan2(y2 - y1, x2 - x1 + 1e-6))

    return [
        min(1.0, count / 80.0),
        min(1.0, float(np.mean(lengths)) / gray.shape[1]),
        min(1.0, float(np.std(angles)) / np.pi),
    ]  # 3 values


def _scratch_mask_ratio(gray: np.ndarray) -> float:
    """
    Morphological thin-line detector.
    Erode then subtract — isolates 1-3px lines (scratches).
    Returns ratio of scratch-like pixels.
    """
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    eroded_h = cv2.erode(gray, kernel_h)
    eroded_v = cv2.erode(gray, kernel_v)
    lines_h  = cv2.subtract(gray, eroded_h)
    lines_v  = cv2.subtract(gray, eroded_v)
    combined = cv2.add(lines_h, lines_v)
    _, mask   = cv2.threshold(combined, 15, 255, cv2.THRESH_BINARY)
    return float(np.count_nonzero(mask)) / float(gray.size)  # 1 value


def _lbp_variance_grid(gray: np.ndarray, grid: int = 8) -> list:
    """
    LBP proxy: variance in non-overlapping grid cells.
    Captures fine-texture irregularity (surface roughness from scratches).
    Returns grid*grid values.
    """
    h, w = gray.shape
    bh, bw = h // grid, w // grid
    feats = []
    for r in range(grid):
        for c in range(grid):
            patch = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            feats.append(float(patch.std()) / 128.0)
    return feats  # grid^2 values


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    256-dim feature vector:
      [0:48]   colour histograms (3ch × 16 bins)
      [48:49]  global edge density
      [49:65]  4×4 variance grid (LBP proxy, coarse)
      [65:73]  Gabor 4-orientation (mean+std each)
      [73:76]  Hough line density (count, length, angle_std)
      [76:77]  scratch mask ratio
      [77:141] 8×8 LBP variance grid (fine)
      [141:143] gradient magnitude (mean, std)
      [143:144] dark-region ratio
      [144:256] zero-padded to 256
    """
    img  = cv2.resize(img_bgr, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    # Colour histograms (48)
    for ch in range(3):
        hist, _ = np.histogram(img[:, :, ch], bins=16, range=(0, 256))
        feats.extend(hist / (hist.sum() + 1e-8))

    # Global edge density (1)
    edges = cv2.Canny(gray, 50, 150)
    feats.append(edges.mean() / 255.0)

    # Coarse 4×4 variance grid (16)
    h, w = gray.shape
    bs = 32
    for r in range(0, h, bs):
        for c in range(0, w, bs):
            feats.append(float(gray[r:r+bs, c:c+bs].std()) / 128.0)

    # Gabor 4-orientation (8)
    feats.extend(_gabor_features(gray))

    # Hough line density (3)
    feats.extend(_hough_line_density(gray))

    # Scratch mask ratio (1)
    feats.append(_scratch_mask_ratio(gray))

    # Fine 8×8 LBP variance grid (64)
    feats.extend(_lbp_variance_grid(gray, grid=8))

    # Gradient magnitude stats (2)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gm = np.sqrt(gx**2 + gy**2)
    feats.extend([gm.mean() / 255.0, gm.std() / 255.0])

    # Dark-region ratio (1)
    feats.append(float((gray < 60).mean()))

    # Pad / trim to 256
    feats = feats[:256]
    while len(feats) < 256:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_per_class: int = 350):
    print(f"Generating synthetic images ({n_per_class} per class) ...")
    X, y_part, y_damage = [], [], []
    combos = [(p, d) for p in PARTS for d in DAMAGES]
    total  = len(combos) * n_per_class
    done   = 0

    for part, damage in combos:
        for _ in range(n_per_class):
            img  = make_synthetic_image(part, damage)
            feat = extract_features(img)
            X.append(feat)
            y_part.append(part)
            y_damage.append(damage)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{total} samples generated ...")

    return np.array(X), np.array(y_part), np.array(y_damage)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PART CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def train_part_classifier(X, y_part):
    print("\nTraining Part Classifier ...")
    le = LabelEncoder()
    y  = le.fit_transform(y_part)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                               random_state=42, stratify=y)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=22, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ))
    ])
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"  Part Classifier accuracy: {acc*100:.1f}%")
    print(classification_report(y_te, clf.predict(X_te),
                                 target_names=list(le.classes_), zero_division=0))

    path = os.path.join(MODELS_DIR, "part_classifier.pkl")
    joblib.dump({"model": clf, "encoder": le, "classes": list(le.classes_)}, path)
    print(f"  Saved -> {path}")
    return clf, le


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DAMAGE CLASSIFIER  (now MULTI-CLASS, not binary)
# ─────────────────────────────────────────────────────────────────────────────

def train_damage_classifier(X, y_damage):
    print("\nTraining Damage Classifier (multi-class) ...")
    le = LabelEncoder()
    y  = le.fit_transform(y_damage)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                               random_state=42, stratify=y)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, max_depth=22, min_samples_leaf=2,
            class_weight="balanced",   # handles no_damage imbalance
            n_jobs=-1, random_state=42
        ))
    ])
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"  Damage Classifier accuracy: {acc*100:.1f}%")
    print(classification_report(y_te, clf.predict(X_te),
                                 target_names=list(le.classes_), zero_division=0))

    path = os.path.join(MODELS_DIR, "damage_classifier.pkl")
    joblib.dump({"model": clf, "encoder": le, "classes": list(le.classes_)}, path)
    print(f"  Saved -> {path}")
    return clf, le


# ─────────────────────────────────────────────────────────────────────────────
# 6.  COST REGRESSOR  (unchanged logic, updated to use new damage classes)
# ─────────────────────────────────────────────────────────────────────────────

COST_BASE = {
    ("bumper",     "dent"):    (4000,  9000),
    ("bumper",     "scratch"): (1500,  4500),
    ("bumper",     "crack"):   (3000,  8000),
    ("bumper",     "shatter"): (8000,  18000),
    ("door",       "dent"):    (6000,  20000),
    ("door",       "scratch"): (2000,  6000),
    ("door",       "crack"):   (5000,  15000),
    ("door",       "shatter"): (15000, 35000),
    ("hood",       "dent"):    (7000,  22000),
    ("hood",       "scratch"): (2500,  7500),
    ("hood",       "crack"):   (8000,  25000),
    ("hood",       "shatter"): (20000, 45000),
    ("headlight",  "dent"):    (2000,  5000),
    ("headlight",  "scratch"): (1000,  3000),
    ("headlight",  "crack"):   (3000,  8000),
    ("headlight",  "shatter"): (5000,  15000),
    ("windshield", "dent"):    (3000,  8000),
    ("windshield", "scratch"): (1500,  4000),
    ("windshield", "crack"):   (5000,  18000),
    ("windshield", "shatter"): (12000, 40000),
    ("tyre",       "dent"):    (1000,  3000),
    ("tyre",       "scratch"): (500,   2000),
    ("tyre",       "crack"):   (2000,  6000),
    ("tyre",       "shatter"): (4000,  12000),
    ("fender",     "dent"):    (5000,  15000),
    ("fender",     "scratch"): (2000,  6000),
    ("fender",     "crack"):   (4000,  12000),
    ("fender",     "shatter"): (10000, 25000),
    ("mirror",     "dent"):    (1500,  4000),
    ("mirror",     "scratch"): (800,   2500),
    ("mirror",     "crack"):   (2000,  5000),
    ("mirror",     "shatter"): (3000,  8000),
}

SEG_MULT = {"hatchback": 0.8, "sedan": 1.0, "suv": 1.35, "luxury": 2.3, "ev": 1.6}
SEV_MULT = {"minor": 0.65, "moderate": 1.0, "severe": 1.55, "critical": 2.2}
DAMAGE_COSTS = [d for d in DAMAGES if d != "no_damage"]


def generate_cost_data(n: int = 15000):
    rng  = np.random.default_rng(42)
    le_p = LabelEncoder().fit(PARTS)
    le_d = LabelEncoder().fit(DAMAGE_COSTS)
    le_s = LabelEncoder().fit(list(SEG_MULT.keys()))
    le_v = LabelEncoder().fit(["minor", "moderate", "severe", "critical"])
    rows, costs = [], []
    for _ in range(n):
        part   = rng.choice(PARTS)
        damage = rng.choice(DAMAGE_COSTS)
        seg    = rng.choice(list(SEG_MULT.keys()))
        sev    = rng.choice(["minor", "moderate", "severe", "critical"])
        age    = int(rng.integers(0, 16))
        panels = int(rng.integers(1, 4))
        lo, hi = COST_BASE.get((part, damage), (3000, 10000))
        cost = (
            rng.uniform(lo, hi)
            * SEG_MULT[seg]
            * SEV_MULT[sev]
            * (1 + 0.03 * age)
            * (1 + 0.15 * (panels - 1))
            * rng.uniform(0.88, 1.12)
        )
        rows.append([
            le_p.transform([part])[0],
            le_d.transform([damage])[0],
            le_s.transform([seg])[0],
            le_v.transform([sev])[0],
            age, panels
        ])
        costs.append(cost)
    return np.array(rows), np.array(costs), le_p, le_d, le_s, le_v


def train_cost_regressor():
    print("\nTraining Cost Regressor ...")
    X, y, le_p, le_d, le_s, le_v = generate_cost_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ))
    ])
    reg.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, reg.predict(X_te))
    print(f"  Cost Regressor MAE: Rs.{mae:,.0f}")
    path = os.path.join(MODELS_DIR, "cost_regressor.pkl")
    joblib.dump({
        "model": reg,
        "le_part": le_p, "le_damage": le_d,
        "le_segment": le_s, "le_severity": le_v,
        "cost_base": COST_BASE,
    }, path)
    print(f"  Saved -> {path}")
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# 7.  METADATA
# ─────────────────────────────────────────────────────────────────────────────

def save_meta():
    meta = {
        "parts":      PARTS,
        "damages":    DAMAGES,
        "segments":   list(SEG_MULT.keys()),
        "severities": ["minor", "moderate", "severe", "critical"],
        "feature_dim": 256,
        "cost_base":  {f"{k[0]}|{k[1]}": list(v) for k, v in COST_BASE.items()},
    }
    path = os.path.join(MODELS_DIR, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AutoScan AI — Improved Model Training")
    print("  Scratch-aware feature extraction (256-dim)")
    print("=" * 60)

    X, y_part, y_damage = generate_dataset(n_per_class=350)
    print(f"\nDataset: {len(X)} samples x {X.shape[1]} features")

    train_part_classifier(X, y_part)
    train_damage_classifier(X, y_damage)
    train_cost_regressor()
    save_meta()

    print("\nAll models trained and saved to ./models/")
    print("Drop-in replacement: copy models/ to abc/models/")
    print("Update inference.py: extract_features() -> import from this file")