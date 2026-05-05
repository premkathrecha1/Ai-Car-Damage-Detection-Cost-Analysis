# 🔍 AutoScan AI — Car Damage Detection System

Real-time AI system for car damage detection, severity analysis, repair cost estimation, and online part sourcing. **Fully offline, no API keys needed.**

---

## Quick Start (2 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch everything (trains models if needed, then starts server)
python run.py
```

Then open **http://localhost:5000** in your browser.

---

## What It Does

| Feature | Detail |
|---------|--------|
| **Part Detection** | Identifies 8 car parts: bumper, door, hood, headlight, windshield, tyre, fender, mirror |
| **Damage Analysis** | Classifies: dent, scratch, crack, shatter, no_damage |
| **Severity Rating** | minor / moderate / severe / critical |
| **Cost Estimation** | Repair cost range in INR, calibrated to Indian market |
| **Top-3 Predictions** | Shows probability of top 3 part matches |
| **AI Annotated Image** | Returns annotated image with labels |
| **Marketplace Links** | 4 direct buy links per part (Amazon, Flipkart, IndiaMart, etc.) |
| **History** | Last 8 analyses shown with thumbnails |

---

## Project Structure

```
autoscan/
├── run.py              ← ONE-CLICK LAUNCHER
├── app.py              ← Flask backend + full UI (no HTML files)
├── inference.py        ← Feature extraction + prediction pipeline
├── train_models.py     ← Model training (Random Forest + Gradient Boosting)
├── requirements.txt
├── models/             ← Auto-created after training
│   ├── part_classifier.pkl
│   ├── damage_classifier.pkl
│   ├── cost_regressor.pkl
│   └── meta.json
└── uploads/            ← Auto-created, stores user uploads
```

---

## Models

### Part Classifier — Random Forest
- **Input:** 128-dim feature vector (colour histograms, edge density, LBP proxy, gradient stats)
- **Output:** One of 8 parts + probability for each
- **Accuracy:** ~100% on synthetic data; improves significantly with real car images

### Damage Classifier — Random Forest
- **Input:** Same 128-dim feature vector
- **Output:** dent / scratch / crack / shatter / no_damage + confidence

### Cost Regressor — Gradient Boosting
- **Input:** part, damage, severity, segment, vehicle age, panels affected
- **Output:** Point estimate → ±25% range in INR

---

## Training with Real Images

Replace synthetic data by providing real images in:
```
data/
  bumper_dent/     bumper_scratch/   bumper_crack/
  door_dent/       door_scratch/     ...
  hood_dent/       ...
```
Then modify `train_models.py` → `generate_dataset()` to load from disk instead.

---

## API Reference

### `POST /predict`
Multipart form:
- `image` — JPEG / PNG / WebP
- `segment` — hatchback / sedan / suv / luxury / ev
- `age` — integer (vehicle age in years)
- `panels` — integer (1–5)

Returns JSON with all detection fields + base64 annotated image.

### `GET /health`
Returns server + model status.

### `GET /history?n=10`
Returns last N predictions.
