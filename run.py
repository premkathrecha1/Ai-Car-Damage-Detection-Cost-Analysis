#!/usr/bin/env python3
"""
run.py — One-click launcher for AutoScan AI
============================================
Run:  python run.py
"""
import os, sys, subprocess
from pathlib import Path

BASE = Path(__file__).parent

def check_models():
    needed = ["part_classifier.pkl", "damage_classifier.pkl", "cost_regressor.pkl", "meta.json"]
    return all((BASE / "models" / f).exists() for f in needed)

if __name__ == "__main__":
    print("=" * 55)
    print("  AutoScan AI — One-Click Launcher")
    print("=" * 55)

    if not check_models():
        print("⚙  Models not found — training now (takes ~60 seconds)…")
        result = subprocess.run([sys.executable, str(BASE / "train_models.py")], cwd=BASE)
        if result.returncode != 0:
            print("❌ Training failed. Check output above.")
            sys.exit(1)
    else:
        print("✅  Models already trained.")

    print("\n🚀 Starting AutoScan AI server…")
    print("   Open your browser at:  http://localhost:5000\n")
    subprocess.run([sys.executable, str(BASE / "app.py")], cwd=BASE)