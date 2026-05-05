"""
app.py — AutoScan AI  (Complete Enhanced Version)
===================================================
New features vs v1:
  - Damage heatmap visualization
  - Damage region bounding-box detection
  - Surface roughness index
  - Colour deviation analysis
  - PDF report download
  - Batch multi-image upload (up to 6)
  - Side-by-side comparison view
  - Dark metrics dashboard
  - All served from Python — no HTML files

Run:  python app.py   →   http://localhost:5000
"""

import os, uuid, io, json, threading, webbrowser, tempfile, zipfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response
from PIL import Image

from inference import predict
from vision_analysis import full_vision_analysis
from report_generator import generate_pdf_report

UPLOAD_DIR  = Path(__file__).parent / "uploads"
REPORTS_DIR = Path(__file__).parent / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024   # 60 MB for batch

HISTORY = []


# ─────────────────────────────────────────────────────────────────────────────
# FULL UI  (embedded Python string)
# ─────────────────────────────────────────────────────────────────────────────
UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AutoScan AI — Car Damage Detector</title>
<style>
:root{--bg:#0d0f14;--s1:#141720;--s2:#1c2030;--s3:#252840;--bdr:#2e3350;--bdr2:#404878;
      --acc:#00d4ff;--acc2:#0099cc;--grn:#00e676;--red:#ff5252;--amb:#ffab40;--pur:#ce93d8;
      --txt:#e8eaf6;--muted:#7986cb;--r:14px}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(0,212,255,.06),transparent);pointer-events:none;z-index:0}

/* NAV */
nav{position:sticky;top:0;z-index:200;display:flex;align-items:center;justify-content:space-between;padding:.8rem 2rem;background:rgba(13,15,20,.92);backdrop-filter:blur(20px);border-bottom:1px solid var(--bdr)}
.logo{display:flex;align-items:center;gap:.6rem;font-size:1.2rem;font-weight:700;letter-spacing:-.02em}
.logo-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--acc),#0055ff);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1rem}
.logo span{color:var(--acc)}
.nav-right{display:flex;align-items:center;gap:.5rem}
.pill{padding:.28rem .75rem;border-radius:100px;font-size:.72rem;border:1px solid var(--bdr);color:var(--muted);display:flex;align-items:center;gap:.3rem}
.pill.green{border-color:rgba(0,230,118,.3);color:var(--grn)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--grn);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* TABS */
.tabs{display:flex;gap:.3rem;padding:.5rem 2rem 0;position:relative;z-index:1}
.tab-btn{padding:.5rem 1.2rem;border-radius:8px 8px 0 0;font-size:.85rem;font-weight:500;cursor:pointer;color:var(--muted);background:transparent;border:1px solid transparent;border-bottom:none;transition:all .2s}
.tab-btn.active{color:var(--txt);background:var(--s1);border-color:var(--bdr)}

/* HERO */
.hero{text-align:center;padding:2.5rem 1rem 1.5rem;position:relative;z-index:1}
.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);border-radius:100px;padding:.28rem .9rem;font-size:.72rem;color:var(--acc);margin-bottom:1.2rem;text-transform:uppercase;letter-spacing:.06em}
h1{font-size:clamp(1.8rem,5vw,3.4rem);font-weight:800;line-height:1.05;letter-spacing:-.04em;margin-bottom:.8rem}
h1 em{background:linear-gradient(90deg,var(--acc),#7c83ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-style:normal}
.hero-sub{font-size:.95rem;color:var(--muted);max-width:520px;margin:0 auto 1.8rem;line-height:1.7}

/* FEATURE PILLS */
.features{display:flex;flex-wrap:wrap;justify-content:center;gap:.5rem;max-width:900px;margin:0 auto 2rem;padding:0 1rem;position:relative;z-index:1}
.feat{background:var(--s2);border:1px solid var(--bdr);border-radius:100px;padding:.3rem .85rem;font-size:.75rem;color:var(--muted);display:flex;align-items:center;gap:.35rem}
.feat b{color:var(--acc)}

/* MAIN GRID */
.container{position:relative;z-index:1;max-width:1160px;margin:0 auto;padding:0 1rem 4rem;display:grid;grid-template-columns:380px 1fr;gap:1.5rem;align-items:start}
@media(max-width:820px){.container{grid-template-columns:1fr}}

/* CARD */
.card{background:var(--s1);border:1px solid var(--bdr);border-radius:var(--r);overflow:hidden;margin-bottom:1.2rem}
.card:last-child{margin-bottom:0}
.card-head{padding:.85rem 1.2rem;border-bottom:1px solid var(--bdr);display:flex;align-items:center;justify-content:space-between;font-weight:600;font-size:.9rem}
.card-head .icon{font-size:1rem;margin-right:.4rem}
.card-body{padding:1.1rem}

/* UPLOAD / DROP */
.drop-zone{border:2px dashed var(--bdr2);border-radius:10px;min-height:200px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.75rem;cursor:pointer;transition:all .25s;position:relative;overflow:hidden}
.drop-zone:hover,.drop-zone.over{border-color:var(--acc);background:rgba(0,212,255,.04)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.dz-icon{font-size:2.2rem}
.dz-title{font-weight:600}
.dz-sub{font-size:.78rem;color:var(--muted)}
#prev-wrap{display:none;width:100%;height:200px;position:relative}
#prev-img{width:100%;height:100%;object-fit:cover;border-radius:8px}
.clear-btn{position:absolute;top:.5rem;right:.5rem;background:rgba(0,0,0,.8);border:1px solid var(--bdr);color:var(--txt);border-radius:7px;padding:.22rem .55rem;font-size:.7rem;cursor:pointer;transition:background .2s}
.clear-btn:hover{background:var(--red)}

/* PARAMS */
.params{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.9rem}
.param label{display:block;font-size:.68rem;color:var(--muted);margin-bottom:.28rem;text-transform:uppercase;letter-spacing:.06em}
.param select,.param input[type=number]{width:100%;background:var(--s2);border:1px solid var(--bdr);border-radius:8px;padding:.48rem .7rem;color:var(--txt);font-size:.85rem;outline:none;font-family:inherit;transition:border-color .2s}
.param select:focus,.param input[type=number]:focus{border-color:var(--acc)}

/* ANALYSE BUTTON */
.btn-analyze{width:100%;margin-top:.9rem;background:linear-gradient(90deg,var(--acc2),var(--acc));color:#000;border:none;border-radius:10px;padding:.85rem;font-size:.95rem;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:.55rem;transition:all .2s}
.btn-analyze:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(0,212,255,.3)}
.btn-analyze:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.spin{display:none;width:17px;height:17px;border:2px solid rgba(0,0,0,.25);border-top-color:#000;border-radius:50%;animation:sp .7s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}

/* SECONDARY BUTTONS */
.btn-sm{padding:.4rem .85rem;border-radius:8px;font-size:.78rem;font-weight:600;cursor:pointer;border:1px solid var(--bdr);background:var(--s2);color:var(--txt);transition:all .2s;display:inline-flex;align-items:center;gap:.35rem}
.btn-sm:hover{border-color:var(--acc);color:var(--acc)}
.btn-sm.primary{background:var(--acc);color:#000;border-color:var(--acc)}
.btn-sm.primary:hover{background:var(--acc2);border-color:var(--acc2);color:#000}

/* STATUS BADGE */
.status-badge{display:flex;align-items:center;gap:.8rem;padding:.9rem 1.1rem;border-radius:10px;font-weight:600;font-size:.9rem}
.status-badge.damaged{background:rgba(255,82,82,.08);border:1px solid rgba(255,82,82,.25);color:var(--red)}
.status-badge.safe{background:rgba(0,230,118,.08);border:1px solid rgba(0,230,118,.25);color:var(--grn)}
.badge-icon{font-size:1.5rem}

/* DETECTION GRID */
.det-grid{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:.8rem}
.det-card{background:var(--s2);border:1px solid var(--bdr);border-radius:9px;padding:.75rem .9rem}
.det-label{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:.22rem}
.det-value{font-weight:700;font-size:1rem;text-transform:capitalize}
.sev-minor{color:var(--grn)}.sev-moderate{color:var(--amb)}.sev-severe{color:var(--red)}.sev-critical{color:var(--pur)}.sev-no_damage{color:var(--acc)}

/* CONFIDENCE BAR */
.conf-bar-wrap{margin-top:.8rem}
.conf-bar-label{display:flex;justify-content:space-between;font-size:.7rem;color:var(--muted);margin-bottom:.28rem}
.conf-bar-bg{height:5px;background:var(--s3);border-radius:100px;overflow:hidden}
.conf-bar-fill{height:100%;border-radius:100px;transition:width .8s ease}

/* COST BLOCK */
.cost-block{background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.18);border-radius:9px;padding:1rem 1.1rem}
.cost-label{font-size:.68rem;color:rgba(0,212,255,.65);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem}
.cost-range{font-size:1.5rem;font-weight:800;color:var(--acc);letter-spacing:-.02em}
.cost-mid{font-size:.8rem;color:var(--muted);margin-top:.2rem}

/* TOP-3 */
.top3{display:flex;flex-direction:column;gap:.38rem;margin-top:.6rem}
.top3-row{display:flex;align-items:center;gap:.55rem;font-size:.8rem}
.top3-name{min-width:85px;text-transform:capitalize}
.top3-bar-bg{flex:1;height:4px;background:var(--s3);border-radius:100px;overflow:hidden}
.top3-bar{height:100%;background:linear-gradient(90deg,var(--acc2),var(--acc));border-radius:100px;transition:width .6s ease}
.top3-pct{min-width:34px;text-align:right;color:var(--muted);font-size:.75rem}

/* TIPS */
.tip-box{background:var(--s2);border-left:3px solid var(--acc);border-radius:0 8px 8px 0;padding:.65rem .9rem;font-size:.83rem;line-height:1.6;margin-bottom:.5rem}
.tip-box.warn{border-left-color:var(--amb)}.tip-box.danger{border-left-color:var(--red)}

/* LINKS GRID */
.links-grid{display:grid;grid-template-columns:1fr 1fr;gap:.45rem}
.link-card{background:var(--s2);border:1px solid var(--bdr);border-radius:8px;padding:.55rem .75rem;display:flex;align-items:center;gap:.45rem;text-decoration:none;color:var(--txt);font-size:.8rem;transition:all .2s}
.link-card:hover{border-color:var(--acc);color:var(--acc)}

/* VISION TABS */
.vision-tabs{display:flex;gap:.3rem;margin-bottom:.8rem;flex-wrap:wrap}
.vtab{padding:.3rem .75rem;border-radius:6px;font-size:.75rem;cursor:pointer;border:1px solid var(--bdr);color:var(--muted);background:transparent;transition:all .2s}
.vtab.active{border-color:var(--acc);color:var(--acc);background:rgba(0,212,255,.08)}
.vision-panel{display:none}.vision-panel.active{display:block}
.vision-img{width:100%;border-radius:8px;display:block}

/* METRIC ROW */
.metric-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin-bottom:.8rem}
.metric-card{background:var(--s2);border:1px solid var(--bdr);border-radius:9px;padding:.75rem .9rem;text-align:center}
.metric-val{font-size:1.5rem;font-weight:800;color:var(--acc)}
.metric-lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:.2rem}

/* BATCH */
#batch-tab{display:none}
.batch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.6rem;margin-bottom:1rem}
.batch-item{background:var(--s2);border:1px solid var(--bdr);border-radius:9px;overflow:hidden;position:relative}
.batch-img{width:100%;height:90px;object-fit:cover}
.batch-label{padding:.4rem .5rem;font-size:.72rem;color:var(--muted)}
.batch-status{position:absolute;top:.3rem;right:.3rem;width:18px;height:18px;border-radius:50%;font-size:.7rem;display:flex;align-items:center;justify-content:center}
.batch-status.done{background:var(--grn);color:#000}
.batch-status.err{background:var(--red);color:#fff}

/* COMPARE */
#compare-tab{display:none}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
.compare-card{background:var(--s2);border:1px solid var(--bdr);border-radius:9px;overflow:hidden}
.compare-img{width:100%;height:130px;object-fit:cover}
.compare-body{padding:.7rem}
.compare-title{font-size:.82rem;font-weight:600;text-transform:capitalize;margin-bottom:.3rem}
.compare-row{display:flex;justify-content:space-between;font-size:.76rem;color:var(--muted);margin-bottom:.15rem}
.compare-row b{color:var(--txt)}

/* HISTORY */
.hist-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:.6rem}
.hist-card{background:var(--s1);border:1px solid var(--bdr);border-radius:9px;overflow:hidden;cursor:pointer;transition:all .2s}
.hist-card:hover{border-color:var(--bdr2);transform:translateY(-2px)}
.hist-img-wrap{width:100%;height:80px;overflow:hidden}
.hist-img{width:100%;height:100%;object-fit:cover}
.hist-body{padding:.55rem .7rem}
.hist-part{font-size:.8rem;font-weight:600;text-transform:capitalize}
.hist-sub{font-size:.72rem;color:var(--muted);margin-top:.1rem;text-transform:capitalize}
.hist-cost{font-size:.77rem;color:var(--acc);margin-top:.2rem}

/* EMPTY */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:220px;color:var(--muted);gap:.7rem;font-size:.88rem}
.empty-icon{font-size:2.2rem;opacity:.4}

/* LOADING OVERLAY */
#loading-overlay{display:none;position:fixed;inset:0;background:rgba(13,15,20,.88);z-index:500;align-items:center;justify-content:center;flex-direction:column;gap:.9rem}
#loading-overlay.show{display:flex}
.loading-spinner{width:48px;height:48px;border:3px solid var(--bdr);border-top-color:var(--acc);border-radius:50%;animation:sp .8s linear infinite}
.loading-text{color:var(--acc);font-size:.9rem;font-weight:500}
.loading-sub{font-size:.78rem;color:var(--muted)}

footer{text-align:center;padding:1.4rem;font-size:.75rem;color:var(--muted);border-top:1px solid var(--bdr);position:relative;z-index:1}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="logo"><div class="logo-icon">🔍</div>Auto<span>Scan</span> AI</div>
  <div class="nav-right">
    <div class="pill green"><span class="dot"></span>AI Online</div>
    <div class="pill">Flask + OpenCV</div>
    <div class="pill">Offline</div>
  </div>
</nav>

<!-- TABS -->
<div class="tabs">
  <button class="tab-btn active" onclick="switchMainTab('single',this)">🔍 Single Analysis</button>
  <button class="tab-btn" onclick="switchMainTab('batch',this)">📦 Batch Upload</button>
  <button class="tab-btn" onclick="switchMainTab('compare',this)">⚖️ Compare</button>
  <button class="tab-btn" onclick="switchMainTab('history',this)">🕘 History</button>
</div>

<!-- HERO -->
<section class="hero">
  <div class="hero-badge">⚡ Random Forest + Gradient Boosting + OpenCV</div>
  <h1>AI car damage<br><em>detection & analysis</em></h1>
  <p class="hero-sub">Upload any car part photo — detect damage, get repair costs, heatmaps, region maps, and buy parts online. 100% offline.</p>
</section>

<!-- FEATURES -->
<div class="features">
  <div class="feat">🎯 <b>8</b> parts detected</div>
  <div class="feat">🔬 <b>5</b> damage types</div>
  <div class="feat">🌡️ Damage heatmap</div>
  <div class="feat">📦 Region detection</div>
  <div class="feat">📐 Surface roughness index</div>
  <div class="feat">🎨 Colour deviation analysis</div>
  <div class="feat">💰 INR cost estimate</div>
  <div class="feat">🛒 Live marketplace links</div>
  <div class="feat">📄 PDF report download</div>
  <div class="feat">📦 Batch processing</div>
  <div class="feat">⚖️ Side-by-side compare</div>
  <div class="feat">🔒 100% offline</div>
</div>

<!-- ═══ SINGLE TAB ═══════════════════════════════════════════════════════════ -->
<div id="single-tab">
<div class="container">

  <!-- LEFT PANEL -->
  <div>
    <div class="card">
      <div class="card-head"><span><span class="icon">📷</span>Upload Image</span></div>
      <div class="card-body">
        <div class="drop-zone" id="dz">
          <div id="dz-content">
            <div class="dz-icon">📸</div>
            <div class="dz-title">Drop a car image here</div>
            <div class="dz-sub">JPG · PNG · WebP · up to 20 MB</div>
          </div>
          <div id="prev-wrap">
            <img id="prev-img" alt="Preview"/>
            <button class="clear-btn" onclick="clearImg(event)">✕</button>
          </div>
          <input type="file" id="file-input" accept="image/*"/>
        </div>

        <div class="params">
          <div class="param">
            <label>Vehicle Segment</label>
            <select id="segment">
              <option value="hatchback">Hatchback</option>
              <option value="sedan" selected>Sedan</option>
              <option value="suv">SUV / MUV</option>
              <option value="luxury">Luxury</option>
              <option value="ev">Electric Vehicle</option>
            </select>
          </div>
          <div class="param">
            <label>Vehicle Age (yrs)</label>
            <input type="number" id="age" value="3" min="0" max="25"/>
          </div>
          <div class="param">
            <label>Panels Affected</label>
            <input type="number" id="panels" value="1" min="1" max="5"/>
          </div>
          <div class="param">
            <label>Expected Part</label>
            <select id="expected-part">
              <option value="">Auto Detect</option>
              <option value="bumper">Bumper</option>
              <option value="door">Door</option>
              <option value="hood">Hood / Bonnet</option>
              <option value="headlight">Headlight</option>
              <option value="windshield">Windshield</option>
              <option value="tyre">Tyre</option>
              <option value="fender">Fender</option>
              <option value="mirror">Side Mirror</option>
            </select>
          </div>
        </div>

        <button class="btn-analyze" id="analyze-btn" onclick="analyze()">
          <div class="spin" id="spin"></div>
          <span id="btn-txt">🔍 Analyze Damage</span>
        </button>
      </div>
    </div>

    <!-- Annotated image -->
    <div class="card" id="ann-card" style="display:none">
      <div class="card-head">
        <span><span class="icon">🖼️</span>AI Annotated Image</span>
        <button class="btn-sm" onclick="downloadAnnotated()">⬇ Save</button>
      </div>
      <div class="card-body" style="padding:.7rem">
        <img id="annotated-img" alt="Annotated" style="width:100%;border-radius:8px"/>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL — results -->
  <div id="result-col">
    <div class="empty" id="empty-state">
      <div class="empty-icon">🚗</div>
      <div>Upload a car image to begin AI analysis</div>
    </div>
  </div>

</div>
</div><!-- /single-tab -->

<!-- ═══ BATCH TAB ════════════════════════════════════════════════════════════ -->
<div id="batch-tab" style="display:none;max-width:1160px;margin:0 auto;padding:0 1rem 4rem;position:relative;z-index:1">
  <div class="card">
    <div class="card-head"><span><span class="icon">📦</span>Batch Upload — Analyse up to 6 images at once</span></div>
    <div class="card-body">
      <div class="drop-zone" id="batch-dz" style="min-height:120px">
        <div class="dz-icon">📸</div>
        <div class="dz-title">Drop multiple car images here</div>
        <div class="dz-sub">Up to 6 images · JPG · PNG · WebP</div>
        <input type="file" id="batch-input" accept="image/*" multiple/>
      </div>
      <div style="display:flex;gap:.6rem;margin-top:1rem;align-items:center">
        <div class="param" style="flex:1">
          <label>Vehicle Segment (all)</label>
          <select id="batch-segment">
            <option value="hatchback">Hatchback</option>
            <option value="sedan" selected>Sedan</option>
            <option value="suv">SUV / MUV</option>
            <option value="luxury">Luxury</option>
            <option value="ev">EV</option>
          </select>
        </div>
        <div class="param" style="flex:.5">
          <label>Age (yrs)</label>
          <input type="number" id="batch-age" value="3" min="0" max="25"/>
        </div>
        <button class="btn-sm primary" onclick="runBatch()" style="margin-top:1.4rem">▶ Run Batch</button>
        <button class="btn-sm" onclick="downloadBatchReport()" id="batch-dl" style="margin-top:1.4rem;display:none">⬇ ZIP Reports</button>
      </div>
    </div>
  </div>
  <div id="batch-results"></div>
</div>

<!-- ═══ COMPARE TAB ══════════════════════════════════════════════════════════ -->
<div id="compare-tab" style="display:none;max-width:1160px;margin:0 auto;padding:0 1rem 4rem;position:relative;z-index:1">
  <div class="card">
    <div class="card-head"><span><span class="icon">⚖️</span>Compare Two Damage Analyses</span></div>
    <div class="card-body">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
        <div>
          <label style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Image A</label>
          <div class="drop-zone" id="cmp-dz-a" style="min-height:130px">
            <div id="cmp-a-content"><div class="dz-icon" style="font-size:1.6rem">📷</div><div class="dz-sub">Drop image A</div></div>
            <div id="cmp-a-prev" style="display:none;width:100%;height:130px;position:relative"><img id="cmp-img-a" style="width:100%;height:100%;object-fit:cover;border-radius:8px" alt="A"/><button class="clear-btn" onclick="clearCmp('a',event)">✕</button></div>
            <input type="file" id="cmp-input-a" accept="image/*"/>
          </div>
        </div>
        <div>
          <label style="font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;display:block;margin-bottom:.35rem">Image B</label>
          <div class="drop-zone" id="cmp-dz-b" style="min-height:130px">
            <div id="cmp-b-content"><div class="dz-icon" style="font-size:1.6rem">📷</div><div class="dz-sub">Drop image B</div></div>
            <div id="cmp-b-prev" style="display:none;width:100%;height:130px;position:relative"><img id="cmp-img-b" style="width:100%;height:100%;object-fit:cover;border-radius:8px" alt="B"/><button class="clear-btn" onclick="clearCmp('b',event)">✕</button></div>
            <input type="file" id="cmp-input-b" accept="image/*"/>
          </div>
        </div>
      </div>
      <button class="btn-analyze" id="cmp-btn" onclick="runCompare()" style="margin-top:0">
        <div class="spin" id="cmp-spin"></div>
        <span id="cmp-btn-txt">⚖️ Compare Both Images</span>
      </button>
    </div>
  </div>
  <div id="compare-results"></div>
</div>

<!-- ═══ HISTORY TAB ══════════════════════════════════════════════════════════ -->
<div id="history-tab" style="display:none;max-width:1160px;margin:0 auto;padding:0 1rem 4rem;position:relative;z-index:1">
  <div class="card">
    <div class="card-head">
      <span><span class="icon">🕘</span>Analysis History</span>
      <button class="btn-sm" onclick="clearHistory()">🗑 Clear</button>
    </div>
    <div class="card-body">
      <div class="hist-grid" id="hist-grid">
        <div class="empty"><div class="empty-icon">📂</div><div>No analyses yet</div></div>
      </div>
    </div>
  </div>
</div>

<footer>AutoScan AI · Random Forest + Gradient Boosting + OpenCV · Flask 3 · 100% Offline &amp; Private · &copy; 2025</footer>

<!-- LOADING -->
<div id="loading-overlay">
  <div class="loading-spinner"></div>
  <div class="loading-text" id="loading-text">Analyzing image…</div>
  <div class="loading-sub" id="loading-sub">Running AI models + CV analysis</div>
</div>

<script>
// ── state ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let lastResult   = null;
let lastVision   = null;
let lastB64Ann   = null;
let batchFiles   = [];
let batchResults = [];
let cmpFiles     = {a: null, b: null};
const historyStore = [];

// ── MAIN TABS ──────────────────────────────────────────────────────────────
function switchMainTab(tab, btn) {
  ['single','batch','compare','history'].forEach(t => {
    document.getElementById(t+'-tab').style.display = t===tab ? '' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// ── SINGLE UPLOAD ─────────────────────────────────────────────────────────
const dz = document.getElementById('dz');
dz.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('over')});
dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
dz.addEventListener('drop', e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files[0])loadFile(e.dataTransfer.files[0])});
document.getElementById('file-input').addEventListener('change',e=>{if(e.target.files[0])loadFile(e.target.files[0])});

function loadFile(f){
  selectedFile=f;
  document.getElementById('prev-img').src=URL.createObjectURL(f);
  document.getElementById('dz-content').style.display='none';
  document.getElementById('prev-wrap').style.display='block';
  document.getElementById('result-col').innerHTML='<div class="empty"><div class="empty-icon">✅</div><div>Image loaded — press Analyze</div></div>';
  document.getElementById('ann-card').style.display='none';
}
function clearImg(e){
  e.stopPropagation();selectedFile=null;
  document.getElementById('file-input').value='';
  document.getElementById('dz-content').style.display='';
  document.getElementById('prev-wrap').style.display='none';
  document.getElementById('result-col').innerHTML='<div class="empty" id="empty-state"><div class="empty-icon">🚗</div><div>Upload a car image to begin AI analysis</div></div>';
  document.getElementById('ann-card').style.display='none';
}

// ── ANALYZE ───────────────────────────────────────────────────────────────
async function analyze(){
  if(!selectedFile){alert('Please select an image first.');return}
  setLoading(true,'Analyzing image…','Running AI models + CV analysis');
  const fd=new FormData();
  fd.append('image',selectedFile);
  fd.append('segment',document.getElementById('segment').value);
  fd.append('age',document.getElementById('age').value);
  fd.append('panels',document.getElementById('panels').value);
  const ep=document.getElementById('expected-part').value; if(ep)fd.append('expected_part',ep);
  try{
    const res=await fetch('/predict',{method:'POST',body:fd});
    const d=await res.json();
    if(!res.ok)throw new Error(d.error||'Server error');
    lastResult=d; lastVision=d.vision; lastB64Ann=d.annotated_img;
    showResults(d);
    addHistory(d);
  }catch(err){alert('Error: '+err.message)}
  finally{setLoading(false)}
}

function setLoading(on,txt='',sub=''){
  document.getElementById('analyze-btn').disabled=on;
  document.getElementById('spin').style.display=on?'block':'none';
  document.getElementById('btn-txt').textContent=on?'Analyzing…':'🔍 Analyze Damage';
  const ov=document.getElementById('loading-overlay');
  if(on){ov.classList.add('show');document.getElementById('loading-text').textContent=txt;document.getElementById('loading-sub').textContent=sub}
  else ov.classList.remove('show');
}

// ── RENDER RESULTS ────────────────────────────────────────────────────────
function showResults(d){
  const sev=d.severity||'no_damage';
  const confColor=d.damage_conf>75?'#00e676':d.damage_conf>55?'#ffab40':'#ff5252';
  let html='';

  // Status
  if(d.is_damaged)
    html+=`<div class="status-badge damaged"><span class="badge-icon">⚠️</span><div><div>Damage Detected</div><div style="font-size:.78rem;font-weight:400;opacity:.75;margin-top:.1rem">${d.damage} · ${d.part}</div></div></div>`;
  else
    html+=`<div class="status-badge safe"><span class="badge-icon">✅</span><div><div>Normal Damage Detected</div><div style="font-size:.78rem;font-weight:400;opacity:.75;margin-top:.1rem">Part appears in good condition</div></div></div>`;

  // Detection
  html+=`<div class="card"><div class="card-head"><span><span class="icon">📊</span>Detection Results</span>
    <div style="display:flex;gap:.4rem">
      <button class="btn-sm" onclick="downloadPDF()">📄 PDF Report</button>
    </div></div><div class="card-body">
    <div class="det-grid">
      <div class="det-card"><div class="det-label">Detected Part</div><div class="det-value">${d.part}</div></div>
      <div class="det-card"><div class="det-label">Damage Type</div><div class="det-value ${d.is_damaged?'':'sev-no_damage'}">${d.damage}</div></div>
      <div class="det-card"><div class="det-label">Severity</div><div class="det-value sev-${sev}">${sev}</div></div>
      <div class="det-card"><div class="det-label">Part Confidence</div><div class="det-value">${d.part_conf}%</div></div>
    </div>
    <div class="conf-bar-wrap">
      <div class="conf-bar-label"><span>Damage Confidence</span><span>${d.damage_conf}%</span></div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" id="dmg-bar" style="width:0%;background:${confColor}"></div></div>
    </div>
    ${d.top3_parts?`<div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-top:.8rem;margin-bottom:.4rem">Top Part Predictions</div>
    <div class="top3">${d.top3_parts.map(p=>`<div class="top3-row"><span class="top3-name">${p.part}</span><div class="top3-bar-bg"><div class="top3-bar" style="width:${p.prob}%"></div></div><span class="top3-pct">${p.prob}%</span></div>`).join('')}</div>`:''}
  </div></div>`;

  // Vision metrics
  if(d.vision){
    const v=d.vision;
    html+=`<div class="card"><div class="card-head"><span><span class="icon">🧠</span>Computer Vision Metrics</span></div><div class="card-body">
    <div class="metric-row">
      <div class="metric-card"><div class="metric-val" style="color:${v.overall_damage_score>65?'#ff5252':v.overall_damage_score>35?'#ffab40':'#00e676'}">${v.overall_damage_score}</div><div class="metric-lbl">Damage Score</div></div>
      <div class="metric-card"><div class="metric-val" style="color:#7986cb">${v.surface_roughness.sri}</div><div class="metric-lbl">Roughness Index</div></div>
      <div class="metric-card"><div class="metric-val" style="color:#ce93d8">${v.colour_deviation.deviation_score}</div><div class="metric-lbl">Colour Deviation</div></div>
    </div>
    <div class="vision-tabs">
      <button class="vtab active" onclick="switchVTab('heatmap',this)">🌡️ Heatmap</button>
      <button class="vtab" onclick="switchVTab('regions',this)">📦 Regions</button>
    </div>
    <div class="vision-panel active" id="vpanel-heatmap">
      <img class="vision-img" src="data:image/jpeg;base64,${v.heatmap_b64}" alt="Heatmap"/>
      <div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">Red = high damage probability zone. Generated from edge density, gradient magnitude and local variance.</div>
    </div>
    <div class="vision-panel" id="vpanel-regions">
      <img class="vision-img" src="data:image/jpeg;base64,${v.region_detection.annotated_b64}" alt="Regions"/>
      <div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">Found <b style="color:var(--txt)">${v.region_detection.region_count}</b> damage region(s) covering <b style="color:var(--txt)">${v.region_detection.coverage_pct}%</b> of the image. Red=large, orange=medium, yellow=small.</div>
    </div>
    <div style="margin-top:.8rem;font-size:.8rem;line-height:1.6">
      <div style="color:var(--muted)">Surface: <b style="color:var(--txt)">${v.surface_roughness.label}</b></div>
      <div style="color:var(--muted)">Colour: <b style="color:var(--txt)">${v.colour_deviation.label}</b></div>
      <div style="color:var(--muted)">Risk Level: <b style="color:var(--txt)">${v.risk_level}</b></div>
    </div>
    </div></div>`;
  }

  // Cost
  if(d.is_damaged&&d.cost){
    html+=`<div class="card"><div class="card-head"><span><span class="icon">💰</span>Repair Cost Estimate (INR)</span></div><div class="card-body">
    <div class="cost-block"><div class="cost-label">Estimated Range</div><div class="cost-range">${d.cost.formatted}</div><div class="cost-mid">Most likely: ${d.cost.midformatted}</div></div>
    <div style="font-size:.75rem;color:var(--muted);margin-top:.6rem">Approximate Indian market rates. Varies by city, model &amp; service centre.</div>
    </div></div>`;
  }

  // Tips
  if(d.repair_tip){
    const tc=sev==='critical'?'danger':sev==='severe'?'warn':'';
    html+=`<div class="card"><div class="card-head"><span><span class="icon">💡</span>Repair Advice</span></div><div class="card-body">
    <div class="tip-box ${tc}"><b>Repair:</b> ${d.repair_tip}</div>
    ${d.severity_tip?`<div class="tip-box warn"><b>Urgency:</b> ${d.severity_tip}</div>`:''}
    </div></div>`;
  }

  // Marketplace links
  if(d.part_links&&d.part_links.length){
    html+=`<div class="card"><div class="card-head"><span><span class="icon">🛒</span>Buy Replacement Part Online</span></div><div class="card-body">
    <div class="links-grid">${d.part_links.map(l=>`<a href="${l.url}" target="_blank" class="link-card">🔗 ${l.site}</a>`).join('')}</div>
    <div style="font-size:.72rem;color:var(--muted);margin-top:.6rem">⚠️ Verify part compatibility with your vehicle before purchasing.</div>
    </div></div>`;
  }

  document.getElementById('result-col').innerHTML=html;
  setTimeout(()=>{const b=document.getElementById('dmg-bar');if(b)b.style.width=d.damage_conf+'%'},100);

  if(d.annotated_img){
    document.getElementById('annotated-img').src='data:image/jpeg;base64,'+d.annotated_img;
    document.getElementById('ann-card').style.display='block';
  }
}

function switchVTab(id,btn){
  document.querySelectorAll('.vtab').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.vision-panel').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('vpanel-'+id).classList.add('active');
}

// ── PDF DOWNLOAD ──────────────────────────────────────────────────────────
async function downloadPDF(){
  if(!lastResult){alert('Run an analysis first.');return}
  setLoading(true,'Generating PDF report…','Building charts and layout');
  try{
    const fd=new FormData();
    fd.append('result',JSON.stringify(lastResult));
    fd.append('vision',JSON.stringify(lastVision));
    const res=await fetch('/generate_report',{method:'POST',body:fd});
    if(!res.ok){const j=await res.json();throw new Error(j.error||'Failed')}
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download='autoscan_report.pdf';a.click();
    URL.revokeObjectURL(url);
  }catch(err){alert('PDF error: '+err.message)}
  finally{setLoading(false)}
}

function downloadAnnotated(){
  if(!lastB64Ann)return;
  const a=document.createElement('a');
  a.href='data:image/jpeg;base64,'+lastB64Ann;
  a.download='autoscan_annotated.jpg';a.click();
}

// ── BATCH UPLOAD ──────────────────────────────────────────────────────────
const batchDz=document.getElementById('batch-dz');
batchDz.addEventListener('dragover',e=>{e.preventDefault();batchDz.classList.add('over')});
batchDz.addEventListener('dragleave',()=>batchDz.classList.remove('over'));
batchDz.addEventListener('drop',e=>{e.preventDefault();batchDz.classList.remove('over');loadBatch(e.dataTransfer.files)});
document.getElementById('batch-input').addEventListener('change',e=>loadBatch(e.target.files));

function loadBatch(files){
  batchFiles=[...files].slice(0,6);
  renderBatchPreviews();
}
function renderBatchPreviews(){
  const c=document.getElementById('batch-results');
  if(!batchFiles.length){c.innerHTML='';return}
  let h=`<div class="card" style="margin-top:1rem"><div class="card-head"><span>${batchFiles.length} image(s) queued</span></div><div class="card-body"><div class="batch-grid">`;
  batchFiles.forEach((f,i)=>{
    h+=`<div class="batch-item" id="bitem-${i}"><img class="batch-img" src="${URL.createObjectURL(f)}" alt=""/><div class="batch-label">${f.name.slice(0,20)}</div><div class="batch-status" id="bst-${i}"></div></div>`;
  });
  h+=`</div></div></div>`;
  c.innerHTML=h;
}

async function runBatch(){
  if(!batchFiles.length){alert('Upload at least one image.');return}
  batchResults=[];
  setLoading(true,'Processing batch…',`0 / ${batchFiles.length} done`);
  const seg=document.getElementById('batch-segment').value;
  const age=document.getElementById('batch-age').value;

  for(let i=0;i<batchFiles.length;i++){
    document.getElementById('loading-sub').textContent=`${i+1} / ${batchFiles.length} done`;
    const fd=new FormData();
    fd.append('image',batchFiles[i]);
    fd.append('segment',seg);fd.append('age',age);fd.append('panels','1');
    try{
      const r=await fetch('/predict',{method:'POST',body:fd});
      const d=await r.json();
      batchResults.push({...d,filename:batchFiles[i].name});
      const st=document.getElementById('bst-'+i);
      if(st){st.textContent='✓';st.className='batch-status done'}
    }catch(e){
      batchResults.push({error:e.message,filename:batchFiles[i].name});
      const st=document.getElementById('bst-'+i);
      if(st){st.textContent='!';st.className='batch-status err'}
    }
  }
  setLoading(false);
  renderBatchResults();
}

function renderBatchResults(){
  let html=`<div class="card" style="margin-top:1rem"><div class="card-head"><span>Batch Results (${batchResults.length})</span><button class="btn-sm primary" id="batch-dl" onclick="downloadBatchCSV()">⬇ Export CSV</button></div><div class="card-body">`;
  html+=`<table style="width:100%;border-collapse:collapse;font-size:.82rem">
    <tr style="color:var(--muted);border-bottom:1px solid var(--bdr)">
      <th style="text-align:left;padding:.5rem .6rem">File</th>
      <th>Part</th><th>Damage</th><th>Severity</th><th>Confidence</th><th>Est. Cost</th>
    </tr>`;
  batchResults.forEach(d=>{
    if(d.error){
      html+=`<tr style="border-bottom:1px solid var(--bdr)"><td style="padding:.5rem .6rem;color:var(--muted)">${d.filename}</td><td colspan="5" style="color:var(--red);text-align:center">Error: ${d.error}</td></tr>`;
    } else {
      const sc=d.severity==='critical'?'var(--pur)':d.severity==='severe'?'var(--red)':d.severity==='moderate'?'var(--amb)':'var(--grn)';
      html+=`<tr style="border-bottom:1px solid var(--bdr)">
        <td style="padding:.5rem .6rem;color:var(--muted)">${d.filename.slice(0,22)}</td>
        <td style="text-align:center;text-transform:capitalize">${d.part}</td>
        <td style="text-align:center;text-transform:capitalize">${d.damage}</td>
        <td style="text-align:center;color:${sc};font-weight:600">${d.severity}</td>
        <td style="text-align:center">${d.damage_conf}%</td>
        <td style="text-align:center;color:var(--acc)">${d.cost?d.cost.formatted:'N/A'}</td>
      </tr>`;
      addHistory(d);
    }
  });
  html+=`</table></div></div>`;
  document.getElementById('batch-results').innerHTML+=html;
}

function downloadBatchCSV(){
  let csv='File,Part,Part Conf,Damage,Damage Conf,Severity,Risk,Coverage%,Cost Low,Cost Mid,Cost High\n';
  batchResults.forEach(d=>{
    if(!d.error){
      csv+=`${d.filename},${d.part},${d.part_conf},${d.damage},${d.damage_conf},${d.severity},`;
      csv+=`${d.vision?d.vision.risk_level:''},${d.vision?d.vision.region_detection.coverage_pct:''},`;
      csv+=`${d.cost?d.cost.low:''},${d.cost?d.cost.mid:''},${d.cost?d.cost.high:''}\n`;
    }
  });
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='autoscan_batch.csv';a.click();
}

// ── COMPARE ───────────────────────────────────────────────────────────────
['a','b'].forEach(side=>{
  const dz=document.getElementById('cmp-dz-'+side);
  dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('over')});
  dz.addEventListener('dragleave',()=>dz.classList.remove('over'));
  dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files[0])loadCmp(side,e.dataTransfer.files[0])});
  document.getElementById('cmp-input-'+side).addEventListener('change',e=>{if(e.target.files[0])loadCmp(side,e.target.files[0])});
});

function loadCmp(side,f){
  cmpFiles[side]=f;
  document.getElementById('cmp-img-'+side).src=URL.createObjectURL(f);
  document.getElementById('cmp-'+side+'-content').style.display='none';
  document.getElementById('cmp-'+side+'-prev').style.display='block';
}
function clearCmp(side,e){
  e.stopPropagation();cmpFiles[side]=null;
  document.getElementById('cmp-input-'+side).value='';
  document.getElementById('cmp-'+side+'-content').style.display='';
  document.getElementById('cmp-'+side+'-prev').style.display='none';
}

async function runCompare(){
  if(!cmpFiles.a||!cmpFiles.b){alert('Upload both images to compare.');return}
  document.getElementById('cmp-btn').disabled=true;
  document.getElementById('cmp-spin').style.display='block';
  document.getElementById('cmp-btn-txt').textContent='Comparing…';
  setLoading(true,'Comparing both images…','Running parallel AI analysis');
  try{
    const [rA,rB]=await Promise.all(['a','b'].map(async side=>{
      const fd=new FormData();fd.append('image',cmpFiles[side]);fd.append('segment','sedan');fd.append('age','3');fd.append('panels','1');
      const res=await fetch('/predict',{method:'POST',body:fd});return res.json();
    }));
    renderCompare(rA,rB);
  }catch(err){alert('Compare error: '+err.message)}
  finally{
    document.getElementById('cmp-btn').disabled=false;
    document.getElementById('cmp-spin').style.display='none';
    document.getElementById('cmp-btn-txt').textContent='⚖️ Compare Both Images';
    setLoading(false);
  }
}

function renderCompare(a,b){
  function card(d,label){
    const sev=d.severity||'no_damage';
    const sc=sev==='critical'?'var(--pur)':sev==='severe'?'var(--red)':sev==='moderate'?'var(--amb)':'var(--grn)';
    return `<div class="compare-card">
      <img class="compare-img" src="data:image/jpeg;base64,${d.annotated_img||''}" alt="${label}"/>
      <div class="compare-body">
        <div class="compare-title">${label}</div>
        <div class="compare-row"><span>Part</span><b>${d.part}</b></div>
        <div class="compare-row"><span>Damage</span><b>${d.damage}</b></div>
        <div class="compare-row"><span>Severity</span><b style="color:${sc}">${sev}</b></div>
        <div class="compare-row"><span>Confidence</span><b>${d.damage_conf}%</b></div>
        ${d.vision?`<div class="compare-row"><span>Damage Score</span><b>${d.vision.overall_damage_score}</b></div>
        <div class="compare-row"><span>Coverage</span><b>${d.vision.region_detection.coverage_pct}%</b></div>`:''}
        <div class="compare-row"><span>Est. Cost</span><b style="color:var(--acc)">${d.cost?d.cost.formatted:'N/A'}</b></div>
      </div>
    </div>`;
  }

  // Winner
  const aScore=a.vision?a.vision.overall_damage_score:0;
  const bScore=b.vision?b.vision.overall_damage_score:0;
  const winner=aScore>bScore?'Image A is more severely damaged':'Image B is more severely damaged';
  const costDiff=a.cost&&b.cost?Math.abs(a.cost.mid-b.cost.mid):null;

  let html=`<div class="card" style="margin-top:1rem">
    <div class="card-head"><span>Comparison Results</span></div>
    <div class="card-body">
    <div style="background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.2);border-radius:9px;padding:.8rem 1rem;margin-bottom:1rem;font-size:.88rem">
      <b style="color:var(--acc)">Verdict:</b> ${winner}${costDiff?` · Cost difference: <b style="color:var(--acc)">₹${costDiff.toLocaleString('en-IN')}</b>`:''}
    </div>
    <div class="compare-grid">${card(a,'Image A')}${card(b,'Image B')}</div>
    </div></div>`;
  document.getElementById('compare-results').innerHTML=html;
}

// ── HISTORY ───────────────────────────────────────────────────────────────
function addHistory(d){
  historyStore.unshift(d);
  if(historyStore.length>20)historyStore.pop();
  renderHistory();
}
function renderHistory(){
  const grid=document.getElementById('hist-grid');
  if(!historyStore.length){grid.innerHTML='<div class="empty"><div class="empty-icon">📂</div><div>No analyses yet</div></div>';return}
  grid.innerHTML=historyStore.map(d=>`
    <div class="hist-card">
      <div class="hist-img-wrap"><img class="hist-img" src="data:image/jpeg;base64,${d.annotated_img||''}" alt=""/></div>
      <div class="hist-body">
        <div class="hist-part">${d.part} · ${d.damage}</div>
        <div class="hist-sub sev-${d.severity}">${d.severity}</div>
        ${d.cost?`<div class="hist-cost">${d.cost.formatted}</div>`:'<div class="hist-cost" style="color:var(--grn)">No damage</div>'}
      </div>
    </div>`).join('');
}
function clearHistory(){historyStore.length=0;renderHistory()}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return UI, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    ext  = (file.filename or "img.jpg").rsplit(".", 1)[-1].lower()
    if ext not in {"jpg","jpeg","png","webp","bmp"}:
        return jsonify({"error": "Unsupported file type"}), 400

    segment = request.form.get("segment", "sedan")
    age     = int(request.form.get("age", 3))
    panels  = int(request.form.get("panels", 1))

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # save upload
    fname = f"{uuid.uuid4().hex}.jpg"
    pil_img.save(UPLOAD_DIR / fname, "JPEG", quality=85)

    try:
        result = predict(pil_img, segment=segment, age=age, panels=panels)
        vision = full_vision_analysis(pil_img)
        result["vision"] = vision
        result["vehicle_segment"] = segment
        result["vehicle_age"]     = age
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    HISTORY.append(result)
    if len(HISTORY) > 100:
        HISTORY.pop(0)

    return jsonify(result)


@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        result_raw = request.form.get("result", "{}")
        vision_raw = request.form.get("vision", "{}")
        result = json.loads(result_raw)
        vision = json.loads(vision_raw)
    except Exception as e:
        return jsonify({"error": f"Invalid data: {e}"}), 400

    out_path = str(REPORTS_DIR / f"report_{uuid.uuid4().hex[:8]}.pdf")
    try:
        generate_pdf_report(result, vision, out_path)
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {e}"}), 500

    return send_file(out_path, as_attachment=True,
                     download_name="autoscan_report.pdf",
                     mimetype="application/pdf")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "history": len(HISTORY),
                    "models": "loaded", "features": [
                        "heatmap","region_detection","surface_roughness",
                        "colour_deviation","pdf_report","batch","compare"
                    ]})


@app.route("/history")
def history_route():
    n = min(int(request.args.get("n", 10)), 50)
    return jsonify(HISTORY[-n:])


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────

def open_browser():
    import time; time.sleep(1.3)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    print("=" * 60)
    print("  AutoScan AI v2 — Enhanced Car Damage Detection")
    print("  ► http://localhost:5000")
    print("  Features: Heatmap · Regions · PDF · Batch · Compare")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)