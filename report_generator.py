"""
report_generator.py — Professional PDF report using ReportLab + Matplotlib
"""
import io, base64
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

C_DARK   = colors.HexColor("#0d0f14")
C_BLUE   = colors.HexColor("#00aacc")
C_LIGHT  = colors.HexColor("#e8eaf6")
C_MUTED  = colors.HexColor("#7986cb")
C_GREEN  = colors.HexColor("#00c853")
C_RED    = colors.HexColor("#ff5252")
C_AMBER  = colors.HexColor("#ffab40")
C_PURPLE = colors.HexColor("#ce93d8")
C_SURF   = colors.HexColor("#1c2030")
C_SURF2  = colors.HexColor("#252840")
C_BDR    = colors.HexColor("#2e3350")

SEV_COL = {"minor":C_GREEN,"moderate":C_AMBER,"severe":C_RED,
           "critical":C_PURPLE,"no_damage":C_BLUE}


def _b64_to_rl(b64: str, w_cm, h_cm):
    buf = io.BytesIO(base64.b64decode(b64))
    return RLImage(buf, width=w_cm*cm, height=h_cm*cm)


def _gauge(value, label):
    fig, ax = plt.subplots(figsize=(3,1.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#1c2030"); ax.set_facecolor("#1c2030")
    col = "#00c853" if value<35 else "#ffab40" if value<65 else "#ff5252"
    ax.add_patch(mpatches.Wedge((.5,.15),.42,0,180,width=.12,facecolor="#252840",edgecolor="none"))
    deg = 180 - int(value/100*180)
    ax.add_patch(mpatches.Wedge((.5,.15),.42,deg,180,width=.12,facecolor=col,edgecolor="none"))
    ax.text(.5,.17,f"{value:.0f}",ha="center",va="center",fontsize=17,fontweight="bold",color="white")
    ax.text(.5,-.05,label,ha="center",va="center",fontsize=6.5,color="#7986cb")
    ax.set_xlim(0,1); ax.set_ylim(-.15,.6); ax.axis("off")
    plt.tight_layout(pad=.2)
    buf = io.BytesIO(); plt.savefig(buf,format="png",dpi=100,facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0)
    return buf


def _bar_top3(top3):
    fig, ax = plt.subplots(figsize=(4.5,1.6))
    fig.patch.set_facecolor("#1c2030"); ax.set_facecolor("#1c2030")
    names = [p["part"].capitalize() for p in reversed(top3)]
    probs = [p["prob"] for p in reversed(top3)]
    ax.barh(names, probs, color=["#00aacc","#7986cb","#404878"], height=.5)
    for i,(n,p) in enumerate(zip(names,probs)):
        ax.text(p+.8, i, f"{p:.1f}%", va="center", color="white", fontsize=8)
    ax.set_xlim(0,115); ax.tick_params(colors="white",labelsize=8)
    for s in ax.spines.values(): s.set_color("#2e3350")
    ax.set_xlabel("Confidence (%)", color="#7986cb", fontsize=7)
    plt.tight_layout(pad=.4)
    buf = io.BytesIO(); plt.savefig(buf,format="png",dpi=100,facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0)
    return buf


def _cost_bar(cost):
    fig, ax = plt.subplots(figsize=(5,0.9))
    fig.patch.set_facecolor("#1c2030"); ax.set_facecolor("#1c2030")
    lo,mid,hi = cost["low"],cost["mid"],cost["high"]
    ax.barh([0],[hi-lo],left=[lo],height=.5,color="#1c3050",edgecolor="none")
    ax.barh([0],[mid-lo],left=[lo],height=.5,color="#00aacc",alpha=.6,edgecolor="none")
    ax.axvline(mid,color="#00d4ff",linewidth=2,linestyle="--")
    ax.text(lo,.45,f"₹{lo:,}",ha="left",va="bottom",color="#7986cb",fontsize=7)
    ax.text(mid,.45,f"₹{mid:,}",ha="center",va="bottom",color="#00d4ff",fontsize=7,fontweight="bold")
    ax.text(hi,.45,f"₹{hi:,}",ha="right",va="bottom",color="#7986cb",fontsize=7)
    ax.set_xlim(lo*.9,hi*1.12); ax.axis("off")
    plt.tight_layout(pad=.2)
    buf = io.BytesIO(); plt.savefig(buf,format="png",dpi=100,facecolor=fig.get_facecolor()); plt.close(fig); buf.seek(0)
    return buf


def _sty(name="", parent="Normal", **kw):
    ss = getSampleStyleSheet()
    return ParagraphStyle(name or "S", parent=ss[parent], **kw)


def generate_pdf_report(result: dict, vision: dict, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm)

    now  = datetime.now().strftime("%d %B %Y, %H:%M")
    rid  = datetime.now().strftime("RPT-%Y%m%d-%H%M%S")
    sev  = result.get("severity","no_damage")
    scol = SEV_COL.get(sev, C_BLUE)
    story = []

    # ── header ────────────────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("<b><font color='#00aacc'>AutoScan</font> AI</b>",
                  _sty(fontSize=20, fontName="Helvetica-Bold", textColor=C_LIGHT)),
        Paragraph(f"<font color='#7986cb'>Report ID: {rid}<br/>Generated: {now}</font>",
                  _sty(fontSize=8, fontName="Helvetica", textColor=C_MUTED, alignment=TA_RIGHT)),
    ]], colWidths=[10*cm,7*cm])
    hdr.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",(0,0),(-1,-1),1.5,C_BLUE),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    story += [hdr, Spacer(1,.4*cm)]

    # ── title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Car Damage Analysis Report",
        _sty(fontSize=20,fontName="Helvetica-Bold",textColor=C_LIGHT,spaceAfter=4)))
    story.append(Paragraph(
        f"Segment: <b>{result.get('vehicle_segment','sedan').capitalize()}</b>  |  "
        f"Age: <b>{result.get('vehicle_age',3)} yrs</b>",
        _sty(fontSize=9,fontName="Helvetica",textColor=C_MUTED)))
    story.append(Spacer(1,.4*cm))

    # ── status banner ─────────────────────────────────────────────────────────
    if result.get("is_damaged"):
        btxt = f"⚠  DAMAGE DETECTED — {result['damage'].upper()} on {result['part'].upper()}"
        bbg,bfg = colors.HexColor("#2a0a0a"), C_RED
    else:
        btxt = "✓  NO SIGNIFICANT DAMAGE DETECTED"
        bbg,bfg = colors.HexColor("#0a2a0f"), C_GREEN

    banner = Table([[Paragraph(f"<b>{btxt}</b>",
        _sty(fontSize=11,fontName="Helvetica-Bold",textColor=bfg,alignment=TA_CENTER))]],
        colWidths=[17*cm])
    banner.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bbg),
        ("BOX",(0,0),(-1,-1),1,bfg),("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9)]))
    story += [banner, Spacer(1,.4*cm)]

    # ── detection grid ────────────────────────────────────────────────────────
    story.append(Paragraph("Detection Summary",
        _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))

    def dc(lbl,val,col=C_LIGHT):
        return [Paragraph(f"<font color='#7986cb' size='8'>{lbl}</font>",
                          _sty(fontSize=8,fontName="Helvetica")),
                Paragraph(f"<b>{val}</b>",
                          _sty(fontSize=11,fontName="Helvetica-Bold",textColor=col))]

    dt = Table([
        dc("DETECTED PART",result.get("part","—").capitalize()) +
        dc("PART CONFIDENCE",f"{result.get('part_conf',0)}%"),
        dc("DAMAGE TYPE",result.get("damage","—").capitalize()) +
        dc("DAMAGE CONFIDENCE",f"{result.get('damage_conf',0)}%"),
        dc("SEVERITY",sev.upper(),scol) +
        dc("RISK LEVEL",vision.get("risk_level","—"),scol),
        dc("COVERAGE",f"{vision['region_detection']['coverage_pct']}%") +
        dc("REGIONS FOUND",str(vision["region_detection"]["region_count"])),
    ], colWidths=[4.2*cm,4.2*cm,4.2*cm,4.4*cm])
    dt.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#1c2030"),colors.HexColor("#252840")]),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("BOX",(0,0),(-1,-1),.5,C_BDR),("INNERGRID",(0,0),(-1,-1),.3,C_BDR),
    ]))
    story += [dt, Spacer(1,.4*cm)]

    # ── gauges ────────────────────────────────────────────────────────────────
    story.append(Paragraph("AI Damage Metrics",
        _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))

    g1 = RLImage(_gauge(vision["overall_damage_score"],"Overall Damage Score"),5.5*cm,3.2*cm)
    g2 = RLImage(_gauge(vision["surface_roughness"]["sri"],"Surface Roughness Index"),5.5*cm,3.2*cm)
    g3 = RLImage(_gauge(vision["colour_deviation"]["deviation_score"],"Colour Deviation Score"),5.5*cm,3.2*cm)
    gt = Table([[g1,g2,g3]], colWidths=[5.8*cm,5.8*cm,5.8*cm])
    gt.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("BACKGROUND",(0,0),(-1,-1),C_SURF),("BOX",(0,0),(-1,-1),.5,C_BDR)]))
    story.append(gt)

    sub_row = [vision["surface_roughness"]["label"],
               f"Lap. var: {vision['surface_roughness']['laplacian_var']}",
               vision["colour_deviation"]["label"]]
    st = Table([[Paragraph(s,_sty(fontSize=7,fontName="Helvetica",textColor=C_MUTED,alignment=TA_CENTER))
                 for s in sub_row]], colWidths=[5.8*cm,5.8*cm,5.8*cm])
    st.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_SURF),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [st, Spacer(1,.4*cm)]

    # ── top-3 + cost ──────────────────────────────────────────────────────────
    story.append(Paragraph("Part Confidence & Cost Estimate",
        _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))

    top3_img = RLImage(_bar_top3(result["top3_parts"]),8.5*cm,3.0*cm) if result.get("top3_parts") else Spacer(8.5*cm,3*cm)

    if result.get("cost"):
        cost_content = [
            RLImage(_cost_bar(result["cost"]),8.2*cm,1.8*cm),
            Paragraph(
                f"<font color='#00d4ff' size='15'><b>{result['cost']['formatted']}</b></font><br/>"
                f"<font color='#7986cb' size='8'>Most likely: {result['cost']['midformatted']}</font>",
                _sty(fontSize=9,fontName="Helvetica",textColor=C_LIGHT,alignment=TA_CENTER,leading=18)),
        ]
        inner = Table([[c] for c in cost_content], colWidths=[8.2*cm])
        inner.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("BACKGROUND",(0,0),(-1,-1),C_SURF)]))
    else:
        inner = Paragraph("No cost estimate — Normal damage detected.",
                          _sty(fontSize=9,fontName="Helvetica",textColor=C_MUTED))

    ct = Table([[top3_img, inner]], colWidths=[8.8*cm,8.2*cm])
    ct.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("BACKGROUND",(0,0),(-1,-1),C_SURF),("BOX",(0,0),(-1,-1),.5,C_BDR),
        ("INNERGRID",(0,0),(-1,-1),.3,C_BDR),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),8)]))
    story += [ct, Spacer(1,.4*cm)]

    # ── vision images ─────────────────────────────────────────────────────────
    story.append(Paragraph("Computer Vision Analysis",
        _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))
    vt = Table([
        [_b64_to_rl(vision["heatmap_b64"],8.5,6.5), _b64_to_rl(vision["region_detection"]["annotated_b64"],8.5,6.5)],
        [Paragraph("Damage Heatmap (red = high damage prob.)",_sty(fontSize=7,fontName="Helvetica",textColor=C_MUTED,alignment=TA_CENTER)),
         Paragraph("Damage Region Detection (coloured boxes by severity)",_sty(fontSize=7,fontName="Helvetica",textColor=C_MUTED,alignment=TA_CENTER))],
    ], colWidths=[8.5*cm,8.5*cm])
    vt.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("BACKGROUND",(0,0),(-1,-1),C_SURF),
        ("BOX",(0,0),(-1,-1),.5,C_BDR),("INNERGRID",(0,0),(-1,-1),.3,C_BDR),
        ("TOPPADDING",(0,1),(-1,-1),4),("BOTTOMPADDING",(0,1),(-1,-1),8)]))
    story += [vt, Spacer(1,.4*cm)]

    # ── repair advice ─────────────────────────────────────────────────────────
    story.append(Paragraph("Repair Recommendations",
        _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))
    def row(lbl,val):
        return [Paragraph(f"<font color='#7986cb' size='8'>{lbl}</font>",
                          _sty(fontSize=8,fontName="Helvetica")),
                Paragraph(val or "—",_sty(fontSize=9,fontName="Helvetica",textColor=C_LIGHT,leading=13))]
    at = Table([
        row("REPAIR TIP", result.get("repair_tip","")),
        row("URGENCY",    result.get("severity_tip","")),
        row("SURFACE",    vision["surface_roughness"]["label"]),
        row("COLOUR",     vision["colour_deviation"]["label"]),
    ], colWidths=[3.5*cm,13.5*cm])
    at.setStyle(TableStyle([
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#14161e"),colors.HexColor("#1c2030")]),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0),(-1,-1),8),
        ("BOX",(0,0),(-1,-1),.5,C_BDR),("INNERGRID",(0,0),(-1,-1),.3,C_BDR),
    ]))
    story += [at, Spacer(1,.4*cm)]

    # ── marketplace links ─────────────────────────────────────────────────────
    if result.get("part_links"):
        story.append(Paragraph("Online Replacement Part Sources",
            _sty(fontSize=13,fontName="Helvetica-Bold",textColor=C_BLUE,spaceBefore=8,spaceAfter=6)))
        lk = Table([[
            Paragraph(f"<link href='{l['url']}'><u>{l['site']}</u></link>",
                      _sty(fontSize=9,fontName="Helvetica",textColor=C_BLUE,alignment=TA_CENTER))
            for l in result["part_links"]
        ]], colWidths=[4.25*cm]*4)
        lk.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),C_SURF),
            ("BOX",(0,0),(-1,-1),.5,C_BDR),("INNERGRID",(0,0),(-1,-1),.3,C_BDR),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
        story.append(lk)

    # ── footer ────────────────────────────────────────────────────────────────
    story += [Spacer(1,.5*cm), HRFlowable(width="100%",thickness=.5,color=C_MUTED), Spacer(1,.2*cm)]
    story.append(Paragraph(
        "AutoScan AI — Offline AI-based car damage detection. Cost estimates are approximate "
        "Indian market rates. Consult a certified mechanic before any repair decision.",
        _sty(fontSize=7,fontName="Helvetica",textColor=C_MUTED,alignment=TA_CENTER,leading=11)))

    doc.build(story)
    return output_path