"""
vision_analysis.py — Advanced CV analysis: heatmap, regions, roughness, colour
"""
import cv2, numpy as np, io, base64
from PIL import Image

def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_b64(img, quality=88):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""

def generate_heatmap(img_bgr):
    img  = cv2.resize(img_bgr, (480, 360))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gm   = np.sqrt(gx**2 + gy**2)
    gm   = cv2.normalize(gm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = gray.shape
    var_map = np.zeros_like(gray, dtype=np.float32)
    bs = 16
    for r in range(0, h-bs, bs//2):
        for c in range(0, w-bs, bs//2):
            v = float(gray[r:r+bs, c:c+bs].std())
            var_map[r:r+bs, c:c+bs] = np.maximum(var_map[r:r+bs, c:c+bs], v)
    var_map = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges   = cv2.Canny(gray, 30, 100)
    heat    = (0.50*gm.astype(np.float32) + 0.30*var_map.astype(np.float32) + 0.20*edges.astype(np.float32)).clip(0,255).astype(np.uint8)
    heat    = cv2.GaussianBlur(heat, (21,21), 0)
    hmap    = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, hmap, 0.45, 0)
    bar_w   = overlay.shape[1]
    bar     = np.tile(np.linspace(0,255,bar_w,dtype=np.uint8),(20,1))
    bar_c   = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
    cv2.putText(bar_c,"LOW",(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    cv2.putText(bar_c,"HIGH",(bar_w-42,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    return bgr_to_b64(np.vstack([overlay, bar_c]))

def detect_damage_regions(img_bgr):
    img   = cv2.resize(img_bgr,(480,360))
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray,(15,15),0)
    diff  = cv2.absdiff(gray,blur)
    _,th  = cv2.threshold(diff,18,255,cv2.THRESH_BINARY)
    ker   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    th    = cv2.morphologyEx(th,cv2.MORPH_CLOSE,ker)
    th    = cv2.morphologyEx(th,cv2.MORPH_OPEN,ker)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ann   = img.copy()
    total = img.shape[0]*img.shape[1]
    dmg_area = 0; regions = []
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < 400: continue
        x,y,w,h = cv2.boundingRect(cnt)
        dmg_area += a
        col = (0,0,255) if a>total*.10 else (0,140,255) if a>total*.04 else (0,255,255)
        cv2.rectangle(ann,(x,y),(x+w,y+h),col,2)
        cv2.putText(ann,f"{int(a/100)}px²",(x,max(y-5,10)),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
        regions.append({"x":x,"y":y,"w":w,"h":h,"area":int(a)})
    cov = round(dmg_area/total*100,1)
    cv2.putText(ann,f"Regions: {len(regions)}  Coverage: {cov}%",(8,ann.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return {"annotated_b64":bgr_to_b64(ann),"regions":regions,"region_count":len(regions),"coverage_pct":cov}

def surface_roughness_index(img_bgr):
    img  = cv2.resize(img_bgr,(256,256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()
    h,w  = gray.shape; bs=32; total_std=0.0; count=0
    for r in range(0,h,bs):
        for c in range(0,w,bs):
            total_std += gray[r:r+bs,c:c+bs].std(); count+=1
    mean_std = total_std/max(count,1)
    sri = min(100.0, lap_var/600.0*60 + mean_std/40.0*40)
    label = ("Smooth (Good)" if sri<20 else "Slightly Rough (Minor Damage)" if sri<40
             else "Moderately Rough (Moderate Damage)" if sri<65 else "Very Rough (Severe Damage)")
    return {"sri":round(sri,1),"label":label,"laplacian_var":round(float(lap_var),2),"mean_std":round(float(mean_std),2)}

def colour_deviation_analysis(img_bgr):
    img  = cv2.resize(img_bgr,(256,256))
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L    = lab[:,:,0]
    l_std = float(L.std())
    dark  = float((L<50).mean()*100)
    bright= float((L>200).mean()*100)
    dev   = min(100.0, l_std*0.8 + float(lab[:,:,1].std())*0.1 + float(lab[:,:,2].std())*0.1)
    label = ("Uniform surface" if dev<15 else "Slight variation" if dev<30
             else "Significant variation (damage likely)" if dev<55 else "High variation (severe damage)")
    return {"deviation_score":round(dev,1),"label":label,"dark_ratio_pct":round(dark,1),"bright_ratio_pct":round(bright,1),"l_std":round(l_std,2)}

def full_vision_analysis(pil_img):
    bgr  = pil_to_bgr(pil_img)
    h    = generate_heatmap(bgr)
    r    = detect_damage_regions(bgr)
    s    = surface_roughness_index(bgr)
    c    = colour_deviation_analysis(bgr)
    overall = min(100.0, s["sri"]*0.40 + c["deviation_score"]*0.30 + min(r["coverage_pct"]*2,40.0)*0.30)
    risk = ("Low" if overall<25 else "Moderate" if overall<50 else "High" if overall<75 else "Critical")
    return {"overall_damage_score":round(overall,1),"risk_level":risk,
            "heatmap_b64":h,"region_detection":r,"surface_roughness":s,"colour_deviation":c}