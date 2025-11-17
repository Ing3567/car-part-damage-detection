import os, io, json, hashlib, uuid, traceback
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.params import Body
import boto3
from botocore.config import Config

import cv2
import numpy as np
import torch
from ultralytics import YOLO

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



S3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name=os.getenv("MINIO_REGION", "us-east-1"),
    config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
)
SRC_BUCKET = os.environ["SRC_BUCKET"]
DST_BUCKET = os.environ["DST_BUCKET"]


YOLO_WEIGHTS  = os.environ["YOLO_WEIGHTS"]
YOLO_CONF     = float(os.getenv("YOLO_CONF", "0.35"))
yolo = YOLO(YOLO_WEIGHTS)


LABELS_JSON   = os.getenv("LABELS_JSON", "")
IMG_SIZE      = int(os.getenv("IMG_SIZE", "224"))


_resnet_model = None
_resnet_mode  = "dual" 
type_classes: list[str] = []
sev_classes:  list[str] = []



def get_obj_bytes(bucket: str, key: str) -> bytes:
    r = S3.get_object(Bucket=bucket, Key=key)
    return r["Body"].read()

def put_obj_bytes(bucket: str, key: str, data: bytes, content_type: str) -> Dict[str, Any]:
    S3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    sha = hashlib.sha256(data).hexdigest()
    return {"bucket": bucket, "key": key, "sha256": sha}

def encode_jpg(img, quality=90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def imdecode(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def letterbox_crop(img, x1,y1,x2,y2, pad=0.06):
    h,w = img.shape[:2]
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad), int(bh*pad)
    xx1, yy1 = max(0, x1-px), max(0, y1-py)
    xx2, yy2 = min(w, x2+px), min(h, y2+py)
    return img[yy1:yy2, xx1:xx2].copy()

def draw_boxes(img, boxes, names):
    canvas = img.copy()
    for b in boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0]); conf = float(b.conf[0])
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,200,255),2)
        label = f"{names[cls]} {conf:.2f}"
        cv2.putText(canvas,label,(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(20,20,20),3,cv2.LINE_AA)
        cv2.putText(canvas,label,(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return canvas

def _load_labels():
    global type_classes, sev_classes, _resnet_mode
    type_classes, sev_classes = [], []
    if LABELS_JSON and os.path.isfile(LABELS_JSON):
        with open(LABELS_JSON, "r", encoding="utf-8") as f:
            labels = json.load(f)
        type_classes = labels.get("type_classes", []) or []
        sev_classes  = labels.get("severity_classes", []) or []
    if len(type_classes) == 0:
        _resnet_mode = "severity_only"
    else:
        _resnet_mode = "dual"

import torch.nn as nn
import torchvision.models as tvm


def _try_load_torchscript(path: str):
    try:
        m = torch.jit.load(path, map_location=device)
        m.eval()
        return m
    except Exception:
        return None

def _try_load_state_dict(path: str):
    num_sev  = max(1, len(sev_classes))
    num_type = max(1, len(type_classes))


    model = tvm.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    if _resnet_mode == "dual":

        model.fc = nn.Identity() 

    else: 
        model.fc = nn.Linear(num_ftrs, num_sev)

    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and ("state_dict" in sd or "model" in sd):
        sd = sd.get("state_dict", sd.get("model"))
    

    model.load_state_dict(sd, strict=False) 
    return model.to(device).eval()

def _init_resnet():
    """เลือกโหลด TorchScript ก่อน ถ้าไม่สำเร็จค่อยลอง state_dict"""
    global _resnet_model
    _load_labels()
    
    resnet_weights = os.getenv("RESNET_WEIGHTS")
    if not resnet_weights or not os.path.isfile(resnet_weights):
        _resnet_model = None
        print("WARN: RESNET_WEIGHTS is not set or file not found. ResNet classification is disabled.")
        return
        
    m = _try_load_torchscript(resnet_weights)
    if m is None:
        print(f"INFO: Could not load TorchScript, trying state_dict for {resnet_weights}...")
        try:
            m = _try_load_state_dict(resnet_weights)
        except Exception as e:
            print(f"ERROR: Failed to load ResNet state_dict: {e}")
            m = None

    _resnet_model = m
    if _resnet_model:
        print(f"INFO: ResNet model loaded successfully in '{_resnet_mode}' mode.")
    else:
        print("ERROR: ResNet model loading failed.")

_init_resnet() 

def pt_preprocess_bgr(img_bgr, size=224):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]; s = max(h, w)
    canvas = np.full((s, s, 3), 114, np.uint8)
    y0 = (s - h) // 2; x0 = (s - w) // 2
    canvas[y0:y0+h, x0:x0+w] = img
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    canvas = (canvas - mean) / std
    x = torch.from_numpy(canvas.transpose(2,0,1)).unsqueeze(0).to(device)
    return x

def cls_resnet50(img_bgr):
    if _resnet_model is None:
        return None

    x = pt_preprocess_bgr(img_bgr, IMG_SIZE)
    out = _resnet_model(x)


    if _resnet_mode == "dual":
        if isinstance(out, dict):
            tp = out.get("type") or out.get("type_head") or out.get("out_type")
            sp = out.get("severity") or out.get("sev_head") or out.get("out_sev")
        else:
            tp, sp = out  # tuple
        if isinstance(tp, torch.Tensor): tp = torch.softmax(tp, dim=1)
        if isinstance(sp, torch.Tensor): sp = torch.softmax(sp, dim=1)
        tp = tp[0].detach().cpu().numpy()
        sp = sp[0].detach().cpu().numpy()
        ti, si = int(tp.argmax()), int(sp.argmax())
        return {
            "type": {"label": type_classes[ti], "conf": float(tp[ti])},
            "severity": {"label": sev_classes[si], "conf": float(sp[si])}
        }
    else:
        if isinstance(out, dict):
            sp = out.get("severity") or out.get("sev_head") or list(out.values())[0]
        else:
            sp = out
        if isinstance(sp, torch.Tensor): sp = torch.softmax(sp, dim=1)
        sp = sp[0].detach().cpu().numpy()
        si = int(sp.argmax())
        return {
            "type": None,  
            "severity": {"label": sev_classes[si], "conf": float(sp[si])}
        }
    

def _softmax(xs):
    import math
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _argmax(d: dict[str, float]) -> tuple[str, float]:
    if not d:
        return None, None
    lab = max(d, key=lambda k: d[k])
    return lab, d[lab]

def normalize_head(head: dict | None, classes: list[str]) -> dict | None:
    if not head:
        return None

    out = {"label": None, "conf": None, "probs": None}

    
    if isinstance(head.get("probs"), dict) and head["probs"]:
        probs = {k: float(v) for k, v in head["probs"].items()}
        for c in classes:
            probs.setdefault(c, 0.0)
        lab, conf = _argmax(probs)
        out["probs"] = probs
        out["label"] = head.get("label") or lab
        out["conf"]  = float(head.get("conf") if head.get("conf") is not None else conf)

    
    elif isinstance(head.get("scores"), (list, tuple)) and classes:
        scores = [float(x) for x in head["scores"]]
        probs_vec = _softmax(scores)
        probs = {c: probs_vec[i] if i < len(probs_vec) else 0.0 for i, c in enumerate(classes)}
        lab, conf = _argmax(probs)
        out["probs"] = probs
        out["label"] = head.get("label") or lab
        out["conf"]  = float(head.get("conf") if head.get("conf") is not None else conf)

    elif isinstance(head.get("logits"), (list, tuple)) and classes:
        logits = [float(x) for x in head["logits"]]
        probs_vec = _softmax(logits)
        probs = {c: probs_vec[i] if i < len(probs_vec) else 0.0 for i, c in enumerate(classes)}
        lab, conf = _argmax(probs)
        out["probs"] = probs
        out["label"] = head.get("label") or lab
        out["conf"]  = float(head.get("conf") if head.get("conf") is not None else conf)

    
    else:
        out["label"] = head.get("label")
        out["conf"]  = float(head.get("conf")) if head.get("conf") is not None else None
        if isinstance(head.get("probs"), (list, tuple)) and classes:
            vec = [float(x) for x in head["probs"]]
            out["probs"] = {c: vec[i] if i < len(vec) else 0.0 for i, c in enumerate(classes)}
        if out["conf"] is None and out["label"]:
            out["conf"] = 1.0

    
    if out["conf"] is not None:
        out["conf"] = max(0.0, min(1.0, float(out["conf"])))
        out["conf"] = round(out["conf"], 4)

    return out if out["label"] else None


from collections import Counter

def summarize_crops(crops: list[dict]) -> dict:
    parts = Counter()
    type_hist = Counter()
    sev_hist = Counter()
    worst_item = None

    
    sev_order = sev_classes if len(sev_classes) > 0 else ["minor", "moderate", "severe"]
    sev_rank = {lab: i for i, lab in enumerate(sev_order)}  # ค่าน้อย = เบากว่า

    for c in crops:
        parts[c.get("part", "-")] += 1
       
        t = (c.get("type") or {}).get("label")
        if t: type_hist[t] += 1
        
        s = (c.get("severity") or {}).get("label")
        if s: sev_hist[s] += 1

        
        if s in sev_rank:
            if worst_item is None:
                worst_item = c
            else:
                ws = (worst_item.get("severity") or {}).get("label")
                pick = False
                if ws not in sev_rank:
                    pick = True
                else:
                    # มากกว่า = รุนแรงกว่า
                    if sev_rank[s] > sev_rank[ws]:
                        pick = True
                    elif sev_rank[s] == sev_rank[ws]:
                        # เท่ากันดูคอนฟิเดนซ์
                        sc = (c.get("severity") or {}).get("conf") or 0.0
                        wc = (worst_item.get("severity") or {}).get("conf") or 0.0
                        if sc > wc:
                            pick = True
                if pick:
                    worst_item = c

    
    def mode_or_none(cnt: Counter):
        if not cnt:
            return None
        lab, num = cnt.most_common(1)[0]
        return {"label": lab, "count": int(num)}

    return {
        "parts_count": sum(parts.values()),
        "parts_hist": dict(parts),
        "type_mode": mode_or_none(type_hist),      # อาจเป็น None ถ้าไม่มี type
        "severity_mode": mode_or_none(sev_hist),   # อาจเป็น None ถ้าไม่มี severity
        "worst_item": worst_item,                  # อาจเป็น None ถ้าไม่มี severity
    }


def processed_marker_key(src_key: str) -> str:
    return f"_markers/{hashlib.md5(src_key.encode()).hexdigest()}.done"

def is_already_processed(dst_bucket: str, src_key: str) -> bool:
    mk = processed_marker_key(src_key)
    try:
        S3.head_object(Bucket=dst_bucket, Key=mk)
        return True
    except Exception:
        return False

def mark_processed(dst_bucket: str, src_key: str):
    mk = processed_marker_key(src_key)
    put_obj_bytes(dst_bucket, mk, b"ok", "text/plain")



def run_pipeline(source_bucket: str, source_key: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # idempotency
    if is_already_processed(DST_BUCKET, source_key):
        print(f"INFO: Skipping already processed key: {source_key}")
        return {"status": "skip", "reason": "already processed", "key": source_key}

    try:
        raw = get_obj_bytes(source_bucket, source_key)
        img = imdecode(raw)
        if img is None:
            return {"status": "error", "reason": "cannot decode image", "key": source_key}
    except Exception as e:
        print(f"ERROR: Failed to download or decode image {source_bucket}/{source_key}: {e}")
        return {"status": "error", "reason": f"image download/decode failed: {e}", "key": source_key}


    
    res = yolo(img, conf=YOLO_CONF, verbose=False)[0]
    names = res.names
    stem = os.path.splitext(os.path.basename(source_key))[0]

    
    annot = draw_boxes(img, res.boxes, names)
    annot_key = f"yolo/annot/{stem}.jpg"
    put_obj_bytes(DST_BUCKET, annot_key, encode_jpg(annot), "image/jpeg")

   
    items = []
    for i, b in enumerate(res.boxes, start=1):
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0]); conf = float(b.conf[0]); part = names[cls]
        crop = letterbox_crop(img, x1, y1, x2, y2, 0.06)
        crop_key = f"yolo/crops/{part}/{stem}_{i}.jpg"
        put_obj_bytes(DST_BUCKET, crop_key, encode_jpg(crop), "image/jpeg")

        det = {
            "part": part,
            "bbox": [x1, y1, x2, y2],
            "det_conf": round(conf, 3),
            "crop_key": crop_key
        }

        
        r2 = cls_resnet50(crop)
        if r2:
            t_norm = normalize_head(r2.get("type"), type_classes)
            s_norm = normalize_head(r2.get("severity"), sev_classes)
            if t_norm:
                det["type"] = t_norm      # => {"label","conf","probs"?}
            if s_norm:
                det["severity"] = s_norm  # => {"label","conf","probs"?}
            if isinstance(r2.get("model"), dict):
                det["resnet_meta"] = r2["model"]

        items.append(det)

    
    summary = summarize_crops(items)

    payload = {
        "source_bucket": source_bucket,
        "source_key": source_key,
        "meta": meta or {},
        "annot_key": annot_key,
        "crops": items,          
        "summary": summary,
        "labels": {
            "type_classes": type_classes,
            "severity_classes": sev_classes
        }
    }

    
    json_key = f"yolo/json/{stem}.json"
    put_obj_bytes(
        DST_BUCKET,
        json_key,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json"
    )

    mark_processed(DST_BUCKET, source_key)

    return {
        "status": "ok",
        "json_key": json_key,
        "annot_key": annot_key,
        "count": len(items),
        "payload": payload
    }


# ------------ FastAPI App ------------
app = FastAPI(title="YOLO→Crop→ResNet50 Webhook", version="1.1.0")

@app.get("/health")
def health():
    return {"ok": True, "resnet_loaded": _resnet_model is not None, "mode": _resnet_mode}


@app.post("/process")
def process_manual(body: dict = Body(...)):
    try:
        key = body.get("key")
        if not key:
            return JSONResponse({"ok": False, "error": "field 'key' is required"}, status_code=400)


        bucket = body.get("bucket") or SRC_BUCKET
        meta   = body.get("meta") or {"source": "manual"}

        out = run_pipeline(bucket, key, meta=meta)   

        if out.get("status") != "ok":
            return JSONResponse({"ok": False, "error": out}, status_code=400)


        return JSONResponse({"ok": True, "payload": out["payload"]}, status_code=200)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": f"internal error: {e}"}, status_code=500)