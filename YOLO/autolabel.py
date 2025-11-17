# auto_label_yolo11.py  (ready for YOLOv11)  -- layout = ./content/{train,valid,test}/{images,labels}
import os, argparse, shutil, cv2, json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def xyxy_to_xywhn(x1, y1, x2, y2, w, h):
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def write_yolo_det_txt(txt_path: Path, dets, img_w, img_h):
    """YOLO detection label format (no confidence!): class cx cy w h (normalized 0..1)"""
    lines = []
    for d in dets:
        cls = d["class_id"]
        x1,y1,x2,y2 = d["xyxy"]
        cx, cy, ww, hh = xyxy_to_xywhn(x1, y1, x2, y2, img_w, img_h)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
    # สร้างไฟล์เสมอ แม้ไม่มีดีเทคชัน (ไฟล์ว่าง) เพื่อให้ชื่อภาพ-เลเบลจับคู่กัน 1:1
    txt_path.write_text("\n".join(lines), encoding="utf-8")

def write_yolo_seg_txt(txt_path: Path, segs, img_w, img_h):
    """YOLO segmentation label format: class x1 y1 x2 y2 ... (normalized)"""
    lines = []
    for s in segs:
        cls = s["class_id"]
        poly = []
        for (x, y) in s["polygon"]:
            poly.append(str(max(0.0, min(1.0, x / img_w))))
            poly.append(str(max(0.0, min(1.0, y / img_h))))
        lines.append(" ".join([str(cls)] + poly))
    txt_path.write_text("\n".join(lines), encoding="utf-8")

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def process_image(model, img_path: Path, out_images: Path, out_labels: Path,
                  conf=0.35, save_visual=False, class_filter=None, segmentation=False):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read: {img_path}")
        return None
    h, w = img.shape[:2]

    r = model(img, conf=conf, verbose=False)[0]

    # กำหนดไฟล์คู่กัน images/*.jpg ↔ labels/*.txt
    out_img_path = out_images / img_path.name
    out_txt_path = out_labels / (img_path.stem + ".txt")
    ensure_dir(out_images); ensure_dir(out_labels)

    # คัดกรองผล
    det_results = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls[0])
            if class_filter and cls not in class_filter:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
            if x2 > x1 and y2 > y1:
                det_results.append({"class_id": cls, "xyxy": (x1,y1,x2,y2)})

    seg_results = []
    if segmentation and getattr(r, "masks", None) is not None and r.masks is not None and r.masks.xy is not None:
        for i, poly in enumerate(r.masks.xy):
            cls = int(r.boxes.cls[i])
            if class_filter and cls not in class_filter:
                continue
            seg_results.append({
                "class_id": cls,
                "polygon": [(float(x), float(y)) for x,y in poly]
            })

    # เขียน label (det หรือ seg) — สร้างไฟล์เสมอ
    if segmentation and len(seg_results) > 0:
        write_yolo_seg_txt(out_txt_path, seg_results, w, h)
    else:
        write_yolo_det_txt(out_txt_path, det_results, w, h)

    # คัดลอก/บันทึกรูป
    if save_visual:
        canvas = img.copy()
        for d in det_results:
            x1,y1,x2,y2 = d["xyxy"]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.imwrite(str(out_images / f"annot_{img_path.name}"), canvas)
        save_json(out_images / f"{img_path.stem}.json", {
            "image": img_path.name, "width": w, "height": h,
            "items": [
                {"class_id": d["class_id"], "name": r.names[d["class_id"]],
                 "bbox_xyxy": list(d["xyxy"])}
                for d in det_results
            ]
        })
        if segmentation and len(seg_results) > 0:
            save_json(out_images / f"{img_path.stem}_seg.json", {
                "image": img_path.name, "width": w, "height": h,
                "items": [
                    {"class_id": s["class_id"], "name": r.names[s["class_id"]],
                     "polygon": s["polygon"]}
                    for s in seg_results
                ]
            })
    else:
        shutil.copy2(img_path, out_img_path)

def iter_images(folder: Path):
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts:
            yield p

def extract_and_process_video(model, video_path: Path, out_images: Path, out_labels: Path,
                              conf=0.35, frame_stride=5, save_visual=False, class_filter=None, segmentation=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return
    i = 0
    ensure_dir(out_images)  # ให้แน่ใจว่าโฟลเดอร์มี ก่อนบันทึกเฟรมชั่วคราว
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % frame_stride != 0:
            i += 1; continue
        tmp_name = f"{video_path.stem}_f{i:06d}.jpg"
        tmp_path = out_images / tmp_name
        cv2.imwrite(str(tmp_path), frame)
        process_image(model, tmp_path, out_images, out_labels, conf, save_visual, class_filter, segmentation)
        i += 1
    cap.release()

def maybe_write_data_yaml(content_root: Path, names):
    """สร้าง ./data.yaml อัตโนมัติ ให้ตรงกับโครงสร้าง ./content/<split>/images"""
    data_yaml = Path("data.yaml")
    if data_yaml.exists():
        return
    # names อาจเป็น list หรือ dict
    if isinstance(names, dict):
        names_dict = {int(k): v for k, v in names.items()}
    else:
        names_dict = {i: n for i, n in enumerate(names)}

    path_str = str(content_root.resolve()).replace("\\", "/")
    lines = [
        f"path: {path_str}",
        "train: train/images",
        "val: valid/images",
        "# test: test/images",
        "names:"
    ] + [f"  {k}: {v}" for k, v in sorted(names_dict.items())]

    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] wrote data.yaml -> {data_yaml.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to yolo11*.pt (detect หรือ *-seg)")
    ap.add_argument("--source", required=True, help="path to images folder or a video file")
    # ค่าเริ่มต้น out='./content'
    ap.add_argument("--out", default="./content", help="output root (./content/<split>/{images,labels})")
    ap.add_argument("--split", default="train", choices=["train","valid","test"], help="subfolder name")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=5, help="frame stride for video")
    ap.add_argument("--save_visual", action="store_true", help="save annotated preview & json per image")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="keep only these class ids")
    ap.add_argument("--segmentation", action="store_true", help="if model is *-seg and you want seg labels")
    ap.add_argument("--write_data_yaml", action="store_true", help="also create ./data.yaml once")
    args = ap.parse_args()

    model = YOLO(args.weights)

    # โฟลเดอร์ปลายทางให้ตรงภาพ: ./content/<split>/{images,labels}
    out_root = Path(args.out)
    out_images = out_root / args.split / "images"
    out_labels = out_root / args.split / "labels"
    ensure_dir(out_images); ensure_dir(out_labels)

    # สร้าง data.yaml (ครั้งแรก) — ใช้ root เป็น ./content
    if args.write_data_yaml:
        # ถ้า args.out == "./content" ก็ใช้ตามนั้น
        content_root = out_root
        maybe_write_data_yaml(content_root, getattr(model.model, "names", getattr(model, "names", {})))

    src = Path(args.source)
    if src.is_dir():
        imgs = list(iter_images(src))
        for p in tqdm(imgs, desc=f"auto-label images -> {args.split}"):
            process_image(model, p, out_images, out_labels,
                          conf=args.conf, save_visual=args.save_visual,
                          class_filter=set(args.classes) if args.classes else None,
                          segmentation=args.segmentation)
        print(f"Done images → {out_root.resolve()}")
    elif src.is_file():
        extract_and_process_video(model, src, out_images, out_labels,
                                  conf=args.conf, frame_stride=args.stride,
                                  save_visual=args.save_visual,
                                  class_filter=set(args.classes) if args.classes else None,
                                  segmentation=args.segmentation)
        print(f"Done video → {out_root.resolve()}")
    else:
        raise SystemExit("source not found")

if __name__ == "__main__":
    main()
