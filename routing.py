import os
import json
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import boto3
from ultralytics import YOLO


load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    region_name=os.getenv("MINIO_REGION", "us-east-1")
)


BUCKET_SRC = os.environ["SRC_BUCKET"]
BUCKET_DST = os.environ["DST_BUCKET"]


print("Loading YOLO...")
yolo_model = YOLO(os.environ["YOLO_WEIGHTS"])
YOLO_CONF = 0.35


print("Loading ResNet & Labels...")
labels_path = os.getenv("LABELS_JSON", "")
type_list = []
sev_list = []


if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        type_list = data.get("type_classes", [])
        sev_list = data.get("severity_classes", [])


resnet_model = None
resnet_path = os.getenv("RESNET_WEIGHTS")
if resnet_path and os.path.exists(resnet_path):
    try:
        resnet_model = torch.jit.load(resnet_path, map_location=device)
        resnet_model.eval()
        print("ResNet loaded successfully.")
    except Exception as e:
        print(f"Error loading ResNet: {e}")
else:
    print("ResNet model not found.")



def crop_image(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    
    pad_x = int((x2 - x1) * 0.06)
    pad_y = int((y2 - y1) * 0.06)

    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return img[y1:y2, x1:x2].copy()



app = FastAPI()

@app.get("/health")
def check_health():
    return {"status": "ok", "resnet_ready": resnet_model is not None}


@app.post("/process")
def process_image(body: dict = Body(...)):
    try:
        
        file_key = body.get("key")
        bucket_name = body.get("bucket") or BUCKET_SRC

        if not file_key:
            return JSONResponse({"ok": False, "error": "No key provided"}, status_code=400)

        print(f"Processing: {file_key}")

        
        file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_data = file_obj["Body"].read()

        
        np_arr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"ok": False, "error": "Cannot decode image"}, status_code=400)

        
        results = yolo_model(img, conf=YOLO_CONF)[0]
        
        
        detected_items = []
        img_annotated = img.copy()
        file_name = os.path.splitext(os.path.basename(file_key))[0]

        
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            part_name = results.names[cls_id]

            
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_annotated, f"{part_name} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            crop = crop_image(img, [x1, y1, x2, y2])

            
            crop_key = f"yolo/crops/{part_name}/{file_name}_{i+1}.jpg"
            _, crop_buf = cv2.imencode(".jpg", crop)
            s3.put_object(Bucket=BUCKET_DST, Key=crop_key, Body=crop_buf.tobytes(), ContentType="image/jpeg")

            
            item_data = {
                "part": part_name,
                "bbox": [x1, y1, x2, y2],
                "det_conf": round(conf, 3),
                "crop_key": crop_key,
                "type": None,
                "severity": None
            }

            
            if resnet_model:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_crop, (224, 224))
                tensor_img = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor_img = (tensor_img - mean) / std
                input_batch = tensor_img.unsqueeze(0).to(device)

                
                output = resnet_model(input_batch)


                try:
                    if isinstance(output, tuple):
                        out_type, out_sev = output
                    elif isinstance(output, dict):
                        out_type = output.get("type")
                        out_sev = output.get("severity")
                    else:
                        out_type = None
                        out_sev = output

                    if out_type is not None and len(type_list) > 0:
                        probs = torch.softmax(out_type, dim=1)[0].cpu().numpy()
                        idx = probs.argmax()
                        item_data["type"] = {
                            "label": type_list[idx],
                            "conf": round(float(probs[idx]), 4)
                        }


                    if out_sev is not None and len(sev_list) > 0:
                        probs = torch.softmax(out_sev, dim=1)[0].cpu().numpy()
                        idx = probs.argmax()
                        item_data["severity"] = {
                            "label": sev_list[idx],
                            "conf": round(float(probs[idx]), 4)
                        }

                except Exception as e:
                    print(f"ResNet predict error: {e}")

            
            detected_items.append(item_data)

        annot_key = f"yolo/annot/{file_name}.jpg"
        _, annot_buf = cv2.imencode(".jpg", img_annotated)
        s3.put_object(Bucket=BUCKET_DST, Key=annot_key, Body=annot_buf.tobytes(), ContentType="image/jpeg")


        parts_count = len(detected_items)
        
        payload = {
            "source_key": file_key,
            "annot_key": annot_key,
            "count": parts_count,
            "crops": detected_items
        }

        json_key = f"yolo/json/{file_name}.json"
        s3.put_object(
            Bucket=BUCKET_DST, 
            Key=json_key, 
            Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        print("Done.")
        return JSONResponse({"ok": True, "payload": payload})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)