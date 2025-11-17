# infer_folder_ts.py  (ใช้ TorchScript)
import csv, time
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

IMG_SIZE = 224
ROOT = Path("ResNet50\Car_parts\data3a\Test")
OUT_CSV = Path("./preds.csv")
TS_PATH = Path("runs_resnet50/model_ts.pt")
CKPT_PATH = Path("runs_resnet50/best_resnet50.pt")

ckpt = torch.load(CKPT_PATH, map_location="cpu")
classes = ckpt["classes"]

ts = torch.jit.load(str(TS_PATH), map_location="cpu").eval()

tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
imgs = [p for p in ROOT.rglob("*") if p.suffix.lower() in exts]

t0 = time.time()
with torch.no_grad(), OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image","pred_class","pred_idx","score"])
    for p in imgs:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        logits = ts(x)
        prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax())
        w.writerow([str(p), classes[idx], idx, float(prob[idx])])
dt = time.time()-t0
print(f"Done {len(imgs)} images in {dt:.2f}s  -> {OUT_CSV.resolve()}")


