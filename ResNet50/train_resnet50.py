import argparse, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import onnx, onnxscript

def get_transforms(img_size=224):
    mean = [0.485, 0.456, 0.406] 
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_tf, val_tf

def make_loaders(data_root, img_size, batch, workers):
    train_tf, val_tf = get_transforms(img_size)

    train_path = Path(data_root)/"train"
    val_path   = Path(data_root)/"valid"

    if not train_path.exists():
        train_path = Path(data_root)/"training"
    if not val_path.exists():
        val_path = Path(data_root)/"validation"

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_path,   transform=val_tf)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)
    return train_ds, val_ds, train_ld, val_ld

def compute_class_weights(ds):
    counts = np.bincount([y for _,y in ds.samples], minlength=len(ds.classes))
    weights = counts.sum() / (counts + 1e-9)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

def build_model(num_classes, freeze_until=0):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, num_classes)
    )
    return model

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, preds_all, gts_all = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        preds = out.argmax(1)
        preds_all.append(preds.cpu().numpy())
        gts_all.append(y.cpu().numpy())
    preds_all = np.concatenate(preds_all); gts_all = np.concatenate(gts_all)
    f1 = f1_score(gts_all, preds_all, average="macro")
    acc = (preds_all==gts_all).mean()
    return np.mean(losses), acc, f1, preds_all, gts_all

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="root of crops/ (มี train/ valid/)")
    p.add_argument("--out", default="./runs_resnet50")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--freeze_backbone", action="store_true", help="freeze เกือบทั้งหมด ยกเว้น head")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, train_ld, val_ld = make_loaders(args.data, args.imgsz, args.batch, args.workers)
    class_names = train_ds.classes
    num_classes = len(class_names)

    model = build_model(num_classes, freeze_until=1 if args.freeze_backbone else 0).to(device)


    weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_ld)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_f1, best_path = -1, None

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

        val_loss, val_acc, val_f1, preds, gts = evaluate(model, val_ld, device, criterion)
        dt = time.time()-t0
        print(f"[{epoch:03d}/{args.epochs}] train_loss={epoch_loss/len(train_ld):.4f} "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} ({dt:.1f}s)")


        if val_f1 > best_f1:
            best_f1 = val_f1
            best_path = Path(args.out)/"best_resnet50.pt"
            torch.save({
                "model": model.state_dict(),
                "classes": class_names,
                "img_size": args.imgsz
            }, best_path)
            cm = confusion_matrix(gts, preds)
            print("Confusion Matrix:\n", cm)
            print(classification_report(gts, preds, target_names=class_names))

    print(f"[DONE] Best F1={best_f1:.4f} -> {best_path}")


    if best_path:
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model.eval()
    device = next(model.parameters()).device
    dummy  = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)


    ts_path = Path(args.out) / "model_ts.pt"
    traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)


    model_cpu = model.to("cpu").eval()
    dummy_cpu = dummy.to("cpu")
    onnx_path = Path(args.out) / "model.onnx"
    torch.onnx.export(
        model_cpu, dummy_cpu, onnx_path,
        input_names=["images"], output_names=["logits"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"Saved TorchScript: {ts_path}\nSaved ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
