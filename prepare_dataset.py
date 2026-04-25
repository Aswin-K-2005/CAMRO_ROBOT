"""
Earthquake rescue human detection - dataset prep.
No API key needed. Downloads COCO automatically.

Requirements:
    pip install ultralytics albumentations opencv-python tqdm pyyaml
"""
import json, shutil, urllib.request, zipfile, random, os
from pathlib import Path
import cv2, yaml
from tqdm import tqdm

DATASET_DIR = Path("dataset")
MERGED_DIR  = DATASET_DIR / "merged"
RUNS_DIR    = Path("runs/rescue")
BASE_MODEL  = "yolov8n.pt"
EPOCHS      = 60
IMG_SIZE    = 640
BATCH       = 16
COCO_MAX    = 2000

def _progress(c, b, t):
    if t > 0:
        print(f"\r  {min(c*b/t*100,100):.1f}%", end="", flush=True)

def _download(url, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest, _progress)
    print()

def _unzip(src, dest):
    Path(dest).mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {src}...")
    with zipfile.ZipFile(src) as z:
        z.extractall(dest)

def download_coco():
    ann_dir  = DATASET_DIR / "coco"          # ← extract here, not inside annotations/
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / "annotations" / "instances_val2017.json"   # ← correct path

    if not ann_file.exists():
        print("[INFO] Downloading COCO 2017 annotations (~241 MB)...")
        _download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                ann_dir / "annotations.zip")
        _unzip(ann_dir / "annotations.zip", ann_dir)   # extracts → dataset/coco/annotations/

    print("[INFO] Parsing COCO for person class...")
    with open(ann_file) as f:
        coco = json.load(f)

    pid = next(c["id"] for c in coco["categories"] if c["name"] == "person")
    img_ids = list({a["image_id"] for a in coco["annotations"] if a["category_id"] == pid})
    if COCO_MAX:
        img_ids = img_ids[:COCO_MAX]

    id2info = {i["id"]: i for i in coco["images"]}
    anns_by = {}
    for a in coco["annotations"]:
        if a["category_id"] == pid:
            anns_by.setdefault(a["image_id"], []).append(a)

    out_img = MERGED_DIR / "train" / "images"
    out_lbl = MERGED_DIR / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {len(img_ids)} COCO images...")
    for iid in tqdm(img_ids):
        info = id2info[iid]
        fname = info["file_name"]
        W, H  = info["width"], info["height"]
        dst   = out_img / f"coco_{fname}"
        if not dst.exists():
            try:
                urllib.request.urlretrieve(
                    "http://images.cocodataset.org/val2017/" + fname, dst)
            except Exception as e:
                print(f"[WARN] {fname}: {e}")
                continue
        lines = []
        for a in anns_by.get(iid, []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            lines.append(f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        (out_lbl / f"coco_{Path(fname).stem}.txt").write_text("\n".join(lines))
    print(f"[INFO] COCO done.")

def check_widerperson():
    wp_zip = DATASET_DIR / "widerperson.zip"
    wp_dir = DATASET_DIR / "widerperson"
    if wp_dir.exists() and any(wp_dir.rglob("*.jpg")):
        _convert_widerperson(wp_dir); return
    if wp_zip.exists():
        print("[INFO] Extracting WiderPerson...")
        _unzip(wp_zip, wp_dir)
        _convert_widerperson(wp_dir); return
    print("[SKIP] WiderPerson not found (optional).")
    print("       Download: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/")
    print(f"       Place at: {wp_zip.resolve()}")

def _convert_widerperson(wp_dir):
    out_img = MERGED_DIR / "train" / "images"
    out_lbl = MERGED_DIR / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    ann = wp_dir / "Annotations" / "train.txt"
    if not ann.exists():
        print("[WARN] WiderPerson annotation missing."); return
    lines = ann.read_text().splitlines()
    count, i = 0, 0
    while i < len(lines):
        name = lines[i].strip(); i += 1
        if i >= len(lines): break
        n = int(lines[i].strip()); i += 1
        src = wp_dir / "Images" / name
        if not src.exists(): i += n; continue
        img = cv2.imread(str(src))
        if img is None: i += n; continue
        H, W = img.shape[:2]
        ylines = []
        for _ in range(n):
            p = lines[i].strip().split(); i += 1
            if int(p[0]) not in (1, 2, 3): continue
            x1,y1,x2,y2 = int(p[1]),int(p[2]),int(p[3]),int(p[4])
            bw,bh = x2-x1, y2-y1
            if bw<=0 or bh<=0: continue
            ylines.append(f"0 {(x1+bw/2)/W:.6f} {(y1+bh/2)/H:.6f} {bw/W:.6f} {bh/H:.6f}")
        stem = f"wp_{Path(name).stem}"
        shutil.copy(src, out_img / (stem + src.suffix))
        (out_lbl / (stem + ".txt")).write_text("\n".join(ylines))
        count += 1
    print(f"[INFO] WiderPerson done — {count} images")

def augment():
    try:
        import albumentations as A
    except ImportError:
        print("[WARN] pip install albumentations  — skipping augmentation"); return
    aug = A.Compose([
        A.CoarseDropout(max_holes=14, max_height=80, max_width=80,
                        min_holes=4, min_height=20, min_width=20, fill_value=0, p=0.85),
        A.RandomBrightnessContrast(brightness_limit=(-0.6,0.05), p=0.8),
        A.GaussNoise(var_limit=(30,100), p=0.6),
        A.MotionBlur(blur_limit=9, p=0.4),
        A.ToGray(p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.3),
    ])
    img_dir = MERGED_DIR / "train" / "images"
    lbl_dir = MERGED_DIR / "train" / "labels"
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"[INFO] Augmenting {len(imgs)} images...")
    for p in tqdm(imgs):
        img = cv2.imread(str(p))
        if img is None: continue
        out = aug(image=img)["image"]
        cv2.imwrite(str(img_dir / (p.stem+"_aug"+p.suffix)), out)
        lbl = lbl_dir / (p.stem+".txt")
        if lbl.exists():
            shutil.copy(lbl, lbl_dir / (p.stem+"_aug.txt"))
    print("[INFO] Augmentation done.")

def make_val_split():
    imgs = list((MERGED_DIR/"train"/"images").glob("*"))
    random.shuffle(imgs)
    n = max(1, int(len(imgs)*0.1))
    vi = MERGED_DIR/"valid"/"images"; vl = MERGED_DIR/"valid"/"labels"
    vi.mkdir(parents=True, exist_ok=True); vl.mkdir(parents=True, exist_ok=True)
    for p in imgs[:n]:
        lbl = MERGED_DIR/"train"/"labels"/(p.stem+".txt")
        shutil.move(str(p), vi/p.name)
        if lbl.exists(): shutil.move(str(lbl), vl/lbl.name)
    print(f"[INFO] Val split: {n} images")

def write_yaml():
    data = {"path": str(MERGED_DIR.resolve()), "train": "train/images",
            "val": "valid/images", "nc": 1, "names": ["person"]}
    yp = MERGED_DIR / "data.yaml"
    with open(yp, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yp


def train(data_yaml):
    from ultralytics import YOLO
    m = YOLO(BASE_MODEL)
    m.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,          # was 16, keep it at 8 (YOLO already tried reducing)
        project=str(RUNS_DIR),
        name="rescue_finetune",
        workers=2,        # was 8 — THIS is the main fix, fewer shared memory pipes
        device=0,
        amp=True,
        cache=False,      # keep False, caching needs more RAM
        hsv_v=0.6, degrees=15, flipud=0.3,
        mosaic=1.0, mixup=0.1, copy_paste=0.2, patience=15
    )

if __name__ == "__main__":
    print("="*50)
    print("  Rescue Human Detection - Dataset Prep")
    print("="*50)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    download_coco()
    check_widerperson()
    augment()
    make_val_split()
    train(write_yaml())
