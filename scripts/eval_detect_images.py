"""Zero-shot SAM 3 detection on the Monash human-labelled test frames (4x4 tiles).

uv run python scripts/eval_detect_images.py $BEECENSUS_DATA/yolo_testing out/det.json bee,insect
then: uv run python scripts/score_detect_images.py out/det.json
"""
import sys, glob, os, json, torch, numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
root, out_path = sys.argv[1], sys.argv[2]
prompts = sys.argv[3].split(",")
NAMES = ["Hymenoptera", "Flower", "Syrphidae", "Lepidoptera"]
dev = "cuda"
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(dev).eval()
proc = Sam3Processor.from_pretrained("facebook/sam3")

def detect(img, prompt):
    inp = proc(images=img, text=prompt, return_tensors="pt").to(dev)
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        o = model(**inp)
    r = proc.post_process_object_detection(o, threshold=0.2, target_sizes=[img.size[::-1]])[0]
    return [(b.float().tolist(), float(s)) for b, s in zip(r["boxes"], r["scores"])]

def tiled(img, prompt, nx=4, ny=4, pad=40):
    W, H = img.size; tw, th = W // nx, H // ny; dets = []
    for i in range(nx):
        for j in range(ny):
            x0, y0 = max(0, i*tw - pad), max(0, j*th - pad)
            x1, y1 = min(W, (i+1)*tw + pad), min(H, (j+1)*th + pad)
            for (b, s) in detect(img.crop((x0, y0, x1, y1)), prompt):
                dets.append(([b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0], s))
    # merge duplicates across tile overlap (simple NMS by center distance)
    dets.sort(key=lambda d: -d[1]); keep = []
    for b, s in dets:
        c = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        if all(abs(c[0]-(k[0][0]+k[0][2])/2) > 20 or abs(c[1]-(k[0][1]+k[0][3])/2) > 20 for k in keep):
            keep.append((b, s))
    return keep

res = []
files = sorted(glob.glob(os.path.join(root, "*.png")))
for f in files:
    lab = f[:-4] + ".txt"
    if not os.path.exists(lab): continue
    img = Image.open(f).convert("RGB"); W, H = img.size
    gts = []
    for l in open(lab):
        p = l.split()
        if not p: continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        gts.append((NAMES[c], [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]))
    if not any(n != "Flower" for n, _ in gts): continue
    row = {"file": os.path.basename(f), "gt": gts}
    for pr in prompts:
        pass
        row[f"tile4:{pr}"] = tiled(img, pr)
    res.append(row)
    print(len(res), os.path.basename(f), flush=True)
json.dump(res, open(out_path, "w"))
