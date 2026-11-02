"""Recall per class and false positives per frame for eval_detect_images.py output, at several score thresholds."""
import sys, json, numpy as np
res = json.load(open(sys.argv[1]))
def hit(gt, b, pad=10):
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    return gt[0]-pad <= cx <= gt[2]+pad and gt[1]-pad <= cy <= gt[3]+pad
keys = [k for k in res[0] if ":" in k]
print("images", len(res), "insect GT", sum(1 for r in res for n, _ in r["gt"] if n != "Flower"))
for k in keys:
    for th in (0.2, 0.3, 0.5):
        tp = {}; fp = 0; tot = {}
        for r in res:
            dets = [b for b, s in r[k] if s >= th]
            ins = [(n, g) for n, g in r["gt"] if n != "Flower"]
            flw = [g for n, g in r["gt"] if n == "Flower"]
            for n, g in ins:
                tot[n] = tot.get(n, 0) + 1
                if any(hit(g, b) for b in dets): tp[n] = tp.get(n, 0) + 1
            for b in dets:
                if not any(hit(g, b) for _, g in ins): fp += 1
        rec = {n: f"{tp.get(n,0)}/{tot[n]}" for n in tot}
        allr = sum(tp.values()) / sum(tot.values())
        print(f"{k:16s} th={th} recall={allr:.2f} {rec} FP/img={fp/len(res):.2f}")
