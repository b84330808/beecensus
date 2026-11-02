"""Draw a survey result on the footage: flowers (green once visited), insect trails, labels, counters."""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

FFMPEG = shutil.which("ffmpeg") or str(Path(__file__).resolve().parents[3] / "demo" / "bin" / "ffmpeg.exe")
COLORS = {  # BGR
    "honeybee": (0, 200, 255), "bumblebee": (0, 140, 255), "solitary_bee": (0, 230, 200), "wasp": (0, 255, 255),
    "hoverfly": (255, 120, 40), "other_fly": (200, 160, 120), "butterfly_moth": (255, 80, 220), "beetle": (80, 80, 255),
    "small_insect": (180, 180, 180), "other_insect": (200, 200, 200), "unknown": (200, 200, 200),
}
NICE = {"honeybee": "honeybee", "bumblebee": "bumblebee", "solitary_bee": "solitary bee", "wasp": "wasp",
        "hoverfly": "hoverfly", "other_fly": "fly", "butterfly_moth": "moth/butterfly", "beetle": "beetle",
        "small_insect": "small insect", "other_insect": "insect", "unknown": "insect"}


def group_of(ins: dict) -> str:
    return (ins.get("label") or {}).get("group") or "unknown"


def render(res: dict, out: str | Path, t_start: float | None = None, t_end: float | None = None,
           width: int = 1280, trail_s: float = 2.0) -> Path:
    fps = res["fps"]
    t_start = res["t_start"] if t_start is None else t_start
    t_end = res["t_end"] if t_end is None else t_end
    insects = [i for i in res["insects"] if group_of(i) != "not_insect"]
    pts = {i["id"]: np.column_stack([i["frame"], i["cx"], i["cy"]]) for i in insects}
    grp = {i["id"]: group_of(i) for i in insects}
    first_visit = {}
    for v in res["visits"]:
        first_visit.setdefault(v["flower"], v["start"])
    W, H = res["size"]
    s = width / W
    size = (width, int(round(H * s / 2)) * 2)
    cap = cv2.VideoCapture(res["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_start * fps))
    ff = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    first_seen = sorted(int(p[0, 0]) for p in pts.values())
    for f in range(int(t_start * fps), int(t_end * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        for fl in res["flowers"]:
            visited = fl["id"] in first_visit and first_visit[fl["id"]] <= f
            c = (90, 220, 90) if visited else (235, 235, 235)
            cv2.circle(fr, (int(fl["x"] * s), int(fl["y"] * s)), int(fl["r"] * s * 1.2), c, 2 if visited else 1, cv2.LINE_AA)
        for iid, p in pts.items():
            c = COLORS.get(grp[iid], (200, 200, 200))
            past = p[(p[:, 0] <= f) & (p[:, 0] >= f - trail_s * fps)]
            for a, b in zip(past[:-1], past[1:]):
                cv2.line(fr, (int(a[1] * s), int(a[2] * s)), (int(b[1] * s), int(b[2] * s)), c, 2, cv2.LINE_AA)
            now = p[np.abs(p[:, 0] - f) <= 2]
            if len(now) and p[0, 0] <= f <= p[-1, 0]:
                x, y = int(now[0][1] * s), int(now[0][2] * s)
                cv2.circle(fr, (x, y), 22, c, 3, cv2.LINE_AA)
                cv2.putText(fr, NICE.get(grp[iid], grp[iid]), (x + 26, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(fr, NICE.get(grp[iid], grp[iid]), (x + 26, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
        n_vis = sum(1 for v in res["visits"] if v["start"] <= f)
        n_fl = sum(1 for k, v in first_visit.items() if v <= f)
        n_ins = sum(1 for a in first_seen if a <= f)
        hud = f"{f / fps:6.1f}s   insects {n_ins}   flower visits {n_vis}   flowers visited {n_fl}/{len(res['flowers'])}"
        cv2.rectangle(fr, (0, 0), (size[0], 44), (0, 0, 0), -1)
        cv2.putText(fr, hud, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(fr.tobytes())
    ff.stdin.close()
    ff.wait()
    cap.release()
    return Path(out)
