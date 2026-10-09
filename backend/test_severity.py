"""Checks the severity logic WITHOUT downloading any model (CLIP and YOLO are replaced by stand-ins).
Usage:  python test_severity.py
This proves the maths and wiring. It does NOT prove the real models are accurate:
for that, use eval_severity.py with photos you have rated yourself."""
import os, tempfile
os.environ["FIXCHECK_MODE"] = "demo"
import numpy as np
from PIL import Image
import checks

fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name, extra)
    fails += 0 if cond else 1

tmp = tempfile.mkdtemp()
def img(name, grey):
    p = os.path.join(tmp, name); Image.new("RGB", (320, 240), (grey,) * 3).save(p); return p

# ---- CLIP ordinal scoring: bigger problem descriptions winning => higher severity
lv = checks.CFG["garbage"]["levels"]            # 5 levels, 2+1+1+1+1 = 6 prompts
def logits_for(winner):                          # winner = level index that CLIP likes best
    out = []
    for i, (_, _, prompts) in enumerate(lv):
        out += [20.0 if i == winner else 10.0] * len(prompts)
    return out
r = [checks.ordinal_scores(logits_for(i), lv) for i in range(5)]
check("clean photo -> prob ~0, severity 0", r[0]["prob"] < 0.01 and r[0]["severity"] == 0.0, r[0])
check("severity rises with each level", r[1]["severity"] < r[2]["severity"] < r[3]["severity"] < r[4]["severity"],
      [round(x["severity"], 2) for x in r[1:]])
check("problem levels have prob ~1", all(x["prob"] > 0.99 for x in r[1:]))
check("a confident tiny-litter photo is NOT high severity", r[1]["severity"] < 0.3, round(r[1]["severity"], 2))

# ---- YOLO coverage: overlapping boxes counted once, masks supported
b = [(0, 0, 100, 100), (50, 50, 150, 150)]       # in a 200x200 photo: union = 17500/40000
f = checks._union_fraction(None, b, (200, 200))
check("box union counts overlap once", abs(f - 0.4375) < 0.03, round(f, 3))
m = np.zeros((50, 50)); m[:25, :] = 1            # top half
check("mask coverage", abs(checks._union_fraction([m], [(0, 0, 200, 200)], (200, 200)) - 0.5) < 0.02)
sev = lambda c, cov, n=1: checks._coverage_severity(c, cov, n)
check("pothole severity grows with size", sev("pothole", 0.005) < sev("pothole", 0.05) < sev("pothole", 0.25))
check("more spots = slightly worse", sev("pothole", 0.05, 3) > sev("pothole", 0.05, 1))
check("severity capped at 1", sev("pothole", 0.9, 9) <= 1.0)

# ---- darkness for streetlights
check("darker night photo = worse", checks._darkness_severity(10) > checks._darkness_severity(70) > checks._darkness_severity(105))

# ---- whole pipeline with stand-in models
class Arr(list):
    def tolist(self): return list(self)
class Boxes:
    def __init__(s, xyxy, conf, cls): s.xyxy, s.conf, s.cls = Arr(xyxy), Arr(conf), Arr(cls)
class Res:
    def __init__(s, boxes, names): s.boxes, s.names, s.orig_shape, s.masks = boxes, names, (240, 320), None
class FakeYolo:
    def __init__(s, res): s.res = res
    def __call__(s, *a, **k): return [s.res]

checks._state["clip"] = object()
checks._state["proc"] = object()

for cat, win in (("pothole", 2), ("garbage", 4), ("waterlogging", 1)):
    levels = checks.CFG[cat]["levels"]
    def fl(path, texts, levels=levels, win=win):
        return [20.0 if i == win else 10.0 for i, (_, _, ps) in enumerate(levels) for _ in ps]
    checks._clip_logits = fl
    checks._state["yolo"] = {}
    out = checks.verify(img(cat + ".jpg", 90), cat)
    check(f"{cat}: CLIP-only gives prob, severity, summary", out["prob"] > .9 and 0 < out["severity"] <= 1 and out["measures"]["summary"], (out["severity"], out["measures"]["summary"]))

# YOLO + CLIP: pothole big vs small
checks._state["yolo"] = {"pothole": FakeYolo(Res(Boxes([(0, 0, 160, 120)], [.8], [0]), {0: "pothole"}))}
big = checks.verify(img("p1.jpg", 90), "pothole")
checks._state["yolo"] = {"pothole": FakeYolo(Res(Boxes([(0, 0, 16, 12)], [.8], [0]), {0: "pothole"}))}
small = checks.verify(img("p2.jpg", 90), "pothole")
check("big pothole scores higher than small one", big["severity"] > small["severity"], (big["severity"], small["severity"]))
check("engine reported as yolo+clip", big["engine"] == "yolo+clip")
checks._state["yolo"] = {"pothole": FakeYolo(Res(Boxes([], [], []), {0: "pothole"}))}
none = checks.verify(img("p3.jpg", 90), "pothole")
check("detector sees nothing -> prob 0, severity 0 (fix accepted)", none["prob"] == 0 and none["severity"] == 0)

# streetlight: dark night photo worse than dim-but-visible; daytime gets a warning
checks._state["yolo"] = {}
lv = checks.CFG["streetlight"]["levels"]
checks._clip_logits = lambda p, t: [20.0 if i == 2 else 10.0 for i, (_, _, ps) in enumerate(lv) for _ in ps]
dark = checks.verify(img("s_dark.jpg", 12), "streetlight")
dim = checks.verify(img("s_dim.jpg", 75), "streetlight")
day = checks.verify(img("s_day.jpg", 200), "streetlight")
check("darker streetlight scene = higher severity", dark["severity"] > dim["severity"], (dark["severity"], dim["severity"]))
check("daytime streetlight photo carries a warning", day["measures"]["notes"] and "darkness" not in [s["name"] for s in day["measures"]["signals"]])
# streetlight detector with on/off classes
checks._state["yolo"] = {"streetlight": FakeYolo(Res(Boxes([(10, 10, 30, 90), (200, 10, 220, 90)], [.7, .8], [0, 1]), {0: "lamp_off", 1: "lamp_on"}))}
two = checks.verify(img("s_two.jpg", 40), "streetlight")
check("streetlight detector: 1 of 2 lamps off", two["measures"]["lamps_off"] == 1 and two["measures"]["lamps_on"] == 1, two["measures"]["summary"])

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
raise SystemExit(1 if fails else 0)
