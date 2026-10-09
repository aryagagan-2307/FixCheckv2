"""Runs the whole demo flow without a browser.   Usage:  python smoke_test.py
Uses demo mode and a temporary data folder, so it never touches your real data."""
import os, sys, tempfile, random, io
os.environ["FIXCHECK_MODE"] = "demo"
os.environ["FIXCHECK_DATA"] = tempfile.mkdtemp()
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import main  # noqa: E402

def pic(seed):
    r = random.Random(seed)
    im = Image.new("RGB", (320, 240), (80,) * 3)
    d = ImageDraw.Draw(im)
    for _ in range(8):
        x, y = r.randint(0, 250), r.randint(0, 180)
        g = r.randint(20, 220)
        d.rectangle([x, y, x + r.randint(30, 120), y + r.randint(30, 100)], fill=(g, g, g))
    b = io.BytesIO(); im.save(b, "JPEG"); return b.getvalue()

LAT, LON = 13.0827, 80.2707
fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name, extra)
    fails += 0 if cond else 1

def file(name, seed): return {"photo": (name, pic(seed), "image/jpeg")}

with TestClient(main.app) as c:
    check("status reports demo mode", c.get("/api/status").json()["mode"] == "demo")

    r = c.post("/api/complaints", files=file("pothole_a.jpg", 1), data=dict(category="pothole", lat=LAT, lon=LON, near_school="true", ward="Ward 1")).json()
    check("1 genuine pothole accepted", r["accepted"] and not r["merged"]); cid = r["complaint"]["id"]

    r = c.post("/api/complaints", files=file("pothole_a_again.jpg", 1), data=dict(category="pothole", lat=LAT, lon=LON)).json()
    check("2 same photo twice rejected as reused", (not r["accepted"]) and r["code"] == "reused_photo")

    r = c.post("/api/complaints", files=file("random_thing.jpg", 2), data=dict(category="pothole", lat=LAT + 0.01, lon=LON)).json()
    check("3 non-pothole photo gives 'uncertain'", (not r["accepted"]) and r["code"] == "uncertain")

    r = c.post("/api/complaints", files=file("pothole_b.jpg", 3), data=dict(category="pothole", lat=LAT + 0.00018, lon=LON)).json()
    check("4 report 20 m away is merged", r["accepted"] and r["merged"] and r["complaint"]["duplicate_count"] == 1)

    created = c.get(f"/api/complaints/{cid}").json()["created_at"]

    r = c.post(f"/api/complaints/{cid}/close", files=file("fixed_c.jpg", 4), data=dict(lat=LAT + 0.01, lon=LON)).json()
    check("5 far-away closure reopens (location)", (not r["passed"]) and any(f["code"] == "location" for f in r["failures"]) and r["complaint"]["status"] == "reopened")
    check("  age was not reset", r["complaint"]["created_at"] == created)

    r = c.post(f"/api/complaints/{cid}/close", files=file("pothole_a_copy.jpg", 1), data=dict(lat=LAT, lon=LON)).json()
    codes = [f["code"] for f in r["failures"]]
    check("6 original photo reused as proof reopens", (not r["passed"]) and "same_as_original" in codes, codes)
    check("  reopen_count is 2 and escalated", r["complaint"]["reopen_count"] == 2 and "zonal" in r["complaint"]["escalation"])

    r = c.post(f"/api/complaints/{cid}/close", files=file("fixed_d.jpg", 5), data=dict(lat=LAT, lon=LON)).json()
    check("7 genuine fix accepted", r["passed"] and r["complaint"]["status"] == "closed_pending", r["failures"])

    r = c.post("/api/complaints", files=file("garbage_1.jpg", 6), data=dict(category="garbage", lat=LAT + 0.02, lon=LON)).json()
    gid = r["complaint"]["id"]
    r = c.post(f"/api/complaints/{gid}/close", files=file("fixed_c_again.jpg", 4), data=dict(lat=LAT + 0.02, lon=LON)).json()
    check("8 proof photo reused from another closure is caught", any(f["code"] == "reused" for f in r["failures"]), [f["code"] for f in r["failures"]])

    r = c.post(f"/api/complaints/{cid}/confirm", data=dict(confirmed="false"))
    check("9 disputing without a photo is refused", r.status_code == 400)
    r = c.post(f"/api/complaints/{cid}/confirm", data=dict(confirmed="true")).json()
    check("10 resident confirms, complaint closes", r["status"] == "closed")

    items = c.get("/api/complaints?rain=true").json()
    check("11 list sorted by priority, closed last", items[-1]["status"] == "closed" and all(items[i]["priority"] >= items[i + 1]["priority"] for i in range(len(items) - 2)))

    c.post("/api/demo/seed")
    items = c.get("/api/complaints").json()
    check("12 demo data loads (5 complaints)", len(items) == 5, [i["id"] for i in items])
    hs = [main.checks.hash_distance(a["photo"] and main.checks.photo_hash(main.UPLOADS / a["photo"]), main.checks.photo_hash(main.UPLOADS / b["photo"])) for i, a in enumerate(items) for b in items[i + 1:]]
    check("  seed photos are all different", min(hs) > main.checks.HASH_SAME, min(hs))
    check("  frontend is served", "FixCheck" in c.get("/").text)

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
