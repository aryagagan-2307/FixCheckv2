import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

import checks
from db import UPLOADS, Attempt, Complaint, Event, engine, init_db
from seed import seed

DUP_RADIUS_KM = 0.05        # same spot = within 50 m
LOCATION_TOLERANCE_KM = 0.05
FAST_CLOSE_SECONDS = 60     # warning only
ACTIVE = ("open", "reopened", "closed_pending")
LABELS = {"open": "Open", "reopened": "Reopened", "closed_pending": "Fix submitted, awaiting resident",
          "closed": "Closed and confirmed"}
ESCALATION = {1: "Officer and supervisor notified. 48 hours to show proof.",
              2: "Escalated to the zonal officer.",
              3: "Escalated to the deputy commissioner."}


@asynccontextmanager
async def lifespan(app):
    init_db()
    checks.load_models()
    # warm-up so the first demo request is not slow
    yield


app = FastAPI(title="FixCheck prototype", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------
def now():
    return datetime.utcnow()


def iso(dt):
    return dt.isoformat() + "Z"


async def save_upload(f: UploadFile) -> str:
    name = f"{uuid.uuid4().hex[:8]}_{checks.safe_name(f.filename)}"
    (UPLOADS / name).write_bytes(await f.read())
    return name


def drop(name):
    (UPLOADS / name).unlink(missing_ok=True)


def valid_image(name):
    try:
        checks.photo_hash(UPLOADS / name)
        return True
    except Exception:
        return False


def ward_factors(session):
    counts = {}
    for c in session.exec(select(Complaint).where(Complaint.status.in_(ACTIVE))).all():
        if c.ward:
            counts[c.ward] = counts.get(c.ward, 0) + 1
    if len(counts) < 2:
        return {}
    mean = sum(counts.values()) / len(counts)
    return {w: max(0.8, min(1.25, mean / n)) for w, n in counts.items()}


def to_dict(c, session, rain=False, factors=None, detail=False):
    age_h = (now() - c.created_at).total_seconds() / 3600
    score, reasons = checks.priority(c.severity, c.near_school, c.duplicate_count, age_h / 24, rain,
                                     (factors or {}).get(c.ward, 1.0))
    measures = json.loads(c.measures or "{}")
    # what the photo models saw goes first, so the officer sees WHY the severity is what it is
    reasons = ([measures["summary"]] if measures.get("summary") else []) + measures.get("notes", []) + reasons
    d = {
        "id": c.id, "category": c.category, "ward": c.ward, "lat": c.lat, "lon": c.lon,
        "near_school": c.near_school, "status": c.status, "status_label": LABELS[c.status],
        "created_at": iso(c.created_at), "age_hours": round(age_h, 2),
        "duplicate_count": c.duplicate_count, "reopen_count": c.reopen_count,
        "dispute_count": c.dispute_count,
        "public_disputed": c.dispute_count >= 2 or (c.dispute_count >= 1 and c.model_mismatch),
        "confidence": round(c.confidence, 2), "severity": c.severity,
        "priority": score, "reasons": reasons, "photo": c.photo, "measures": measures,
        "escalation": ESCALATION.get(min(c.reopen_count, 3)) if c.reopen_count else None,
    }
    if detail:
        atts = session.exec(select(Attempt).where(Attempt.complaint_id == c.id).order_by(Attempt.at)).all()
        evs = session.exec(select(Event).where(Event.complaint_id == c.id).order_by(Event.at)).all()
        d["attempts"] = [{"at": iso(a.at), "photo": a.photo, "passed": a.passed,
                          "failures": json.loads(a.failures), "warnings": json.loads(a.warnings)} for a in atts]
        d["events"] = [{"at": iso(e.at), "kind": e.kind, "text": e.text, "photo": e.photo} for e in evs]
        d["dispute_photo"] = next((e.photo for e in reversed(evs) if e.kind == "disputed"), None)
    return d


def get_or_404(session, cid):
    c = session.get(Complaint, cid)
    if not c:
        raise HTTPException(404, "Complaint not found")
    return c


def photo_already_used(session, h, exclude_complaint_original=None):
    """Return text if this photo matches any stored complaint or closure photo."""
    for c in session.exec(select(Complaint)).all():
        if c.id != exclude_complaint_original and checks.hash_distance(h, c.photo_hash) <= checks.HASH_SAME:
            return f"complaint #{c.id}"
    for a in session.exec(select(Attempt)).all():
        if checks.hash_distance(h, a.photo_hash) <= checks.HASH_SAME:
            return f"a closure on complaint #{a.complaint_id}"
    return None


def reopen(session, c, text):
    c.status = "reopened"
    c.reopen_count += 1       # created_at is NOT touched: the age keeps running
    esc = ESCALATION.get(min(c.reopen_count, 3))
    if c.reopen_count >= 4:
        esc = "Escalated to the commissioner. Repeated false closure flagged."
    session.add(Event(complaint_id=c.id, at=now(), kind="rejected", text=f"{text} Reopened. {esc}"))


# ---------- API ----------
@app.get("/api/status")
def status():
    return checks.info()


@app.post("/api/complaints")
async def create_complaint(photo: UploadFile = File(...), category: str = Form(...), lat: float = Form(...),
                           lon: float = Form(...), near_school: bool = Form(False), ward: str = Form("")):
    if category not in checks.CATEGORIES:
        raise HTTPException(400, "Unknown category")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "Invalid coordinates")
    name = await save_upload(photo)
    if not valid_image(name):
        drop(name)
        raise HTTPException(400, "That file is not a readable image")
    path = UPLOADS / name
    h = checks.photo_hash(path)
    with Session(engine) as s:
        used = photo_already_used(s, h)
        if used:
            drop(name)
            return {"accepted": False, "code": "reused_photo",
                    "message": f"This photo was already submitted as {used}. Please take a new photo in the app."}
        v = checks.verify(path, category)
        if v["prob"] < checks.THRESHOLD:
            drop(name)
            return {"accepted": False, "code": "uncertain", "confidence": round(v["prob"], 2),
                    "message": f"We could not clearly see a {category} in this photo "
                               f"({v['prob']*100:.0f}% confidence). Please retake it closer to the problem."}
        # duplicate: same category, within 50 m, still active
        for other in s.exec(select(Complaint).where(Complaint.category == category,
                                                    Complaint.status.in_(("open", "reopened")))).all():
            if checks.km(lat, lon, other.lat, other.lon) <= DUP_RADIUS_KM:
                other.duplicate_count += 1
                s.add(other)
                s.add(Event(complaint_id=other.id, at=now(), kind="duplicate", photo=name,
                            text=f"Another resident reported the same spot. Merged (now {other.duplicate_count + 1} reports)."))
                s.commit()
                s.refresh(other)
                return {"accepted": True, "merged": True, "complaint": to_dict(other, s, factors=ward_factors(s))}
        c = Complaint(category=category, ward=ward.strip(), lat=lat, lon=lon, near_school=near_school,
                      created_at=now(), photo=name, photo_hash=h, confidence=v["prob"], severity=v["severity"],
                      measures=json.dumps(v["measures"]))
        s.add(c)
        s.commit()
        s.refresh(c)
        s.add(Event(complaint_id=c.id, at=c.created_at, kind="filed",
                    text=f"Complaint filed. Evidence check: {category} seen, {v['prob']*100:.0f}% confidence ({v['engine']}). "
                         f"Severity {v['severity']:.2f}: {v['measures'].get('summary', '')}"))
        s.commit()
        return {"accepted": True, "merged": False, "complaint": to_dict(c, s, factors=ward_factors(s))}


@app.get("/api/complaints")
def list_complaints(rain: bool = False):
    with Session(engine) as s:
        f = ward_factors(s)
        items = [to_dict(c, s, rain, f) for c in s.exec(select(Complaint)).all()]
    items.sort(key=lambda d: (d["status"] == "closed", -d["priority"]))
    return items


@app.get("/api/complaints/{cid}")
def get_complaint(cid: int, rain: bool = False):
    with Session(engine) as s:
        return to_dict(get_or_404(s, cid), s, rain, ward_factors(s), detail=True)


@app.post("/api/complaints/{cid}/close")
async def close_complaint(cid: int, photo: UploadFile = File(...), lat: float = Form(...), lon: float = Form(...)):
    name = await save_upload(photo)
    if not valid_image(name):
        drop(name)
        raise HTTPException(400, "That file is not a readable image")
    path = UPLOADS / name
    h = checks.photo_hash(path)
    with Session(engine) as s:
        c = get_or_404(s, cid)
        if c.status not in ("open", "reopened"):
            drop(name)
            raise HTTPException(400, f"Complaint is '{c.status}', it cannot be closed now")
        failures, warnings = [], []

        dist = checks.km(c.lat, c.lon, lat, lon)
        if dist > LOCATION_TOLERANCE_KM:
            failures.append({"code": "location",
                             "text": f"Closure photo was taken {dist*1000:.0f} m from the original complaint."})
        if checks.hash_distance(h, c.photo_hash) <= checks.HASH_SAME:
            failures.append({"code": "same_as_original",
                             "text": "The closure photo is the same as the original complaint photo."})
        else:
            used = photo_already_used(s, h, exclude_complaint_original=c.id)
            if used:
                failures.append({"code": "reused", "text": f"This photo was already used for {used}."})
        v = checks.verify(path, c.category)
        if v["prob"] >= checks.THRESHOLD:
            failures.append({"code": "still_visible",
                             "text": f"The {c.category} is still visible in the closure photo "
                                     f"({v['prob']*100:.0f}% confidence)."})
        if (now() - c.created_at).total_seconds() < FAST_CLOSE_SECONDS:
            warnings.append("Closed unusually fast after filing.")

        s.add(Attempt(complaint_id=c.id, at=now(), photo=name, photo_hash=h, lat=lat, lon=lon,
                      passed=not failures, failures=json.dumps(failures), warnings=json.dumps(warnings)))
        if failures:
            c.model_mismatch = any(f["code"] == "still_visible" for f in failures)
            reopen(s, c, "Closure rejected: " + " ".join(f["text"] for f in failures))
        else:
            c.status = "closed_pending"
            s.add(Event(complaint_id=c.id, at=now(), kind="fix_submitted", photo=name,
                        text="Fix submitted with proof. All automatic checks passed. Waiting for the resident to confirm."))
        s.add(c)
        s.commit()
        s.refresh(c)
        return {"passed": not failures, "failures": failures, "warnings": warnings,
                "complaint": to_dict(c, s, factors=ward_factors(s), detail=True)}


@app.post("/api/complaints/{cid}/confirm")
async def confirm(cid: int, confirmed: bool = Form(...), photo: Optional[UploadFile] = File(None)):
    with Session(engine) as s:
        c = get_or_404(s, cid)
        if c.status != "closed_pending":
            raise HTTPException(400, "There is no submitted fix to confirm")
        if confirmed:
            c.status = "closed"
            s.add(Event(complaint_id=c.id, at=now(), kind="closed", text="Resident confirmed the fix. Closed."))
        else:
            if photo is None:
                raise HTTPException(400, "A photo is required to dispute a closure.")
            name = await save_upload(photo)
            if not valid_image(name):
                drop(name)
                raise HTTPException(400, "That file is not a readable image")
            v = checks.verify(UPLOADS / name, c.category)
            c.dispute_count += 1
            c.model_mismatch = c.model_mismatch or v["prob"] >= checks.THRESHOLD
            s.add(Event(complaint_id=c.id, at=now(), kind="disputed", photo=name,
                        text=f"Resident disputed the fix with a photo (problem seen: {v['prob']*100:.0f}%)."))
            reopen(s, c, "Fix disputed by the resident.")
        s.add(c)
        s.commit()
        s.refresh(c)
        return to_dict(c, s, factors=ward_factors(s), detail=True)


@app.post("/api/demo/seed")
def demo_seed():
    with Session(engine) as s:
        seed(s)
    return {"ok": True}


# static files: photos, then the front-end at "/"
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")
app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="frontend")
