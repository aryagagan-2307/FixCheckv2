"""Demo data. Photos are generated placeholders, NOT real photos."""
import random
from datetime import datetime, timedelta

from PIL import Image, ImageDraw
from sqlmodel import Session, select

from db import UPLOADS, Attempt, Complaint, Event, engine
import checks

BASE = (13.0827, 80.2707)   # central Chennai


def make_image(name, label, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (640, 480), (70 + rnd.randint(0, 40),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(9):
        x, y = rnd.randint(0, 500), rnd.randint(0, 380)
        g = rnd.randint(30, 200)
        d.rectangle([x, y, x + rnd.randint(40, 200), y + rnd.randint(30, 160)], fill=(g, g, g))
    x, y = rnd.randint(150, 400), rnd.randint(120, 300)
    d.ellipse([x, y, x + rnd.randint(60, 150), y + rnd.randint(40, 100)], fill=(15, 15, 15))
    d.text((12, 12), f"PLACEHOLDER: {label}", fill=(242, 183, 5))
    img.save(UPLOADS / name, "JPEG")
    return name


def _add(session, *, cat, ward, dlat, dlon, school, age_hours, sev, dups, seed):
    photo = make_image(f"seed_{cat}_{seed}.jpg", cat, seed)
    c = Complaint(category=cat, ward=ward, lat=BASE[0] + dlat, lon=BASE[1] + dlon, near_school=school,
                  created_at=datetime.utcnow() - timedelta(hours=age_hours), photo=photo,
                  photo_hash=checks.photo_hash(UPLOADS / photo), confidence=0.9, severity=sev,
                  duplicate_count=dups)
    session.add(c)
    session.commit()
    session.refresh(c)
    session.add(Event(complaint_id=c.id, at=c.created_at, kind="filed",
                      text="Complaint filed (demo data)."))
    return c


def seed(session: Session):
    for model in (Event, Attempt, Complaint):
        for row in session.exec(select(model)).all():
            session.delete(row)
    session.commit()

    _add(session, cat="pothole", ward="Ward 12", dlat=0.002, dlon=0.001, school=True,
         age_hours=5 * 24, sev=1.0, dups=2, seed=1)
    _add(session, cat="garbage", ward="Ward 12", dlat=-0.004, dlon=0.003, school=False,
         age_hours=2 * 24, sev=0.6, dups=0, seed=2)
    _add(session, cat="streetlight", ward="Ward 12", dlat=0.006, dlon=-0.002, school=False,
         age_hours=24, sev=0.4, dups=1, seed=3)
    _add(session, cat="waterlogging", ward="Ward 7", dlat=-0.01, dlon=-0.008, school=False,
         age_hours=3, sev=0.8, dups=0, seed=4)

    # A complaint that was "closed" once from the wrong place, and got reopened.
    c = _add(session, cat="pothole", ward="Ward 31", dlat=0.012, dlon=0.009, school=False,
             age_hours=6 * 24, sev=0.6, dups=3, seed=5)
    proof = make_image("seed_proof_5.jpg", "claimed fix", 55)
    c.status, c.reopen_count, c.model_mismatch = "reopened", 1, False
    session.add(c)
    session.add(Attempt(complaint_id=c.id, at=datetime.utcnow() - timedelta(days=1), photo=proof,
                        photo_hash=checks.photo_hash(UPLOADS / proof), lat=c.lat + 0.02, lon=c.lon,
                        passed=False,
                        failures='[{"code":"location","text":"Closure photo was taken 2.2 km from the original complaint."}]',
                        warnings="[]"))
    session.add(Event(complaint_id=c.id, at=datetime.utcnow() - timedelta(days=1), kind="rejected",
                      text="Closure rejected: photo location does not match. Reopened. Supervisor notified, 48 hours to show proof."))
    session.commit()
