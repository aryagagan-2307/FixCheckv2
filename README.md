# FixCheck (prototype)

Verifies that a closed civic complaint is actually fixed. Built for GENCRAFT 2K26, SDG 16.

One command runs everything: the API, the database and the web pages.

## Run it (5 minutes, no ML needed)

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open http://localhost:8000, go to **Officer dashboard**, press **Load demo data**.

Check that everything works without a browser: `pip install httpx` then `python smoke_test.py` (should end with ALL PASSED).

To try it on a phone on the same Wi-Fi: `uvicorn main:app --host 0.0.0.0 --port 8000`, then open `http://<your-laptop-ip>:8000`. The camera button works over http. The "Use my location" button needs https, so type the coordinates instead.

## Demo mode vs live models

The yellow badge at the top shows which mode you are in.

- **DEMO MODE** (default until you install the models): no ML runs. The app decides from the photo's FILE NAME. A name containing `fixed`, `clean`, `repaired` or `after` counts as "problem not visible"; `random`, `blurry`, `unclear` or `notpothole` counts as "not the problem"; anything else counts as "problem visible". Everything else (reused photos, location, duplicates, priority, reopening) is real logic.
- **Live models:** `pip install -r requirements-ml.txt`, run once with internet so CLIP downloads (about 600 MB, then cached).

### How the priority score is built from the photo

Every photo becomes a **severity from 0 to 1** (40% of the priority score; the rest is school, duplicates, age, rain). Severity blends up to three signals, and the officer sees the reasons as chips (e.g. "Pothole: 2 found, 9% of the photo, serious"):

| Category | Signal 1: detector (optional, you train it) | Signal 2: CLIP, no training | Signal 3 |
|---|---|---|---|
| Pothole | how much of the photo the potholes cover, and how many | graded descriptions: smooth / crack / pothole / large / crater | - |
| Garbage | how much of the photo the garbage covers | clean / litter / pile / heap / overflowing | - |
| Waterlogging | how much of the photo the water covers | dry / puddle / shallow / deep / vehicles stuck | - |
| Streetlight | lamps OFF vs ON in the photo | lit / dim / switched off / fully dark | how dark the photo is (night photos only) |

- With **no detector** (`backend/weights/<category>.pt` missing) a category uses CLIP alone. Every category works that way out of the box.
- With a detector, the detector decides *whether the problem is there*, and severity is 70% detector + 30% CLIP.
- A **streetlight cannot be judged from a daytime photo**. The app warns and the report page tells residents to photograph it after dark.
- Train detectors with `train_detectors.py` (Colab).

### Check and tune the accuracy (do this before the demo)

```bash
python test_severity.py      # logic test, no models needed: should end with ALL PASSED
python eval_severity.py      # compares the app with YOUR ratings of real photos (instructions at the top of the file)
```

The numbers in `CFG` in `checks.py` (the `coverage` anchors) are starting guesses, not measured values. `eval_severity.py` shows how far off they are on your photos; adjust, rerun, repeat. Quote the Spearman score from your own test photos in the pitch, not "accurate".

Do not present demo mode as real AI. If the models fail on the day, say so honestly and show the badge.

## The 5-minute demo (in this order)

1. **Report** a pothole photo (name it `pothole_1.jpg`). It is verified and gets a priority.
2. Report the **same file again**: rejected as a reused photo.
3. Report a **second photo 20 m away** (change the latitude by 0.0002): merged as a duplicate.
4. **Dashboard**: tick "Rain expected" and watch priorities change. Open the pothole.
5. **Close with proof**, press "Simulate: photo taken far away": rejected, reopened, supervisor notified, age keeps counting.
6. **Close with the original photo**: rejected (same photo as the original).
7. **Close with a new photo** named `fixed_1.jpg` at the right location: accepted, waiting for the resident.
8. **Resident confirms**, or disputes with a photo.

For a stronger start, press "Load demo data" first. It includes a complaint that was already falsely closed once.

## What is real and what is not

Real in this prototype: in-order verification pipeline, perceptual-hash reuse detection (including photos reused across other closures), location check, 50 m duplicate merging, transparent priority score with reasons, quiet/busy ward adjustment, automatic reopening that never resets the age, escalation ladder text, right to dispute with photo evidence, "Disputed" label only after two disputes or one plus a model mismatch, server-side timestamps.

Simplified or not built (say these as future scope):
- Escalation is shown as text only. No real notifications.
- Severity is only as good as the models and your test photos. Without trained detectors it comes from CLIP's graded descriptions, which is far better than before but not measured until you run `eval_severity.py`.
- Existing complaints made before this update have no photo-reasons (their `measures` is empty). Delete `backend/data/` for a clean start.
- No gallery-upload blocking, mock-GPS detection or on-device face/plate blurring. The camera button is a browser file input.
- No WhatsApp or call fallback, no login, no auto-RTI draft, no public false-closure report by ward, no real weather feed (the "Rain expected" box is a toggle).
- SQLite instead of PostgreSQL/PostGIS.
- The demo photos are generated placeholders, not real images.

## Disclosure (the rules ask for this)

Fill in before submitting:
- AI models: OpenAI CLIP (`openai/clip-vit-base-patch32`, via Hugging Face Transformers); Ultralytics YOLOv8 fine-tuned by us for each category you trained (note: Ultralytics is AGPL-3.0)
- Datasets: *(one line per trained category: Roboflow dataset name, link and licence)*
- AI assistants used while building: *(list)*
- Libraries: FastAPI, SQLModel, Pillow, ImageHash

## Files

```
backend/main.py        API endpoints and the closure/duplicate logic
backend/checks.py      severity models, hashing, distance, priority score
backend/weights/       your trained <category>.pt files go here
backend/test_severity.py   logic tests, no models needed
backend/eval_severity.py   scores the app against your own photo ratings
backend/db.py          database tables
backend/seed.py        demo data
backend/smoke_test.py  end-to-end test, no browser
frontend/index.html    the whole web app
train_detectors.py     Colab steps for the YOLO models (all four categories)
```
Data (database and photos) is stored in `backend/data/`. Delete that folder to reset everything.
