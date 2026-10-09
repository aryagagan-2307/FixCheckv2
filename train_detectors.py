"""Train one YOLO model per category in Google Colab (free GPU). Run each block as its own cell.

Pothole, garbage and waterlogging: the app measures how much of the photo the problem covers,
so SEGMENTATION datasets (outlines, "yolov8-seg") are best; plain box datasets also work.
Streetlight: you need a dataset with classes for lamp ON and lamp OFF (class names containing
off / broken / dark / unlit count as the problem). If you cannot find one, skip it: the app
still scores streetlights from CLIP + how dark the photo is.

Output for each: backend/weights/<category>.pt   (pothole.pt, garbage.pt, waterlogging.pt, streetlight.pt)
Any category without a file keeps working with CLIP only.

Steps per category:
 1. Colab: Runtime -> Change runtime type -> GPU.
 2. roboflow.com (free) -> Universe -> search the term below -> pick one with a licence you can use
    and at least ~500 images -> Download -> YOLOv8 format -> "show download code" -> paste it.
 3. Write the dataset name, link and licence in the README Disclosure section.
Search terms:  pothole | garbage OR waste OR litter | flood OR "standing water" | "street light" OR "street lamp"
"""

# --- cell 1 ---
# !pip install ultralytics roboflow

# --- cell 2: paste YOUR Roboflow download snippet here (it sets `dataset`) ---
# dataset = project.version(1).download("yolov8")      # use "yolov8" for boxes, or the segmentation export

# --- cell 3: train. Change CATEGORY and the base model for each run ---
# from ultralytics import YOLO
# CATEGORY = "garbage"                         # pothole | garbage | waterlogging | streetlight
# model = YOLO("yolov8n-seg.pt")               # segmentation data. For box data use "yolov8n.pt"
# model.train(data=f"{dataset.location}/data.yaml", epochs=40, imgsz=640, batch=16)
# print(model.val())                           # look at mAP50: above ~0.5 is workable, below 0.3 needs better data

# --- cell 4: sanity check on photos the model has never seen ---
# m = YOLO("runs/segment/train/weights/best.pt")   # runs/detect/... for box models
# m("your_test_photo.jpg", conf=0.25, save=True)   # open the saved image and look at it

# --- cell 5: download, rename to <category>.pt, put in backend/weights/ ---
# from google.colab import files
# import shutil; shutil.copy("runs/segment/train/weights/best.pt", f"{CATEGORY}.pt")
# files.download(f"{CATEGORY}.pt")
