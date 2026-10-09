"""Measures how close the app's severity is to YOUR judgement. This is how you tune it.

1. Put 15-30 photos per category in backend/eval/photos/ (real ones, a mix of mild and bad).
2. Create backend/eval/labels.csv:
       file,category,severity
       pothole_01.jpg,pothole,3
       garbage_04.jpg,garbage,1
       ...
   severity = your rating: 0 = problem not there / fixed, 1 = minor, 2 = serious, 3 = severe.
   (Ask a teammate to rate them too, and use the average. Rate BEFORE you look at the scores.)
3. Run:  python eval_severity.py

For each category it prints Spearman rank correlation (1.0 = same ordering as you; above ~0.7
is decent, below 0.5 means that category needs a trained detector or different anchors) and
the average gap. Adjust the `coverage` anchors in checks.py CFG, rerun, repeat.
"""
import csv
import sys
from pathlib import Path

import checks

HERE = Path(__file__).parent / "eval"


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    labels = HERE / "labels.csv"
    if not labels.exists():
        sys.exit(__doc__)
    checks.load_models()
    print("engines:", checks.info()["engines"], "\n")
    by_cat = {}
    for row in csv.DictReader(labels.open()):
        path = HERE / "photos" / row["file"]
        if not path.exists():
            print("missing", path)
            continue
        v = checks.verify(path, row["category"])
        human = float(row["severity"]) / 3
        by_cat.setdefault(row["category"], []).append((row["file"], human, v["severity"], v["prob"]))
        print(f"{row['category']:<13}{row['file']:<28} you={human:.2f}  app={v['severity']:.2f}  presence={v['prob']:.2f}")
    print()
    for cat, rows in by_cat.items():
        h, a = [r[1] for r in rows], [r[2] for r in rows]
        mae = sum(abs(x - y) for x, y in zip(h, a)) / len(rows)
        miss = sum(1 for r in rows if (r[1] > 0) != (r[3] >= checks.THRESHOLD))
        print(f"{cat:<13} n={len(rows):<3} spearman={spearman(h, a):.2f}  avg gap={mae:.2f}  "
              f"present/absent wrong on {miss}/{len(rows)}")


if __name__ == "__main__":
    main()
