import sys, csv, json, random, zipfile, unicodedata, hashlib, shutil
from pathlib import Path
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

VL, VS, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
EVAL_PLACES = ("모래고개입구", "장현동")
N_YES, N_NO, SIDE, SEED = 400, 400, 360, 42
SMOKE = {"흑색연기", "백색/회색연기"}
NEG_OBJ = {"구름", "안개/연무", "굴뚝연기"}
rng = random.Random(SEED)

def nfc(s):
    return unicodedata.normalize("NFC", s)

def real_name(i):
    if i.flag_bits & 0x800:
        return i.filename
    raw = i.filename.encode("cp437", errors="ignore")
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return i.filename

def bbox_of(a):
    pts = []
    def walk(x):
        if isinstance(x, (int, float)):
            pts.append(float(x))
        elif isinstance(x, list):
            for y in x:
                walk(y)
    walk(a.get("segmentation"))
    if len(pts) >= 6:
        return min(pts[0::2]), min(pts[1::2]), max(pts[0::2]), max(pts[1::2])
    b = a.get("bbox")
    if b and len(b) == 4:
        return b[0], b[1], b[0] + b[2], b[1] + b[3]
    return None

def biggest(lab, names):
    cat = {c["id"]: c["name"] for c in lab.get("categories", [])}
    best = None
    for a in lab.get("annotations") or []:
        if cat.get(a.get("category_id")) not in names:
            continue
        b = bbox_of(a)
        if not b or min(b[2] - b[0], b[3] - b[1]) < 24:
            continue
        if best is None or (b[2] - b[0]) * (b[3] - b[1]) > (best[2] - best[0]) * (best[3] - best[1]):
            best = b
    return best

def crop(im, b):
    W, H = im.size
    s = max(max(b[2] - b[0], b[3] - b[1]) * 1.4, 64)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    c = im.crop((max(0, cx - s / 2), max(0, cy - s / 2), min(W, cx + s / 2), min(H, cy + s / 2)))
    c.thumbnail((SIDE, SIDE))
    return c

def split_of(vid):
    return "val" if int(hashlib.md5("_".join(vid).encode()).hexdigest(), 16) % 100 < 15 else "train"

eval_rows = list(csv.DictReader(open("docs/eval_set.csv", encoding="utf-8-sig")))
eval_names = {r["file_name"]: r for r in eval_rows}
pos, neg, eval_lab = {}, {}, {}
with zipfile.ZipFile(VL) as zf:
    for info in zf.infolist():
        name = nfc(real_name(info).split("/")[-1])
        if not name.endswith(".json"):
            continue
        p = name.split("_")
        if len(p) < 7:
            continue
        is_eval = name in eval_names
        if not is_eval and any(p[2].endswith(e) for e in EVAL_PLACES):
            continue
        lab = json.loads(zf.read(info).decode("utf-8"))
        if is_eval:
            eval_lab[name] = lab
            continue
        if (lab.get("environment") or {}).get("device") != "drone":
            continue
        vid = (p[2], p[3], p[4])
        if p[0] == "PositiveRealDB":
            b = biggest(lab, SMOKE)
            if b:
                pos.setdefault(vid, []).append((name, b))
        elif p[0] == "NegativeDB":
            neg.setdefault(vid, []).append((name, biggest(lab, NEG_OBJ)))
if not eval_lab:
    sys.exit("평가셋 라벨을 못 찾았어요. 여기서 멈출게요")
print(f"연기 영상 {len(pos)}개, 오탐 유발 영상 {len(neg)}개, 평가셋 라벨 {len(eval_lab)}/{len(eval_rows)}")

def pick(groups, n, per_vid):
    vids = sorted(groups)
    rng.shuffle(vids)
    out = []
    for k in range(per_vid):
        for v in vids:
            files = sorted(groups[v])
            if k < len(files) and len(out) < n:
                out.append((v, rng.choice(files) if k == 0 else files[k]))
    return out

yes_pick = pick(pos, N_YES, 2)
no_pick = pick(neg, N_NO, 3)
model = YOLO("best.pt")
vs = zipfile.ZipFile(VS)
jpg = {nfc(real_name(i).split("/")[-1]): i for i in vs.infolist() if i.filename.lower().endswith(".jpg")}
if OUT.exists():
    shutil.rmtree(OUT)
meta, missing = [], 0

def open_img(json_name):
    i = jpg.get(json_name[:-5] + ".jpg")
    return Image.open(BytesIO(vs.read(i))).convert("RGB") if i else None

def neg_box(im, label_box):
    r = model.predict(im, imgsz=960, conf=0.02, verbose=False)[0]
    if len(r.boxes):
        k = int(r.boxes.conf.argmax())
        return r.boxes.xyxy[k].tolist(), "탐지오인"
    if label_box:
        return label_box, "라벨물체"
    W, H = im.size
    s = rng.uniform(0.1, 0.3) * min(W, H)
    x, y = rng.uniform(0, W - s), rng.uniform(0, H - s)
    return (x, y, x + s, y + s), "무작위"

def save(split, label, name, im, box, source, vid):
    d = OUT / split / ("yes" if label else "no")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{len(meta):05d}.jpg"
    crop(im, box).save(f, quality=92)
    meta.append([split, label, f"{split}/{d.name}/{f.name}", source, "_".join(vid), name[:-5] + ".jpg"])

for n, (vid, (name, box)) in enumerate(yes_pick, 1):
    im = open_img(name)
    if im is None:
        missing += 1
        continue
    save(split_of(vid), 1, name, im, box, "연기라벨", vid)
    if n % 100 == 0:
        print(f"  연기 조각 {n}/{len(yes_pick)}")
for n, (vid, (name, lbox)) in enumerate(no_pick, 1):
    im = open_img(name)
    if im is None:
        missing += 1
        continue
    box, src = neg_box(im, lbox)
    save(split_of(vid), 0, name, im, box, src, vid)
    if n % 100 == 0:
        print(f"  오탐 조각 {n}/{len(no_pick)}")
for r in eval_rows:
    lab, im = eval_lab.get(r["file_name"]), open_img(r["file_name"])
    if im is None or lab is None:
        missing += 1
        continue
    vid = (r["place"], r["camera_id"], r["video_id"])
    if r["smoke"] == "1":
        b = biggest(lab, SMOKE)
        if b is None:
            b, _ = neg_box(im, None)
        save("eval", 1, r["file_name"], im, b, "연기라벨", vid)
    else:
        b, src = neg_box(im, biggest(lab, NEG_OBJ))
        save("eval", 0, r["file_name"], im, b, src, vid)

with open(OUT / "meta.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["split", "label", "file", "source", "video", "original"])
    w.writerows(meta)
from collections import Counter
c = Counter((m[0], m[1]) for m in meta)
s = Counter((m[0], m[3]) for m in meta if m[1] == 0)
for sp in ("train", "val", "eval"):
    print(f"{sp:6s} 연기 {c[(sp, 1)]:4d}  오탐 {c[(sp, 0)]:4d}")
print("오탐 조각 출처:", dict(s))
print("사진 못 찾음:", missing)
print("압축 파일:", shutil.make_archive(str(OUT), "zip", root_dir=OUT))
