# Step-by-step: run my pretrained model comparison

Follow these in order. Everything runs in Google Colab — nothing needs to be
installed on my laptop. Total hands-on time is about 20 minutes, plus 1-2 hours
of waiting while it trains.

---

## Step 1 — Open Colab and turn on the GPU

1. Go to https://colab.research.google.com
2. **File → New notebook**
3. **Runtime → Change runtime type**
4. Under "Hardware accelerator" pick **T4 GPU**, then **Save**

This matters. On CPU this experiment takes most of a day; on the T4 it's
roughly an hour.

Check it worked — paste into the first cell and press Shift+Enter:

```python
!nvidia-smi
```

If a table appears showing "Tesla T4", the GPU is on. If it says
"command not found", the runtime type didn't save — redo step 3.

---

## Step 2 — Install the library

New cell, Shift+Enter:

```python
!pip install ultralytics -q
```

Takes about a minute. A warning about restarting the runtime can be ignored.

---

## Step 3 — Get my code

```python
!git clone -b claude/pretrained-models-types-qeivhm https://github.com/rishi902/lesson20.git
%cd lesson20
```

---

## Step 4 — Download the dataset from Roboflow

The dataset is not in the repo (885 images is too big for git, and it isn't
mine to redistribute). Download it fresh:

1. Go to https://universe.roboflow.com/aiorbits/bulky-waste-2-0-ifbkq/dataset/26
2. Click **Download this Dataset**
3. Format: **YOLOv8**
4. Choose **Show download code**, and copy the snippet it gives me

It looks like this — paste my own version (with my own API key) into a cell:

```python
!pip install roboflow -q
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_MY_KEY_HERE")
project = rf.workspace("aiorbits").project("bulky-waste-2-0-ifbkq")
dataset = project.version(26).download("yolov8")
print(dataset.location)
```

That prints a path like `/content/lesson20/Bulky-Waste-2-0-26`. Note it down —
Step 5 needs it.

---

## Step 5 — Point the script at the dataset

Roboflow's download includes its own `data.yaml` with the correct paths
already filled in. Use that one rather than the copy in my repo:

```python
DATA = dataset.location + "/data.yaml"
print(open(DATA).read())
```

The printed output should list the 5 class names
(`bin_empty`, `bin_full`, `bulk_waste`, `normal_waste`, `vehicle`) and show
`train:` / `val:` / `test:` paths. If the class names are there, I'm good.

---

## Step 6 — Quick sanity check before the long run

Don't launch the full experiment blind. Run one model for one epoch first — it
takes about 2 minutes and catches a broken dataset path immediately:

```python
!python compare_pretrained.py --data "$DATA" --epochs 1 --models yolov8n.pt
```

Wait — `$DATA` is a Python variable, so pass it properly:

```python
!python compare_pretrained.py --data {DATA} --epochs 1 --models yolov8n.pt
```

If this finishes and prints a small results table, everything is wired up
correctly. If it errors about missing images, the path from Step 4 is wrong.

---

## Step 7 — The real run

```python
!python compare_pretrained.py --data {DATA} --epochs 10
```

This trains all 5 models back to back. Expect roughly 1-2 hours on a T4.

**Keep the browser tab open.** Colab disconnects idle sessions and I lose the
run. The script writes `results/comparison.csv` after every single model, so
even if it does disconnect, whatever finished is saved.

If it's taking too long, cancel and run the shorter version — four models
instead of five, dropping the slowest:

```python
!python compare_pretrained.py --data {DATA} --epochs 10 \
  --models yolov8n.pt yolov8s.pt yolo11n.pt yolo11s.pt
```

---

## Step 8 — Get my results out

```python
import pandas as pd
df = pd.read_csv("results/comparison.csv")
df
```

Then download both files to my computer:

```python
from google.colab import files
files.download("results/comparison.csv")
files.download("results/comparison.png")
```

The winning model is the top row (the table is sorted by mAP50-95).

---

## Step 9 — Train the final production model

Now that I know the winner, train just that one properly — longer, for the real
numbers the team reports:

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")   # <-- replace with whichever model actually won
model.train(data=DATA, epochs=100, batch=32, lr0=0.001,
            optimizer="AdamW", imgsz=640, patience=20,
            name="final_production")
```

`patience=20` stops early if validation performance stops improving for 20
epochs, which prevents wasting time and guards against overfitting.

Then evaluate on the **test** split — the one no training or model selection
touched, so it's the honest number:

```python
metrics = model.val(data=DATA, split="test")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
```

Download the trained weights and the charts Ultralytics generates:

```python
files.download("runs/detect/final_production/weights/best.pt")
files.download("runs/detect/final_production/results.png")
files.download("runs/detect/final_production/confusion_matrix.png")
```

The confusion matrix is worth keeping — the report asks for error analysis, and
it will show directly whether `bin_empty` and `bin_full` are being confused.

---

## Step 10 — Get some prediction images for the report

Section 5 of the report asks for qualitative results — pictures of the model
working, and pictures of it failing:

```python
model.predict(source=dataset.location + "/test/images", save=True, conf=0.4)
```

Saved images land in `runs/detect/predict/`. Browse them in Colab's file panel
(folder icon on the left), pick 2-3 good detections and 1-2 obvious mistakes,
and download them. The failures are worth more marks than the successes — the
rubric explicitly asks what didn't work and why.

---

## Step 11 — Write it up

Open `report_section_draft.md` in the repo and fill in every `___` from
`comparison.csv`. Paste the finished text into the team doc under sections 4
and 5, and fill in the blank **Base Weights** line with whichever model won.

---

## If something breaks

| Error | Fix |
|---|---|
| `CUDA out of memory` | Drop `yolov8m.pt` from `--models`, or add `--epochs 5` |
| `dataset not found` | The `--data` path is wrong; re-run Step 5 and check the printed yaml |
| Colab disconnected mid-run | `results/comparison.csv` still has the models that finished; re-run only the missing ones via `--models` |
| `nvidia-smi: not found` | GPU isn't on; redo Step 1 |
