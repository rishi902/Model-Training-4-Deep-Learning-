# Bulky Waste Detection — Pretrained Model Comparison

My individual contribution to the C384 Final Assessment group project (Bulky
Waste 2.0 detector). The team's shared task is multi-class object detection on
CCTV frames for illegal bulky-waste dumping; my assigned experiment is
**comparing types of pretrained models**.

## What this experiment does

Everything except the starting weights is held constant — batch size 32,
AdamW, lr0 = 0.001, 640x640, same seed — so any difference in the results is
attributable to the pretrained checkpoint alone. The batch size comes from Wei
Kang's separate batch-size grid search, which found 32 optimal.

Candidates cover two axes:

| Axis | Checkpoints | Question it answers |
|---|---|---|
| Model size | `yolov8n` → `yolov8s` → `yolov8m` | Does more capacity help on an 885-image dataset, or does it overfit? |
| Architecture family | `yolov8n/s` vs `yolo11n/s` | Does the newer YOLO11 backbone beat YOLOv8 at comparable size? |

All candidates are pretrained on COCO, so this is transfer learning in every
case — the comparison is between *which* pretrained feature extractor to
fine-tune, not pretrained vs. from scratch.

Metrics recorded per model: mAP50, mAP50-95, precision, recall, parameter
count, inference latency (ms/image), and wall-clock training time. Latency is
included because the deployment target is continuous CCTV processing, so the
best model is not automatically the most accurate one.

## Dataset

Roboflow **Bulky Waste 2.0**, version 26 (CC BY 4.0) —
https://universe.roboflow.com/aiorbits/bulky-waste-2-0-ifbkq/dataset/26

- 885 images, YOLOv8 format bounding-box annotations
- 5 classes: `bin_empty`, `bin_full`, `bulk_waste`, `normal_waste`, `vehicle`
- Pre-applied by Roboflow: auto-orientation, resize to 640x640 (stretch)
- Augmentation (3 versions per source image): 50% horizontal flip, brightness
  ±25%, exposure ±15%, salt-and-pepper noise on 2% of pixels

## Running it

Needs a GPU. Google Colab (Runtime → Change runtime type → T4 GPU) is fine.

```bash
pip install -r requirements.txt

# Download the dataset export from Roboflow so that train/, valid/, test/
# sit next to data.yaml, then:
python compare_pretrained.py --data data.yaml --epochs 10
```

Outputs land in `results/`:
- `comparison.csv` — the results table for the report
- `comparison.png` — grouped bar chart of mAP with latency overlaid

Ten epochs is enough to rank the candidates. Once the winner is known, retrain
only that one for the full production run (50–100 epochs) and report those
final numbers.

To cut runtime if Colab is slow, drop `yolov8m`:

```bash
python compare_pretrained.py --data data.yaml --epochs 10 \
  --models yolov8n.pt yolov8s.pt yolo11n.pt yolo11s.pt
```

## Acknowledgements

- Dataset: Roboflow Universe, "Bulky Waste 2.0" by aiorbits (CC BY 4.0)
- Framework: Ultralytics YOLO
- COCO-pretrained weights: Ultralytics
