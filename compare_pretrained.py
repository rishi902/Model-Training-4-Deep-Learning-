"""
Pretrained-model comparison for the Bulky Waste 2.0 detector.

My individual contribution to the C384 FA project: everything except the
starting weights is held fixed (batch size 32, from Wei Kang's batch-size
grid search), and only the pretrained checkpoint is varied. The goal is to
pick the base weights for the team's final production model.

Usage:
    python compare_pretrained.py --data data.yaml --epochs 10
    python compare_pretrained.py --data data.yaml --epochs 10 --models yolov8n.pt yolov8s.pt
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# The candidates. Two axes on purpose so the report can talk about both:
#   - model SIZE  (n -> s -> m): more parameters, slower inference
#   - model FAMILY (v8 vs 11):   newer architecture at a similar size
DEFAULT_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]

# Fixed for every run so the comparison is fair.
FIXED = dict(batch=32, lr0=0.001, optimizer="AdamW", imgsz=640, seed=0)


def run_one(weights, data, epochs, project):
    """Fine-tune one pretrained checkpoint and return its validation metrics."""
    model = YOLO(weights)
    n_params = sum(p.numel() for p in model.model.parameters())

    start = time.time()
    model.train(
        data=data,
        epochs=epochs,
        name=Path(weights).stem,
        project=project,
        exist_ok=True,
        verbose=False,
        **FIXED,
    )
    train_minutes = (time.time() - start) / 60

    # Evaluate on the validation split with the best epoch's weights.
    metrics = model.val(data=data, imgsz=FIXED["imgsz"], verbose=False)

    return {
        "model": weights,
        "params_M": round(n_params / 1e6, 2),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        # ms per image at inference - matters because this runs on CCTV feeds.
        "inference_ms": round(float(metrics.speed["inference"]), 2),
        "train_minutes": round(train_minutes, 1),
    }


def plot(df, out_path):
    """Bar chart of mAP per model, with inference speed overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["mAP50"], width=0.4, label="mAP50")
    ax.bar([i + 0.2 for i in x], df["mAP50_95"], width=0.4, label="mAP50-95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=15)
    ax.set_ylabel("mAP")
    ax.set_title(f"Pretrained model comparison (batch=32, {FIXED['optimizer']}, lr0={FIXED['lr0']})")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(list(x), df["inference_ms"], color="black", marker="o", label="inference (ms)")
    ax2.set_ylabel("inference latency (ms/image)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"chart -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--project", default="runs/pretrained_comparison")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    rows = []
    for weights in args.models:
        print(f"\n=== {weights} ===")
        try:
            rows.append(run_one(weights, args.data, args.epochs, args.project))
        except Exception as exc:  # usually CUDA OOM on the larger variants
            print(f"skipped {weights}: {exc}")
        # Save after every model so a crash halfway doesn't lose earlier runs.
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)

    if not rows:
        print("no runs completed")
        return

    df = pd.DataFrame(rows).sort_values("mAP50_95", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "comparison.csv", index=False)
    plot(df, out_dir / "comparison.png")

    print("\n" + df.to_string(index=False))
    print(f"\nBest by mAP50-95: {df.iloc[0]['model']}")


if __name__ == "__main__":
    main()
