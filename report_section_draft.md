# Pretrained model comparison — write-up for the group report

Results below are from the actual run: 5 COCO-pretrained checkpoints, 10 epochs
each, all other hyperparameters fixed (batch 32, AdamW, lr0 0.001, 640x640,
seed 0). Raw output is in `results/comparison.csv`.

Paste section 4 text under **4. Model Training**, alongside the batch-size
paragraph. Paste section 5 text under **5. Results and Evaluation**.

---

## 4. Model Training — Pretrained Model Selection

To determine which pretrained checkpoint to fine-tune for the production model,
I evaluated five COCO-pretrained YOLO variants over 10 epochs each. All other
hyperparameters were held constant at the values established by the batch-size
study (batch = 32, optimizer = AdamW, lr0 = 0.001, imgsz = 640, seed = 0), so
that any difference in performance is attributable to the starting weights
alone.

The candidates were chosen to span two axes:

- **Capacity:** YOLOv8n, YOLOv8s and YOLOv8m (2.6M to 25.9M parameters), to
  test whether a larger backbone helps or overfits on a dataset of 771
  training images.
- **Architecture generation:** YOLOv8n/s against YOLO11n/s, to test whether the
  newer backbone improves accuracy at a comparable parameter budget.

All five use transfer learning rather than training from scratch. With 771
training images across 5 classes, training a detector from random
initialisation would not converge to a usable model. The COCO-pretrained
backbone already encodes generic edge, texture and object-shape features, so
fine-tuning only needs to adapt the later layers to waste-specific appearance.
This was visible in the training logs: every variant reached a usable mAP
within 6-7 epochs.

**Finding:** YOLOv8s achieved the highest mAP50-95 at 0.3856, ahead of YOLO11s
(0.3786), YOLOv8m (0.3694), and the two nano variants (both 0.3536).

The most informative result is that **YOLOv8m performed worse than YOLOv8s
despite having 2.3x the parameters**, while taking 2.4x longer per inference.
On a dataset of this size the additional capacity provides no benefit — there
is not enough training signal to fit the larger backbone's parameters, and the
extra depth also slows convergence within a fixed 10-epoch budget.

Equally notable is that the newer architecture generation gave no advantage:
YOLOv8n and YOLO11n tied exactly at 0.3536 mAP50-95, and YOLOv8s beat YOLO11s
by only 0.007, which is within run-to-run variation on a 73-image validation
set. Architecture generation mattered far less than model size for this task.

**Deployment consideration:** the system is intended to process continuous CCTV
feeds, so inference latency is a design constraint rather than a footnote.
YOLO11n was the fastest variant at 5.45 ms per image — 30% faster than YOLOv8n
(7.72 ms) at identical accuracy, and 4.5x faster than YOLOv8m. YOLOv8s costs
10.35 ms for the best accuracy in the study.

YOLOv8s was selected as the base weights for the production model: it gives the
highest accuracy at a latency that remains well within real-time requirements
for frame-sampled CCTV processing. YOLO11n remains the better choice if the
system is later deployed to edge hardware, where its 30% latency advantage at
equal accuracy would matter more than 0.03 mAP.

### Final Production Model Configuration

- **Base Weights:** yolov8s.pt (selected by this experiment)
- **Epochs:** 100 (early stopping, patience = 20)
- **Batch Size:** 32
- **Optimizer:** AdamW
- **Initial Learning Rate:** 0.001
- **Resolution:** 640 x 640

---

## 5. Results and Evaluation — Pretrained model comparison

Insert `results/comparison.png` as a figure alongside this table.

| Model | Params (M) | mAP50 | mAP50-95 | Precision | Recall | Inference (ms) |
|---|---|---|---|---|---|---|
| YOLOv8s | 11.17 | 0.5101 | **0.3856** | 0.6262 | 0.4928 | 10.35 |
| YOLO11s | 9.46 | 0.4994 | 0.3786 | 0.5583 | 0.5099 | 10.53 |
| YOLOv8m | 25.90 | 0.5054 | 0.3694 | 0.5612 | 0.5390 | 24.54 |
| YOLOv8n | 3.16 | 0.4846 | 0.3536 | 0.5745 | 0.4925 | 7.72 |
| YOLO11n | 2.62 | 0.4821 | 0.3536 | 0.5656 | 0.4791 | 5.45 |

**Analysis**

The total spread across all five variants is only 0.032 mAP50-95 (0.3536 to
0.3856). The choice of pretrained checkpoint therefore has a modest effect on
this task compared with factors such as dataset size and training duration —
a useful finding in itself, since it indicates that effort is better spent on
data quality than on architecture selection.

Precision and recall traded off rather than moving together. YOLOv8m achieved
the highest recall (0.5390) but middling precision (0.5612), while YOLOv8n had
the highest precision (0.5745) with lower recall (0.4925). For this
application, recall is arguably the more important metric: a missed pile of
bulky waste means no alert is raised at all, whereas a false alarm only costs
an operator a few seconds to dismiss. If the deployed system prioritises
catching every incident, YOLOv8m's recall advantage would justify
reconsidering it despite its lower mAP.

**Limitations of this comparison**

1. **All models were still improving when training stopped.** YOLOv8m rose from
   0.344 to 0.369 mAP50-95 on its final epoch. The 10-epoch budget therefore
   measures how quickly each model converges, not its ceiling. Larger models
   converge more slowly, so YOLOv8m is likely penalised by the budget rather
   than genuinely inferior; a longer comparison could change the ranking.
2. **The validation set contains only 73 images (432 instances).** Differences
   of 0.01 mAP or less between adjacent variants are within noise and should
   not be treated as meaningful. Only the nano-versus-small gap (0.032) is
   large enough to be confident about.
3. **Each configuration was trained once with a single seed.** Repeating each
   run with several seeds and reporting the mean would give a more reliable
   ranking, but was not feasible within the available compute budget.
4. **Only the checkpoint was varied.** The optimal learning rate may differ by
   model size, so each variant was not necessarily evaluated at its own best
   settings — larger models often benefit from a lower learning rate.

---

## Notes for section 3 (Dataset) — corrections needed

Two things the group doc currently gets wrong, both visible in the training
logs:

1. **The split is not 70/20/10.** The actual v26 export is 771 train / 73 valid
   / 41 test, which is **87 / 8 / 5%**. Either the doc should be corrected to
   the real figures, or the dataset re-split in Roboflow before submission.
   Section 3 asks explicitly for the partitioning, and a marker who opens the
   dataset will see the discrepancy.
2. **The export is a mixed detect/segment dataset.** 2,778 polygon segments
   accompany 3,939 boxes in the training split. Ultralytics discards the
   segments and trains on bounding boxes only. Worth one line in the dataset
   description, since it shows the logs were actually read.

## Note on comparability with the batch-size study

The batch-size study reports mAP50 = 0.5351 / mAP50-95 = 0.3988 at batch 32,
which is slightly above the best result here (0.5101 / 0.3856 for YOLOv8s).
Since both studies used 10 epochs, AdamW and lr0 = 0.001, the difference
suggests the two runs did not use an identical base checkpoint or dataset
version. This should be reconciled before submission so the report presents one
consistent set of figures.
