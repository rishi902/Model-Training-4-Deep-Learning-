# My section for the group report — fill in after running the experiment

Paste this into the team doc under **4. Model Training** (Wei Kang's batch-size
paragraph sits alongside it) and **5. Results and Evaluation**. All the `___`
are placeholders — fill them from `results/comparison.csv`. Do not invent
numbers; the marks for Results & Evaluation depend on the analysis, not on the
numbers being high.

---

## 4. Model Training — Pretrained Model Selection (my contribution)

To determine which pretrained checkpoint to fine-tune for the production model,
I evaluated ___ COCO-pretrained YOLO variants over ___ epochs each. All other
hyperparameters were held constant at the values established by the batch-size
study (batch = 32, optimizer = AdamW, lr0 = 0.001, imgsz = 640, seed = 0), so
that any performance difference is attributable to the starting weights alone.

The candidates spanned two axes:

- **Capacity:** YOLOv8n, YOLOv8s and YOLOv8m, to test whether a larger backbone
  helps or overfits on a dataset of only 885 images.
- **Architecture generation:** YOLOv8n/s against YOLO11n/s, to test whether the
  newer backbone and neck improve accuracy at comparable parameter counts.

All variants use transfer learning rather than training from scratch. With 885
images across 5 classes, training a detector from random initialisation would
not converge to a usable model; the COCO-pretrained backbone already encodes
generic edge, texture and object-shape features, so fine-tuning only has to
adapt the later layers to waste-specific appearance.

**Finding:** ___ achieved the highest mAP50-95 (___), compared with ___ for
___. [Then explain *why* — pick whichever pattern your numbers actually show:]

- If the small models won: the dataset is too small to supply enough gradient
  signal for the larger backbone's extra parameters, so YOLOv8m overfits — its
  training loss keeps falling while validation mAP plateaus or drops.
- If the larger models won: the extra capacity captures the fine-grained
  texture differences that separate `bin_full` from `bin_empty` and
  `bulk_waste` from `normal_waste`, which the nano backbone under-fits.
- If YOLO11 beat YOLOv8 at equal size: the improved backbone extracts stronger
  features at the same parameter budget.

**Deployment consideration:** the system processes continuous CCTV feeds, so
inference latency is a real constraint, not a footnote. ___ ran at ___ ms per
image versus ___ ms for ___ — a ___% latency cost for a ___ point mAP gain.
[State which trade-off you chose and defend it.]

### Final Production Model Configuration

- **Base Weights:** ___ (selected by this experiment)
- **Epochs:** ___
- **Batch Size:** 32
- **Optimizer:** AdamW
- **Initial Learning Rate:** 0.001
- **Resolution:** 640 x 640

---

## 5. Results and Evaluation — Pretrained model comparison

Insert `results/comparison.csv` as a table and `results/comparison.png` as a
figure, then write the analysis.

| Model | Params (M) | mAP50 | mAP50-95 | Precision | Recall | Inference (ms) |
|---|---|---|---|---|---|---|
| ___ | ___ | ___ | ___ | ___ | ___ | ___ |

Points worth making in the analysis (the rubric explicitly asks you to analyse,
not describe):

- Which classes each model handled worst. Check the per-class mAP in the
  Ultralytics output — `bin_empty` vs `bin_full` is a status distinction rather
  than an object distinction, so confusion between them is expected and is
  worth calling out.
- Whether precision and recall moved together or traded off across variants.
- Whether 10 epochs was enough to rank the models fairly, or whether the larger
  variants were still improving when training stopped — this is a genuine
  limitation of the comparison and stating it earns more than hiding it.

## 6. Discussion — limitations of this experiment

- Each configuration was trained once with a single seed, so small mAP
  differences between adjacent variants may be run-to-run noise rather than a
  real effect.
- The 10-epoch budget favours models that converge quickly, which may
  understate the larger backbones.
- Only the checkpoint was varied; the optimal learning rate may differ per
  model size, so each variant was not necessarily at its own best settings.
- 885 source images is small for 5 classes; more data would likely change which
  capacity is optimal.
