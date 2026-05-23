# Pipeline Overview — Wildlife ReID Synthetic Data Pipeline

## Summary

This document gives a complete overview of the three-notebook pipeline built for the GZGC Wildlife Re-Identification project.

---

## The Problem We Are Solving

Wildlife Re-Identification (ReID) requires training data — specifically, multiple labeled images of the **same individual animal from different viewpoints**. In the wild, this is nearly impossible to collect at scale because:

- You cannot control how animals are photographed
- Two photos of the same zebra can look completely different due to viewing angle
- Manual labeling is expensive and slow

**Our solution:** Generate synthetic 3D models from real photos and render them from controlled viewpoints, giving us unlimited labeled training data.

---

## Pipeline Flow

```
INPUT: 4,948 real GZGC wildlife photos
           ↓
[NOTEBOOK 1] — gzgc_single_object.ipynb
SAM3 Segmentation → Gaussian Splatting → 360° GIF renders
           ↓
OUTPUT: .gif (renders), .glb (meshes), .ply (gaussians) → HuggingFace
           ↓
[NOTEBOOK 2] — run_shards.ipynb
Papermill + ThreadPoolExecutor → DeepLabCut on all 8 shards in parallel
           ↓
OUTPUT: .json files (39 keypoints × 300 frames per GIF) → HuggingFace
           ↓
[NOTEBOOK 3] — viewpoint_estimation.ipynb
Eye tracking → Sine wave → Trough detection → 8 canonical viewpoints
           ↓
OUTPUT: reference-frames CSV → HuggingFace
           ↓
[CURRENT] — FiftyOne Human Annotation
Review results → Tag bad samples → Verified reference frames
           ↓
[UPCOMING] — ReID Model Training
Viewpoint-matched pairs → Metric learning → ReID model
```

---

## Why This Pipeline Matters

Viewpoint estimation is the **bridge** between raw synthetic renders and structured training data. Without knowing which frame corresponds to which viewing direction:

- We cannot create controlled training pairs
- The ReID model cannot learn to distinguish viewpoint from identity
- We cannot construct hard negatives for metric learning

---

## Data Organization

The dataset is split into **shards** (subsets) for manageable parallel processing:
- Shards 2 through 9 (8 shards total)
- Each shard contains a subset of the 4,948 GZGC images
- All outputs are organized by shard: `renders/shard_009/`, `meshes/shard_009/`, etc.

---

## Key Numbers

| Metric | Value |
|---|---|
| Total real photos | 4,948 |
| Shards processed | 8 (IDs 2–9) |
| Keypoints per frame | 39 |
| Frames per GIF | 300 |
| Canonical viewpoints extracted | 8 per animal |
| Parallel workers used | 8 |

---

*See individual notebook notes for detailed explanations of each stage.*
