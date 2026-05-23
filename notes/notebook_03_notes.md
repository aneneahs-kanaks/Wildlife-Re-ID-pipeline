# Notebook 3 Notes — Viewpoint Estimation
### File: `viewpoint_estimation.ipynb`

---

## What This Notebook Does

This is the most algorithmically interesting notebook in the pipeline. It takes the 360° GIF renders and their corresponding DeepLabCut JSON keypoint files and **automatically determines which frame in each GIF corresponds to which viewing direction** — left, right, front, back, and the four diagonals.

---

## The Core Problem

Each GIF has 300 frames showing the animal rotating 360°. But which frame is the animal facing left? Which is front? Without knowing this, we cannot create properly labeled training data for the ReID model.

The solution: **track the eyes across all frames and find the sine wave trough.**

---

## The Sine Wave Insight

When an animal rotates 360° at constant speed, the x-position of each eye traces a **perfect sinusoidal path**. This is because:

> A point rotating in a circle, when projected onto the x-axis, mathematically produces a sine wave.

This is literally the geometric definition of a sine wave — and it is the core insight that makes this algorithm work.

**The trough (minimum point) of the sine wave = the frame where the animal faces directly left** — because that is when the eyes are at their leftmost x-position on the image.

---

## Why Only the Eyes?

Although 39 keypoints are available, only the eyes are used. They satisfy all three requirements:

| Requirement | Eyes | Legs | Nose | Ears |
|---|---|---|---|---|
| Periodic signal | ✅ YES | ❌ NO | ✅ YES | ✅ YES |
| Clean stable signal | ✅ YES | ❌ NO | ⚠️ PARTIAL | ⚠️ PARTIAL |
| Symmetric pair | ✅ YES | ✅ YES | ❌ NO | ✅ YES |
| Rarely occluded | ✅ YES | ❌ NO | ⚠️ PARTIAL | ❌ NO |
| **All met** | ✅ **YES** | ❌ NO | ❌ NO | ❌ NO |

Legs fail because they move independently (walking/posing) — their signal is noisy and non-periodic. Eyes are fixed on the head and follow a clean rotation path.

**Key insight:** The supervisor essentially converted a computer vision problem into a signal processing problem — an Electrical Engineering way of thinking.

---

## Step-by-Step Algorithm

### Step 1 — Data Reformatting

The DeepLabCut JSON is organized by frame. This is reformatted to be organized by keypoint — making it easy to extract one keypoint across all frames.

**Before reformatting (by frame):**
```
Frame 0: [nose, left_eye, right_eye, ...]
Frame 1: [nose, left_eye, right_eye, ...]
```

**After reformatting (by keypoint):**
```
left_eye:  {frame0: (x,y,conf), frame1: (x,y,conf), ...}
right_eye: {frame0: (x,y,conf), frame1: (x,y,conf), ...}
```

This is called **data reshaping** — same data, reorganized for convenience.

### Step 2 — Extract Eye X-Positions
```python
lxs = [v[0] for v in reformated_res['left_eye'].values()]
rxs = [v[0] for v in reformated_res['right_eye'].values()]
```
Two arrays of 300 values each — the x-coordinate of each eye in every frame.

### Step 3 — Forward/Backward Fill
Some frames have missing detections (confidence = -1, eye not visible). These gaps are filled:
- **Forward fill** — replace missing value with the last known value going forward
- **Backward fill** — for gaps at the start, use the next known value going backward

```python
def forward_backward_fill(arr, missing_rep=-1):
    # Forward fill first, then backward fill any remaining gaps
    ...
```

### Step 4 — Signal Shifting
The signal is shifted so the trough aligns to a standard position, making processing consistent across all animals regardless of where in the 300 frames the trough happens to fall.

```python
shift_amnt = -(np.argmin(np.array(rxs_ff)) - 75)
lxs_ff = np.roll(lxs_ff, shift_amnt)
rxs_ff = np.roll(rxs_ff, shift_amnt)
```

### Step 5 — Fold and Mirror
The signal is folded at the halfway point to create an "approaching" and "leaving" component for each eye — giving 4 signals total. This makes averaging more robust.

```python
def fold_halfway(arr, image_width=512):
    arr2 = arr.reshape(2, -1).copy()
    arr2[1] = image_width - arr2[1]
    arr2[1] = arr2[1][::-1]
    return arr2
```

### Step 6 — Savitzky-Golay Smoothing and Averaging
All 4 signals are smoothed using a Savitzky-Golay filter and then averaged:

```python
from scipy.signal import savgol_filter

def average_and_smooth_v2(left_arr, right_arr, window_size=21):
    smoothed_arrs = [savgol_filter(arr, window_length=21, polyorder=3) 
                     for arr in [arr1, arr2, arr3, arr4]]
    avg_arr_smooth = np.mean(smoothed_arrs, axis=0)
    return smoothed_arrs, avg_arr_smooth
```

**What is Savitzky-Golay filter?**
A smoothing filter that fits a polynomial to a sliding window of data points. It removes noise while preserving the shape of the signal — perfect for cleaning up the sine wave without distorting its trough position.

### Step 7 — Find the Trough
```python
trough_index = np.argmin(avg_arr_smooth)
```
`np.argmin` returns the index of the minimum value — the trough of the sine wave — which is the frame where the animal faces directly left.

### Step 8 — Extract 8 Canonical Viewpoints
```python
orientations = ['left', 'back-left', 'back', 'back-right', 
                'right', 'front-right', 'front', 'front-left']

for i, orientation in enumerate(orientations):
    steps = int(37.5 * i)
    ind = (trough_index + steps) % 300
```
Starting from the trough (left view), every 37.5 frames gives the next viewpoint direction, covering the full 360° with 8 evenly spaced canonical views.

---

## FiftyOne Visualization

After processing all GIFs, results are loaded into FiftyOne with grouped views:

| Group | Content |
|---|---|
| lxs | Left eye x-position plot |
| rxs | Right eye x-position plot |
| lrxs_ff | Both eyes after forward/backward fill |
| lxrxs_ff_shifted | Both eyes after shifting |
| lxrs_fold | Folded and mirrored signals |
| smoothed_average | Final smoothed signal with trough marked |
| viewpoints | Source image + 8 canonical viewpoint frames |
| threed | Interactive 3D GLB mesh viewer |

Human annotators review each animal and tag incorrect results as "bad". Bad samples get trough_index = -1 in the final CSV export.

---

## Output

A CSV file per shard containing:
- `name` — GIF filename
- `trough_index` — the frame index of the left-facing viewpoint (-1 if bad)
- `dropped_frames_left` — number of frames where left eye was not detected
- `dropped_frames_right` — number of frames where right eye was not detected

This CSV is uploaded to HuggingFace as the verified reference frames dataset.

---

## Key Learnings from This Notebook

- Signal processing thinking can elegantly solve computer vision problems
- Data reshaping (reformatting) is a fundamental skill in data science
- Missing data handling (forward/backward fill) is essential in real-world pipelines
- Smoothing before finding extrema prevents noise from giving wrong results
- Human verification is still needed even after a well-designed algorithm

---

*Part of the DSAIL-ReID Wildlife Re-Identification Pipeline*
