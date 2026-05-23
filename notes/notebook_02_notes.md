# Notebook 2 Notes — Parallel Keypoint Detection
### File: `run_shards.ipynb`

---

## What This Notebook Does

This notebook **automates and parallelizes** the DeepLabCut keypoint detection process across all 8 dataset shards. Instead of manually running each shard one at a time, it uses Papermill and Python's ThreadPoolExecutor to process all shards simultaneously.

---

## Key Concept: Why Parallel Processing?

The dataset has 8 shards. Running them one by one would mean:
- Run shard 2 → wait → run shard 3 → wait → ... → run shard 9
- Total time = time_per_shard × 8

Running them in parallel means:
- Run all 8 shards at the same time
- Total time ≈ time_per_shard × 1

This is a huge time saving — especially when each shard can take many minutes to process.

---

## Step-by-Step Breakdown

### 1. Install and Import Papermill
```python
!pip install papermill
import papermill as pm
```

**What is Papermill?**
A Python tool that lets you run a Jupyter notebook programmatically — like running a script — and pass different parameters each time. This means you can take one notebook and run it 8 times with different SHARD_ID values automatically.

### 2. Configure Shards and Workers
```python
SHARD_IDS = [2, 3, 4, 5, 6, 7, 8, 9]  # All shards to process
WORKERS = 8                              # Run 8 at the same time
```

### 3. Define the Run Function
```python
def run_shard(shard_id):
    output_notebook = f'Shard_{shard_id}_Keypoints_detection.ipynb'
    return pm.execute_notebook(
        'Keypoints_detection.ipynb',     # Input notebook
        output_notebook,                  # Output (executed copy)
        parameters=dict(SHARD_ID=shard_id),  # Pass the shard ID
    )
```
For each shard, Papermill:
1. Takes the keypoints detection notebook as a template
2. Injects the SHARD_ID parameter
3. Executes it
4. Saves the executed version as a separate output notebook

### 4. Run All Shards in Parallel
```python
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = [executor.submit(run_shard, i) for i in SHARD_IDS]
    for future in as_completed(futures):
        result = future.result()
        print('Finished one shard')
```
ThreadPoolExecutor creates a pool of 8 worker threads. Each worker picks up one shard and processes it independently. As each shard finishes, it reports completion.

---

## What DeepLabCut Does (The Underlying Process)

DeepLabCut is a state-of-the-art animal pose estimation library. For each frame of each GIF it:
1. Runs a deep neural network (HRNet + Faster RCNN) on the frame
2. Detects the location of all 39 anatomical keypoints
3. Assigns a confidence score to each detection
4. Returns x, y, confidence for all 39 keypoints

**Model used:** `superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2_before_adapt`

This is a pre-trained model from DeepLabCut's SuperAnimal collection — trained on many quadruped animals and fine-tuned for this type of wildlife data.

---

## The 39 Keypoints

Every single frame has all 39 keypoints detected simultaneously:

| Region | Keypoints |
|---|---|
| Head | nose, upper_jaw, lower_jaw, mouth_end_right, mouth_end_left |
| Eyes | right_eye, left_eye |
| Ears | right_earbase, right_earend, left_earbase, left_earend |
| Antlers | right_antler_base, right_antler_end, left_antler_base, left_antler_end |
| Neck | neck_base, neck_end, throat_base, throat_end |
| Back | back_base, back_end, back_middle |
| Tail | tail_base, tail_end |
| Front Legs | front_left_thai, front_left_knee, front_left_paw, front_right_thai, front_right_knee, front_right_paw |
| Back Legs | back_left_thai, back_left_knee, back_left_paw, back_right_thai, back_right_knee, back_right_paw |
| Body | belly_bottom, body_middle_right, body_middle_left |

---

## Output JSON Structure

For each GIF, one JSON file is produced organized by frame:

```
[
  {  // Frame 0
     "bodyparts": [
       [[x, y, conf],  // nose
        [x, y, conf],  // upper_jaw
        [x, y, conf],  // lower_jaw
        ...            // all 39 keypoints
       ]
     ]
  },
  {  // Frame 1
     ...
  },
  ...  // up to Frame 299
]
```

A confidence of **-1** means the keypoint was not detected in that frame — this is handled in Notebook 3 using forward/backward fill.

---

## Key Learnings from This Notebook

- **Automation > manual repetition** — Papermill makes notebook automation clean and simple
- **Parallel processing** dramatically reduces total pipeline time
- Every GIF produces one JSON file with 300 frames × 39 keypoints = 11,700 keypoint detections per animal
- Missing detections (-1) are expected and handled in the next stage

---

## What This Notebook Produces

| Output | Format | Used In |
|---|---|---|
| Keypoint detections | .json | Notebook 3 (viewpoint estimation) |
| Executed notebooks | .ipynb | Audit trail / debugging |

---

*Part of the DSAIL-ReID Wildlife Re-Identification Pipeline*
