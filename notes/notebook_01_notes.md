# Notebook 1 Notes — 3D Model Generation
### File: `gzgc_single_object.ipynb`

---

## What This Notebook Does

This is the **first notebook** in the pipeline and the starting point of the entire project. It takes real wildlife photographs from the GZGC dataset and converts each individual animal into a 3D model, then renders a 360° animated GIF of that model.

---

## Step-by-Step Breakdown

### 1. Configuration
```python
SHARD_ID = None       # Which shard you are processing
ASSIGNEE = ''         # Your name — for tracking who processed what
REPO_ID = 'DSAIL-ReID/synthetic-gzgc'
IMAGES_ROOT = HOME / 'gzgc/gzgc.coco/images/train2020'
```
The notebook is designed to be reproducible and trackable — every output is tagged with the assignee name and shard ID.

### 2. Load SAM3 Segmentation Masks
Pre-computed segmentation masks from SAM3 (Segment Anything Model 3) are loaded from a CSV file. These masks tell us exactly which pixels in each photo belong to the animal.

**What is a segmentation mask?**
A binary image map — pixels belonging to the animal = 1, background = 0. Stored in RLE (Run-Length Encoding) format to save space.

**What is SAM3?**
A text-prompted segmentation model. You tell it "zebra" or "giraffe" and it finds and outlines those animals in the photo automatically.

### 3. Filter Masks by Size
```python
MASK_SIZE_THRESHOLD = 0.1
mask_size_filter = (curr_shard_df_['w'] > MASK_SIZE_THRESHOLD) & 
                   (curr_shard_df_['h'] > MASK_SIZE_THRESHOLD)
```
Animals that appear small in the photo (far from camera, or heavily occluded) produce poor quality 3D models. Any mask whose width or height is less than 10% of the image size is discarded.

**Why this matters:** A low quality 3D model produces a low quality GIF which produces unreliable viewpoint labels — garbage in, garbage out.

### 4. Run Gaussian Splatting Inference
```python
output = inference(image_arr, fullmask, seed=42)
```
This is the core 3D reconstruction step. Given:
- The original photo (image_arr)
- The animal's segmentation mask (fullmask)

The Gaussian Splatting model generates a 3D representation of the animal.

**What is Gaussian Splatting?**
A cutting-edge 3D reconstruction technique that represents a 3D scene using millions of tiny 3D Gaussian blobs. Each blob has a position, size, orientation, and color. Together they form the shape and appearance of the animal. It can create a 3D model from a single 2D image.

### 5. Export 3D Outputs
```python
output["gs"].save_ply(...)   # Gaussian splat point cloud
output['glb'].export(...)    # Standard 3D mesh
```
Two 3D formats are saved:
- **PLY** — raw gaussian splat data (for future research use)
- **GLB** — standard 3D mesh format (viewable in FiftyOne, Blender, web browsers)

### 6. Render 360° GIF
```python
video = render_video(
    scene_gs,
    r=1,
    fov=90,
    pitch_deg=15,
    yaw_start_deg=-45,
    resolution=512,
)
```
The 3D model is rendered from a full 360° rotation. Parameters:
- `fov=90` — field of view (camera zoom)
- `pitch_deg=15` — slight upward camera angle (realistic wildlife photography angle)
- `yaw_start_deg=-45` — starting rotation angle
- `resolution=512` — 512×512 pixel output
- **300 frames total at 30fps**

### 7. Upload to HuggingFace
All outputs (GIFs, GLBs, PLYs) are uploaded to the shared HuggingFace dataset repository with a commit message tagging the assignee.

---

## Key Learnings from This Notebook

- **One real photo → one 3D model → one 360° GIF** per animal instance
- Some images contain multiple animals — each gets its own mask, 3D model, and GIF
- The `mask_index` column uniquely identifies each animal instance within an image
- GPU (CUDA) is required for Gaussian Splatting inference — this notebook requires an L4 GPU or higher

---

## What This Notebook Produces

| Output | Format | Used In |
|---|---|---|
| 360° animated renders | .gif | Notebook 3 (viewpoint estimation) |
| 3D meshes | .glb | FiftyOne visualization |
| Gaussian splats | .ply | Future research use |
| Source images | .jpg | FiftyOne reference display |

---

*Part of the DSAIL-ReID Wildlife Re-Identification Pipeline*
