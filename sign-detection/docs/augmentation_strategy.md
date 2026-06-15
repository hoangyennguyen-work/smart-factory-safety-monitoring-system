# Augmentation Strategy

Purpose: document offline and online augmentation choices for small safety signs in CCTV footage.

## Notebook 03 Scope

Notebook 03 creates offline augmentation from the training split only:

```text
data/generated/splits_original/train/
```

Validation and test splits remain original and untouched.

## Geometric Augmentation

Geometric augmentation simulates rotation, scale, translation, and mild CCTV viewpoint changes. Because these transforms move objects, YOLO bounding boxes are updated by transforming bbox corners and clipping them to the augmented image canvas.

## Photometric Augmentation

Photometric augmentation simulates IR/grayscale CCTV, harsh sunlight, low light, shadows, brightness, and contrast shifts. These transforms do not move objects, so labels are copied unchanged.

## Weather and Quality Augmentation

Weather and quality augmentation simulates blur, JPEG compression, sensor noise, dirty lens artifacts, and low-resolution upsampling. These transforms also keep geometry fixed, so labels are copied unchanged.

## Label Update Rules

Geometry-changing transforms must update labels. Non-geometric transforms can copy labels unchanged. Empty-label no-sign images remain valid after augmentation and keep empty label files.

## Synthetic Placement

Synthetic placement is disabled by default. If enabled later, pasted signs must generate correct updated labels before the data can be used.
