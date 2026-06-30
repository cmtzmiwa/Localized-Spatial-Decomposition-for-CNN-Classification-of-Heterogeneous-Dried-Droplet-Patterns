# Localized Spatial Decomposition for CNN Classification of Heterogeneous Dried Droplet Patterns

This repository was developed to support research on the automated classification of methotrexate (MTX) dry-droplet patterns using convolutional neural networks (CNNs).

The dataset presents a collection of images of dried MTX droplets at four concentration levels: **100%, 80%, 60%, and 40%**. The lower concentrations correspond to samples adulterated with **20%, 40%, and 60% water**, respectively.

Each image includes regions of interest (ROIs) extracted from both the central and peripheral areas of each droplet at multiple rotational angles. This localized spatial decomposition captures morphological variations and helps enhance model generalization.

## Dataset Categories

Images are organized into two main categories:

* **Control**
* **Experimental**

Each category contains complete droplet images and extracted ROIs, organized by water content percentage:

* **40% W.C.**
* **60% W.C.**
* **80% W.C.**

## Repository Structure

```text
├── README.md
├── requirements.txt
└── code/
    ├── 01_Droplet_Spatial_Decomposition.py
    ├── 02_Patch_Dataset_Partitioning.py
    └── 03_VGG16_Patch_Evaluation.py
│
├── MTX Dataset/
├── Control/
│   ├── Droplets/
│   │   ├── 40% W.C./
│   │   │   ├── DSC_2234.JPG
│   │   │   ├── DSC_2235.JPG
│   │   │   └── ...
│   │   ├── 60% W.C./
│   │   │   ├── DSC_2234.JPG
│   │   │   ├── DSC_2235.JPG
│   │   │   └── ...
│   │   └── 80% W.C./
│   │       ├── DSC_2234.JPG
│   │       ├── DSC_2235.JPG
│   │       └── ...
│   │
│   └── ROIs/
│       ├── 40% W.C./
│       │   ├── 1_Control_DSC_2241_Circle_0.jpg
│       │   ├── 2_Control_DSC_2241_Circle_30.jpg
│       │   ├── 3_Control_DSC_2241_Center1_30.jpg
│       │   ├── 4_Control_DSC_2241_Center2_30.jpg
│       │   └── ...
│       ├── 60% W.C./
│       │   ├── 1_Control_DSC_2241_Circle_0.jpg
│       │   ├── 2_Control_DSC_2241_Circle_30.jpg
│       │   ├── 3_Control_DSC_2241_Center1_30.jpg
│       │   ├── 4_Control_DSC_2241_Center2_30.jpg
│       │   └── ...
│       └── 80% W.C./
│           ├── 1_Control_DSC_2241_Circle_0.jpg
│           ├── 2_Control_DSC_2241_Circle_30.jpg
│           ├── 3_Control_DSC_2241_Center1_30.jpg
│           ├── 4_Control_DSC_2241_Center2_30.jpg
│           └── ...
│
└── Experimental/
    ├── Droplets/
    │   ├── 40% W.C./
    │   │   ├── DSC_3756.JPG
    │   │   ├── DSC_3757.JPG
    │   │   └── ...
    │   ├── 60% W.C./
    │   │   ├── DSC_3719.JPG
    │   │   ├── DSC_3720.JPG
    │   │   └── ...
    │   └── 80% W.C./
    │       ├── DSC_3675.JPG
    │       ├── DSC_3676.JPG
    │       └── ...
    │
    └── ROIs/
        ├── 40% W.C./
        │   ├── 1_Experimental_DSC_7137_Circle_0.jpg
        │   ├── 2_Experimental_DSC_7137_Circle_30.jpg
        │   ├── 3_Experimental_DSC_7137_Center1_30.jpg
        │   ├── 4_Experimental_DSC_7137_Center2_30.jpg
        │   └── ...
        ├── 60% W.C./
        │   ├── 1_Experimental_DSC_6940_Circle_0.jpg
        │   ├── 2_Experimental_DSC_6940_Circle_30.jpg
        │   ├── 3_Experimental_DSC_6940_Center1_30.jpg
        │   ├── 4_Experimental_DSC_6940_Center2_30.jpg
        │   └── ...
        └── 80% W.C./
            ├── 1_Experimental_DSC_3688_Circle_0.jpg
            ├── 2_Experimental_DSC_3688_Circle_30.jpg
            ├── 3_Experimental_DSC_3688_Center1_30.jpg
            ├── 4_Experimental_DSC_3688_Center2_30.jpg
            └── ...
```

## Levels Explained

| Level   | Description             | Example                            |
| ------- | ----------------------- | ---------------------------------- |
| Level 1 | Category                | `Control` or `Experimental`        |
| Level 2 | Subcategory             | `Droplets` or `ROIs`               |
| Level 3 | Water content / session | `40% W.C.`, `60% W.C.`, `80% W.C.` |
| Level 4 | Image files             | `1_Control_DSC_2241_Circle_0.jpg`  |

## Image Naming Convention

ROI images follow a structured naming scheme that encodes metadata:

```text
<Index>_<Category>_<ImageID>_<Region>_<Angle>.jpg
```

### Example

```text
1_Control_DSC_2241_Circle_0.jpg
```

### Name Segments Definition

| Segment    | Meaning                               |
| ---------- | ------------------------------------- |
| `1`        | Image index                           |
| `Control`  | Category: `Control` or `Experimental` |
| `DSC_2241` | Unique ID of the sample               |
| `Circle`   | Region of extraction                  |
| `0`        | Sampling angle                        |

## Usage

Python example to load the dataset:

```python
import os
import cv2

root_dir = "path/to/MTX Dataset"

for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)

    for subcategory in os.listdir(category_path):
        subcategory_path = os.path.join(category_path, subcategory)

        for session in os.listdir(subcategory_path):
            session_path = os.path.join(subcategory_path, session)

            for img_file in os.listdir(session_path):
                img_path = os.path.join(session_path, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue

                # Add image processing or model inference here
                print(f"Loaded image: {img_path}")
```

## Notes

* `Droplets/` contains complete dried-droplet images.
* `ROIs/` contains localized image patches extracted from different droplet regions.
* Peripheral ROIs are sampled at different rotational angles.
* Central ROIs capture internal morphological structures of the dried droplets.
* The dataset is intended for CNN-based classification and evaluation of heterogeneous dry-droplet patterns.

