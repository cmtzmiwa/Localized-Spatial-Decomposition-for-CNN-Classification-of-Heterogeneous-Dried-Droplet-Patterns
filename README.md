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

## Code Availability and Reproducibility

This repository includes the Python scripts used to reproduce the main computational stages reported in the manuscript. The scripts are provided as supplementary material to support transparency, reproducibility, and reuse of the proposed localized spatial decomposition methodology for CNN-based classification of heterogeneous dried-droplet patterns.

The source code is organized in the `code/` folder:

```text
code/
├── 01_Droplet_Spatial_Decomposition.py
├── 02_Patch_Dataset_Partitioning.py
└── 03_VGG16_Patch_Evaluation.py
```

## Script Description

| Script                                | Purpose                                                                                                                                                                                                                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `01_Droplet_Spatial_Decomposition.py` | Detects the dried droplet using the Hough transform and extracts localized image patches from peripheral and central regions. The script rotates each image at 30-degree intervals and saves crown and center patches into class-specific folders.                                   |
| `02_Patch_Dataset_Partitioning.py`    | Organizes the extracted patches into training, validation, and testing subsets according to the partition previously defined for complete droplet images. This ensures that all patches from the same physical droplet remain in the same subset, reducing the risk of data leakage. |
| `03_VGG16_Patch_Evaluation.py`        | Evaluates trained VGG16-based models on the localized droplet patches. The script loads trained models, predicts test patches, computes classification metrics, evaluates center and peripheral regions separately, and exports the results to Excel files.                          |

## Recommended Workflow

The scripts should be executed in the following order:

```text
1. Spatial decomposition of complete droplet images
2. Patch-level dataset partitioning
3. VGG16 model evaluation on localized patches
```

### Step 1: Droplet Spatial Decomposition

Run the following script to detect the droplet region and extract localized patches:

```bash
python code/01_Droplet_Spatial_Decomposition.py
```

This script generates localized patches from each complete dried-droplet image. The extracted ROIs include:

* Peripheral/crown patches sampled at 30-degree intervals.
* Central patches extracted from the inner region of the droplet.

The output patches are saved into folders according to the corresponding class condition.

### Step 2: Patch Dataset Partitioning

After extracting the ROIs, run:

```bash
python code/02_Patch_Dataset_Partitioning.py
```

This script assigns each localized patch to the training, validation, or testing subset according to the original partition of the complete droplet images.

This step is important because all patches extracted from the same physical droplet must remain in the same subset. This avoids information leakage between training and testing data.

### Step 3: VGG16 Patch Evaluation

Finally, run:

```bash
python code/03_VGG16_Patch_Evaluation.py
```

This script evaluates trained VGG16 models using the localized droplet patches. It computes global classification metrics and also reports region-specific performance for center and circle/ring patches.

## Requirements

The scripts were developed in Python and require the following main libraries:

```text
numpy
opencv-python
tensorflow
scikit-learn
natsort
openpyxl
```

To install the required packages, use:

```bash
pip install -r requirements.txt
```

## Dataset Path Configuration

Before running the scripts, users must update the dataset and output paths inside each Python file.

For example:

```python
base_patch_directory = "path/to/MTX Dataset/..."
base_model_directory = "path/to/trained/models/..."
```

The placeholder paths should be replaced with the local paths where the dataset, extracted patches, trained models, and output folders are stored.

## Notes

* The code preserves the original experimental workflow used in the manuscript.
* The spatial decomposition script extracts patches from both peripheral and central droplet regions.
* The partitioning script keeps all patches from the same droplet in the same subset to prevent data leakage.
* The evaluation script computes global and region-specific metrics.
* Users should verify that folder names and file naming conventions match the expected structure before executing the scripts.
