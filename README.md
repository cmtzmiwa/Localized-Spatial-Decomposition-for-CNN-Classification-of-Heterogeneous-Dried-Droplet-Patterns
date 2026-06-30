# Localized-Spatial-Decomposition-for-CNN-Classification-of-Heterogeneous-Dried-Droplet-Patterns
This repository was developed to support research on the automated classification of methotrexate (MTX) dry-droplet patterns using CNNs. The dataset presents a collection of images of dried MTX droplets at four concentration levels, where the lower concentrations correspond to samples adulterated with 20%, 40%, and 60% water.

- This repository was developed to support research on the automated classification of methotrexate (MTX) dry-droplet patterns using convolutional neural networks (CNNs). 
- The datase presents a collection of images of dried MTX droplets at four concentration levels (100%, 80%, 60%, and 40%), where the lower concentrations correspond to samples adulterated with 20%, 40%, and 60% water. 
- Each image includes regions of interest (ROIs) extracted from both the central and peripheral areas of each droplet at multiple rotational angles, capturing morphological variations 
  to enhance model generalization.

Images are organized into four levels of folders and comprise two main categories: 
- Control
- Experimental

each with images of complete droplets and ROIs, organized by water content percentages:
- 40% 
- 60%
- 80%


###### Repository Structure ######

/MTX Dataset/
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
│   │   ├── 80% W.C./
│   │   │   ├── DSC_2234.JPG
│   │   │   ├── DSC_2235.JPG
│   │   │   └── ...
│   │   └── 
│   │
│   ├── ROIs/
│   │   ├── 40% W.C./
│   │   │   ├── 1_Control_DSC_2241_Circle_0.jpg
│   │   │   ├── 2_Control_DSC_2241_Circle_30.jpg
│   │   │   ├── 3_Control_DSC_2241_Center1_30.jpg
│   │   │   ├── 4_Control_DSC_2241_Cetner2_30.jpg
│   │   │   └── ...
│   │   ├── 60% W.C./
│   │   │   ├── 1_Control_DSC_2241_Circle_0.jpg
│   │   │   ├── 2_Control_DSC_2241_Circle_30.jpg
│   │   │   ├── 3_Control_DSC_2241_Center1_30.jpg
│   │   │   ├── 4_Control_DSC_2241_Cetner2_30.jpg
│   │   │   └── ...
│   │   ├── 80% W.C./
│   │   │   ├── 1_Control_DSC_2241_Circle_0.jpg
│   │   │   ├── 2_Control_DSC_2241_Circle_30.jpg
│   │   │   ├── 3_Control_DSC_2241_Center1_30.jpg
│   │   │   ├── 4_Control_DSC_2241_Cetner2_30.jpg
│   │   │   └── ...
│   │   └──
│   └──      
└──       


├── Experimental/
│   ├── Droplets/
│   │   ├── 40% W.C./
│   │   │   ├── DSC_3756.JPG
│   │   │   ├── DSC_3757.JPG
│   │   │   └── ...
│   │   ├── 60% W.C./
│   │   │   ├── DSC_3719.JPG
│   │   │   ├── DSC_3720.JPG
│   │   │   └── ...
│   │   ├── 80% W.C./
│   │   │   ├── DSC_3675.JPG
│   │   │   ├── DSC_3676.JPG
│   │   │   └── ...
│   │   └── 
│   │
│   ├── ROIs/
│   │   ├── 40% W.C./
│   │   │   ├── 1_Experimental_DSC_7137_Circle_0.jpg
│   │   │   ├── 2_Experimental_DSC_7137_Circle_30.jpg
│   │   │   ├── 3_Experimental_DSC_7137_Center1_30.jpg
│   │   │   ├── 4_Experimental_DSC_7137_Cetner2_30.jpg
│   │   │   └── ...
│   │   ├── 60% W.C./
│   │   │   ├── 1_Experimental_DSC_6940_Circle_0.jpg
│   │   │   ├── 2_Experimental_DSC_6940_Circle_30.jpg
│   │   │   ├── 3_Experimental_DSC_6940_Center1_30.jpg
│   │   │   ├── 4_Experimental_DSC_6940_Cetner2_30.jpg
│   │   │   └── ...
│   │   ├── 80% W.C./
│   │   │   ├── 1_Experimental_DSC_3688_Circle_0.jpg
│   │   │   ├── 2_Experimental_DSC_3688_Circle_30.jpg
│   │   │   ├── 3_Experimental_DSC_3688_Center1_30.jpg
│   │   │   ├── 4_Experimental_DSC_3688_Cetner2_30.jpg
│   │   │   └── ...
│   │   └──
│   └──      
└──    


##### Levels Explained #####

Level 1 – Category: Control or Experimental

Level 2 – Subcategory: Droplets or ROIs

Level 3 – Water Content / Session: 40% W.C., 60% W.C., 80% W.C.

Level 4 – Images: Actual image files (e.g., 1_Control_DSC_2241_Circle_0.jpg)


##### Image Naming Convention #####

Images follow a structured naming scheme to encode metadata:

<Index>_<Category>_<ImageID>_<Region>_<Angle>.jpg

- Example: <1>_<Control>_<DSC_2241>_<Circle>_<0>.jpg


Name segments definition:

| Segment    | Meaning                                                    |
| ---------- | ---------------------------------------------------------- |
| `1`        | Image index                                                |
| `Control`  | Category (`Control` or `Experimental`)                     |
| `DSC_2241` | Unique ID of the sample                                    |
| `Circle`   | Region of extraction                                       |
| `0`        | Sampling angle                                             |


###### Usage ######

Python example to load the dataset:

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
