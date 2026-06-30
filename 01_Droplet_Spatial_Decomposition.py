"""
01_Droplet_Spatial_Decomposition.py

This script:
1. Reads dried droplet images from a selected folder.
2. Detects the droplet circle using the Hough transform.
3. Uses the detected circle center and radius to define crown and center regions.
4. Rotates each image at 30-degree intervals.
5. Extracts 12 peripheral/crown patches and 8 central patches per droplet.
6. Saves the extracted patches into class-specific folders.


Author:
    Carlos Alberto Martínez-Miwa
    
Authorship and permission

I, Carlos Alberto Martínez-Miwa, declare that this source code is my original work 
and was developed as part of the research presented in the associated manuscript.
 
I hereby authorize my co-author Mario Castelan to upload and maintain this code in 
a public GitHub repository as supplementary material accompanying the publication. 
This authorization is granted exclusively for academic, scientific, and reproducibility 
purposes related to this work.    

"""

# 1. Libraries
import glob
import os

import cv2
import numpy as np


# 2. Input configuration
print(os.getcwd())

# Percentage condition used to select the input folder.
# Original convention:
# - 10 is treated as Control.
# - Any other value is treated as Experimental.
percentage_condition = 4

input_path = (
      str(percentage_condition)
      + "/*.*"
)


# 3. Processing parameters
# Image resizing percentages used in the original script.
DSC_SCALE_PERCENT = 15
DEFAULT_SCALE_PERCENT = 50

# Hough circle detection parameters used in the original script.
HOUGH_DP = 1
HOUGH_MIN_DIST = 2500
HOUGH_PARAM_1 = 100
HOUGH_PARAM_2 = 30
HOUGH_MIN_RADIUS = 0
HOUGH_MAX_RADIUS = 0
MIN_ACCEPTED_RADIUS = 100

# Patch extraction parameters used in the original script.
N_ANGLES = 12
ROTATION_SCALE = 1
CROWN_PATCH_HALF_SIZE = 50
CROWN_PATCH_VERTICAL_OFFSET = 10
CENTER_PATCH_HALF_SIZE = CROWN_PATCH_HALF_SIZE * 2
CENTER_PATCH_ANGLES = (30, 90)

# Output index used in the original naming convention.
patch_index = 0


# 4. Main loop over all images
image_files = glob.glob(input_path)

for image_index, image_file in enumerate(image_files):

    # 4.1. Read image and file information
    file_name = os.path.splitext(os.path.basename(image_file))[0]

    input_image = cv2.imread(image_file)

    # 4.2. Resize image
    # Images whose name starts with "DSC" are resized to 15%.
    # All other images are resized to 50%.

    if file_name[0:3] == "DSC":
        scale_percent = DSC_SCALE_PERCENT
    else:
        scale_percent = DEFAULT_SCALE_PERCENT

    resized_width = int(input_image.shape[1] * scale_percent / 100)
    resized_height = int(input_image.shape[0] * scale_percent / 100)
    resized_dimensions = (resized_width, resized_height)

    resized_image = cv2.resize(
        input_image,
        resized_dimensions,
        interpolation=cv2.INTER_AREA,
    )

    # 4.3. Preprocess image for Hough circle detection
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(grayscale_image, 5)

    detected_circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        HOUGH_DP,
        HOUGH_MIN_DIST,
        param1=HOUGH_PARAM_1,
        param2=HOUGH_PARAM_2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )


    input_folder_name = os.path.splitext(input_path)[0].split("/")[-2]


    # 5. Patch extraction when a circle is detected
    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))

        for detected_circle in detected_circles[0, :]:

            circle_center_x = detected_circle[0]
            circle_center_y = detected_circle[1]
            circle_radius = detected_circle[2]

            # Only process circles with radius >= 100 pixels.
            if circle_radius >= MIN_ACCEPTED_RADIUS:

                # 5.1. Define class label and output folder
                if percentage_condition == 10:
                    class_label = "Control"
                else:
                    class_label = "Experimental"

                output_folder = (
                    str(percentage_condition)
                    + "%/"
                    + class_label
                    + "/"
                )


                try:
                    os.makedirs(output_folder)
                except OSError:
                    pass

                # 5.2. Prepare image for rotation-based extraction
                extraction_image = resized_image
                image_height, image_width = extraction_image.shape[:2]
                rotation_center = (circle_center_x, circle_center_y)


                # 5.3. Extract crown patches at 30-degree intervals
                for angle_index in range(N_ANGLES):

                    rotation_angle = (360 / N_ANGLES) * angle_index
                    rotation_angle_int = int(rotation_angle)

                    # The crown patch is extracted from the upper point of the circle
                    # after rotating the image around the detected droplet center.
                    crown_patch_center = (
                        rotation_center[0],
                        rotation_center[1] - circle_radius,
                    )

                    crown_x1 = crown_patch_center[0] - CROWN_PATCH_HALF_SIZE
                    crown_y1 = (
                        crown_patch_center[1]
                        - CROWN_PATCH_HALF_SIZE
                        + CROWN_PATCH_VERTICAL_OFFSET
                    )
                    crown_x2 = crown_patch_center[0] + CROWN_PATCH_HALF_SIZE
                    crown_y2 = (
                        crown_patch_center[1]
                        + CROWN_PATCH_HALF_SIZE
                        + CROWN_PATCH_VERTICAL_OFFSET
                    )

                    rotation_matrix = cv2.getRotationMatrix2D(
                        rotation_center,
                        rotation_angle,
                        ROTATION_SCALE,
                    )

                    rotated_image = cv2.warpAffine(
                        extraction_image,
                        rotation_matrix,
                        (image_width, image_height),
                    )


                    # 5.4. Define center patch coordinates
                    center_x1 = rotation_center[0] - CENTER_PATCH_HALF_SIZE
                    center_y1 = rotation_center[1] - CENTER_PATCH_HALF_SIZE
                    center_x2 = rotation_center[0] + CENTER_PATCH_HALF_SIZE
                    center_y2 = rotation_center[1] + CENTER_PATCH_HALF_SIZE

                    center_x_mid = center_x1 + int((center_x2 - center_x1) / 2)
                    center_y_mid = center_y1 + int((center_y2 - center_y1) / 2)

                    # 5.5. Extract crown and center patches
                    crown_patch = rotated_image[crown_y1:crown_y2, crown_x1:crown_x2]

                    center_patch_1 = rotated_image[
                        center_y1:center_y_mid,
                        center_x1:center_x_mid,
                    ]
                    center_patch_2 = rotated_image[
                        center_y_mid:center_y2,
                        center_x1:center_x_mid,
                    ]
                    center_patch_3 = rotated_image[
                        center_y_mid:center_y2,
                        center_x_mid:center_x2,
                    ]
                    center_patch_4 = rotated_image[
                        center_y1:center_y_mid,
                        center_x_mid:center_x2,
                    ]


                    # 5.6. Save crown patch
                    patch_index += 1
                    crown_patch_name = (
                        output_folder
                        + str(patch_index)
                        + "_"
                        + class_label
                        + "_"
                        + file_name
                        + "_Circle_"
                        + str(rotation_angle_int)
                        + ".jpg"
                    )
                    cv2.imwrite(crown_patch_name, crown_patch)

                    # 5.7. Save center patches only at 30 and 90 degrees
                    if rotation_angle_int in CENTER_PATCH_ANGLES:

                        center_patches = [
                            ("Center1", center_patch_1),
                            ("Center2", center_patch_2),
                            ("Center3", center_patch_3),
                            ("Center4", center_patch_4),
                        ]

                        for center_patch_label, center_patch in center_patches:
                            patch_index += 1
                            center_patch_name = (
                                output_folder
                                + str(patch_index)
                                + "_"
                                + class_label
                                + "_"
                                + file_name
                                + "_"
                                + center_patch_label
                                + "_"
                                + str(rotation_angle_int)
                                + ".jpg"
                            )
                            cv2.imwrite(center_patch_name, center_patch)


            # 6. Log images whose detected circle radius is too small
            else:
                no_hough_file = (
                    "No Hough_"
                    + str(percentage_condition)
                    + "%_New.txt"
                )

                with open(no_hough_file, "a") as file_handle:
                    file_handle.write(input_folder_name + "/" + file_name + ".jpg" + "\n")
