"""
03_VGG16_Patch_Evaluation.py

This script evaluates trained VGG16 models on localized droplet patches.
For each water-content condition and each experimental run, the script:

1. Loads the corresponding trained Keras model.
2. Reads the test patches assigned to the current run.
3. Predicts the class of each patch.
4. Computes global classification metrics.
5. Computes separate precision values for center and circle/ring patches.
6. Exports the results to a formatted Excel file.


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

# Imports
import gc
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from excel_exporter import export_to_excel_grouped


# Hardware configuration
# This reproduces the original execution mode by disabling GPU usage.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Image configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224


# Helper functions
def preprocess_image(image_path):
    """
    Load, resize, normalize, and expand dimensions of an input image.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Preprocessed image ready for model prediction.
    """
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0

    return image_array


def safe_division(numerator, denominator):
    """
    Compute a division while avoiding division-by-zero errors.
    """
    if denominator == 0:
        return 0

    return numerator / denominator


def get_patch_region(filename_parts):
    """
    Identify whether a patch corresponds to the circle/ring or center region.

    The original naming convention uses:
    - 'Ci' for circle/ring patches
    - 'Ce' for center patches
    """
    region_code = filename_parts[-2][0:2]

    if region_code == "Ci":
        return "Circle"

    if region_code == "Ce":
        return "Center"

    return None


def update_confusion_code(predicted_label, true_label, experimental_label):
    """
    Assign the original confusion-code convention used in the manuscript code.

    Original convention:
    0 -> TP
    1 -> FP
    2 -> TN
    3 -> FN
    """
    if predicted_label == "0" and true_label == "0":
        return 0

    if predicted_label == "0" and true_label == experimental_label:
        return 1

    if predicted_label == experimental_label and true_label == experimental_label:
        return 2

    if predicted_label == experimental_label and true_label == "0":
        return 3

    return None


def update_region_confusion_code(confusion_code, patch_region):
    """
    Assign the original region-specific confusion-code convention.

    Center:
    10 -> TP_Ce, 11 -> FP_Ce, 12 -> TN_Ce, 13 -> FN_Ce

    Circle/Ring:
    20 -> TP_Ci, 21 -> FP_Ci, 22 -> TN_Ci, 23 -> FN_Ci
    """
    if patch_region == "Center":
        return 10 + confusion_code

    if patch_region == "Circle":
        return 20 + confusion_code

    return None


# Main configuration
# Original script evaluated this water-content condition.
water_content_list = [6]

base_model_directory = "/..."
base_patch_directory = "/..."


# Main evaluation loop
for water_content in water_content_list:

    # Metric containers across the 10 experimental runs
    precision_experimental_values = []
    recall_experimental_values = []
    f1_experimental_values = []

    accuracy_values = []

    precision_control_values = []
    recall_control_values = []
    f1_control_values = []

    subset_numbers = []

    precision_experimental_center_values = []
    precision_experimental_circle_values = []

    precision_control_center_values = []
    precision_control_circle_values = []

    true_positive_values = []
    false_positive_values = []
    true_negative_values = []
    false_negative_values = []

    circle_patch_counts = []
    center_patch_counts = []

    all_predicted_labels = []
    all_true_labels = []

    # Evaluate the 10 trained models / partitions
    for run_index in range(1, 11):

        K.clear_session()
        gc.collect()

        circle_patch_count = 0
        center_patch_count = 0

        subset_number = run_index
        test_label = "P" + str(water_content)

        model_path = (
            base_model_directory
            + "/P"
            + str(water_content)
            + "0_"
            + str(subset_number)
            + ".h5"
        )

        # Load the trained model for the current run.
        model = tf.keras.models.load_model(model_path, compile=False)

        run_directory = (
            base_patch_directory
            + "/"
            + str(water_content)
            + "%/P"
            + str(water_content)
            + "0_"
            + str(run_index)
        )

        test_images_directory = run_directory + "/test/Mixed/"
        training_directory = run_directory + "/train/"


        # Recover class-label mapping from the training folder
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            training_directory,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1,
            class_mode="categorical",
        )

        class_indices = train_generator.class_indices
        class_labels = {value: key for key, value in class_indices.items()}


        # Containers for current run
        test_patch_files = os.listdir(test_images_directory)

        print("CATS:", len(test_patch_files))

        test_vector = []
        confusion_codes = []
        region_labels = []
        region_confusion_codes = []

        predicted_labels = []
        true_labels = []


        # Predict each test patch
        for patch_index, patch_filename in enumerate(test_patch_files):

            filename_without_extension = os.path.splitext(patch_filename)[0]
            filename_parts = filename_without_extension.split("_")

            patch_path = os.path.join(test_images_directory, patch_filename)

            preprocessed_patch = preprocess_image(patch_path)
            prediction = model.predict(preprocessed_patch)

            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_probability = prediction[0][predicted_class]
            predicted_class_label = class_labels[predicted_class]


            # Original code used the second character of the folder label
            # to identify the class: '0' for control and water-content value
            # for the experimental class.
            predicted_class_code = predicted_class_label[1]
            true_class_code = filename_parts[0][1]
            experimental_class_code = str(water_content)


            # Store prediction and true labels
            if predicted_class_code == experimental_class_code:
                test_vector.append(0)
                predicted_labels.append(0)
                all_predicted_labels.append(0)

            elif predicted_class_code == "0":
                test_vector.append(1)
                predicted_labels.append(1)
                all_predicted_labels.append(1)

            if true_class_code == experimental_class_code:
                true_labels.append(0)
                all_true_labels.append(0)

            if true_class_code == "0":
                true_labels.append(1)
                all_true_labels.append(1)


            # Global confusion-code assignment
            confusion_code = update_confusion_code(
                predicted_label=predicted_class_code,
                true_label=true_class_code,
                experimental_label=experimental_class_code,
            )

            confusion_codes.append(confusion_code)


            # Region-specific confusion-code assignment
            patch_region = get_patch_region(filename_parts)
            region_labels.append(patch_region)

            if patch_region == "Circle":
                circle_patch_count += 1

            if patch_region == "Center":
                center_patch_count += 1

            region_confusion_code = update_region_confusion_code(
                confusion_code=confusion_code,
                patch_region=patch_region,
            )

            region_confusion_codes.append(region_confusion_code)


        # Confusion matrix counts
        true_positives = confusion_codes.count(0)
        false_positives = confusion_codes.count(1)
        true_negatives = confusion_codes.count(2)
        false_negatives = confusion_codes.count(3)


        # Region-specific confusion matrix counts
        true_positives_center = region_confusion_codes.count(10)
        false_positives_center = region_confusion_codes.count(11)
        true_negatives_center = region_confusion_codes.count(12)
        false_negatives_center = region_confusion_codes.count(13)

        true_positives_circle = region_confusion_codes.count(20)
        false_positives_circle = region_confusion_codes.count(21)
        true_negatives_circle = region_confusion_codes.count(22)
        false_negatives_circle = region_confusion_codes.count(23)

        print("TP_Ce:", true_positives_center)
        print("FP_Ce:", false_positives_center)
        print("TN_Ce:", true_negatives_center)
        print("FN_Ce:", false_negatives_center)

        print("TP_Ci:", true_positives_circle)
        print("FP_Ci:", false_positives_circle)
        print("TN_Ci:", true_negatives_circle)
        print("FN_Ci:", false_negatives_circle)


        # Region-specific precision values
        precision_experimental_center = safe_division(
            true_positives_center,
            true_positives_center + false_positives_center,
        )

        precision_experimental_circle = safe_division(
            true_positives_circle,
            true_positives_circle + false_positives_circle,
        )

        precision_control_center = safe_division(
            true_negatives_center,
            true_negatives_center + false_negatives_center,
        )

        precision_control_circle = safe_division(
            true_negatives_circle,
            true_negatives_circle + false_negatives_circle,
        )


        # Global classification metrics
        precision_experimental = safe_division(
            true_positives,
            true_positives + false_positives,
        )

        recall_experimental = safe_division(
            true_positives,
            true_positives + false_negatives,
        )

        f1_experimental = safe_division(
            2 * precision_experimental * recall_experimental,
            precision_experimental + recall_experimental,
        )

        recall_control = safe_division(
            true_negatives,
            true_negatives + false_positives,
        )

        accuracy = safe_division(
            true_positives + true_negatives,
            true_positives + false_positives + true_negatives + false_negatives,
        )

        precision_control = safe_division(
            true_negatives,
            true_negatives + false_negatives,
        )

        f1_control = safe_division(
            2 * precision_control * recall_control,
            precision_control + recall_control,
        )


        # Store current-run metrics
        precision_experimental_values.append(float(precision_experimental))
        recall_experimental_values.append(float(recall_experimental))
        f1_experimental_values.append(float(f1_experimental))

        accuracy_values.append(float(accuracy))

        precision_control_values.append(float(precision_control))
        recall_control_values.append(float(recall_control))
        f1_control_values.append(float(f1_control))

        subset_numbers.append(float(subset_number))

        precision_experimental_center_values.append(float(precision_experimental_center))
        precision_experimental_circle_values.append(float(precision_experimental_circle))

        precision_control_center_values.append(float(precision_control_center))
        precision_control_circle_values.append(float(precision_control_circle))

        true_positive_values.append(true_positives)
        false_positive_values.append(false_positives)
        true_negative_values.append(true_negatives)
        false_negative_values.append(false_negatives)

        circle_patch_counts.append(circle_patch_count)
        center_patch_counts.append(center_patch_count)

 

    # Export results for the current water-content condition
    rows = list(
        zip(
            subset_numbers,
            accuracy_values,
            precision_experimental_values,
            recall_experimental_values,
            f1_experimental_values,
            precision_control_values,
            recall_control_values,
            f1_control_values,
            true_positive_values,
            false_positive_values,
            true_negative_values,
            false_negative_values,
            precision_experimental_center_values,
            precision_control_center_values,
            center_patch_counts,
            precision_experimental_circle_values,
            precision_control_circle_values,
            circle_patch_counts,
        )
    )

    group_headers = [
        (
            "General (" + str(water_content) + "%)",
            ["Subset Num", "Accuracy"],
        ),
        (
            "Experimental (" + str(water_content) + "%)",
            ["Precision", "Recall", "F1Score"],
        ),
        (
            "Control (" + str(water_content) + "%)",
            ["Precision", "Recall", "F1Score"],
        ),
        (
            "TP FP TN FN (" + str(water_content) + "%)",
            ["True Positives", "False Positives", "True Negatives", "False Negatives"],
        ),
        (
            "Patches Center (" + str(water_content) + "%)",
            ["Precision (Experimental)", "Precision (Control)", "Num Patches"],
        ),
        (
            "Patches Circle (" + str(water_content) + "%)",
            ["Precision (Experimental)", "Precision (Control)", "Num Patches"],
        ),
    ]

    output_file = (
        "/.../"
        + str(water_content)
        + "%.xlsx"
    )

    export_to_excel_grouped(output_file, rows, group_headers)


    # Cleanup
    del model
    K.clear_session()
    gc.collect()
