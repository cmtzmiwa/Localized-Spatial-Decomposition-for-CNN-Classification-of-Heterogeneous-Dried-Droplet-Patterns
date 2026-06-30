"""
02_Patch_Dataset_Partitioning.py

This script organizes the localized droplet patches into the training, validation, and testing subsets 
according to the partition previously defined for the complete droplet images. 
For each water-content condition and each of the ten experimental runs, the script:

1. Reads the partition of the complete droplet images for the current experimental run.
2. Matches each localized patch to its corresponding complete droplet using the droplet identifier contained in the filename.
3. Assigns the patch to the same subset (training, validation, or testing) as its corresponding complete droplet.
4. Renames and copies the patch to the appropriate output directory while preserving the class label.
5. Generates a text file containing the identifiers of the patches assigned to each subset for traceability.

The implementation preserves the original partitioning strategy and guarantees that all localized patches extracted from the same physical droplet remain within the same dataset subset, thereby preventing information leakage during model training and evaluation.

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

import os
import shutil

from natsort import natsorted


# Configuration
water_content = '6%'

class_names = ['/Experimental', '/Control']
class_ids = [str(water_content[0]), '0']

patch_output_root = (
    '/.../'
    + water_content + '/'
)

experimental_patch_root = (
    '/.../'
)

control_patch_directory = (
    '/.../10%/Control'
)

complete_image_root = (
    '/.../'
)

subset_folders = {
    'train': 'train/',
    'test': 'test/',
    'val': 'validation/'
}


# Helper functions
def get_patch_droplet_id(patch_filename):
    """
    Extract the droplet identifier from a localized patch filename.

    This function preserves the original filename parsing rules:
    - If the third field starts with 'D', the fourth field is used.
    - If the third field starts with 'p', the last two characters of
      the third field are used.
    """
    patch_parts = patch_filename.split('_')

    if patch_parts[2][0] == 'D':
        return patch_parts[3]

    if patch_parts[2][0] == 'p':
        return patch_parts[2][-2:]

    return None


def get_complete_image_droplet_id(image_filename):
    """
    Extract the droplet identifier from a complete droplet image filename.

    This function preserves the original filename parsing rules:
    - If the third field starts with 'D', the fourth field is used,
      removing the file extension.
    - If the third field starts with 'p', characters from position 4
      onward are used, removing the file extension.
    """
    image_parts = image_filename.split('_')

    if image_parts[2][0] == 'D':
        return image_parts[3][:-4]

    if image_parts[2][0] == 'p':
        return image_parts[2][4:-4]

    return None


def create_directory(directory_path):
    """Create a directory if it does not already exist."""
    try:
        os.makedirs(directory_path)
    except OSError:
        pass


def copy_matching_patches(
    patch_files,
    complete_image_files,
    source_patch_directory,
    output_subset_directory,
    subset_name,
    class_id,
    run_index
):
    """
    Copy localized patches into the corresponding subset folder.

    Patches are copied when their droplet identifier matches the identifier
    of a complete image previously assigned to the selected subset.
    A text file with the copied patch paths is also generated for traceability.
    """
    id_file_path = (
        output_subset_directory
        + 'Patches_ID_' + subset_name + '_' + str(run_index) + '.txt'
    )

    # Preserve the original behavior: create a new ID file and raise an
    # error if the file already exists.
    open(id_file_path, 'x').close()

    for patch_filename in patch_files:
        patch_id = get_patch_droplet_id(patch_filename)

        for complete_image_filename in complete_image_files:
            complete_image_id = get_complete_image_droplet_id(complete_image_filename)

            if patch_id == complete_image_id:
                renamed_patch_path = (
                    output_subset_directory
                    + 'P' + class_id + '_Cat' + patch_filename
                )

                source_patch_path = os.path.join(source_patch_directory, patch_filename)
                shutil.copy(source_patch_path, renamed_patch_path)

                with open(id_file_path, 'a') as id_file:
                    id_file.write(renamed_patch_path + '\n')


# Clean previous output directory
try:
    shutil.rmtree(patch_output_root, ignore_errors=False, onerror=None)
except OSError:
    pass


# Main processing loop
for class_index in range(0, 2):

    class_name = class_names[class_index]
    class_id = class_ids[class_index]

    print('COND:', class_id)

    # Select source directory for experimental or control patches.
    if class_id == str(water_content[0]):
        patch_source_directory = experimental_patch_root + water_content + '/' + class_name

    if class_id == '0':
        patch_source_directory = control_patch_directory


    # Process the 10 independent runs.
    for run_index in range(1, 11):

        run_name = 'P' + str(water_content)[0] + '0_' + str(run_index) + '/'

        patch_output_directory = patch_output_root + run_name
        complete_image_directory = complete_image_root + water_content + '/' + run_name

        print(patch_output_directory)

        # Create output folders for this run and class
        create_directory(patch_output_directory)

        train_output_directory = (
            patch_output_directory + subset_folders['train'] + 'P' + class_id + '/'
        )
        test_output_directory = (
            patch_output_directory + subset_folders['test'] + 'P' + class_id + '/'
        )
        validation_output_directory = (
            patch_output_directory + subset_folders['val'] + 'P' + class_id + '/'
        )

        create_directory(train_output_directory)
        create_directory(test_output_directory)
        create_directory(validation_output_directory)


        # Read source patch files and complete-image partitions
        patch_files = natsorted(os.listdir(patch_source_directory))
        print(patch_files)

        train_images = natsorted(os.listdir(
            complete_image_directory + subset_folders['train'] + 'P' + class_id + '/'
        ))

        test_images = natsorted(os.listdir(
            complete_image_directory + subset_folders['test'] + 'P' + class_id + '/'
        ))

        validation_images = natsorted(os.listdir(
            complete_image_directory + subset_folders['val'] + 'P' + class_id + '/'
        ))


        # Copy patches according to the complete-image partition
        copy_matching_patches(
            patch_files=patch_files,
            complete_image_files=train_images,
            source_patch_directory=patch_source_directory,
            output_subset_directory=train_output_directory,
            subset_name='train',
            class_id=class_id,
            run_index=run_index
        )

        copy_matching_patches(
            patch_files=patch_files,
            complete_image_files=test_images,
            source_patch_directory=patch_source_directory,
            output_subset_directory=test_output_directory,
            subset_name='test',
            class_id=class_id,
            run_index=run_index
        )

        copy_matching_patches(
            patch_files=patch_files,
            complete_image_files=validation_images,
            source_patch_directory=patch_source_directory,
            output_subset_directory=validation_output_directory,
            subset_name='val',
            class_id=class_id,
            run_index=run_index
        )

        print('All Files Renamed')
