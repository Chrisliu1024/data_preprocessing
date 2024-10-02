import os
import zipfile
from loguru import logger

zip_subfolders = ['image', 'cloud_las', 'calib_extract']

def rename_folder(root_folder, old_folder, new_folder):
    for folder_name, subfolders, filenames in os.walk(root_folder):
        if old_folder in subfolders:
            old_folder_path = os.path.join(folder_name, old_folder)
            new_folder_path = os.path.join(folder_name, new_folder)
            os.rename(old_folder_path, new_folder_path)
            logger.info(f'Renamed {old_folder_path} to {new_folder_path}')


def get_target_folders(root_folder):
    target_folders = []
    for folder_name, subfolders, filenames in os.walk(root_folder):
        if all(subfolder in subfolders for subfolder in zip_subfolders):
            target_folders.append(folder_name)
    return target_folders

def compress_folders_with_image_and_cloud_las(root_folder, output_zip_folder):
    # Get the target folders
    target_folders = get_target_folders(root_folder)
    # Compress the target folders to zip files
    for folder in target_folders:
        folder_name = os.path.basename(folder)
        zip_file = os.path.join(output_zip_folder, folder_name + '.zip')
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the image, cloud_las and calib_extract subfolders to the zip file
            for subfolder in zip_subfolders:
                subfolder_path = os.path.join(folder, subfolder)
                for dirpath, dirnames, filenames in os.walk(subfolder_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        zipf.write(file_path, os.path.relpath(file_path, folder))
                        logger.info(f'Added {file_path} to {zip_file}')
            logger.info(f'Compressed {folder} to {zip_file}')


# Usage example
# Get the folder path from cmd line, and trim the whitespace
folder_path = input("Enter the folder path: ").strip()
# Get the parent folder of the folder path
parent_folder = os.path.dirname(folder_path)
# Get the directory name of the folder path
folder_name = os.path.basename(folder_path)
# Create a new folder with the name of the original folder, and append '_zip'
output_zip_folder = os.path.join(parent_folder, folder_name + '_zip')
if not os.path.exists(output_zip_folder):
    os.makedirs(output_zip_folder)
# rename the image_raw folder to image
rename_folder(folder_path, 'image_raw', 'image')
# compress the folders to zip files
compress_folders_with_image_and_cloud_las(folder_path, output_zip_folder)
logger.info('Compression complete')