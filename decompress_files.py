import os
import tarfile
import datetime
from loguru import logger

batch_num = 10

def get_all_files(folder_path, suffix):
    # Get all files in the folder with the specified suffix
    files = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(suffix):
                files.append(os.path.join(dirpath, filename))
    return files

def extract_tar_files(folder_path, destination_folder):
    # Get a list of all tar files in the folder
    tar_files = get_all_files(folder_path, '.tar')
    # Extract the tar files to the destination folder, with a batch of 10 files, and put them into a subfolder
    batch_size = batch_num
    for i in range(0, len(tar_files), batch_size):
        batch_files = tar_files[i:i+batch_size]
        subfolder = os.path.join(destination_folder, f'batch_{i//batch_size}')
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for file in batch_files:
            # Use tarfile module to extract tar files
            with tarfile.open(os.path.join(folder_path, file), 'r|*') as tar:
                # get exception info
                try: 
                    tar.extractall(subfolder)
                    logger.info(f'Extracted {file} to {subfolder}')
                except Exception as e:
                    logger.error(f'Error extracting {file}: {e}')

# Get the folder path from cmd line, and trim the whitespace
folder_path = input("Enter the folder path: ").strip()
# Get current date，with yyyyMMdd format
date = datetime.datetime.now().strftime('%Y%m%d')
# Get the parent folder of the folder path
parent_folder = os.path.dirname(folder_path)
# Get the directory name of the folder path
folder_name = os.path.basename(folder_path)
# Create a new folder with the name of the original folder, and append the current date
destination_folder = os.path.join(parent_folder, folder_name + '_extract_' + date)
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
# Extract the tar files
extract_tar_files(folder_path, destination_folder)
logger.info('Extraction complete')