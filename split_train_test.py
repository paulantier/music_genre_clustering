import os
import shutil
import random

def split_dataset(source_dir, train_dir, test_dir, split_ratio=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Create category directories in train and test folders
        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)
        
        if not os.path.exists(train_category_path):
            os.makedirs(train_category_path)
        
        if not os.path.exists(test_category_path):
            os.makedirs(test_category_path)
        
        # Get all files in the current category directory
        files = os.listdir(category_path)
        files = [f for f in files if os.path.isfile(os.path.join(category_path, f))]
        
        # Shuffle files to ensure random splitting
        random.shuffle(files)
        
        split_index = int(len(files) * split_ratio)
        test_files = files[:split_index]
        train_files = files[split_index:]

        # Copy files to the train and test directories
        for file_name in train_files:
            shutil.copy(os.path.join(category_path, file_name), os.path.join(train_category_path, file_name))
        
        for file_name in test_files:
            shutil.copy(os.path.join(category_path, file_name), os.path.join(test_category_path, file_name))

# Paths to the source, train, and test directories
source_directory = "MELs"
train_directory = "MELs_train"
test_directory = "MELs_test"

# Split the dataset
split_dataset(source_directory, train_directory, test_directory)
