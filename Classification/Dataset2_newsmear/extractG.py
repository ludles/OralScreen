# -*- coding: utf-8 -*-

import cv2, os
from glob import glob
from tqdm import tqdm

DIR_TRAIN_DATA = "./data_train/"
DIR_TRAIN_DATA_BACKUP = "./data_train_backup/"
DIR_TEST_DATA = "./data_test/"
DIR_TEST_DATA_BACKUP = "./data_test_backup/"
if not os.path.exists(DIR_TRAIN_DATA_BACKUP):
    os.rename(DIR_TRAIN_DATA, DIR_TRAIN_DATA_BACKUP)
    os.makedirs(DIR_TRAIN_DATA + 'Cancer/')
    os.makedirs(DIR_TRAIN_DATA + 'Healthy/')
if not os.path.exists(DIR_TEST_DATA_BACKUP):
    os.rename(DIR_TEST_DATA, DIR_TEST_DATA_BACKUP)
    os.makedirs(DIR_TEST_DATA + 'Cancer/')
    os.makedirs(DIR_TEST_DATA + 'Healthy/')
    
#dir_data = ["./data_train/", "./data_test/"]
#dir_data_backup = ["./data_train_backup/", "./data_test_backup/"]
#for i in range(len(dir_data)):
    
# %%
for img_path in tqdm(glob(DIR_TRAIN_DATA_BACKUP + 'Cancer/*')):
    img = cv2.imread(img_path)[:,:,1]
    cv2.imwrite(DIR_TRAIN_DATA + 'Cancer/' + os.path.basename(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print("Images in \'" + DIR_TRAIN_DATA + "Cancer/\' converted to G channel.")

for img_path in tqdm(glob(DIR_TRAIN_DATA_BACKUP + 'Healthy/*')):
    img = cv2.imread(img_path)[:,:,1]
    cv2.imwrite(DIR_TRAIN_DATA + 'Healthy/' + os.path.basename(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print("Images in \'" + DIR_TRAIN_DATA + "Healthy/\' converted to G channel.")

for img_path in tqdm(glob(DIR_TEST_DATA_BACKUP + 'Cancer/*')):
    img = cv2.imread(img_path)[:,:,1]
    cv2.imwrite(DIR_TEST_DATA + 'Cancer/' + os.path.basename(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print("Images in \'" + DIR_TEST_DATA + "Cancer/\' converted to G channel.")

for img_path in tqdm(glob(DIR_TEST_DATA_BACKUP + 'Healthy/*')):
    img = cv2.imread(img_path)[:,:,1]
    cv2.imwrite(DIR_TEST_DATA + 'Healthy/' + os.path.basename(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
print("Images in \'" + DIR_TEST_DATA + "Healthy/\' converted to G channel.")

