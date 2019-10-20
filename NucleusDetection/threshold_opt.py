# -*- coding: utf-8 -*-

import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from model import buildModel_U_net
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc
import scipy.ndimage as ndimage
from skimage.io import imread, imshow
import cv2
import time
import random

#%%
# Set some parameters
file_train = "../VOC2007_old/2007_train.txt"
val_set = ['000024', '000057']

IMG_WIDTH = 1024
IMG_HEIGHT = 512
seed = 42
random.seed = seed
np.random.seed = seed

def read_image(line_in_file, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT):
    img_path = line_in_file.split()[0]
    img = cv2.imread(img_path)
#    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    return img

def read_mask(line_in_file, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT, resize=True):
    img_path = line_in_file.split()[0]
    img_name = img_path.split('/')[-1].split('.')[0]
    if resize == True:
        mask_path = img_path[:-10] + '../Masks/disk/' + img_name + '_mask.jpg'
        mask = cv2.imread(mask_path, 0)
#    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        mask = cv2.resize(mask, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    elif resize == False:
        mask_path = img_path[:-10] + '../Masks/point/' + img_name + '_mask.jpg'
        mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def is_val(line_in_file):
    img_path = line_in_file.split()[0]
    img_name = img_path.split('/')[-1].split('.')[0]
    return (img_name in val_set)

def step_decay(epoch):
    step = 16
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)

def train_set():
    train_set = []
    with open(file_train, 'r') as f:
        for line in f.readlines():
            img_path = line.split()[0]
            img_name = img_path.split('/')[-1].split('.')[0]
            if img_name not in val_set:
                train_set.append(img_name)
    return train_set

train_set = train_set()

#%%
# load data
def load_data():
    with open(file_train, 'r') as f:
        X_train = np.array([read_image(line) for line in f.readlines() if not is_val(line)])
        
    with open(file_train, 'r') as f:
        X_test = np.array([read_image(line) for line in f.readlines() if is_val(line)])
    
    with open(file_train, 'r') as f:
        Y_train = np.array([read_mask(line) for line in f.readlines() if not is_val(line)])
    Y_train = np.where(Y_train == 255, True, False)
    
    with open(file_train, 'r') as f:
        Y_test = np.array([read_mask(line) for line in f.readlines() if is_val(line)])
    Y_test = np.where(Y_test == 255, True, False)

#    data = np.concatenate((X_train, X_test))
    anno = np.concatenate((Y_train, Y_test))
    anno = 100.0 * (anno > 0)
    anno = [ndimage.gaussian_filter(np.squeeze(anno[i]), sigma=(1, 1), order=0) for i in range(len(anno))]
    anno = np.asarray(anno, dtype = 'float32')
    anno = np.expand_dims(anno, axis=-1)
    
#    mean = np.mean(data)
#    std = np.std(data)
#    
#    data_ = (data - mean) / std
    
#    train_data = data_[:12]
    train_data = (X_train - np.mean(X_train)) / np.std(X_train)
    train_anno = anno[:len(train_set)]
    
#    val_data = data_[12:]
    val_data = (X_test - np.mean(X_test)) / np.std(X_test)
    val_anno = anno[len(train_set):]
    
    return train_data, train_anno, val_data, val_anno


train_data, train_anno, val_data, val_anno = load_data()


#%% Creat the model

model = buildModel_U_net(input_dim = train_data.shape[1:])

model.load_weights('trained_model.hdf5')
#%% Detection
def detect(data=val_data, threshold=0.6):
    

    start = time.time()
    A = model.predict(val_data)
    print("\nConsumed time: \t%.2f\t s\n" % (time.time()-start))
    #mean_diff = np.average(np.abs(np.sum(np.sum(A,1),1)-np.sum(np.sum(val_anno,1),1))) / (100.0)
    #print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
    
    preds_test = np.where(A > 0, A / 100, A)
    preds_test = (preds_test + 1) / 2
    #preds_test = (A + 100) / 200
    #preds_test_t = (preds_test > 0.7).astype(np.uint8)
    preds_test_t = (preds_test > threshold).astype(np.uint8)
    
    return preds_test_t

preds_test_t = detect(val_data, threshold=0.59)

# Show the results
#imshow(np.squeeze(preds_test_t)[1])
#imshow(np.squeeze(Y_test)[1])

# Save predicted masks
def save_masks(data_set=val_set):
    """ Save predicted masks to directory
    data_set -- list of img_name
    """
    for i in range(len(data_set)):
        pred_mask = cv2.resize(np.squeeze(preds_test_t)[i], (6496,3360), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./PredMasks/{}_pred_mask.jpg".format(data_set[i]), 
                    pred_mask*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

save_masks(val_set)