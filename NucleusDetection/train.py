# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:12:53 2017

@author: Weidi Xie

@Description: This is the file used for training, loading images, annotation, training with model.
"""

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

#    test_data = data_[12:]
    test_data = (X_test - np.mean(X_test)) / np.std(X_test)
    test_anno = anno[len(train_set):]

    return train_data, train_anno, test_data, test_anno


train_data, train_anno, test_data, test_anno = load_data()


#%% Creat the model
print('-'*30)
print('Creating and compiling the fully convolutional regression networks.')
print('-'*30)    
   
model = buildModel_U_net(input_dim = train_data.shape[1:])
model_checkpoint = ModelCheckpoint('cell_counting.hdf5', monitor='loss', save_best_only=True)
#model.summary()
print('...Fitting model...')
print('-'*30)
change_lr = LearningRateScheduler(step_decay)

datagen = ImageDataGenerator(
    featurewise_center = False,  # set input mean to 0 over the dataset
    samplewise_center = False,  # set each sample mean to 0
    featurewise_std_normalization = False,  # divide inputs by std of the dataset
    samplewise_std_normalization = False,  # divide each input by its std
    zca_whitening = False,  # apply ZCA whitening
    rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
    zoom_range = 0.3,
    shear_range = 0.,
    horizontal_flip = True,  # randomly flip images
    vertical_flip = True, # randomly flip images
    fill_mode = 'constant',
    dim_ordering = 'tf')  

#%% Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(train_data,
                                 train_anno,
                                 batch_size = 1
                                 ),
                    steps_per_epoch = train_data.shape[0],
                    epochs = 100,
                    callbacks = [model_checkpoint, change_lr],
                    initial_epoch=0)
#%% Detection
def detect(data=test_data, threshold=0.6):
    
    model.load_weights('trained_model.hdf5')
    start = time.time()
    A = model.predict(data)
    print("\nConsumed time: \t%.2f\t s\n" % (time.time()-start))
    #mean_diff = np.average(np.abs(np.sum(np.sum(A,1),1)-np.sum(np.sum(test_anno,1),1))) / (100.0)
    #print('After training, the difference is : {} cells per image.'.format(np.abs(mean_diff)))
    
    preds_test = np.where(A > 0, A / 100, A)
    preds_test = (preds_test + 1) / 2
    #preds_test = (A + 100) / 200
    #preds_test_t = (preds_test > 0.7).astype(np.uint8)
    preds_test_t = (preds_test > threshold).astype(np.uint8)
    
    return preds_test_t                                       

preds_test_t = detect(test_data, 0.59)

# Show the results
#imshow(np.squeeze(preds_test_t)[1])
#imshow(np.squeeze(Y_test)[1])

#%% Save predicted masks
def save_masks(data_set=val_set):
    """ Save predicted masks to directory
    data_set -- list of img_name
    """
    for i in range(len(data_set)):
        pred_mask = cv2.resize(np.squeeze(preds_test_t)[i], (6496,3360), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./PredMasks/{}_pred_mask.jpg".format(data_set[i]), 
                    pred_mask*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

save_masks(val_set)


#%% Save sample Pmap

Pmap = cv2.resize(preds_test[0,:,:,0], (6496,3360), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("{}_PMap.jpg".format(val_set[0]), Pmap*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

Pmap_color =  cv2.imread("{}_PMap.jpg".format(val_set[0]), cv2.IMREAD_GRAYSCALE)
Pmap_color = cv2.applyColorMap(Pmap_color, cv2.COLORMAP_JET)
cv2.imwrite("{}_PMap_color.jpg".format(val_set[0]), Pmap_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#%%
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = Axes3D(fig)
#X, Y = np.meshgrid(np.arange(Pmap.shape[1]), np.arange(Pmap.shape[0]))
#ax.plot_surface(X, Y, Pmap, rstride=10, cstride=10, cmap='rainbow')
#plt.show()