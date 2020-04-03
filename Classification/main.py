# -*- coding: utf-8 -*-
#%% Packages

import numpy as np
import os, matplotlib
#matplotlib.use('Agg')
#from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from sklearn.utils import class_weight

#import matplotlib.pyplot as plt
from glob import glob
import cv2, random, argparse
import utils

# %% Command line arguements

parser = argparse.ArgumentParser(description='Framework for training and evaluation.')
parser.add_argument(
        '--dataset', '-d', 
        help="1 -- smear baseline, 2 -- smear pipeline, 3 -- LBC pipeline", 
        type=int, 
        choices=[1, 2, 3], 
        default=1)
parser.add_argument(
        '--architecture', '-a', 
        help="choose a network architecture", 
        choices=['ResNet50', 'DenseNet201'], 
        default='ResNet50')
parser.add_argument(
        '--pretrain', '-p', 
        help="use pre-trained weights on ImageNet", 
        type=int, 
        choices=[0, 1], 
        default=0)
parser.add_argument(
        '--fold', '-f', 
        help="Dataset 1&2: 3 folds; Dataset 3: 2 folds.", 
        type=int, 
        choices=[1, 2, 3], 
        default=1)
parser.add_argument(
        '--index', '-i', 
        help="index for multiple training to get STD", 
        type=int, 
#        choices=[1, 2, 3], 
        default=1)
parser.add_argument(
        '--mode', '-m',
        help="train or test", 
        choices=['train', 'test'], 
        default='train')
parser.add_argument(
        '--savefile', '-s', 
        help="if save results to csv files", 
        type=int, 
        choices=[0, 1], 
        default=0)
args = parser.parse_args()


# %%    Parameters
#args.dataset = 1
#args.architecture = 'ResNet50'
#args.pretrain = 1
#args.fold = 1
#args.index = 1
#args.mode = 'train'

DATASET = args.dataset
ARCHI_NAME = args.architecture
PRETRAIN = args.pretrain
FOLD = args.fold
INDEX = args.index
MODE = args.mode


# log dir
#if ARCHI_NAME == 'ResNet50':
#    PRETRAIN = 0
#    DIR_LOG = f"./logs/resScratch/fold{FOLD}/"
#elif ARCHI_NAME == 'DenseNet201':
#    if PRETRAIN == 0:
#        DIR_LOG = f"./logs/denseScratch/fold{FOLD}/"
#    else:
#        DIR_LOG = f"./logs/densePretrain/fold{FOLD}/"
        
DIR_LOG = f"./logs/dataset_{DATASET}/{ARCHI_NAME}_pre{PRETRAIN}/"
if not os.path.exists(DIR_LOG):
    os.makedirs(DIR_LOG)

WEIGHT_PATH = DIR_LOG + f"data{DATASET}_{ARCHI_NAME}_pre{PRETRAIN}_fold{FOLD}_{INDEX}.hdf5"

# training parameter
if ARCHI_NAME == 'ResNet50':
    if DATASET == 1:
        BATCH_SIZE = 128
        EPOCHS = 30
    else:
        BATCH_SIZE = 512
        EPOCHS = 50
elif ARCHI_NAME == 'DenseNet201':
    if DATASET == 1:
        BATCH_SIZE = 128
        EPOCHS = 20
    else:
        BATCH_SIZE = 256
        EPOCHS = 30
if PRETRAIN == 1:
    EPOCHS = 5

# data dir
if DATASET in [1, 2]:
    DIR_TRAIN_DATA = f"./Datasets/dataset{DATASET}/data_train{FOLD}/"
    DIR_TEST_DATA = f"./Datasets/dataset{DATASET}/data_test{FOLD}/"
elif DATASET == 3:
    if FOLD == 1:
        DIR_TRAIN_DATA = f"./Datasets/dataset{DATASET}/train/"
        DIR_TEST_DATA = f"./Datasets/dataset{DATASET}/test/"
    elif FOLD == 2:
        DIR_TRAIN_DATA = f"./Datasets/dataset{DATASET}/test/"
        DIR_TEST_DATA = f"./Datasets/dataset{DATASET}/train/"
    else:
        raise ValueError("FOLD must be in [1, 2] for Dataset 3.")
    
    
#MODEL_PATH = DIR_LOG + "ResNet_aug.h5"
if PRETRAIN == 1 and DATASET == 1:
    IMG_SHAPE = (80, 80, 1)
    SAMPLE_SHAPE = (80, 80, 1)
else:
    IMG_SHAPE = (80, 80, 3)
    SAMPLE_SHAPE = (80, 80, 3)


# %% Load  data

if MODE == 'train':
    X_train, Y_train = utils.load_set(DIR_TRAIN_DATA, IMG_SHAPE, SAMPLE_SHAPE)
(X_test, Y_test, 
 indices, index_slide, 
 slides_cls0, slides_cls1) = utils.load_set(
         DIR_TEST_DATA, IMG_SHAPE, SAMPLE_SHAPE, is_per_slide=True)

#%% Create the model
if ARCHI_NAME == 'ResNet50':
    model = utils.build_resnet(input_shape=SAMPLE_SHAPE, classes=2, pretrain=PRETRAIN)
elif ARCHI_NAME == 'DenseNet201':
    model = utils.build_densenet(input_shape=SAMPLE_SHAPE, classes=2, pretrain=PRETRAIN)

#%% Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#%% Train with augmentation

if MODE == 'train':

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=utils.aug_non_inter,
            validation_split=0.1) # set validation split
#    elif ARCHI_NAME == 'DenseNet201':
#        train_datagen = ImageDataGenerator(
#                rescale=1./255,
##                featurewise_center=True,
##                featurewise_std_normalization=True,
#                preprocessing_function=utils.aug_non_inter,
#                validation_split=0.1) # set validation split
        
    train_datagen.fit(X_train)
    
    train_generator = train_datagen.flow(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            subset='training') # set as training data
    
    class_weights = class_weight.compute_class_weight(
            'balanced',
            np.argmax(np.unique(Y_train, axis=0), axis=1),
            np.argmax(Y_train, axis=1))
    
    #class_weights = {0: 3.100251889168766, 1: 1.0}
    
    validation_generator = train_datagen.flow(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            subset='validation') # set as validation data
    
    #   Callbacks
    mc = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    if PRETRAIN == 0:
        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    else:
        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=0, verbose=1)
#    if ARCHI_NAME == 'ResNet50':
#        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
#    elif ARCHI_NAME == 'DenseNet201':
#        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=0, verbose=1)
    
    #   Training
    history = model.fit_generator(
            generator = train_generator,
    #        steps_per_epoch = len(train_generator),
            epochs = EPOCHS,
            verbose=1,
            class_weight = class_weights,
            validation_data = validation_generator, 
    #        validation_steps = len(validation_generator),
            callbacks=[mc, es, rp])


# %% Evaluate model
test_datagen = ImageDataGenerator(
#        featurewise_center=True,
#        featurewise_std_normalization=True,
        rescale=1./255)
#test_datagen.fit(X_test)

test_generator = test_datagen.flow(
        X_test, Y_test, 
        shuffle=False, 
        batch_size=BATCH_SIZE)

#   Restore the saved best model
model.load_weights(WEIGHT_PATH)

#   Confution Matrix and Classification Report
#test_generator.reset()
Y_pred = model.predict_generator(
        generator = test_generator, 
        steps=len(test_generator),
        verbose=1)
Y_pred = np.argmax(Y_pred, axis=1)

target_names = ['Cancer', 'Healthy']

dict_metrics = utils.evaluate(Y_test, Y_pred, target_names)
#utils.plot_confusion_matrix(metrics['cm'], target_names, normalize=True)
for metric in dict_metrics:
    print(dict_metrics[metric])
    
if args.savefile == 1:
    utils.write_results(dict_metrics, args)
    utils.write_per_slide_results(
            Y_test, Y_pred, 
            dict_metrics, args, 
            indices, index_slide, slides_cls0, slides_cls1)

# %% Save model
#model.save(MODEL_PATH)

#%% Plot learning curve

if MODE == 'train':
    utils.accuracy_curve(history, DIR_LOG)
#%%

