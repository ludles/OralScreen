# -*- coding: utf-8 -*-
#%% Packages

import numpy as np
import os, csv, cv2
#matplotlib.use('Agg')
#from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout, AveragePooling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils import to_categorical
from glob import glob
from tqdm import tqdm

SAMPLE_SHAPE = (80, 80, 1)
BATCH_SIZE = 128
EPOCHS = 30

#%% identity_block

def identity_block(X, f, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

#%% convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

#%% ResNet50

def ResNet50(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), name = "avg_pool")(X)

    # output layer
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

#%% Create the model
model = ResNet50(input_shape=SAMPLE_SHAPE, classes=2)

#%% Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% Load test data
def load_testset():
    
    def read_sample(img_path):
        if SAMPLE_SHAPE[-1] == 1:
            img = cv2.imread(img_path, 0)
        else:
            img = cv2.imread(img_path)
        if img.shape[:2] != SAMPLE_SHAPE[:2]:
            img = cv2.resize(img, SAMPLE_SHAPE[:2])
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img
    
# =============================================================================
#     if i_slide in slides_test0:
#         X_test = np.array([read_sample(path) for path in glob(dir_cancer + i_slide + '*.jpg')])
#         Y_test = np.concatenate((np.zeros((len(X_test), 1)), [[1.]]), axis=0)
#         Y_test = to_categorical(Y_test)[:-1]
#     elif i_slide in slides_test1:
#         X_test = np.array([read_sample(path) for path in glob(dir_healthy + i_slide + '*.jpg')])
#         Y_test = np.ones((len(X_test), 1))
#         Y_test = to_categorical(Y_test)
# =============================================================================
    
    
    cancer = np.array([read_sample(path) for path in glob(DIR_TEST_DATA + 'Cancer/*')])
    healthy = np.array([read_sample(path) for path in glob(DIR_TEST_DATA + 'Healthy/*')])
    X_test = np.concatenate((cancer, healthy), axis=0)
    
    Y_test = np.concatenate((np.zeros((len(cancer), 1)), np.ones((len(healthy), 1))), axis=0)
    Y_test = to_categorical(Y_test)

    sizes0 = [len(glob(dir_cancer + i_slide + '*.jpg')) for i_slide in slides_test0]
    sizes1 = [len(glob(dir_healthy + i_slide + '*.jpg')) for i_slide in slides_test1]
    sizes = sizes0 + sizes1
    
    indices = [i for i in range(sum(sizes))]
    
    index_slide = {slides_test[i]:indices[sum(sizes[:i]):sum(sizes[:i+1])] for i in range(len(slides_test))}
    
    
    np.random.shuffle(indices)
    X_test = X_test[indices]
    Y_test = Y_test[indices]    
   
    return X_test, Y_test, indices, index_slide

# %%    Save fold results into csv file
f = open('./fold_slide_results_data1.csv', 'a+', newline='')
writer = csv.writer(f)
header = ['Fold', 'Slide', 'Class', 'Accuracy', 'Percentage of cancer prediction']
writer.writerow(header)


for fold in tqdm(['1', '2', '3']):
        
    DIR_LOG = "./logs" + fold + '/'
    if not os.path.exists(DIR_LOG):
        os.makedirs(DIR_LOG)
    DIR_TEST_DATA = "./data_test" + fold + '/'
    WEIGHT_PATH = DIR_LOG + "ResNet_aug.hdf5"

    dir_cancer = DIR_TEST_DATA + 'Cancer/'
    dir_healthy = DIR_TEST_DATA + 'Healthy/'
    slides_test0 = list(set([os.path.basename(patch_path)[:2] for patch_path in glob(dir_cancer + '*.jpg')]))
    slides_test1 = list(set([os.path.basename(patch_path)[:2] for patch_path in glob(dir_healthy + '*.jpg')]))
    slides_test = slides_test0 + slides_test1
    
    
    X_test, Y_test, indices, index_slide = load_testset()
    
    
    # % Evaluate model
    #   Restore the saved best model
    model.load_weights(WEIGHT_PATH)
    
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow(
            X_test, Y_test, 
            shuffle=False, 
            batch_size=BATCH_SIZE)
    
    
    #   predict
    Y_pred = model.predict_generator(
            generator = test_generator, 
            verbose=1)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    
    #   per-glass performance
    for i_slide in slides_test:
        indices_slide = [indices.index(i) for i in index_slide[i_slide]]
        Y_true_slide = Y_true[indices_slide]
        Y_pred_slide = Y_pred[indices_slide]
        acc_slide = accuracy_score(Y_true_slide, Y_pred_slide)
        if i_slide in slides_test0:
            slide_class = 'Cancer\t'
            perc_cancer = acc_slide
        elif i_slide in slides_test1:
            slide_class = 'Healthy'
            perc_cancer = 1 - acc_slide
#        print("{}\t {}\t {:.5f}\t {:.5f}".format(i_slide, slide_class, acc_slide, perc_cancer))
        writer.writerow([fold, i_slide, slide_class, acc_slide, perc_cancer])
   

f.close()
#%%
