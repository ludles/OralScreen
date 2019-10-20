# -*- coding: utf-8 -*-
#%% Packages

import numpy as np
import math, os, matplotlib
matplotlib.use('Agg')
#from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout, AveragePooling2D, MaxPooling2D
from keras.models import Model
#from keras.preprocessing import image
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import preprocess_input
#import pydot
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.utils import class_weight

#import scipy.misc
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils import to_categorical
#from sklearn.model_selection import StratifiedKFold
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from glob import glob
import cv2, random
# %%    Parameters
SAMPLE_SHAPE = (80, 80, 1)
DIR_LOG = "./logs/"
if not os.path.exists(DIR_LOG):
    os.makedirs(DIR_LOG)
DIR_TRAIN_DATA = "./data_train/"
DIR_TEST_DATA = "./data_test/"
WEIGHT_PATH = DIR_LOG + "ResNet_aug.hdf5"
BATCH_SIZE = 512
EPOCHS = 100

# %%    Non-interpolation augmentation
def aug_non_inter(img):
    
    def ori(img):
        return img
    
    def fliph(img):
        return np.fliplr(img)
    
    def flipv(img):
        return np.flipud(img)
    
    def fliphv(img):
        return np.fliplr(np.flipud(img))
    
    def ori90(img):
        return np.rot90(img)
    
    def fliph90(img):
        return np.rot90(np.fliplr(img))
    
    def flipv90(img):
        return np.rot90(np.flipud(img))
    
    def fliphv90(img):
        return np.rot90(np.fliplr(np.flipud(img)))
    
    aug_functions = [ori, fliph, fliphv, fliphv, ori90, fliph90, fliphv90, fliphv90]   
    
    return random.choice(aug_functions)(img)

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
    
    cancer = np.array([read_sample(path) for path in glob(DIR_TEST_DATA + 'Cancer/*')])
    healthy = np.array([read_sample(path) for path in glob(DIR_TEST_DATA + 'Healthy/*')])
    X_test = np.concatenate((cancer, healthy), axis=0)
    
    Y_test = np.concatenate((np.zeros((len(cancer), 1)), np.ones((len(healthy), 1))), axis=0)
    Y_test = to_categorical(Y_test)
    
    # shuffle testing set
    index = [i for i in range(len(X_test))]
    np.random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]
    
    return X_test, Y_test

X_test, Y_test = load_testset()
    
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

#%% Confusion matrix plot

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#%% Train with augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        # rotation_range=180,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # zoom_range=0.1,
        # horizontal_flip=True,
        # vertical_flip=True,
        preprocessing_function=aug_non_inter,
        validation_split=0.1) # set validation split

train_generator = train_datagen.flow_from_directory(
        DIR_TRAIN_DATA,
        target_size=(SAMPLE_SHAPE[0], SAMPLE_SHAPE[1]),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training') # set as training data

class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes), 
        train_generator.classes)

#class_weights = {0: 3.100251889168766, 1: 1.0}

validation_generator = train_datagen.flow_from_directory(
        DIR_TRAIN_DATA, # same directory as training data
        target_size=(SAMPLE_SHAPE[0], SAMPLE_SHAPE[1]),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation') # set as validation data

#   Callbacks
mc = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

#   Training
history = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE),
        epochs = EPOCHS,
        verbose=0,
        class_weight = class_weights,
        validation_data = validation_generator, 
        validation_steps = math.ceil(validation_generator.samples / BATCH_SIZE),
        callbacks=[mc, es, rp])


# %% Evaluate model
test_datagen = ImageDataGenerator(rescale=1./255)

# =============================================================================
# test_generator = test_datagen.flow_from_directory(
#         DIR_TEST_DATA,
#         shuffle=False,
#         target_size=(SAMPLE_SHAPE[0], SAMPLE_SHAPE[1]),
#         color_mode='grayscale',
#         batch_size=BATCH_SIZE,
#         class_mode='categorical')
# =============================================================================
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
        verbose=1)
Y_pred = np.argmax(Y_pred, axis=1)


target_names = ['Cancer', 'Healthy']

print('Classification Report: ResNet')
#print(classification_report(test_generator.classes, Y_pred, target_names=target_names, digits=5))
print(classification_report(np.argmax(Y_test, axis=1), Y_pred, target_names=target_names, digits=5))

print('Confusion Matrix')
#cm = confusion_matrix(test_generator.classes, Y_pred)
cm = confusion_matrix(np.argmax(Y_test, axis=1), Y_pred)
#plot_confusion_matrix(cm, target_names, normalize=True)
print(cm)

#%%

#%% Plot learning curve

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Validation')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Validation')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
#    plt.show()
    plt.savefig(DIR_LOG + 'learning_curve_ResNet.png', bbox_inches='tight', transparent=False)
    
accuracy_curve(history)
#%%

