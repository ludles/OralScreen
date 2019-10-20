# -*- coding: utf-8 -*-

import numpy as np
import os
from model import buildModel_U_net
from keras import backend as K
import cv2
import random
from glob import glob
from tqdm import tqdm
#%%
# Set some parameters
IMG_WIDTH = 1024
IMG_HEIGHT = 512
IMG_CHANNEL = 3
seed = 42
random.seed = seed
np.random.seed = seed

#%% Creat the model
print('#'*30)
print('Creating and compiling the fully convolutional regression networks.')
print('#'*30)    
   
model = buildModel_U_net(input_dim = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
model.load_weights('trained_model.hdf5')

print('#'*30)
print('Model loaded.')
print('#'*30)  
#%% Read image
def read_image(img_path):
    
    img_name = os.path.basename(img_path)[:-4]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = (img - np.mean(img)) / np.std(img)
    
    return img_name, img
    
#%% Detection
def detect(data, threshold=0.59):
#    start = time.time()
    A = model.predict(data)
#    print("\nConsumed time: \t%.2f\t s\n" % (time.time()-start))
    preds_test = np.where(A > 0, A / 100, A)
    preds_test = (preds_test + 1) / 2
    preds_test_t = (preds_test > threshold).astype(np.uint8)
    
    return preds_test_t

#%% Save predicted mask
def save_masks(data_set):
    """ Save predicted masks to directory
    data_set -- preds_test_t
    """
    pred_mask = cv2.resize(np.squeeze(data_set), (6496,3360), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("./PredMasks/{}_pred_mask.jpg".format(img_name), 
                pred_mask*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return

#%%
if __name__ == '__main__':
    #img_paths = glob('./JPEGImages/*.jpg')
    print('#'*30)
    print('Generating predicted masks.')
    print('#'*30)  
    # for each sub image in directory (only predict at z0)
    for img_path in tqdm(glob('./JPEGImages/*_x40_z0_*.jpg')):
        img_name, img = read_image(img_path)
        preds_test_t = detect(img, threshold=0.63)
        save_masks(preds_test_t)


