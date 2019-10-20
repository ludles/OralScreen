# -*- coding: utf-8 -*-

import numpy as np
import os
#from model import buildModel_U_net
import cv2
import time, csv
import random
from keras.models import load_model
from glob import glob
import sys
#%%
# Set some parameters
val_set = [os.path.basename(img_path)[:-len('.jpg')] for img_path in glob('./JPEGImages_testset/*')]

IMG_WIDTH = 1024
IMG_HEIGHT = 512
seed = 42
random.seed = seed
np.random.seed = seed


#%%

def load_test_data():
    
    def read_image(img_path, IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        return img
    
    X_test = np.array([read_image(img_path) for img_path in glob('./JPEGImages_testset/*')])
    val_data = (X_test - np.mean(X_test)) / np.std(X_test)
    return val_data

val_data = load_test_data()

#%% Creat the model

model = load_model('trained_model.h5')

#%% Detection
def detect(threshold):
    

    start = time.time()
    A = model.predict(val_data)
    print("\nConsumed time: \t%.2f\t s\n" % (time.time()-start))
    
    preds_test = np.where(A > 0, A / 100, A)
    preds_test = (preds_test + 1) / 2

    preds_test_t = (preds_test > threshold).astype(np.uint8)
    
    return preds_test_t

# Save predicted masks
def save_masks(data_set=val_set):
    """ Save predicted masks to directory
    data_set -- list of img_name
    """
    for i in range(len(data_set)):
        pred_mask = cv2.resize(np.squeeze(preds_test_t)[i], (6496,3360), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./PredMasks/{}_pred_mask.jpg".format(data_set[i]), 
                    pred_mask*255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    
    with open('./Results/ThreshldOpt.csv', 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold", "Precision", "Recall", "F1-score"])
        
    for threshold in np.linspace(0.5, 0.7, 21):
        preds_test_t = detect(threshold)
        save_masks(val_set)
        os.system('ImageJ --headless -macro particle_analysis.ijm')
        os.system('python3 test.py {}'.format(threshold))
