# -*- coding: utf-8 -*-
"""
draw detection markers on a sample
"""

import numpy as np
import cv2, math, csv, random

#%%
# Set some parameters
file_train = "../VOC2007_old/2007_train.txt"
val_set = ['000024', '000057']

IMG_WIDTH = 1024
IMG_HEIGHT = 512
SEED = 42
random.seed = SEED
np.random.seed = SEED

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

# %%
def read_data(data_set='val'):
    
    boxes = []
    
    with open(file_train, 'r') as f:
        line = f.readlines()[1] 
        if is_val(line):
            boxes.extend(line.split()[1:])    # extract the bounding boxes

    # ground truth boxes
    boxes = [list(map(int, box.split(',')[:-1])) for box in boxes]   # convert to int
    # ground truth positions
    pos = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]
    
    # read results from csv file
    pred_pos = []
    
    img_name = val_set[0]
    with open('./Results/Results_' + img_name + '.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        _pred_pos = [[float(row[-2]), float(row[-1])] for row in f_csv]
        pred_pos.extend(_pred_pos)
    
    return pred_pos, pos

pred_pos, pos = read_data(data_set='val')
# %% draw crosses on detection

def mark_img(img, gts, preds, length, thickness):
    '''
    gts -- list of ground truth positions
    preds -- list of prediction positions
    '''
    
    def draw_cross(img, centre, colour):
        '''
        centre -- [x, y]
        length -- int
        thickness -- int
        colour -- (B, G, R)
        '''
        centre = [round(centre[0]), round(centre[1])]
        cv2.line(img, (centre[0], centre[1] - length), (centre[0], centre[1] + length), colour, thickness)
        cv2.line(img, (centre[0] - length, centre[1]), (centre[0] + length, centre[1]), colour, thickness)
        return
    
    def draw_cross_x(img, centre, colour):
        '''
        centre -- [x, y]
        length -- int
        thickness -- int
        colour -- (B, G, R)
        '''
        offset = length / math.sqrt(2)
        cv2.line(
                img, 
                (round(centre[0] - offset), round(centre[1] - offset)), 
                (round(centre[0] + offset), round(centre[1] + offset)), 
                colour, thickness)
        cv2.line(
                img, 
                (round(centre[0] - offset), round(centre[1] + offset)), 
                (round(centre[0] + offset), round(centre[1] - offset)), 
                colour, thickness)
        return
    
    for gt in gts:
        draw_cross(img, centre=gt, colour=(0, 255, 0))
    for pred in preds:
        draw_cross_x(img, centre=pred, colour=(255, 0, 0))
    
    return img


img = cv2.imread("./PredMasks/000024_pred_mask.jpg")
if img.ndim == 2:
    img = np.expand_dims(img, axis=-1)
img_marked = mark_img(img, pos, pred_pos, 10, 2)
cv2.imwrite('./images/000024_pred_mask_marked.jpg', img_marked, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

#%% 